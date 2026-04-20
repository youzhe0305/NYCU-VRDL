"""DINO: DETR with Improved DeNoising Anchor Boxes for digit detection (HW2).

Implements key DINO components on top of the existing DETR codebase:
  1. Multi-scale backbone features (4 levels)
  2. Deformable attention (pure PyTorch, no custom CUDA ops)
  3. Deformable Transformer encoder & decoder
  4. Contrastive DeNoising (CDN) training
  5. Mixed query selection (positional from encoder, content learnable)
  6. Look forward twice (iterative box refinement with gradient flow)
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


# ---------------------------------------------------------------------------
# Positional encoding (sine, per-level)
# ---------------------------------------------------------------------------

class PositionEmbeddingSine(nn.Module):
    """2-D sine/cosine positional encoding (DETR-style)."""

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, mask):
        """mask: [B, H, W] True = padding."""
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=mask.device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            [pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4
        ).flatten(3)
        pos_y = torch.stack(
            [pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos  # [B, d_model, H, W]


# ---------------------------------------------------------------------------
# Multi-scale backbone
# ---------------------------------------------------------------------------

class ResNet50MultiScaleBackbone(nn.Module):
    """ResNet-50 backbone returning multi-scale features.

    Args:
        pretrained: use ImageNet pretrained weights
        use_c3: if True, return [C3, C4, C5]; if False, return [C4, C5]
            C3 (stride 8) is very large and dominates compute. Dropping it
            saves ~75% of encoder tokens with minimal accuracy loss.
    """

    def __init__(self, pretrained=True, use_c3=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = resnet50(weights=weights)
        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
        )
        self.layer1 = base.layer1  # C2: stride 4,  256 ch
        self.layer2 = base.layer2  # C3: stride 8,  512 ch
        self.layer3 = base.layer3  # C4: stride 16, 1024 ch
        self.layer4 = base.layer4  # C5: stride 32, 2048 ch
        self.use_c3 = use_c3
        if use_c3:
            self.num_channels = [512, 1024, 2048]  # C3, C4, C5
        else:
            self.num_channels = [1024, 2048]  # C4, C5

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        if self.use_c3:
            return [c3, c4, c5]
        return [c4, c5]


# ---------------------------------------------------------------------------
# MLP head
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


# ---------------------------------------------------------------------------
# Pure-PyTorch Multi-Scale Deformable Attention
# ---------------------------------------------------------------------------

class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention (pure PyTorch, no custom CUDA ops).

    For each query, learns sampling offsets and attention weights across
    multiple feature levels and sampling points.
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.d_per_head = d_model // n_heads

        self.sampling_offsets = nn.Linear(
            d_model, n_heads * n_levels * n_points * 2
        )
        self.attention_weights = nn.Linear(
            d_model, n_heads * n_levels * n_points
        )
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        # Initialize offsets as a grid pattern
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = (
            grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        )  # [n_heads, 2]
        for i in range(self.n_points):
            grid_init_i = grid_init * (i + 1)
            # Expand for all levels
            grid_init_level = grid_init_i.unsqueeze(1).repeat(
                1, self.n_levels, 1
            )  # [n_heads, n_levels, 2]
            if i == 0:
                full_init = grid_init_level.unsqueeze(2)
            else:
                full_init = torch.cat(
                    [full_init, grid_init_level.unsqueeze(2)], dim=2
                )
        # full_init: [n_heads, n_levels, n_points, 2]
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(
                full_init.reshape(-1)
            )
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self, query, reference_points, input_flatten, input_spatial_shapes,
        input_level_start_index, input_padding_mask=None,
    ):
        """
        Args:
            query: [B, Len_q, d_model]
            reference_points: [B, Len_q, n_levels, 2] or [B, Len_q, n_levels, 4]
            input_flatten: [B, sum(Hi*Wi), d_model]
            input_spatial_shapes: [n_levels, 2] (H, W)
            input_level_start_index: [n_levels]
            input_padding_mask: [B, sum(Hi*Wi)]
        Returns:
            output: [B, Len_q, d_model]
        """
        B, Len_q, _ = query.shape
        B, Len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(
                input_padding_mask[:, :, None], float(0)
            )
        value = value.view(B, Len_in, self.n_heads, self.d_per_head)

        # Predict offsets and attention weights
        sampling_offsets = self.sampling_offsets(query).view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            B, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        # Compute sampling locations
        if reference_points.shape[-1] == 2:
            # reference_points: [B, Len_q, n_levels, 2] (normalized cx, cy)
            offset_normalizer = input_spatial_shapes.flip(-1)[None, None, None, :, None, :]
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer.to(query.device)
            )
        elif reference_points.shape[-1] == 4:
            # reference_points: [B, Len_q, n_levels, 4] (cx, cy, w, h)
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        # sampling_locations: [B, Len_q, n_heads, n_levels, n_points, 2]

        # ---- Vectorized sampling across all levels ----
        # Pad all levels to the max spatial size and batch into one grid_sample call
        max_H = int(input_spatial_shapes[:, 0].max())
        max_W = int(input_spatial_shapes[:, 1].max())

        # Build padded value maps: [B * n_heads * n_levels, d_per_head, max_H, max_W]
        value_padded = value.new_zeros(
            B, self.n_levels, max_H, max_W, self.n_heads, self.d_per_head
        )
        for lvl in range(self.n_levels):
            H_lvl, W_lvl = int(input_spatial_shapes[lvl, 0]), int(input_spatial_shapes[lvl, 1])
            start = int(input_level_start_index[lvl])
            end = start + H_lvl * W_lvl
            value_padded[:, lvl, :H_lvl, :W_lvl, :, :] = value[:, start:end, :, :].view(
                B, H_lvl, W_lvl, self.n_heads, self.d_per_head
            )

        # [B, n_levels, n_heads, d_per_head, max_H, max_W]
        value_padded = value_padded.permute(0, 1, 4, 5, 2, 3)
        # [B * n_levels * n_heads, d_per_head, max_H, max_W]
        value_padded = value_padded.reshape(
            B * self.n_levels * self.n_heads, self.d_per_head, max_H, max_W
        )

        # Rescale sampling locations from [0,1] (relative to each level) to padded coords
        # sampling_locations: [B, Len_q, n_heads, n_levels, n_points, 2]
        # Scale from normalized [0,1] to padded [0,1] range: loc * (level_size / padded_size)
        scale_factors = input_spatial_shapes.float().flip(-1).to(query.device)  # [n_levels, 2] (W, H)
        padded_size = torch.tensor([max_W, max_H], dtype=torch.float32, device=query.device)
        # [1, 1, 1, n_levels, 1, 2]
        loc_scale = (scale_factors / padded_size)[None, None, None, :, None, :]
        scaled_locations = sampling_locations * loc_scale

        # Convert to grid_sample coords [-1, 1]
        grid = scaled_locations * 2 - 1  # [B, Len_q, n_heads, n_levels, n_points, 2]
        # Reshape: [B, n_levels, n_heads, Len_q, n_points, 2]
        grid = grid.permute(0, 3, 2, 1, 4, 5)
        # [B * n_levels * n_heads, Len_q, n_points, 2]
        grid = grid.reshape(B * self.n_levels * self.n_heads, Len_q, self.n_points, 2)

        # Single batched grid_sample call
        sampled = F.grid_sample(
            value_padded, grid,
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )  # [B * n_levels * n_heads, d_per_head, Len_q, n_points]

        # Reshape: [B, n_levels, n_heads, d_per_head, Len_q, n_points]
        sampled = sampled.view(
            B, self.n_levels, self.n_heads, self.d_per_head, Len_q, self.n_points
        )

        # Apply attention weights: [B, Len_q, n_heads, n_levels, n_points]
        # → [B, n_levels, n_heads, 1, Len_q, n_points]
        w = attention_weights.permute(0, 3, 2, 1, 4).unsqueeze(3)
        # Weighted sum over levels and points
        output = (sampled * w).sum(dim=(1, 5))  # [B, n_heads, d_per_head, Len_q]
        output = output.permute(0, 3, 1, 2).reshape(B, Len_q, self.d_model)

        return self.output_proj(output)


# ---------------------------------------------------------------------------
# Deformable Transformer Encoder Layer
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Adaptive Feature Fusion (AFF)
# ---------------------------------------------------------------------------

class AdaptiveFeatureFusion(nn.Module):
    """Adaptive Feature Fusion (AFF) module.

    For each feature level, computes a learned spatial attention map and
    applies it element-wise to dynamically weight feature importance at
    each position.  This replaces the equal-weight treatment of all levels
    during multi-scale feature fusion.

        A_l = sigma(Conv_l(X_l))          # [B, 1, H_l, W_l]
        Y_l = A_l * X_l                   # broadcast over C

    Reference: "Small Object Detection by DETR via Information Augmentation
    and Adaptive Feature Fusion" (Huang & Wang, 2024), Section 3.3.
    """

    def __init__(self, d_model, n_levels, kernel_size=3):
        super().__init__()
        self.attn_convs = nn.ModuleList([
            nn.Conv2d(d_model, 1, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=True)
            for _ in range(n_levels)
        ])
        self._reset_parameters()

    def _reset_parameters(self):
        for conv in self.attn_convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0.0)

    def forward(self, srcs):
        """
        Args:
            srcs: list of [B, d_model, H_l, W_l] for each level
        Returns:
            list of [B, d_model, H_l, W_l] with adaptive attention applied
        """
        out = []
        for lvl, src in enumerate(srcs):
            attn = torch.sigmoid(self.attn_convs[lvl](src))  # [B, 1, H, W]
            out.append(src * attn)
        return out


# ---------------------------------------------------------------------------
# Deformable Transformer Encoder Layer
# ---------------------------------------------------------------------------

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, d_ffn=2048, dropout=0.1,
                 n_levels=4, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, pos, reference_points, spatial_shapes,
                level_start_index, padding_mask=None):
        # Self-attention (deformable)
        src2 = self.self_attn(
            src + pos, reference_points, src, spatial_shapes,
            level_start_index, padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # FFN
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


# ---------------------------------------------------------------------------
# Deformable Transformer Decoder Layer
# ---------------------------------------------------------------------------

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, d_ffn=2048, dropout=0.1,
                 n_levels=4, n_points=4):
        super().__init__()
        # Self-attention (standard multi-head)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention (deformable)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, query_pos, reference_points, memory,
                spatial_shapes, level_start_index, memory_padding_mask=None,
                self_attn_mask=None):
        # Self-attention
        q = k = tgt + query_pos
        tgt2, _ = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention (deformable)
        tgt2 = self.cross_attn(
            tgt + query_pos, reference_points, memory, spatial_shapes,
            level_start_index, memory_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt


# ---------------------------------------------------------------------------
# Deformable Transformer (Encoder + Decoder)
# ---------------------------------------------------------------------------

def _get_encoder_reference_points(spatial_shapes, valid_ratios, device):
    """Generate reference points for the encoder on a regular grid."""
    reference_points_list = []
    for lvl, (H, W) in enumerate(spatial_shapes):
        H, W = int(H), int(W)
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            indexing="ij",
        )
        ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
        ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
        ref = torch.stack((ref_x, ref_y), dim=-1)  # [B, H*W, 2]
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, dim=1)  # [B, sum(H*W), 2]
    # Expand for all levels
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    # [B, sum(H*W), n_levels, 2]
    return reference_points


def gen_sineembed_for_position(pos_tensor, d_model=256):
    """Generate sine positional embedding from [B, N, 2] or [B, N, 4] coords."""
    half = d_model // 2
    dim_t = torch.arange(half, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000.0 ** (2 * (dim_t // 2) / half)

    x_embed = pos_tensor[:, :, 0:1] * 2 * math.pi  # [B, N, 1]
    y_embed = pos_tensor[:, :, 1:2] * 2 * math.pi

    pos_x = x_embed / dim_t  # [B, N, half]
    pos_y = y_embed / dim_t
    pos_x = torch.stack([pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=-1).flatten(-2)
    pos_y = torch.stack([pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=-1).flatten(-2)

    if pos_tensor.shape[-1] == 2:
        pos = torch.cat([pos_x, pos_y], dim=-1)  # [B, N, d_model]
    elif pos_tensor.shape[-1] == 4:
        w_embed = pos_tensor[:, :, 2:3] * 2 * math.pi
        h_embed = pos_tensor[:, :, 3:4] * 2 * math.pi
        pos_w = w_embed / dim_t
        pos_h = h_embed / dim_t
        pos_w = torch.stack([pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()], dim=-1).flatten(-2)
        pos_h = torch.stack([pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()], dim=-1).flatten(-2)
        pos = torch.cat([pos_x, pos_y, pos_w, pos_h], dim=-1)  # [B, N, d_model*2]
    return pos


class DeformableTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        n_levels=4,
        n_points=4,
        num_queries=900,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.num_queries = num_queries

        # Encoder
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, n_heads, d_ffn, dropout, n_levels, n_points,
        )
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_encoder_layers)]
        )

        # Decoder
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model, n_heads, d_ffn, dropout, n_levels, n_points,
        )
        self.decoder_layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)]
        )
        self.num_decoder_layers = num_decoder_layers

        # Reference point head for decoder (maps 4D anchor → query_pos)
        self.ref_point_head = MLP(d_model * 2, d_model, d_model, 2)

        # Two-stage: encoder output → initial anchors
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        self.enc_score_head = nn.Linear(d_model, 1)  # objectness
        self.enc_bbox_head = MLP(d_model, d_model, 4, 3)

        # Level embedding
        self.level_embed = nn.Parameter(torch.Tensor(n_levels, d_model))
        nn.init.normal_(self.level_embed)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, srcs, masks, pos_embeds, query_embed, bbox_embed_layers,
        class_embed_layers, self_attn_mask=None, targets_for_dn=None,
        dn_meta=None,
    ):
        """
        Args:
            srcs: list of [B, d_model, Hi, Wi] for each level
            masks: list of [B, Hi, Wi] for each level
            pos_embeds: list of [B, d_model, Hi, Wi]
            query_embed: [num_queries, d_model] content queries
            bbox_embed_layers: nn.ModuleList of MLP for bbox per decoder layer
            class_embed_layers: nn.ModuleList of nn.Linear for class per layer
            self_attn_mask: [nq_total, nq_total] for DN masking
            targets_for_dn: DN query tensors (input_query_label, input_query_bbox)
            dn_meta: dict with dn metadata
        """
        # Flatten multi-scale features
        src_flatten = []
        mask_flatten = []
        pos_flatten = []
        spatial_shapes = []

        for lvl in range(self.n_levels):
            B, C, H, W = srcs[lvl].shape
            spatial_shapes.append((H, W))
            src_flatten.append(
                srcs[lvl].flatten(2).transpose(1, 2) + self.level_embed[lvl]
            )
            mask_flatten.append(masks[lvl].flatten(1))
            pos_flatten.append(pos_embeds[lvl].flatten(2).transpose(1, 2))

        src_flatten = torch.cat(src_flatten, dim=1)   # [B, sum(Hi*Wi), d_model]
        mask_flatten = torch.cat(mask_flatten, dim=1)  # [B, sum(Hi*Wi)]
        pos_flatten = torch.cat(pos_flatten, dim=1)    # [B, sum(Hi*Wi), d_model]

        spatial_shapes = torch.tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat([
            spatial_shapes.new_zeros(1),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ])

        # Valid ratios (for handling padding)
        valid_ratios = []
        for lvl in range(self.n_levels):
            H, W = spatial_shapes[lvl]
            valid_H = (~masks[lvl])[:, :, 0].sum(1).float()  # [B]
            valid_W = (~masks[lvl])[:, 0, :].sum(1).float()  # [B]
            valid_ratios.append(
                torch.stack([valid_W / W, valid_H / H], dim=-1)
            )
        valid_ratios = torch.stack(valid_ratios, dim=1)  # [B, n_levels, 2]

        # ---- Encoder ----
        enc_ref_points = _get_encoder_reference_points(
            spatial_shapes, valid_ratios, src_flatten.device,
        )

        memory = src_flatten
        for layer in self.encoder_layers:
            memory = layer(
                memory, pos_flatten, enc_ref_points, spatial_shapes,
                level_start_index, mask_flatten,
            )

        # ---- Two-stage: select top-K encoder features as initial anchors ----
        enc_output = self.enc_output_norm(self.enc_output(memory))
        enc_scores = self.enc_score_head(enc_output).squeeze(-1)  # [B, sum(H*W)]
        # Mask out padding
        enc_scores = enc_scores.masked_fill(mask_flatten, float("-inf"))

        topk = self.num_queries
        topk_scores, topk_idx = torch.topk(enc_scores, topk, dim=1)  # [B, topk]
        topk_feats = torch.gather(
            enc_output, 1, topk_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
        )

        # Predict initial anchor boxes from top-K features
        enc_bbox_pred = self.enc_bbox_head(topk_feats).sigmoid()  # [B, topk, 4]
        # Also store encoder class prediction for auxiliary loss
        enc_class_pred = class_embed_layers[-1](topk_feats)  # [B, topk, num_classes]

        # Mixed query selection: position from encoder, content learnable
        reference_points_init = enc_bbox_pred.detach()  # [B, topk, 4] (cx, cy, w, h)
        # Content queries are learnable (expanded per batch)
        tgt = query_embed.unsqueeze(0).expand(B, -1, -1)  # [B, topk, d_model]

        # ---- Prepend DN queries if training ----
        if dn_meta is not None and targets_for_dn is not None:
            dn_label_embed, dn_bbox = targets_for_dn
            # dn_label_embed: [B, dn_pad_size, d_model]
            # dn_bbox: [B, dn_pad_size, 4]
            tgt = torch.cat([dn_label_embed, tgt], dim=1)
            reference_points_init = torch.cat(
                [dn_bbox, reference_points_init], dim=1
            )

        # ---- Decoder ----
        # Convert reference points to un-sigmoided for refinement
        refpoints = torch.special.logit(reference_points_init.clamp(1e-4, 1 - 1e-4))

        intermediates_hs = []
        intermediates_ref = []

        for lid, layer in enumerate(self.decoder_layers):
            ref_sig = refpoints.sigmoid()  # [B, nq, 4]

            # Generate query positional embedding from reference points
            # For 4D refs: output is [B, nq, d_model*2] (d_model/2 per coordinate × 4)
            query_sine_embed = gen_sineembed_for_position(ref_sig, self.d_model)
            # query_sine_embed: [B, nq, d_model*2] -> project to d_model
            query_pos = self.ref_point_head(query_sine_embed)

            # Reference points for deformable cross-attention: [B, nq, n_levels, 4]
            ref_for_attn = ref_sig[:, :, None, :].expand(
                -1, -1, self.n_levels, -1
            )

            tgt = layer(
                tgt, query_pos, ref_for_attn, memory,
                spatial_shapes, level_start_index, mask_flatten,
                self_attn_mask=self_attn_mask,
            )

            # Predict bbox offset and refine (Look Forward Twice)
            delta_bbox = bbox_embed_layers[lid](tgt)  # [B, nq, 4]

            # b'_i = Update(b_{i-1}, delta_b_i)  -- undetached
            new_ref_undetached = (
                torch.special.logit(ref_sig.clamp(1e-4, 1 - 1e-4)) + delta_bbox
            ).sigmoid()

            # b_i = Detach(b'_i)  -- for next layer input
            refpoints = torch.special.logit(
                new_ref_undetached.detach().clamp(1e-4, 1 - 1e-4)
            )

            intermediates_hs.append(tgt)
            intermediates_ref.append(new_ref_undetached)

        # Stack: [num_dec_layers, B, nq, d_model], [num_dec_layers, B, nq, 4]
        hs_stack = torch.stack(intermediates_hs)
        ref_stack = torch.stack(intermediates_ref)

        return (
            hs_stack, ref_stack, memory,
            enc_class_pred, enc_bbox_pred,
        )


# ---------------------------------------------------------------------------
# Contrastive DeNoising (CDN)
# ---------------------------------------------------------------------------

def prepare_for_cdn(targets, num_classes, d_model, dn_number, label_noise_ratio,
                    box_noise_scale, label_embed, training):
    """Prepare contrastive denoising queries (vectorized).

    Returns:
        (dn_label_embed, dn_bbox): embedded labels and noised boxes
        attn_mask: [pad_size, pad_size] bool mask
        dn_meta: dict with metadata
    """
    if not training or dn_number <= 0:
        return None, None, None

    gt_labels_list = [t["labels"] for t in targets]
    gt_boxes_list = [t["boxes"] for t in targets]
    device = label_embed.weight.device

    max_gt = max(len(lbl) for lbl in gt_labels_list)
    if max_gt == 0:
        return None, None, None

    num_groups = dn_number
    single_pad = 2 * max_gt
    pad_size = single_pad * num_groups
    B = len(targets)

    # Pad GT labels and boxes to max_gt per image: [B, max_gt]
    gt_labels_padded = torch.zeros(B, max_gt, dtype=torch.long, device=device)
    gt_boxes_padded = torch.zeros(B, max_gt, 4, device=device)
    gt_valid = torch.zeros(B, max_gt, dtype=torch.bool, device=device)

    for b in range(B):
        M = len(gt_labels_list[b])
        if M > 0:
            gt_labels_padded[b, :M] = gt_labels_list[b].to(device) - 1  # 0-indexed
            gt_boxes_padded[b, :M] = gt_boxes_list[b].to(device)
            gt_valid[b, :M] = True

    # Expand for all groups: [B, num_groups, max_gt] → [B, num_groups * max_gt]
    gt_lbl_exp = gt_labels_padded[:, None, :].expand(B, num_groups, max_gt).reshape(B, -1)
    gt_box_exp = gt_boxes_padded[:, None, :, :].expand(B, num_groups, max_gt, 4).reshape(B, -1, 4)
    valid_exp = gt_valid[:, None, :].expand(B, num_groups, max_gt).reshape(B, -1)
    # [B, num_groups * max_gt]

    N = num_groups * max_gt  # total per positive/negative

    # --- Label noise (vectorized) ---
    pos_labels = gt_lbl_exp.clone()
    neg_labels = gt_lbl_exp.clone()
    # Random flip mask
    flip_mask_p = torch.rand(B, N, device=device) < (label_noise_ratio * 0.5)
    flip_mask_n = torch.rand(B, N, device=device) < (label_noise_ratio * 0.5)
    random_labels_p = torch.randint(0, num_classes, (B, N), device=device)
    random_labels_n = torch.randint(0, num_classes, (B, N), device=device)
    pos_labels = torch.where(flip_mask_p, random_labels_p, pos_labels)
    neg_labels = torch.where(flip_mask_n, random_labels_n, neg_labels)

    # --- Box noise (vectorized) ---
    w = gt_box_exp[:, :, 2:3]  # [B, N, 1]
    h = gt_box_exp[:, :, 3:4]  # [B, N, 1]
    box_scale = torch.cat([w * 0.5, h * 0.5, w, h], dim=-1)  # [B, N, 4]

    # Positive: small noise
    pos_noise = torch.randn(B, N, 4, device=device) * box_noise_scale
    pos_boxes = (gt_box_exp + pos_noise * box_scale).clamp(0, 1)

    # Negative: larger noise with offset to ensure distance
    neg_noise = torch.randn(B, N, 4, device=device) * (box_noise_scale * 2.0)
    # Add offset to cx, cy to push negatives further away
    offset = torch.ones(B, N, 2, device=device) * box_noise_scale
    offset = offset * torch.cat([w * 0.5, h * 0.5], dim=-1)
    sign = torch.sign(neg_noise[:, :, :2])
    sign[sign == 0] = 1.0
    neg_noise[:, :, :2] = neg_noise[:, :, :2] + sign * offset
    neg_boxes = (gt_box_exp + neg_noise * box_scale).clamp(0, 1)

    # --- Interleave positive and negative: [pos_0, neg_0, pos_1, neg_1, ...] ---
    # dn_labels: [B, pad_size], dn_boxes: [B, pad_size, 4]
    dn_labels = torch.full((B, pad_size), num_classes, dtype=torch.long, device=device)
    dn_boxes = torch.zeros(B, pad_size, 4, device=device)

    # Positive indices: 0, 2, 4, ... within each group
    # pos_labels/boxes are [B, num_groups * max_gt] laid out group-by-group
    for g in range(num_groups):
        base = g * single_pad
        src_start = g * max_gt
        src_end = src_start + max_gt
        pos_idx = torch.arange(max_gt, device=device) * 2 + base      # even slots
        neg_idx = torch.arange(max_gt, device=device) * 2 + 1 + base  # odd slots
        dn_labels[:, pos_idx] = pos_labels[:, src_start:src_end]
        dn_labels[:, neg_idx] = neg_labels[:, src_start:src_end]
        dn_boxes[:, pos_idx] = pos_boxes[:, src_start:src_end]
        dn_boxes[:, neg_idx] = neg_boxes[:, src_start:src_end]

    # Mark invalid (padded GT) entries back to num_classes
    valid_interleaved = torch.zeros(B, pad_size, dtype=torch.bool, device=device)
    for g in range(num_groups):
        base = g * single_pad
        src_start = g * max_gt
        src_end = src_start + max_gt
        pos_idx = torch.arange(max_gt, device=device) * 2 + base
        neg_idx = torch.arange(max_gt, device=device) * 2 + 1 + base
        valid_interleaved[:, pos_idx] = valid_exp[:, src_start:src_end]
        valid_interleaved[:, neg_idx] = valid_exp[:, src_start:src_end]
    dn_labels[~valid_interleaved] = num_classes

    # Embed labels
    dn_label_embed = label_embed(dn_labels)  # [B, pad_size, d_model]

    # Build attention mask (vectorized)
    attn_mask = torch.ones(pad_size, pad_size, dtype=torch.bool, device=device)
    for g in range(num_groups):
        start = g * single_pad
        end = (g + 1) * single_pad
        attn_mask[start:end, start:end] = False

    dn_meta = {
        "pad_size": pad_size,
        "num_dn_group": num_groups,
        "single_pad": single_pad,
        "max_gt": max_gt,
    }

    return (dn_label_embed, dn_boxes), attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_coord, dn_meta):
    """Split DN outputs from matching outputs.

    Returns:
        (matching_class, matching_coord, dn_class, dn_coord)
    """
    if dn_meta is None:
        return outputs_class, outputs_coord, None, None

    pad_size = dn_meta["pad_size"]
    # outputs_class: [num_layers, B, nq, C]
    # outputs_coord: [num_layers, B, nq, 4]
    dn_class = outputs_class[:, :, :pad_size, :]
    dn_coord = outputs_coord[:, :, :pad_size, :]
    matching_class = outputs_class[:, :, pad_size:, :]
    matching_coord = outputs_coord[:, :, pad_size:, :]

    return matching_class, matching_coord, dn_class, dn_coord


# ---------------------------------------------------------------------------
# DINO Model
# ---------------------------------------------------------------------------

class DINO(nn.Module):
    """DINO: DETR with Improved DeNoising Anchor Boxes."""

    def __init__(
        self,
        num_classes=10,
        num_queries=300,
        d_model=256,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        n_levels=4,
        n_points=4,
        pretrained_backbone=True,
        dn_number=100,
        dn_label_noise_ratio=0.5,
        dn_box_noise_scale=0.4,
        use_aff=False,
        aff_kernel_size=3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model
        self.n_levels = n_levels
        self.dn_number = dn_number
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_box_noise_scale = dn_box_noise_scale
        self.use_aff = use_aff

        # Backbone — drop C3 (stride 8) when n_levels <= 3 for speed
        use_c3 = (n_levels >= 4)
        self.backbone = ResNet50MultiScaleBackbone(
            pretrained=pretrained_backbone, use_c3=use_c3,
        )

        # Input projections (one per level)
        self.input_proj = nn.ModuleList()
        for ch in self.backbone.num_channels:  # [512, 1024, 2048]
            self.input_proj.append(nn.Sequential(
                nn.Conv2d(ch, d_model, kernel_size=1),
                nn.GroupNorm(32, d_model),
            ))
        # Extra level via stride-2 conv on C5
        if n_levels > len(self.backbone.num_channels):
            self.input_proj.append(nn.Sequential(
                nn.Conv2d(self.backbone.num_channels[-1], d_model,
                          kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, d_model),
            ))

        # Adaptive Feature Fusion (optional)
        if use_aff:
            self.aff = AdaptiveFeatureFusion(d_model, n_levels, aff_kernel_size)

        self.pos_encoding = PositionEmbeddingSine(num_pos_feats=d_model // 2)

        # Transformer
        self.transformer = DeformableTransformer(
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            n_levels=n_levels,
            n_points=n_points,
            num_queries=num_queries,
        )

        # Learnable content queries (mixed query selection: content is learnable)
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Label embedding for DN (num_classes + 1 for "no object" / padding)
        self.label_embed = nn.Embedding(num_classes + 1, d_model)

        # Prediction heads (shared across decoder layers for simplicity)
        # Per DINO: use separate heads per layer for look-forward-twice
        self.class_embed = nn.ModuleList([
            nn.Linear(d_model, num_classes) for _ in range(num_decoder_layers)
        ])
        self.bbox_embed = nn.ModuleList([
            MLP(d_model, d_model, 4, num_layers=3) for _ in range(num_decoder_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize class bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for cls_layer in self.class_embed:
            nn.init.constant_(cls_layer.bias, bias_value)
        for bbox_layer in self.bbox_embed:
            nn.init.constant_(bbox_layer.layers[-1].weight, 0.0)
            nn.init.constant_(bbox_layer.layers[-1].bias, 0.0)

    def forward(self, images, masks, targets=None):
        """
        Args:
            images: [B, 3, H, W]
            masks: [B, H, W] True = padding
            targets: list of dicts (only needed for DN during training)
        Returns:
            dict with pred_logits, pred_boxes, aux_outputs, etc.
        """
        # ---- Backbone ----
        features = self.backbone(images)  # [C3, C4, C5]

        srcs = []
        feat_masks = []
        pos_embeds = []

        for lvl, feat in enumerate(features):
            src = self.input_proj[lvl](feat)
            B, C, H, W = src.shape
            feat_mask = F.interpolate(
                masks.float().unsqueeze(1), size=(H, W)
            ).squeeze(1).bool()
            pos = self.pos_encoding(feat_mask)
            srcs.append(src)
            feat_masks.append(feat_mask)
            pos_embeds.append(pos)

        # Extra level (stride-2 on C5)
        if self.n_levels > len(features):
            src = self.input_proj[-1](features[-1])
            B, C, H, W = src.shape
            feat_mask = F.interpolate(
                masks.float().unsqueeze(1), size=(H, W)
            ).squeeze(1).bool()
            pos = self.pos_encoding(feat_mask)
            srcs.append(src)
            feat_masks.append(feat_mask)
            pos_embeds.append(pos)

        # ---- Adaptive Feature Fusion (AFF) ----
        if self.use_aff:
            srcs = self.aff(srcs)

        # ---- Contrastive DeNoising ----
        dn_args = None
        attn_mask = None
        dn_meta = None

        if self.training and targets is not None and self.dn_number > 0:
            dn_args, attn_mask_dn, dn_meta = prepare_for_cdn(
                targets, self.num_classes, self.d_model,
                self.dn_number, self.dn_label_noise_ratio,
                self.dn_box_noise_scale, self.label_embed, self.training,
            )
            if dn_meta is not None:
                pad_size = dn_meta["pad_size"]
                total_q = pad_size + self.num_queries
                attn_mask = torch.zeros(
                    total_q, total_q, dtype=torch.bool, device=images.device
                )
                # DN queries mask
                attn_mask[:pad_size, :pad_size] = attn_mask_dn
                # DN queries cannot attend to matching queries
                attn_mask[:pad_size, pad_size:] = True
                # Matching queries cannot attend to DN queries
                attn_mask[pad_size:, :pad_size] = True

        # ---- Transformer ----
        hs, ref, memory, enc_class, enc_bbox = self.transformer(
            srcs, feat_masks, pos_embeds,
            self.query_embed.weight,
            self.bbox_embed, self.class_embed,
            self_attn_mask=attn_mask,
            targets_for_dn=dn_args,
            dn_meta=dn_meta,
        )
        # hs: [num_dec_layers, B, nq, d_model]
        # ref: [num_dec_layers, B, nq, 4]

        # ---- Classify per layer (Look Forward Twice uses refined ref) ----
        outputs_classes = []
        outputs_coords = []
        for lid in range(self.transformer.num_decoder_layers):
            outputs_classes.append(self.class_embed[lid](hs[lid]))
            outputs_coords.append(ref[lid])  # already sigmoided from transformer

        outputs_class = torch.stack(outputs_classes)  # [L, B, nq, C]
        outputs_coord = torch.stack(outputs_coords)   # [L, B, nq, 4]

        # ---- Split DN and matching outputs ----
        (
            matching_class, matching_coord, dn_class, dn_coord,
        ) = dn_post_process(outputs_class, outputs_coord, dn_meta)

        # Final layer outputs
        out = {
            "pred_logits": matching_class[-1],  # [B, num_queries, C]
            "pred_boxes": matching_coord[-1],    # [B, num_queries, 4]
        }

        # Auxiliary outputs
        if self.training:
            out["aux_outputs"] = [
                {
                    "pred_logits": matching_class[i],
                    "pred_boxes": matching_coord[i],
                }
                for i in range(matching_class.shape[0] - 1)
            ]
            # DN outputs
            if dn_meta is not None:
                out["dn_meta"] = dn_meta
                out["dn_class"] = dn_class   # [L, B, pad_size, C]
                out["dn_coord"] = dn_coord   # [L, B, pad_size, 4]

            # Encoder auxiliary output
            out["enc_outputs"] = {
                "pred_logits": enc_class,
                "pred_boxes": enc_bbox,
            }

        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    num_classes=10,
    num_queries=300,
    d_model=256,
    n_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ffn=2048,
    dropout=0.1,
    n_levels=4,
    n_points=4,
    pretrained_backbone=True,
    dn_number=100,
    dn_label_noise_ratio=0.5,
    dn_box_noise_scale=0.4,
    use_aff=False,
    aff_kernel_size=3,
):
    model = DINO(
        num_classes=num_classes,
        num_queries=num_queries,
        d_model=d_model,
        n_heads=n_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ffn=d_ffn,
        dropout=dropout,
        n_levels=n_levels,
        n_points=n_points,
        pretrained_backbone=pretrained_backbone,
        dn_number=dn_number,
        dn_label_noise_ratio=dn_label_noise_ratio,
        dn_box_noise_scale=dn_box_noise_scale,
        use_aff=use_aff,
        aff_kernel_size=aff_kernel_size,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")
    return model
