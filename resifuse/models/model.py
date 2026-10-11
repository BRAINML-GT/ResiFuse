import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from resifuse.models.modules import MotionScout


def select_frames(r1, r2, budget_ratio=0.25, global_ratio=0.3):
    """
    Zero-parameter frame selection based on RVQ residuals only.

    Args:
        r1: Level-1 residual (coarse), [N, T]
        r2: Level-2 residual (total quantization error), [N, T]
        budget_ratio: fraction of frames to select
        global_ratio: fraction of selected frames for global path

    Returns:
        mask_global: [N, T] bool
        mask_local: [N, T] bool
    """
    N, T = r1.shape
    total_k = max(1, int(T * budget_ratio))

    # Select active frames by total quantization error (r2)
    _, top_indices = r2.topk(total_k, dim=-1)
    mask_active = torch.zeros(N, T, dtype=torch.bool, device=r1.device)
    mask_active.scatter_(1, top_indices, True)

    # Among active frames, route by r1 (coarse residual)
    k_global = max(1, int(total_k * global_ratio))
    r1_active = r1.clone()
    r1_active[~mask_active] = -float('inf')
    _, global_indices = r1_active.topk(k_global, dim=-1)
    mask_global = torch.zeros(N, T, dtype=torch.bool, device=r1.device)
    mask_global.scatter_(1, global_indices, True)

    mask_local = mask_active & (~mask_global)

    return mask_global, mask_local


class LearnableFrameSelector(nn.Module):
    """Per-frame scorer from the continuous motion-latent z_e.

    Input per frame:
        z_e[t]           pre-quantization motion latent (D_z = 128)

    Output per frame:
        score            scalar (higher = more important)

    Used by frame_select_mode="learnable" as a fair-information baseline
    against hand-designed selection. The hand-designed residual rule uses
    ||z_e - z_q|| as its selection criterion — a specific function of z_e
    (and z_q). This learnable selector receives z_e directly and is free to
    learn ANY scoring function over it, trained end-to-end with the
    classification loss via straight-through hard top-k.

    The aim is to test whether the residual ||z_e - z_q|| selection rule is
    the right hand-designed function over the motion latent space, or
    whether an end-to-end-trained MLP can find a better score function.
    Notably, the residual r₂ itself is NOT in the input, removing the
    "identity-function shortcut" critique of an [r1, r2, conf] design.
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_e: torch.Tensor) -> torch.Tensor:
        # z_e: [N, L, D_z]
        return self.mlp(z_e).squeeze(-1)   # [N, L]


def differentiable_topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Hard top-k mask in forward, sigmoid-surrogate gradient in backward.

    scores: [B, L] real-valued.
    k:      int, number of top-scoring positions to select per row.

    Returns: [B, L] tensor with values in {0, 1} in the forward pass, but
    a non-zero gradient w.r.t. ``scores`` via the standard straight-through
    trick (hard - soft.detach() + soft, where soft = sigmoid(score - kth)).
    """
    top_vals, top_idx = scores.topk(k, dim=-1)
    hard = torch.zeros_like(scores)
    hard.scatter_(-1, top_idx, 1.0)
    threshold = top_vals[..., -1:]                # k-th largest score per row
    soft = torch.sigmoid(scores - threshold)      # ~1 above threshold, ~0 below
    return hard + soft - soft.detach()


def select_frames_uniform(N, T, budget_ratio=0.25, global_ratio=0.3, device=None):
    """
    Uniform frame selection: evenly spaced frames, same interface as select_frames.
    """
    total_k = max(1, int(T * budget_ratio))
    # Evenly spaced indices
    indices = torch.linspace(0, T - 1, total_k, device=device).long()  # [total_k]
    mask_active = torch.zeros(N, T, dtype=torch.bool, device=device)
    mask_active[:, indices] = True

    k_global = max(1, int(total_k * global_ratio))
    # First k_global of the uniform selection → global, rest → local
    mask_global = torch.zeros(N, T, dtype=torch.bool, device=device)
    mask_global[:, indices[:k_global]] = True

    mask_local = mask_active & (~mask_global)

    return mask_global, mask_local


def collision_contrastive_loss(z_q_indices, v_features, labels, temperature=0.1):
    """
    Codebook collision contrastive loss.
    Same codebook entry + same label = positive, same entry + different label = negative.

    Args:
        z_q_indices: Level-1 codebook index, [N]
        v_features:  cross-attention output, [N, D_z]
        labels:      behavior labels, [N] (-1 = no label)
        temperature: temperature parameter
    Returns:
        loss: scalar
    """
    valid = labels >= 0
    if valid.sum() < 2:
        return torch.tensor(0.0, device=v_features.device)

    indices = z_q_indices[valid]
    feats = F.normalize(v_features[valid], dim=-1)
    labs = labels[valid]

    sim = feats @ feats.T / temperature  # [M, M]

    same_code = (indices.unsqueeze(0) == indices.unsqueeze(1))
    same_label = (labs.unsqueeze(0) == labs.unsqueeze(1))
    eye = torch.eye(len(feats), dtype=torch.bool, device=feats.device)

    positive_mask = same_code & same_label & (~eye)
    in_code_mask = same_code & (~eye)

    has_positive = positive_mask.sum(dim=-1) > 0
    has_negative = (same_code & (~same_label)).sum(dim=-1) > 0
    valid_anchors = has_positive & has_negative

    if valid_anchors.sum() == 0:
        return torch.tensor(0.0, device=v_features.device)

    # InfoNCE: denominator = logsumexp over in-code samples
    sim_masked = sim.clone()
    sim_masked[~in_code_mask] = -float('inf')
    log_denom = torch.logsumexp(sim_masked, dim=-1)  # [M]

    # Average positive similarity
    sim_pos = sim.clone()
    sim_pos[~positive_mask] = 0.0
    num_pos = positive_mask.float().sum(dim=-1).clamp(min=1)
    avg_pos_sim = sim_pos.sum(dim=-1) / num_pos

    per_anchor_loss = -(avg_pos_sim - log_denom)
    loss = per_anchor_loss[valid_anchors].mean()
    return loss


class MLPHead(nn.Module):
    """Three-layer MLP head with LayerNorm and dropout."""
    def __init__(self, dim_in: int, num_classes: int, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden_dim or dim_in
        self.net = nn.Sequential(
            # nn.LayerNorm(dim_in),

            nn.Linear(dim_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),

            # nn.Linear(dim_in, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalAttentionPool(nn.Module):
    """Attention pooling over time with a learnable query."""
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, D]
        scores = (x * self.query.view(1, 1, -1)).sum(dim=-1)  # [B, T]
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return torch.sum(x * attn.unsqueeze(-1), dim=1)


def _transformer_encoder_with_attn(encoder: nn.TransformerEncoder, x: torch.Tensor, key_padding_mask=None):
    attn_weights = None
    for i, layer in enumerate(encoder.layers):
        need_w = i == (len(encoder.layers) - 1)
        attn_out, attn_w = layer.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=need_w,
            average_attn_weights=False,
        )
        x = x + layer.dropout1(attn_out)
        x = layer.norm1(x)
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        x = layer.norm2(x + layer.dropout2(ff))
        if need_w:
            attn_weights = attn_w
    return x, attn_weights


class SocialTransformerAggregator(nn.Module):
    """Fuse per-mouse embeddings with a lightweight transformer and CLS token."""
    def __init__(self, dim: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, A, D]
        B = z.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, z], dim=1)
        out = self.encoder(seq)
        return out[:, 0]


class BiGRUEncoder(nn.Module):
    """Bidirectional GRU encoder with residual connection.

    output = x + proj(GRU(x)),  so initial output ≈ x when proj is near-zero.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim * 2, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(input_dim)
        self.output_dim = input_dim  # same as input for residual

        # Initialize proj near-zero so initial output ≈ input (residual starts silent)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B, T, D]
        pad_mask: [B, T] bool, True where padding should be ignored
        """
        if pad_mask is not None:
            lengths = (~pad_mask).sum(dim=1).clamp(min=1).to(torch.int64)
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.gru(packed)
            gru_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        else:
            gru_out, _ = self.gru(x)
        gru_out = self.dropout(gru_out)
        residual = self.proj(gru_out)  # [B, T, input_dim]
        out = x + residual             # residual connection
        if pad_mask is not None:
            out = out.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return self.out_norm(out)


class ResiFusePretrainer(nn.Module):
    """
    Multi-mouse pretrainer with simplified MotionScout (RVQ only), social attention,
    and vision prediction heads. All dimensions use D_z = vq_latent_dim (128).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.total_kps = config.num_joints  # per subject

        D_z = config.vq_latent_dim  # 128

        self.motion_scout = MotionScout(
            input_dim=config.motion_input_dim,
            vq_hidden_dim=config.vq_hidden_dim,
            vq_latent_dim=D_z,
            codebook1=config.vq_codebook_size,
            codebook2=config.vq_codebook_size2,
            num_stride_blocks=getattr(config, "vq_num_stride_blocks", 3),
            num_attn_layers=getattr(config, "vq_attn_layers", 1),
            attn_heads=getattr(config, "vq_attn_heads", 4),
            num_attn_layers_dec=getattr(config, "vq_attn_layers_dec", None) or getattr(config, "vq_attn_layers", 1),
            beta=config.vq_beta,
            dead_code_threshold=getattr(config, "vq_dead_code_threshold", 1.0),
            revival_warmup_steps=getattr(config, "vq_revival_warmup_steps", 100),
            l2_normalize=getattr(config, "vq_l2_normalize", True),
        )

        # Context transformer (replaces the one formerly inside MotionScout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_z,
            nhead=config.context_num_heads,
            dim_feedforward=D_z * 4,
            batch_first=True,
        )
        self.context_transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.context_num_layers)
        # Keep self.transformer as alias for checkpoint compatibility
        self.transformer = self.context_transformer
        self.mask_token = nn.Parameter(torch.randn(1, 1, D_z))
        self.motion_norm = nn.LayerNorm(D_z)

        # Keypoint prediction head (L_mask)
        self.head = nn.Linear(D_z, self.total_kps * 2)

        # Vision prediction heads (L_pred) — input dim is D_z
        target_dim = int(getattr(config, "vision_target_dim", None) or config.vision_dim)
        pred_hidden = getattr(config, "vision_pred_hidden_dim", D_z)
        pred_dropout = float(getattr(config, "vision_pred_dropout", 0.0) or 0.0)
        horizons = getattr(config, "vision_pred_horizons", None) or [8]
        self.pred_horizons = [int(h) for h in horizons]
        self.pred_heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.LayerNorm(D_z),
                nn.Linear(D_z, D_z),
                nn.GELU(),
                nn.Dropout(pred_dropout),
                nn.Linear(D_z, target_dim),
            )
            for h in self.pred_horizons
        })
        use_adapter = bool(getattr(config, "vision_target_use_adapter", False))
        if use_adapter:
            adapter_hidden = int(getattr(config, "vision_target_adapter_hidden", None) or target_dim)
            adapter_dropout = float(getattr(config, "vision_target_adapter_dropout", 0.0) or 0.0)
            self.target_adapter = nn.Sequential(
                nn.LayerNorm(config.vision_dim),
                nn.Linear(config.vision_dim, adapter_hidden),
                nn.GELU(),
                nn.Dropout(adapter_dropout),
                nn.Linear(adapter_hidden, target_dim),
            )
        if float(getattr(config, "vision_pred_weight", 0.0) or 0.0) <= 0.0:
            for param in self.pred_heads.parameters():
                param.requires_grad = False
            if use_adapter:
                for param in self.target_adapter.parameters():
                    param.requires_grad = False

        # Social encoder — dim D_z
        social_heads = int(getattr(config, "social_num_heads", 0) or 0)
        if social_heads <= 0:
            social_heads = config.num_heads
        social_layers = int(getattr(config, "social_num_layers", 1) or 1)
        social_dropout = float(getattr(config, "social_dropout", 0.1) or 0.1)
        social_layer = nn.TransformerEncoderLayer(
            d_model=D_z,
            nhead=social_heads,
            dim_feedforward=D_z * 4,
            dropout=social_dropout,
            batch_first=True,
        )
        self.social_encoder = nn.TransformerEncoder(social_layer, num_layers=social_layers)
        social_hidden = int(getattr(config, "social_fuse_hidden_dim", None) or D_z)
        self.social_fuse = nn.Sequential(
            nn.LayerNorm(D_z * 2),
            nn.Linear(D_z * 2, social_hidden),
            nn.GELU(),
            nn.Dropout(social_dropout),
            nn.Linear(social_hidden, D_z),
        )

        # ======== Visual residual predictor (pretrain vision for finetune) ========
        D_v = int(getattr(config, "vision_dim", 2048))
        vision_vres_weight = float(getattr(config, "vision_vres_weight", 0.0) or 0.0)
        if vision_vres_weight > 0.0:
            self.visual_residual_predictor = nn.Sequential(
                nn.Linear(D_v, D_z),     # 2048 -> 128  (→ finetune W_v)
                nn.GELU(),
                nn.Linear(D_z, D_z),     # 128 -> 128   (→ finetune mlp[0])
                nn.GELU(),
                nn.Linear(D_z, D_z),     # 128 -> 128   (→ finetune mlp[2])
            )

    def forward(
        self,
        motion_feats: torch.Tensor,
        keypoints: torch.Tensor,
        frames=None,
        mask: torch.Tensor | None = None,
        mask_indices: torch.Tensor | None = None,
        vision_feats: torch.Tensor | None = None,
    ):
        """
        motion_feats/keypoints: [B, A, T, ...]
        mask: [B, A, T] bool, invalid frames
        mask_indices: [B, A, T] optional M-MAE mask
        vision_feats: [B*A, T, D_v] — required only if vision_pred_weight>0 or vision_vres_weight>0
        """
        _vp_w = float(getattr(self.config, "vision_pred_weight", 0.0) or 0.0)
        _vr_w = float(getattr(self.config, "vision_vres_weight", 0.0) or 0.0)
        if (_vp_w > 0.0 or _vr_w > 0.0) and vision_feats is None:
            raise RuntimeError(
                "vision_feats is None but vision_pred_weight>0 or vision_vres_weight>0."
            )
        B, A, T = motion_feats.shape[:3]
        device = motion_feats.device
        valid_mask = mask.to(device) if mask is not None else None

        kps_flat = keypoints.reshape(B * A, T, keypoints.shape[3], 3)
        motion_flat = motion_feats.reshape(B * A, T, -1)

        # 1. RVQ encoding
        scout = self.motion_scout(motion_flat)
        z_q = scout["z_q"]            # [B*A, T, D_z]
        z_e = scout["z_e"]            # [B*A, T, D_z]
        vq_loss = scout["vq_loss"]
        vq_recon = scout["recon"]
        code_indices = scout["code_indices"]

        L = z_q.shape[1]
        stride = max(1, T // L)
        kp_ds = kps_flat[:, ::stride][:, :L]

        # 1b. Visual residual prediction: vision → quantization residual
        #     Teaches "given visual info, what did the RVQ miss?"
        vis_res_loss = z_q.new_tensor(0.0)
        vision_vres_weight = float(getattr(self.config, "vision_vres_weight", 0.0) or 0.0)
        v_raw = vision_feats
        if v_raw is not None and vision_vres_weight > 0.0:
            if v_raw.dim() < 3:
                raise RuntimeError(
                    f"vision_feats must have at least 3 dims, got shape {tuple(v_raw.shape)}"
                )
            if v_raw.shape[-2] != T:
                raise RuntimeError(
                    f"vision_feats time dim mismatch: expected T={T}, got shape {tuple(v_raw.shape)}"
                )
            lead = 1
            for dim in v_raw.shape[:-2]:
                lead *= int(dim)
            if lead != B * A:
                raise RuntimeError(
                    f"vision_feats batch/animal dims must flatten to {B * A}, got shape {tuple(v_raw.shape)}"
                )
            v_raw = v_raw.reshape(B * A, T, v_raw.shape[-1])
            v_residual_pred = self.visual_residual_predictor(v_raw)  # [B*A, T, D_z]
            target_residual = (z_e - z_q).detach()                  # [B*A, T, D_z]
            if v_residual_pred.shape != target_residual.shape:
                raise RuntimeError(
                    "visual residual shape mismatch: "
                    f"pred={tuple(v_residual_pred.shape)} target={tuple(target_residual.shape)}"
                )
            vis_res_loss = F.mse_loss(v_residual_pred, target_residual)

        # 2. Masked prediction: mask + transformer on z_q
        z_q_normed = self.motion_norm(z_q)

        mask_flat = mask_indices.reshape(B * A, T) if mask_indices is not None else None
        if valid_mask is not None:
            invalid_flat = (~valid_mask).reshape(B * A, T)
            if mask_flat is None:
                mask_flat = invalid_flat.to(dtype=motion_flat.dtype)
            else:
                mask_flat = (mask_flat.to(dtype=torch.bool) | invalid_flat).to(dtype=motion_flat.dtype)

        mask_ds = mask_flat[:, ::stride][:, :L] if mask_flat is not None else None

        if mask_ds is not None:
            mask_expanded = mask_ds.unsqueeze(-1).to(dtype=z_q_normed.dtype)
            z_q_masked = z_q_normed * (1 - mask_expanded) + self.mask_token * mask_expanded
        else:
            z_q_masked = z_q_normed
        context = self.context_transformer(z_q_masked)

        # 3. Social attention (multi-animal)
        context_bt = context.view(B, A, L, -1)
        key_padding_mask = None
        valid_ds = None
        if valid_mask is not None:
            valid_ds = valid_mask[:, :, ::stride][:, :, :L]
            valid_ds_f = valid_ds.to(dtype=context_bt.dtype)
            context_bt = context_bt * valid_ds_f.unsqueeze(-1)
            key_padding_mask = ~valid_ds.transpose(1, 2).reshape(B * L, A)

        social_in = context_bt.permute(0, 2, 1, 3).reshape(B * L, A, -1)
        all_masked = None
        if key_padding_mask is not None:
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                social_in = social_in.clone()
                social_in[all_masked] = 0
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked] = False
        social_out = self.social_encoder(social_in, src_key_padding_mask=key_padding_mask)
        social_out = social_out.reshape(B, L, A, -1).permute(0, 2, 1, 3)
        if valid_ds is not None:
            social_out = social_out * valid_ds.to(dtype=social_out.dtype).unsqueeze(-1)
        if all_masked is not None and all_masked.any():
            social_out = social_out.permute(0, 2, 1, 3).reshape(B * L, A, -1)
            social_out[all_masked] = 0
            social_out = social_out.reshape(B, L, A, -1).permute(0, 2, 1, 3)

        context_flat = context_bt.reshape(B * A, L, -1)
        social_flat = social_out.reshape(B * A, L, -1)
        fused = context_flat + self.social_fuse(torch.cat([context_flat, social_flat], dim=-1))
        if valid_ds is not None:
            fused = fused * valid_ds.to(dtype=fused.dtype).reshape(B * A, L, 1)

        # 4. Prediction heads
        pred_kps = self.head(fused).view(B * A, L, self.total_kps, 2)
        if _vp_w > 0.0:
            pred_vis = {k: head(fused) for k, head in self.pred_heads.items()}
        else:
            pred_vis = {}

        return {
            "pred_kps": pred_kps,
            "pred_vis": pred_vis,
            "vq_recon": vq_recon,
            "vq_loss": vq_loss,
            "vis_res_loss": vis_res_loss,
            "code_indices": code_indices,
            "mask_ds": mask_ds,
            "kp_ds": kp_ds,
            "stride": stride,
        }

    @torch.no_grad()
    def get_social_attn_weights(
        self,
        motion_feats: torch.Tensor,
        keypoints: torch.Tensor,
        mask: torch.Tensor | None = None,
        mask_indices: torch.Tensor | None = None,
    ):
        B, A, T = motion_feats.shape[:3]
        valid_mask = mask.to(motion_feats.device) if mask is not None else None

        motion_flat = motion_feats.reshape(B * A, T, -1)
        scout = self.motion_scout(motion_flat)
        z_q = scout["z_q"]
        L = z_q.shape[1]
        stride = max(1, T // L)

        context = self.context_transformer(self.motion_norm(z_q))
        context_bt = context.view(B, A, L, -1)
        key_padding_mask = None
        if valid_mask is not None:
            valid_ds = valid_mask[:, :, ::stride][:, :, :L]
            context_bt = context_bt * valid_ds.to(dtype=context_bt.dtype).unsqueeze(-1)
            key_padding_mask = ~valid_ds.transpose(1, 2).reshape(B * L, A)

        social_in = context_bt.permute(0, 2, 1, 3).reshape(B * L, A, -1)
        _, attn_w = _transformer_encoder_with_attn(self.social_encoder, social_in, key_padding_mask=key_padding_mask)
        return attn_w, key_padding_mask

    @torch.no_grad()
    def update_vq_ema(self):
        self.motion_scout.ema_update()


class ResiFuseClassifier(nn.Module):
    """
    Multi-mouse social classifier v2: simplified MotionScout (RVQ only),
    codebook-guided cross-attention, residual additive fusion.
    All dimensions use D_z = vq_latent_dim (128).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        D_z = config.vq_latent_dim  # 128
        D_v = config.vision_dim     # 2048
        input_dim = config.motion_input_dim

        # Hyperparameters for frame selection & collision loss
        self.budget_ratio = float(getattr(config, "budget_ratio", 0.25))
        self.global_ratio = float(getattr(config, "global_ratio", 0.3))
        self.frame_select_mode = str(getattr(config, "frame_select_mode", "residual")).lower()
        self.use_social_pipeline = bool(getattr(config, "use_social_pipeline", False))
        self.vision_fuse_mode = str(getattr(config, "vision_fuse_mode", "replace")).lower()  # "replace" | "add"

        # Optional: tiny learnable frame-scoring MLP, used only when
        # frame_select_mode == "learnable". Kept None otherwise so the rest
        # of the existing dispatch is unchanged. ~80 parameters.
        self.frame_selector = None
        if self.frame_select_mode == "learnable":
            selector_hidden = int(getattr(config, "learnable_selector_hidden", 16))
            self.frame_selector = LearnableFrameSelector(latent_dim=D_z, hidden_dim=selector_hidden)

        # ======== Frozen Motion Encoder ========
        self.motion_scout = MotionScout(
            input_dim=input_dim,
            vq_hidden_dim=config.vq_hidden_dim,
            vq_latent_dim=D_z,
            codebook1=config.vq_codebook_size,
            codebook2=config.vq_codebook_size2,
            num_stride_blocks=getattr(config, "vq_num_stride_blocks", 3),
            num_attn_layers=getattr(config, "vq_attn_layers", 1),
            attn_heads=getattr(config, "vq_attn_heads", 4),
            num_attn_layers_dec=getattr(config, "vq_attn_layers_dec", None) or getattr(config, "vq_attn_layers", 1),
            beta=config.vq_beta,
            dead_code_threshold=getattr(config, "vq_dead_code_threshold", 1.0),
            revival_warmup_steps=getattr(config, "vq_revival_warmup_steps", 100),
            l2_normalize=getattr(config, "vq_l2_normalize", True),
        )

        # ======== Context transformer ========
        self.motion_norm = nn.LayerNorm(D_z)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_z,
            nhead=config.context_num_heads,
            dim_feedforward=D_z * 4,
            batch_first=True,
        )
        self.context_transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.context_num_layers)

        # ======== Vision temporal encoder (Transformer on vision path only) ========
        vision_seq_heads = int(getattr(config, "vision_seq_heads", 4))
        vision_seq_layers = int(getattr(config, "vision_seq_layers", 2))
        vision_seq_dropout = float(getattr(config, "vision_seq_dropout", 0.1) or 0.0)
        if vision_seq_layers > 0:
            vision_seq_layer = nn.TransformerEncoderLayer(
                d_model=D_z,
                nhead=vision_seq_heads,
                dim_feedforward=D_z * 4,
                dropout=vision_seq_dropout,
                batch_first=True,
            )
            self.vision_seq_encoder = nn.TransformerEncoder(vision_seq_layer, num_layers=vision_seq_layers)
        else:
            self.vision_seq_encoder = None

        # ======== Motion temporal encoder (residual BiGRU for temporal smoothing) ========
        gru_hidden = int(getattr(config, "bigru_hidden_dim", 256) or 256)
        gru_layers = int(getattr(config, "bigru_layers", 1))
        gru_dropout = float(getattr(config, "bigru_dropout", 0.0) or 0.0)
        if gru_layers > 0:
            self.motion_seq_encoder = BiGRUEncoder(
                input_dim=D_z,
                hidden_dim=gru_hidden,
                num_layers=gru_layers,
                dropout=gru_dropout,
            )
        else:
            self.motion_seq_encoder = None

        # ======== Social encoder ========
        social_heads = int(getattr(config, "social_num_heads", 0) or 0)
        social_layers = int(getattr(config, "social_num_layers", 1) or 1)
        social_dropout = float(getattr(config, "social_dropout", 0.1) or 0.1)
        if social_heads <= 0:
            social_heads = config.num_heads
        social_layer = nn.TransformerEncoderLayer(
            d_model=D_z,
            nhead=social_heads,
            dim_feedforward=D_z * 4,
            dropout=social_dropout,
            batch_first=True,
        )
        self.social_encoder = nn.TransformerEncoder(social_layer, num_layers=social_layers)
        social_hidden = int(getattr(config, "social_fuse_hidden_dim", None) or D_z)
        self.social_fuse = nn.Sequential(
            nn.LayerNorm(D_z * 2),
            nn.Linear(D_z * 2, social_hidden),
            nn.GELU(),
            nn.Dropout(social_dropout),
            nn.Linear(social_hidden, D_z),
        )

        # ======== Classification heads ========
        head_hidden = getattr(config, "head_hidden_dim", D_z)
        head_dropout = getattr(config, "head_dropout", 0.1)
        # Animal aggregation mode: "mean" | "concat"
        self.animal_agg_mode = str(getattr(config, "animal_agg_mode", "mean")).lower()
        num_mice = int(getattr(config, "max_neighbors", 2)) + 1  # focal + neighbors
        motion_head_in = D_z * num_mice if self.animal_agg_mode == "concat" else D_z
        self.social_cls_head = MLPHead(motion_head_in, config.head_a_classes, head_hidden, head_dropout)
        # Vision head: takes projected+BiGRU-encoded vision features (D_z)
        vision_head_in = D_z * num_mice if self.animal_agg_mode == "concat" else D_z
        self.vision_cls_head = MLPHead(vision_head_in, config.head_a_classes, head_hidden, head_dropout)
        self.vision_alpha = float(getattr(config, "vision_alpha", 0.5))
        self.num_mice = num_mice

        # ======== Feature fusion path (used when vision_fuse_mode == "feature") ========
        self.vision_projector = nn.Sequential(
            nn.Linear(D_v, D_z),
            nn.LayerNorm(D_z),
        )
        self.motion_feat_norm = nn.LayerNorm(D_z)
        self.fused_feat_norm = nn.LayerNorm(D_z)

        # ======== Optional: mean-center the visual term before fusion ========
        # Unselected frames contribute exactly 0 to the fusion sum while selected
        # frames contribute a LayerNorm-scale vector, so the two groups end up
        # offset by the mean visual embedding — a constant direction that carries
        # no per-frame information but dominates any clustering of the fused
        # feature. Subtracting a running mean of the visual term (estimated over
        # selected frames only) removes that offset and keeps the frame-specific
        # part. Off by default so existing checkpoints/runs are unchanged.
        self.vision_center_mean = bool(getattr(config, "vision_center_mean", False))
        self.vision_center_momentum = float(getattr(config, "vision_center_momentum", 0.99) or 0.99)
        if self.vision_center_mean:
            self.register_buffer("vision_feat_mean", torch.zeros(D_z, dtype=torch.float32))
            self.register_buffer("vision_feat_mean_inited", torch.zeros((), dtype=torch.float32))
        # Unsup DEC hooks (train_entry_unsup_mabe._calibrate_vision_center):
        # when `_vision_center_accum` is a dict, forward accumulates the exact
        # masked sum/count of the visual term; `vision_center_frozen` stops the
        # EMA update so the fused feature space stays fixed under the centers.
        self.vision_center_frozen = False
        self._vision_center_accum = None

    def forward(self, motion_feats, keypoints, pad_mask=None,
                vision_patch_tokens=None, vision_feats=None,
                frame_select_mask=None):
        """
        motion_feats:         [B, A, T, D_in]
        keypoints:            [B, A, T, K, 3]
        pad_mask:             [B, A, T] bool, True = padding
        vision_patch_tokens:  [B, A, T, P, D_v] — for cross-attn (optional)
        vision_feats:         [B, A, T, D_v] — mean-pooled, for vision_cls_head on selected frames
        frame_select_mask:    [B, L] bool — externally computed mask (e.g. from motion uncertainty)
        """
        assert pad_mask is not None
        B, A, T = motion_feats.shape[:3]
        D_z = self.config.vq_latent_dim

        # ======== Stage 0: Frozen Motion Encoder ========
        motion_flat = motion_feats.view(B * A, T, -1)
        with torch.no_grad():
            scout = self.motion_scout(motion_flat)
            z_e = scout["z_e"]
            z_q = scout["z_q"]
            z_q1 = scout["z_q1"]
            z_q2 = scout["z_q2"]
            vq1_indices = scout["code_indices"][0]

        L = z_q.shape[1]

        # ======== Stage 1: When to Look — frame selection by RVQ residuals ========
        r1 = torch.norm(z_e - z_q1, dim=-1)  # [B*A, L]
        r2 = torch.norm(z_e - z_q, dim=-1)   # [B*A, L]

        device, dtype = z_q.device, z_q.dtype
        if self.budget_ratio > 0:
            if frame_select_mask is not None:
                # External mask (e.g. from motion uncertainty) — [B, L] bool
                # Expand to [B*A, L]: same mask for all animals
                mask_global = frame_select_mask.unsqueeze(1).expand(B, A, L).reshape(B * A, L)
                mask_local = torch.zeros_like(mask_global)
            elif self.frame_select_mode == "uniform":
                mask_global, mask_local = select_frames_uniform(
                    B * A, L,
                    budget_ratio=self.budget_ratio,
                    global_ratio=self.global_ratio,
                    device=device,
                )
            elif self.frame_select_mode == "recon_error":
                # Select frames with highest VQ-VAE reconstruction error
                with torch.no_grad():
                    recon = self.motion_scout.vqvae.decode(z_q, target_len=T)
                    recon_error = (recon - motion_flat).pow(2).mean(dim=-1)  # [B*A, T]
                    # Align to L
                    if recon_error.shape[1] != L:
                        stride_re = max(1, recon_error.shape[1] // L)
                        recon_error = recon_error[:, ::stride_re][:, :L]
                total_k = max(1, int(L * self.budget_ratio))
                _, top_idx = recon_error.topk(total_k, dim=-1)
                mask_global = torch.zeros(B * A, L, dtype=torch.bool, device=device)
                mask_global.scatter_(1, top_idx, True)
                mask_local = torch.zeros_like(mask_global)
            elif self.frame_select_mode == "surprise":
                # Select frames with highest temporal change in z_q
                with torch.no_grad():
                    # Compute frame-to-frame difference: [B*A, L-1]
                    diff = (z_q[:, 1:] - z_q[:, :-1]).norm(dim=-1)
                    # Pad first frame with 0 to keep length L
                    surprise = torch.cat([diff[:, :1] * 0, diff], dim=1)  # [B*A, L]
                total_k = max(1, int(L * self.budget_ratio))
                _, top_idx = surprise.topk(total_k, dim=-1)
                mask_global = torch.zeros(B * A, L, dtype=torch.bool, device=device)
                mask_global.scatter_(1, top_idx, True)
                mask_local = torch.zeros_like(mask_global)
            elif self.frame_select_mode == "learnable":
                # Tiny MLP scoring frames from the continuous motion latent
                # z_e (pre-quantization). Trained end-to-end with the
                # classification loss via straight-through hard top-k.
                # Mask comes out as a FLOAT tensor in {0, 1} carrying
                # gradient through the sigmoid surrogate; downstream code
                # handles bool/float uniformly via `_combine_masks`.
                if self.frame_selector is None:
                    raise RuntimeError(
                        "frame_select_mode='learnable' but self.frame_selector is None; "
                        "this should be set in __init__."
                    )
                # Score per (animal, frame) then average across animals so
                # the mask is shared by all A animals of a window — same
                # convention as residual / uniform / online modes.
                scores_flat = self.frame_selector(z_e)                 # [B*A, L]
                scores_b = scores_flat.view(B, A, L).mean(dim=1)       # [B, L]
                total_k = max(1, int(L * self.budget_ratio))
                soft_mask_b = differentiable_topk_mask(scores_b, total_k)  # [B, L] float
                # Broadcast to [B*A, L] (mask shared across animals).
                mask_global = soft_mask_b.unsqueeze(1).expand(B, A, L).reshape(B * A, L).contiguous()
                mask_local = torch.zeros_like(mask_global)             # float, same shape
            else:  # "residual" (default)
                mask_global, mask_local = select_frames(
                    r1, r2,
                    budget_ratio=self.budget_ratio,
                    global_ratio=self.global_ratio,
                )

        else:
            mask_global = torch.zeros(B * A, L, dtype=torch.bool, device=device)
            mask_local = torch.zeros(B * A, L, dtype=torch.bool, device=device)

        # Placeholders kept for output-dict backwards compatibility (downstream
        # code checks these keys with .get() on eval/logging paths).
        _dummy = z_q.new_zeros(())
        attn_global = None
        attn_local = None
        v_global = torch.empty(0, D_z, device=device, dtype=dtype)
        v_local = torch.empty(0, D_z, device=device, dtype=dtype)
        vis_aggregated = torch.zeros(B * A, L, D_z, device=device, dtype=dtype)

        # ======== Motion pipeline ========
        pad_mask_ds = pad_mask[:, :, :L] if pad_mask is not None else None
        pad_flat = pad_mask_ds.reshape(B * A, L) if pad_mask_ds is not None else None

        if self.use_social_pipeline:
            # Full pipeline: z_q → context_transformer → social_encoder → social_fuse → BiGRU
            context = self.context_transformer(self.motion_norm(z_q))
            context_bt = context.view(B, A, L, -1)
            key_padding_mask = None
            if pad_mask_ds is not None:
                context_bt = context_bt.masked_fill(pad_mask_ds.unsqueeze(-1), 0.0)
                key_padding_mask = pad_mask_ds.transpose(1, 2).reshape(B * L, A)
            social_in = context_bt.permute(0, 2, 1, 3).reshape(B * L, A, -1)
            social_out = self.social_encoder(social_in, src_key_padding_mask=key_padding_mask)
            social_out = social_out.reshape(B, L, A, -1).permute(0, 2, 1, 3)
            if pad_mask_ds is not None:
                social_out = social_out.masked_fill(pad_mask_ds.unsqueeze(-1), 0.0)
            context_flat = context_bt.reshape(B * A, L, -1)
            social_flat = social_out.reshape(B * A, L, -1)
            motion_features = context_flat + self.social_fuse(
                torch.cat([context_flat, social_flat], dim=-1)
            )
            _dummy_social = torch.tensor(0.0, device=device)
        else:
            # Simplified: z_q directly (skip context_transformer/social)
            motion_features = z_q
            # DDP: dummy forward so unused params get gradients
            _dummy_social = self.context_transformer.layers[0].self_attn.in_proj_weight.sum() * 0
            _dummy_social = _dummy_social + self.social_encoder.layers[0].self_attn.in_proj_weight.sum() * 0
            _dummy_social = _dummy_social + self.social_fuse[1].weight.sum() * 0

        # ======== Vision path: project → mask → Transformer temporal encoder ========
        vision_encoded = None  # [B*A, L, D_z] or None
        vision_needs_encode = (
            vision_feats is not None
            and self.budget_ratio > 0
            and self.vision_fuse_mode in ("feature", "add", "replace")
        )
        if vision_needs_encode:
            vf = vision_feats.to(dtype=motion_features.dtype)
            if vf.dim() == 4:
                vf = vf.view(B * A, T, -1)
            if vf.shape[1] != L:
                stride_v = max(1, vf.shape[1] // L)
                vf = vf[:, ::stride_v][:, :L]
            vision_proj = self.vision_projector(vf)  # [B*A, L, D_z]
            # Mask unselected frames (zero them out before / after transformer).
            # mask_global is bool for hand-designed modes (residual / uniform /
            # surprise / recon_error / online) and FLOAT in learnable mode (with
            # straight-through gradient). Combine accordingly.
            if mask_global.dtype == torch.bool:
                active_mask_ba = mask_global | mask_local                   # [B*A, L] bool
            else:
                active_mask_ba = (mask_global + mask_local).clamp(0.0, 1.0)  # float [0, 1] w/ grad
            if 0 < self.budget_ratio < 1.0:
                vision_proj = vision_proj * active_mask_ba.unsqueeze(-1).to(vision_proj.dtype)

            if self.vision_seq_encoder is not None:
                # Transformer self-attention with key_padding_mask to ignore unselected frames
                if 0 < self.budget_ratio < 1.0:
                    if active_mask_ba.dtype == torch.bool:
                        vision_pad_mask = ~active_mask_ba
                    else:
                        # Threshold float mask to bool (key_padding_mask must be bool).
                        # Note: this drops gradient for the transformer path; vision_proj
                        # multiplication above already carries gradient to the selector.
                        vision_pad_mask = active_mask_ba < 0.5
                    # Safety: avoid all-masked rows (would produce NaN in attention)
                    all_masked = vision_pad_mask.all(dim=1)
                    if all_masked.any():
                        vision_pad_mask = vision_pad_mask.clone()
                        vision_pad_mask[all_masked, 0] = False
                    vision_encoded = self.vision_seq_encoder(
                        vision_proj, src_key_padding_mask=vision_pad_mask,
                    )
                    vision_encoded = vision_encoded * active_mask_ba.unsqueeze(-1).to(vision_encoded.dtype)
                else:
                    vision_encoded = self.vision_seq_encoder(vision_proj)
            else:
                # vision_seq_layers=0: skip transformer, use projected features directly
                vision_encoded = vision_proj

            # Remove the constant visual offset (see __init__). The mean is
            # accumulated over selected frames only — unselected frames are
            # already zero and must stay zero after centering.
            if self.vision_center_mean and vision_encoded is not None:
                if 0 < self.budget_ratio < 1.0:
                    mask_f = active_mask_ba.unsqueeze(-1).to(vision_encoded.dtype)
                else:
                    mask_f = torch.ones_like(vision_encoded[..., :1])
                accum = self._vision_center_accum
                if accum is not None:
                    with torch.no_grad():
                        accum["sum"] = accum.get("sum", 0.0) + (
                            vision_encoded.detach().float() * mask_f.float()
                        ).sum(dim=(0, 1))
                        accum["count"] = accum.get("count", 0.0) + float(mask_f.float().sum())
                if self.training and not self.vision_center_frozen:
                    with torch.no_grad():
                        s = (vision_encoded.detach().float() * mask_f.float()).sum(dim=(0, 1))
                        c = mask_f.float().sum()
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                            torch.distributed.all_reduce(s)
                            torch.distributed.all_reduce(c)
                        if float(c) > 0:
                            batch_mean = s / c
                            if float(self.vision_feat_mean_inited) == 0.0:
                                self.vision_feat_mean.copy_(batch_mean)
                                self.vision_feat_mean_inited.fill_(1.0)
                            else:
                                m = self.vision_center_momentum
                                self.vision_feat_mean.mul_(m).add_(batch_mean, alpha=1.0 - m)
                mu = self.vision_feat_mean.to(vision_encoded.dtype)
                vision_encoded = (vision_encoded - mu) * mask_f

        # ======== Motion features: no seq encoder (already time-modeled by context_transformer) ========
        # DDP dummy: ensure params get gradients when vision path is inactive
        _dummy_vpath = torch.zeros((), device=device, dtype=dtype)
        if vision_encoded is None:
            _dummy_vpath = _dummy_vpath + sum(p.sum() for p in self.vision_projector.parameters()) * 0
            if self.vision_seq_encoder is not None:
                _dummy_vpath = _dummy_vpath + sum(p.sum() for p in self.vision_seq_encoder.parameters()) * 0
        _dummy_vpath = _dummy_vpath + self.motion_feat_norm.weight.sum() * 0
        _dummy_vpath = _dummy_vpath + self.fused_feat_norm.weight.sum() * 0

        # ======== Fuse (feature mode: before cls_head) ========
        if self.vision_fuse_mode == "feature" and vision_encoded is not None:
            fused = self.motion_feat_norm(motion_features) + vision_encoded
            fused = self.fused_feat_norm(fused)
            motion_encoded = fused
        else:
            motion_encoded = motion_features

        # Snapshot pre-BiGRU fused features for unsupervised DEC consumers; kept
        # separate from the post-BiGRU `motion_encoded` used by the cls_heads.
        fused_features_pre_bigru = motion_encoded.view(B, A, L, -1)

        # Motion BiGRU for temporal smoothing (optional, bigru_layers=0 skips it)
        if self.motion_seq_encoder is not None:
            motion_encoded = self.motion_seq_encoder(motion_encoded, pad_mask=pad_flat)

        # Snapshot post-BiGRU fused features too — DEC can opt into this via
        # `--dec-use-bigru`. When motion_seq_encoder is None this equals
        # fused_features_pre_bigru, so consumers can always read this key.
        fused_features_post_bigru = motion_encoded.view(B, A, L, -1)

        # Multi-animal aggregation → motion logits
        motion_bt = motion_encoded.view(B, A, L, -1)  # [B, A, L, D_z]
        if pad_mask_ds is not None:
            motion_bt = motion_bt.masked_fill(pad_mask_ds.unsqueeze(-1), 0.0)
            valid = (~pad_mask_ds).to(dtype=motion_bt.dtype)  # [B, A, L]
            denom = valid.sum(dim=1).clamp(min=1.0).unsqueeze(-1)  # [B, L, 1]
        else:
            valid = None
            denom = None

        if self.animal_agg_mode == "concat":
            # Concat over animals: [B, L, A*D_z]
            group_motion = motion_bt.permute(0, 2, 1, 3).reshape(B, L, A * motion_bt.shape[-1])
        else:
            # Mean over animals: [B, L, D_z]
            if valid is not None:
                group_motion = (motion_bt * valid.unsqueeze(-1)).sum(dim=1) / denom
            else:
                group_motion = motion_bt.mean(dim=1)

        motion_logits = self.social_cls_head(group_motion)  # [B, L, C]

        # ======== Online frame selection: use motion logits entropy ========
        if self.frame_select_mode == "online" and self.budget_ratio > 0:
            with torch.no_grad():
                probs = torch.sigmoid(motion_logits.detach())  # [B, L, C]
                # Per-class binary entropy, averaged across classes
                H = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
                frame_entropy = H.mean(dim=-1)  # [B, L]
                total_k = max(1, int(L * self.budget_ratio))
                _, top_idx = frame_entropy.topk(total_k, dim=-1)  # highest entropy = most uncertain
                online_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
                online_mask.scatter_(1, top_idx, True)
            # Override frame selection with online mask
            mask_global = online_mask.unsqueeze(1).expand(B, A, L).reshape(B * A, L)
            mask_local = torch.zeros_like(mask_global)

        # ======== Vision head: use BiGRU-encoded vision features (D_z) ========
        # In "feature" mode, vision is already fused before cls_head, skip vision head
        vision_logits = motion_logits.new_zeros(motion_logits.shape)
        if vision_encoded is not None and self.vision_fuse_mode != "feature":
            # vision_encoded: [B*A, L, D_z] (already BiGRU-encoded, sparse on selected frames)
            vision_bt = vision_encoded.view(B, A, L, -1)
            if pad_mask_ds is not None:
                vision_bt = vision_bt.masked_fill(pad_mask_ds.unsqueeze(-1), 0.0)

            if self.animal_agg_mode == "concat":
                group_vision = vision_bt.permute(0, 2, 1, 3).reshape(B, L, A * vision_bt.shape[-1])
            else:
                if valid is not None:
                    group_vision = (vision_bt * valid.unsqueeze(-1)).sum(dim=1) / denom
                else:
                    group_vision = vision_bt.mean(dim=1)

            if mask_global.dtype == torch.bool:
                active_mask = mask_global | mask_local
            else:
                active_mask = (mask_global + mask_local) > 0.5  # threshold float→bool
            frame_active = active_mask.view(B, A, L).any(dim=1)
            if frame_active.any():
                vision_logits[frame_active] = self.vision_cls_head(group_vision[frame_active])

        # ======== Fuse motion + vision logits ========
        if self.vision_fuse_mode == "feature":
            # Feature fusion already done before BiGRU; motion_logits is the final logits
            logits = motion_logits
            # DDP: dummy forward through all vision_cls_head params so they get gradients
            _dummy_vhead = sum(p.sum() for p in self.vision_cls_head.parameters()) * 0
            logits = logits + _dummy_vhead
        elif self.vision_fuse_mode == "add":
            # Add mode: all frames get vision contribution
            logits = motion_logits + self.vision_alpha * vision_logits
        else:
            # Replace mode: selected frames use blended logits, others pure motion
            logits = motion_logits.clone()
            if vision_feats is not None and self.budget_ratio > 0:
                if mask_global.dtype == torch.bool:
                    _combined = mask_global | mask_local
                else:
                    _combined = (mask_global + mask_local) > 0.5
                frame_active = _combined.view(B, A, L).any(dim=1)
                if frame_active.any():
                    logits[frame_active] = (
                        (1 - self.vision_alpha) * motion_logits[frame_active]
                        + self.vision_alpha * vision_logits[frame_active]
                    )
        logits = logits + _dummy + _dummy_social + _dummy_vpath

        # ======== Prepare collision loss inputs ========
        if mask_global.dtype == torch.bool:
            active_mask = mask_global | mask_local
        else:
            active_mask = (mask_global + mask_local) > 0.5
        active_v_flat = torch.empty(0, D_z, device=device, dtype=dtype)
        active_indices_flat = torch.empty(0, device=device, dtype=torch.long)

        return {
            "logits": logits,
            "mask_global": mask_global,
            "mask_local": mask_local,
            "active_v_flat": active_v_flat,
            "active_indices_flat": active_indices_flat,
            "attn_global": attn_global,
            "attn_local": attn_local,
            "vis_aggregated": vis_aggregated,
            "fused_features": fused_features_pre_bigru,
            "fused_features_post_bigru": fused_features_post_bigru,
            "z_q": z_q,
            "r1": r1,
            "r2": r2,
        }

    @torch.no_grad()
    def get_social_attn_weights(
        self,
        motion_feats: torch.Tensor,
        keypoints: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ):
        B, A, T = motion_feats.shape[:3]

        motion_flat = motion_feats.view(B * A, T, -1)
        scout = self.motion_scout(motion_flat)
        z_q = scout["z_q"]
        L = z_q.shape[1]

        context = self.transformer(self.motion_norm(z_q))
        context_bt = context.view(B, A, L, -1)
        key_padding_mask = None
        if pad_mask is not None:
            pad_mask_ds = pad_mask[:, :, :L]
            context_bt = context_bt.masked_fill(pad_mask_ds.unsqueeze(-1), 0.0)
            key_padding_mask = pad_mask_ds.transpose(1, 2).reshape(B * L, A)

        social_in = context_bt.permute(0, 2, 1, 3).reshape(B * L, A, -1)
        _, attn_w = _transformer_encoder_with_attn(self.social_encoder, social_in, key_padding_mask=key_padding_mask)
        return attn_w, key_padding_mask

    @torch.no_grad()
    def update_vq_ema(self):
        self.motion_scout.ema_update()
