import gc
import os
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple
import transformers
from transformers import AutoConfig, AutoImageProcessor
import importlib
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

def _safe_to_pil(tensor):
    """Convert CHW tensor to PIL if torchvision is available."""
    try:
        from torchvision import transforms
        to_pil = transforms.ToPILImage()
        return to_pil(tensor.cpu())
    except Exception:
        return None


def _make_norm1d(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class MotionTransformer(nn.Module):
    """
    Temporal Transformer encoder for motion tokens with attention entropy.
    """
    def __init__(self, input_dim, embed_dim, num_layers, num_heads, max_len=512, dropout=0.0):
        super().__init__()
        # input_dim: dimension of motion features per timestep
        # embed_dim: transformer model dimension
        # num_layers: number of transformer encoder layers
        # num_heads: number of attention heads
        # max_len: maximum supported sequence length for positional embedding
        # dropout: dropout rate inside transformer blocks
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, mask_indices=None, mask_token=None):
        """
        x: [B, T, input_dim]
        Returns: hidden [B, T, D], attn_weights from last layer [B, heads, T, T] or None
        """
        B, T, _ = x.shape
        x = self.input_proj(x)
        pad_mask = None
        if mask_indices is not None:
            mask_token = mask_token if mask_token is not None else 0.0
            if isinstance(mask_token, torch.Tensor):
                mask_token = mask_token.to(x.dtype)
                mask_token_expand = mask_token.expand(B, T, -1)
            else:
                mask_token_expand = mask_token
            mask_expanded = mask_indices.unsqueeze(-1).to(dtype=x.dtype)
            x = x * (1.0 - mask_expanded) + mask_token_expand * mask_expanded
            pad_mask = mask_indices.bool()
        if T > self.pos_embed.shape[1]:
            raise ValueError(f"Time dimension {T} exceeds pos_embed size {self.pos_embed.shape[1]}")
        x = x + self.pos_embed[:, :T, :]

        attn_weights = None
        for i, layer in enumerate(self.layers):
            # replicate TransformerEncoderLayer forward but request weights only on last layer
            need_w = (i == len(self.layers) - 1)
            attn_out, attn_w = layer.self_attn(
                x, x, x,
                key_padding_mask=pad_mask,
                need_weights=need_w,
                average_attn_weights=False,
            )
            x = x + layer.dropout1(attn_out)
            x = layer.norm1(x)
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm2(x + layer.dropout2(ff))
            if need_w:
                attn_weights = attn_w  # [B, heads, T, T]
        return x, attn_weights


class NeighborEncoder(nn.Module):
    """
    Lightweight encoder for neighbor context features.
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_layers: int = 0,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("NeighborEncoder input_dim must be > 0")
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if num_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.encoder = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, input_dim]
        returns: [B, T, embed_dim]
        """
        x = self.proj(x)
        if self.encoder is not None:
            x = self.encoder(x)
        return x


class Residual1DBlock(nn.Module):
    """Simple residual block for 1D conv stacks."""
    def __init__(self, channels, dilation: int = 1):
        super().__init__()
        # channels: number of channels for conv layers
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            _make_norm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            _make_norm1d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ProjectorMLP(nn.Module):
    """Two-layer projector with LayerNorm + activation."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.in_features = in_dim
        self.out_features = out_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class EMAVectorQuantizer(nn.Module):
    """
    VQ layer with EMA codebook update + (optional) L2-normalized codebook
    + dead-code revival from a recent-encoder-output reservoir.

    Two independent stabilization mechanisms:
      1. L2 normalization (controlled by `l2_normalize`, default True).
         When enabled, codebook entries live on the unit sphere (re-normalized
         after every EMA update). This prevents joint norm-collapse where
         encoder and codebook spiral toward zero together, but it also
         compresses encoder magnitude information — the encoder ends up
         producing direction-only signals, which can hurt downstream tasks
         that consume `‖z_e‖` or `‖z_e − z_q‖` as a saliency feature.

      2. Dead-code revival (always on). Any entry whose smoothed cluster size
         falls below `dead_code_threshold` is replaced by a random sample
         from a reservoir of recently-seen encoder outputs. This breaks the
         EMA "decay-to-zero" trap where unused entries get dragged to the
         origin and become permanently unreachable.

    For downstream tasks that need encoder magnitude (e.g. behavior saliency
    or unsupervised clustering on z_e), set `l2_normalize=False`. The
    revival mechanism alone is sufficient to prevent the worst form of
    collapse, while the encoder retains magnitude variation across frames.
    """
    def __init__(
        self,
        codebook_size,
        dim,
        beta=0.25,
        decay=0.99,
        eps=1e-5,
        dead_code_threshold: float = 1.0,
        revival_warmup_steps: int = 100,
        l2_normalize: bool = True,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.dead_code_threshold = dead_code_threshold
        self.revival_warmup_steps = revival_warmup_steps
        self.l2_normalize = l2_normalize

        embed = torch.randn(codebook_size, dim)
        if l2_normalize:
            embed = F.normalize(embed, dim=-1)
        self.register_buffer("codebook", embed)
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embed", embed.clone())
        self.register_buffer("_ema_cluster_size_accum", torch.zeros(codebook_size))
        self.register_buffer("_ema_embed_accum", torch.zeros_like(embed))

        # Reservoir of recent encoder outputs used to revive dead codes.
        # Refreshed each forward; size = codebook_size so we always have enough
        # candidates regardless of dead-entry count.
        self.register_buffer("_revival_buffer", embed.clone())
        self.register_buffer("_revival_buffer_filled", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_num_ema_updates", torch.zeros(1, dtype=torch.long))

    def forward(self, z):
        """
        z: [B, T, D]
        """
        B, T, D = z.shape
        flat = z.reshape(-1, D)  # [N, D]
        if self.l2_normalize:
            dist = (
                flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * flat @ self.codebook.t()
                + 1.0
            )
        else:
            dist = (
                flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * flat @ self.codebook.t()
                + self.codebook.pow(2).sum(dim=1)
            )
        indices = dist.argmin(dim=1)  # [N]
        one_hot = F.one_hot(indices, self.codebook_size).type(flat.dtype)

        quantized = self.codebook[indices].view(B, T, D)

        if self.training:
            with torch.no_grad():
                cluster_size = one_hot.sum(dim=0)
                embed_sum = one_hot.t() @ flat
                self._ema_cluster_size_accum.add_(cluster_size)
                self._ema_embed_accum.add_(embed_sum)
                self._refresh_revival_buffer(flat)

        # ===== Straight-through estimator =====
        quantized_st = z + (quantized - z).detach()

        # ===== VQ losses =====
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(z, quantized.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        return quantized_st, vq_loss, indices.view(B, T)

    @torch.no_grad()
    def _refresh_revival_buffer(self, flat):
        """Replace the revival buffer with a fresh random sample of `flat`."""
        N = flat.shape[0]
        if N == 0:
            return
        K = self.codebook_size
        if N >= K:
            perm = torch.randperm(N, device=flat.device)[:K]
        else:
            perm = torch.randint(0, N, (K,), device=flat.device)
        self._revival_buffer.copy_(flat[perm])
        self._revival_buffer_filled.fill_(1)

    @torch.no_grad()
    def ema_update(self):
        if not hasattr(self, "_ema_cluster_size_accum"):
            return

        # === DDP: aggregate per-rank EMA statistics into a global view ===
        # Without this all-reduce, each rank only updates the codebook from
        # its own local batch (1 / world_size of the data). DDP's
        # broadcast_buffers=True then overwrites all other ranks with
        # rank 0's view, so (world_size - 1) ranks' samples are silently
        # discarded. After the all-reduce, every rank holds the global sum
        # and computes an identical EMA update, so the final codebook is
        # consistent across ranks without relying on broadcast_buffers.
        use_ddp = (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        if use_ddp:
            dist.all_reduce(self._ema_cluster_size_accum, op=dist.ReduceOp.SUM)
            dist.all_reduce(self._ema_embed_accum, op=dist.ReduceOp.SUM)

        cluster_size = self._ema_cluster_size_accum
        embed_sum = self._ema_embed_accum
        if cluster_size.sum().item() <= 0:
            return

        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.ema_embed.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        n = self.ema_cluster_size.sum()
        smoothed = (
            (self.ema_cluster_size + self.eps)
            / (n + self.codebook_size * self.eps)
            * n
        )
        new_codebook = self.ema_embed / smoothed.unsqueeze(1)
        if self.l2_normalize:
            # Project the updated codebook back onto the unit sphere
            new_codebook = F.normalize(new_codebook, dim=-1)
        self.codebook.copy_(new_codebook)

        self._num_ema_updates.add_(1)

        # ===== Dead-code revival =====
        # Skip during warmup so the EMA cluster size has time to stabilize
        if (
            self._num_ema_updates.item() > self.revival_warmup_steps
            and self._revival_buffer_filled.item() > 0
        ):
            dead_mask = self.ema_cluster_size < self.dead_code_threshold
            num_dead = int(dead_mask.sum().item())
            if num_dead > 0:
                # Pool revival candidates across ranks for diversity. Without
                # the all-gather, dead-code revival would only ever sample
                # from rank 0's local batch (broadcast_buffers wins).
                if use_ddp:
                    world_size = dist.get_world_size()
                    gathered = [
                        torch.empty_like(self._revival_buffer)
                        for _ in range(world_size)
                    ]
                    dist.all_gather(gathered, self._revival_buffer)
                    pool = torch.cat(gathered, dim=0)  # [world_size * K, D]
                else:
                    pool = self._revival_buffer

                # Use a deterministic seed so every rank picks the *same*
                # replacements from the gathered pool. This guarantees the
                # codebook stays bit-identical across ranks even if
                # broadcast_buffers were disabled.
                cpu_gen = torch.Generator()
                cpu_gen.manual_seed(int(self._num_ema_updates.item()))
                sample_idx = torch.randint(
                    0, pool.shape[0], (num_dead,), generator=cpu_gen
                ).to(self.codebook.device)
                replacements = pool[sample_idx]
                if self.l2_normalize:
                    replacements = F.normalize(replacements, dim=-1)
                self.codebook[dead_mask] = replacements
                # Reset EMA state for revived entries so they aren't immediately
                # marked dead again next step.
                self.ema_embed[dead_mask] = replacements
                self.ema_cluster_size[dead_mask] = self.dead_code_threshold

        self._ema_cluster_size_accum.zero_()
        self._ema_embed_accum.zero_()

    @torch.no_grad()
    def usage_stats(self) -> dict:
        """Cheap codebook health snapshot for per-epoch logging.

        Reads from buffers that the EMA update already maintains (and
        all-reduces across DDP ranks), so this costs O(K^2 * D) for the
        pairwise distance only — milliseconds even at K=512. There is no
        extra forward pass.

        The cluster-size signal is smoothed with EMA decay, so the effective
        window is roughly 1 / (1 - decay) update steps. With decay=0.99 that
        is ~100 steps, which for typical batch sizes is comparable to one
        epoch. The number is therefore a "rolling recent" usage view, not an
        exact per-epoch count.
        """
        cs = self.ema_cluster_size
        K = int(self.codebook_size)
        total = cs.sum().clamp_min(1e-9)
        probs = cs / total

        used = int((cs >= self.dead_code_threshold).sum().item())

        nonzero = probs[probs > 0]
        if nonzero.numel() > 0:
            entropy = float(-(nonzero * nonzero.log()).sum().item())
        else:
            entropy = 0.0
        max_entropy = math.log(K) if K > 1 else 1.0
        norm_entropy = entropy / max_entropy

        sorted_cs, _ = cs.sort(descending=True)
        top10_frac = float((sorted_cs[:10].sum() / total).item())
        top50_frac = float((sorted_cs[:50].sum() / total).item())

        cb = self.codebook
        norms = cb.norm(dim=-1)
        # Pairwise distance in fp32 for numerical stability
        dists = torch.cdist(cb.float(), cb.float())
        dists.fill_diagonal_(float("inf"))
        min_inter = dists.min(dim=1).values

        # Effective number of clusters: perplexity of the usage distribution.
        # This is a much more meaningful "how many entries are doing real work"
        # metric than `used`, which only checks "non-dead" status. With a low
        # threshold like 0.1, almost any entry that has been touched once will
        # count as used, but its actual probability mass may be tiny.
        # effective_K = exp(H) where H is in nats. Bounded in [1, K].
        effective_k = float(math.exp(entropy))

        return {
            "K": K,
            "used": used,
            "effective_K": effective_k,
            "dead": K - used,
            "norm_entropy": norm_entropy,
            "top10_frac": top10_frac,
            "top50_frac": top50_frac,
            "mean_norm": float(norms.mean().item()),
            "avg_min_inter_dist": float(min_inter.mean().item()),
            "min_inter_dist": float(min_inter.min().item()),
        }


class ResidualVQ(nn.Module):
    """
    Two-level residual vector quantizer with detached residual.

    Why we MUST detach `q1_st` before computing the residual
    --------------------------------------------------------
    The straight-through estimator inside each VQ layer means that, in the
    backward pass, ∂q1_st/∂z = I (it pretends the quantization is the
    identity). If we wrote `residual = z - q1_st` directly, then
    ∂residual/∂z = I − I = 0, and q2's commitment loss
    ‖residual − q2.detach()‖² would propagate **zero** gradient back to the
    encoder. The encoder would only ever be pulled by q1, and the q2
    refinement layer would be effectively dead.

    By detaching q1 (`q1 = q1_st.detach()`), we make the residual a true
    function of `z` again (∂residual/∂z = I), so that q2's commitment loss
    properly pulls the encoder toward the *combined* `q1 + q2` target.

    Note that q1 itself is updated by EMA, not by backward through z, so
    cutting this gradient path costs nothing on the q1 side.
    """
    def __init__(
        self,
        codebook1,
        codebook2,
        dim,
        beta=0.25,
        dead_code_threshold: float = 1.0,
        revival_warmup_steps: int = 100,
        l2_normalize: bool = True,
    ):
        super().__init__()
        self.q1 = EMAVectorQuantizer(
            codebook1,
            dim,
            beta,
            dead_code_threshold=dead_code_threshold,
            revival_warmup_steps=revival_warmup_steps,
            l2_normalize=l2_normalize,
        )
        self.q2 = EMAVectorQuantizer(
            codebook2,
            dim,
            beta,
            dead_code_threshold=dead_code_threshold,
            revival_warmup_steps=revival_warmup_steps,
            l2_normalize=l2_normalize,
        )

    def forward(self, z):
        # Level 1
        q1_st, loss1, idx1 = self.q1(z)
        q1 = q1_st.detach()

        # Residual
        residual = z - q1

        # Level 2
        q2_st, loss2, idx2 = self.q2(residual)

        quantized = q1_st + q2_st
        vq_loss = loss1 + loss2

        return quantized, vq_loss, (idx1, idx2), q1_st, q2_st

    @torch.no_grad()
    def ema_update(self):
        self.q1.ema_update()
        self.q2.ema_update()

    @torch.no_grad()
    def usage_stats(self) -> dict:
        return {"q1": self.q1.usage_stats(), "q2": self.q2.usage_stats()}

class HierarchicalVQVAE(nn.Module):
    """
    Hierarchical VQ-VAE with residual vector quantization (2-level).
    Encoder/decoder are 1D conv stacks along time.
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        codebook1,
        codebook2,
        num_stride_blocks=3,
        num_attn_layers=1,
        attn_heads=4,
        num_attn_layers_dec: int | None = None,
        beta=0.25,
        dead_code_threshold: float = 1.0,
        revival_warmup_steps: int = 100,
        l2_normalize: bool = True,
    ):
        super().__init__()
        # input_dim: input feature dimension (flattened keypoints per timestep)
        # hidden_dim: channels for encoder intermediate conv layers
        # latent_dim: latent dimension after encoder
        # codebook1: size of first codebook
        # codebook2: size of second codebook
        # num_stride_blocks: number of stride-1 conv blocks (preserve temporal length)
        # num_attn_layers/attn_heads: optional lightweight self-attention on latent
        # num_attn_layers_dec: decoder-side self-attention layers (default mirror num_attn_layers)
        # beta: commitment loss scaling
        # dead_code_threshold: revive entries whose smoothed cluster size falls below this
        # revival_warmup_steps: skip revival for the first N ema_update calls
        # l2_normalize: pin codebook to unit sphere (True for compression-style VQ;
        #               False to preserve encoder magnitude variation for downstream
        #               saliency / clustering tasks)
        enc_layers = []
        in_ch = input_dim
        ch = hidden_dim
        for _ in range(num_stride_blocks):
            enc_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, ch, kernel_size=3, stride=1, padding=1),
                    _make_norm1d(ch),
                    nn.ReLU(),
                    Residual1DBlock(ch),
                )
            )
            in_ch = ch
        enc_out_ch = in_ch
        self.encoder = nn.Sequential(*enc_layers)
        self.to_latent = nn.Conv1d(enc_out_ch, latent_dim, kernel_size=3, padding=1)
        self.from_latent = nn.Conv1d(latent_dim, enc_out_ch, kernel_size=3, padding=1)

        self.latent_attn = None
        if num_attn_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=attn_heads,
                dim_feedforward=latent_dim * 4,
                batch_first=True,
                dropout=0.0,
            )
            self.latent_attn = nn.TransformerEncoder(encoder_layer, num_layers=num_attn_layers)

        self.latent_attn_dec = None
        dec_attn_layers = num_attn_layers if num_attn_layers_dec is None else num_attn_layers_dec
        if dec_attn_layers and dec_attn_layers > 0:
            dec_layer = nn.TransformerEncoderLayer(
                d_model=enc_out_ch,
                nhead=attn_heads,
                dim_feedforward=enc_out_ch * 4,
                batch_first=True,
                dropout=0.0,
            )
            self.latent_attn_dec = nn.TransformerEncoder(dec_layer, num_layers=dec_attn_layers)

        self.quantizer = ResidualVQ(
            codebook1,
            codebook2,
            dim=latent_dim,
            beta=beta,
            dead_code_threshold=dead_code_threshold,
            revival_warmup_steps=revival_warmup_steps,
            l2_normalize=l2_normalize,
        )

        dec_layers = []
        dec_ch = enc_out_ch
        for _ in range(num_stride_blocks):
            dec_layers.append(
                nn.Sequential(
                    nn.Conv1d(dec_ch, dec_ch, kernel_size=3, stride=1, padding=1),
                    _make_norm1d(dec_ch),
                    nn.ReLU(),
                    Residual1DBlock(dec_ch),
                )
            )
        self.decoder = nn.Sequential(*dec_layers)
        self.to_out = nn.Conv1d(dec_ch, input_dim, kernel_size=3, padding=1)

    def encode(self, x):
        """
        x: [B, T, C] continuous motion features (including keypoints).
        """
        x = x.transpose(1, 2)  # [B, C, T]
        z = self.encoder(x)
        z = self.to_latent(z)  # [B, D, T_enc]
        z = z.transpose(1, 2)  # [B, T_enc, D]
        if self.latent_attn is not None:
            z = self.latent_attn(z)
        quantized, vq_loss, indices, q1, q2 = self.quantizer(z)
        return z, quantized, vq_loss, indices, q1, q2

    def decode(self, quantized, target_len):
        """
        quantized: [B, T_enc, D]; target_len: original sequence length for trimming.
        """
        z = quantized.transpose(1, 2)  # [B, D, T_enc]
        z = self.from_latent(z)
        if self.latent_attn_dec is not None:
            z = z.transpose(1, 2)  # [B, T_enc, C_dec]
            z = self.latent_attn_dec(z)
            z = z.transpose(1, 2)
        out = self.decoder(z)
        out = self.to_out(out)  # [B, input_dim, T_out]
        out = out[:, :, :target_len]  # trim in case of slight mismatch
        out = out.transpose(1, 2)  # [B, T_out, input_dim]
        return out

    def forward(self, keypoints):
        """
        keypoints: [B, T, P, 3] (P may be the total number of keypoints over all animals)
        returns recon: [B, T, P, 3], vq_loss, indices(tuple)
        """
        B, T, P, _ = keypoints.shape
        x = keypoints.view(B, T, -1)
        z_e, z_q, vq_loss, indices, q1, q2 = self.encode(x)
        recon_flat = self.decode(z_q, target_len=T)
        recon = recon_flat.view(B, T, P, 3)
        return recon, vq_loss, indices, z_q

    @torch.no_grad()
    def ema_update(self):
        self.quantizer.ema_update()

    @torch.no_grad()
    def usage_stats(self) -> dict:
        return self.quantizer.usage_stats()


class MotionScout(nn.Module):
    """
    Motion branch: VQ-VAE encoder + RVQ only (simplified v2).
    No proj, transformer, or gate — just RVQ encoding and decoding.
    """
    def __init__(
        self,
        input_dim: int,
        vq_hidden_dim: int,
        vq_latent_dim: int,
        codebook1: int,
        codebook2: int,
        num_stride_blocks: int,
        num_attn_layers: int,
        attn_heads: int,
        num_attn_layers_dec: int,
        beta: float,
        dead_code_threshold: float = 1.0,
        revival_warmup_steps: int = 100,
        l2_normalize: bool = True,
    ):
        super().__init__()
        self.vqvae = HierarchicalVQVAE(
            input_dim=input_dim,
            hidden_dim=vq_hidden_dim,
            latent_dim=vq_latent_dim,
            codebook1=codebook1,
            codebook2=codebook2,
            num_stride_blocks=num_stride_blocks,
            num_attn_layers=num_attn_layers,
            attn_heads=attn_heads,
            num_attn_layers_dec=num_attn_layers_dec,
            beta=beta,
            dead_code_threshold=dead_code_threshold,
            revival_warmup_steps=revival_warmup_steps,
            l2_normalize=l2_normalize,
        )

    def forward(self, motion_feats: torch.Tensor):
        """
        motion_feats: [B, T, D_in]
        Returns dict with: z_e, z_q, z_q1, z_q2, vq_loss, code_indices, recon
        """
        B, T, _ = motion_feats.shape
        z_e, z_q, vq_loss, code_indices, z_q1, z_q2 = self.vqvae.encode(motion_feats)
        recon = self.vqvae.decode(z_q, target_len=T)
        return {
            "z_e": z_e,           # [B, T, D_z=128] encoder continuous output
            "z_q": z_q,           # [B, T, D_z=128] two-level quantized sum
            "z_q1": z_q1,         # [B, T, D_z=128] Level-1 quantized
            "z_q2": z_q2,         # [B, T, D_z=128] Level-2 quantized
            "vq_loss": vq_loss,
            "code_indices": code_indices,  # (idx1[B,T], idx2[B,T])
            "recon": recon,       # [B, T, D_in]
        }

    @torch.no_grad()
    def ema_update(self):
        self.vqvae.ema_update()

    @torch.no_grad()
    def usage_stats(self) -> dict:
        return self.vqvae.usage_stats()


class CodebookCrossAttention(nn.Module):
    """
    Use RVQ code as query to cross-attend over vision encoder patch tokens.
    """
    def __init__(self, d_code: int, d_vision: int, d_attn: int, d_out: int):
        super().__init__()
        self.d_attn = d_attn
        self.W_q = nn.Linear(d_code, d_attn, bias=False)
        self.W_k = nn.Linear(d_vision, d_attn, bias=False)
        self.W_v = nn.Linear(d_vision, d_out, bias=False)

    def forward(self, z_code: torch.Tensor, patch_tokens: torch.Tensor):
        """
        z_code:       [N, D_z]
        patch_tokens: [N, num_patches, D_v]
        Returns: v [N, D_out], attn_weights [N, num_patches]
        """
        Q = self.W_q(z_code).unsqueeze(1)                # [N, 1, d_attn]
        K = self.W_k(patch_tokens)                        # [N, P, d_attn]
        V = self.W_v(patch_tokens)                        # [N, P, d_out]

        attn = torch.bmm(Q, K.transpose(1, 2)) / (self.d_attn ** 0.5)
        attn = F.softmax(attn, dim=-1)                    # [N, 1, P]

        v = torch.bmm(attn, V).squeeze(1)                # [N, d_out]
        return v, attn.squeeze(1)


class VisionMagnifier(nn.Module):
    """
    Level 2: Visual Magnifier (processes video frames)
    Frozen Qwen-VL Vision Encoder + LoRA Adapter + Temporal Aggregator + Counting Head.
    """
    def __init__(
        self,
        vision_dim: int,
        project_dim: int,
        use_real_model: bool = True,
        qwen_model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        agg_layers: int = 2,
        agg_heads: int = 4,
        max_len: int = 512,
        enable_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_layers: int = 4,
        gate_thresh: float = 0.5,
        gate_topk_ratio: float = 0.0,
        gate_topk: int = 0,
        gate_uniform_topk: bool = False,
        enable_flash_attn: bool = False,
        use_deepstack: bool = True,
        use_frame_diff: bool = True,
        frame_diff_abs: bool = False,
        frame_diff_fuse: str = "residual",
        frame_diff_dt_mode: str = "divide",
        frame_diff_dt_min: float = 1.0,
        use_temporal_conv: bool = True,
        temporal_conv_kernel: int = 3,
        vision_disable_temporal_agg: bool = False,
        enable_grad_checkpoint: bool = False,
        allow_random_fallback: bool = False,
    ):
        super().__init__()
        # vision_dim: dimension of backbone vision features
        # project_dim: projected dimension to motion space
        # use_real_model: whether to load HF Qwen vision tower
        # qwen_model_id: HF model id for Qwen vision
        # agg_layers/agg_heads: temporal aggregator transformer config
        # max_len: max temporal length for aggregator pos embed
        # lora_r/lora_alpha/lora_layers: LoRA config for attention projections
        # gate_thresh: activation threshold for sparse vision
        self.enable_lora = enable_lora
        self.enable_flash_attn = enable_flash_attn
        self.use_real_model = use_real_model
        self.vision_dim = vision_dim
        self.qwen_model_id = qwen_model_id
        self.image_processor = None
        self.backbone = None
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_layers = lora_layers
        self.gate_thresh = gate_thresh
        self.gate_topk_ratio = float(gate_topk_ratio or 0.0)
        self.gate_topk = int(gate_topk or 0)
        self.gate_uniform_topk = bool(gate_uniform_topk)
        self.use_frame_diff = use_frame_diff
        self.use_deepstack = use_deepstack
        self.frame_diff_abs = frame_diff_abs
        self.frame_diff_fuse = frame_diff_fuse
        self.frame_diff_dt_mode = frame_diff_dt_mode
        self.frame_diff_dt_min = frame_diff_dt_min
        self.use_temporal_conv = use_temporal_conv
        self.temporal_conv_kernel = temporal_conv_kernel
        self.disable_temporal_agg = bool(vision_disable_temporal_agg)
        self.enable_grad_checkpoint = enable_grad_checkpoint
        self.allow_random_fallback = bool(allow_random_fallback)
        self._warned_no_qwen = False
        self._warned_hw_misaligned = False
        self._debug_logged = False
        self.current_epoch = None
        self._epoch_usage_epoch = None
        self._epoch_usage_frames = 0
        self._epoch_usage_steps = 0
        self._epoch_usage_calls = 0
        self._epoch_usage_logged_epoch = None
        self._frame_pad_cache: dict[tuple, torch.Tensor] = {}

        if self.use_real_model:
            self._load_qwen_backbone()
        else:
            self._warn_no_qwen()
        if self.use_real_model:
            self._get_vision_params()
            self._assert_vision_dim()

        # Projector (Vision -> Motion Space)
        self.projector = ProjectorMLP(self.vision_dim, project_dim, project_dim)

        if self.frame_diff_fuse not in ("residual", "concat"):
            raise ValueError(f"Unsupported frame_diff_fuse: {self.frame_diff_fuse}")
        if self.frame_diff_dt_mode not in ("none", "divide", "embed", "both"):
            raise ValueError(f"Unsupported frame_diff_dt_mode: {self.frame_diff_dt_mode}")

        if self.use_frame_diff:
            self.delta_mlp = nn.Sequential(
                nn.LayerNorm(project_dim),
                nn.Linear(project_dim, project_dim),
                nn.GELU(),
                nn.Linear(project_dim, project_dim),
            )
            if self.frame_diff_fuse == "concat":
                self.delta_fuse = nn.Linear(project_dim * 2, project_dim)
            if self.frame_diff_dt_mode in ("embed", "both"):
                self.dt_mlp = nn.Sequential(
                    nn.Linear(1, project_dim),
                    nn.GELU(),
                    nn.Linear(project_dim, project_dim),
                )

        if self.use_temporal_conv:
            kernel = max(1, int(temporal_conv_kernel))
            padding = kernel // 2
            self.temporal_dwconv = nn.Conv1d(
                project_dim,
                project_dim,
                kernel_size=kernel,
                padding=padding,
                groups=project_dim,
            )
            self.temporal_pwconv = nn.Conv1d(project_dim, project_dim, kernel_size=1)
            self.temporal_conv_act = nn.GELU()

        # Temporal Aggregator (VarLen Transformer with flash-attn)
        self.agg_pos = nn.Parameter(torch.zeros(1, max_len, project_dim))
        agg_dropout = 0.1
        self.temporal_blocks = nn.ModuleList(
            [
                VarLenTransformerBlock(
                    d_model=project_dim,
                    n_heads=agg_heads,
                    mlp_ratio=4.0,
                    dropout=agg_dropout,
                )
                for _ in range(agg_layers)
            ]
        )
        self._idx_map_cache: dict[tuple[torch.device, int, int], torch.Tensor] = {}
        self._arange_cache: dict[tuple[torch.device, int], torch.Tensor] = {}

        # Counting head (regression)
        self.count_head = nn.Sequential(
            nn.LayerNorm(project_dim),
            nn.Linear(project_dim, 1),
        )

    @staticmethod
    def _get_qwen_model_cls(model_type: str | None, architectures: list[str] | None):
        arch_set = set(architectures or [])
        if model_type == "qwen3_5" or "Qwen3_5ForConditionalGeneration" in arch_set:
            model_cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
            if model_cls is None:
                raise ImportError(
                    "this transformers version does not expose Qwen3_5ForConditionalGeneration; "
                    "upgrade to a version that supports Qwen3.5."
                )
            return model_cls
        if model_type == "qwen3_vl" or "Qwen3VLForConditionalGeneration" in arch_set:
            model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
            if model_cls is None:
                raise ImportError(
                    "this transformers version does not expose Qwen3VLForConditionalGeneration; "
                    "check that the Qwen3-VL dependencies are installed."
                )
            return model_cls
        raise ValueError(
            f"Unsupported Qwen model_type={model_type!r}, architectures={architectures!r}. "
            "VisionMagnifier currently supports only Qwen3-VL / Qwen3.5 multimodal models."
        )

    def _load_qwen_backbone(self):
        try:
            cfg = AutoConfig.from_pretrained(self.qwen_model_id, trust_remote_code=True)
            model_type = getattr(cfg, "model_type", None)
            architectures = getattr(cfg, "architectures", None)
            model_cls = self._get_qwen_model_cls(model_type, architectures)

            try:
                self.image_processor = AutoImageProcessor.from_pretrained(self.qwen_model_id, trust_remote_code=True)
            except Exception as proc_err:
                self.image_processor = None
                print(f"[VisionMagnifier] AutoImageProcessor unavailable for {self.qwen_model_id}: {proc_err}")

            model = model_cls.from_pretrained(
                self.qwen_model_id,
                trust_remote_code=True,
            )
            if self.enable_grad_checkpoint:
                try:
                    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                        model.config.use_cache = False
                    vis_cfg = getattr(model.config, "vision_config", None)
                    if vis_cfg is not None and hasattr(vis_cfg, "use_cache"):
                        vis_cfg.use_cache = False
                except Exception:
                    pass
                try:
                    if hasattr(model, "gradient_checkpointing_enable"):
                        model.gradient_checkpointing_enable()
                        print("[VisionMagnifier] Enabled gradient checkpointing for Qwen backbone.")
                except Exception as ckpt_err:
                    print(f"[VisionMagnifier] Failed to enable gradient checkpointing: {ckpt_err}")
            # Optional: force flash attention if available
            if self.enable_flash_attn:
                try:
                    # Qwen3-VL uses `visual` as the vision tower; fall back to `vision_model` if present.
                    target_cfg = None
                    if hasattr(model, "visual") and hasattr(model.visual, "config"):
                        target_cfg = model.visual.config
                    elif hasattr(model, "vision_model") and hasattr(model.vision_model, "config"):
                        target_cfg = model.vision_model.config
                    if target_cfg is not None:
                        target_cfg._attn_implementation = "flash_attention_2"
                    if hasattr(model, "config") and hasattr(model.config, "vision_config"):
                        model.config.vision_config._attn_implementation = "flash_attention_2"
                    print("[VisionMagnifier] Enabled flash_attention_2 for Qwen vision backbone.")
                except Exception as attn_err:
                    print(f"[VisionMagnifier] Failed to enable flash_attention_2: {attn_err}")
            # grab vision module only
            self.backbone = (
                getattr(model, "visual", None)
                or getattr(model.model, "visual", None)
                or getattr(model, "vision_model", None)
                or getattr(model, "vision_tower", None)
                or getattr(model, "vision_encoder", None)
                or getattr(model, "clip_vision_model", None)
                or model
            )

            if not self.use_deepstack and hasattr(self.backbone, "deepstack_visual_indexes"):
                self.backbone.deepstack_visual_indexes = []
                if hasattr(self.backbone, "deepstack_merger_list"):
                    self.backbone.deepstack_merger_list = nn.ModuleList()
                print("[VisionMagnifier] DeepStack disabled; deepstack_visual_indexes cleared.")

            # Qwen3.5 vision fast-path: Conv3d patch_embed is replaced
            # by an equivalent Linear (cuDNN picks a pathologically slow
            # algorithm for 14x14 grids, ~100s vs <1ms); backbone.forward
            # is swapped for a version that caches per-shape position
            # embeddings and bypasses HF's decorator wrappers. See
            # _patch_qwen35_vision_for_speed for details.
            if type(self.backbone).__name__ == "Qwen3_5VisionModel":
                self._patch_qwen35_vision_for_speed()

            # Force-set _attn_implementation on every vision attention
            # instance: some loaders copy configs per layer, so setting
            # it on the top-level config doesn't always propagate.
            target_impl = "flash_attention_2" if self.enable_flash_attn else "sdpa"
            _patched = 0
            for _mod in self.backbone.modules():
                if "VisionAttention" in type(_mod).__name__ and hasattr(_mod, "config"):
                    try:
                        _mod.config._attn_implementation = target_impl
                        _patched += 1
                    except Exception:
                        pass
            if _patched:
                print(
                    f"[VisionMagnifier] Force-set _attn_implementation='{target_impl}' "
                    f"on {_patched} VisionAttention module(s)."
                )
            # Debug log: current attention implementation + flash_attn availability
            attn_impl = None
            flash_import_ok = False
            flash_import_err = None

            importlib.import_module("flash_attn")
            flash_import_ok = True

            cfg = getattr(self.backbone, "config", None)
            vis_cfg = getattr(cfg, "vision_config", cfg)
            attn_impl = getattr(vis_cfg, "_attn_implementation", None) if vis_cfg is not None else None
            msg = f"[VisionMagnifier] vision attn impl={attn_impl} flash_attn_importable={flash_import_ok}"
            if not flash_import_ok and flash_import_err:
                msg += f" ({flash_import_err})"
                print(msg)
                raise NotImplementedError
            print(msg)
            
            # freeze and detach from language weights to free memory
            self.backbone.requires_grad_(False)
            for attr in ["model", "language_model", "text_model", "llm", "transformer", "language_decoder"]:
                if hasattr(model, attr):
                    try:
                        setattr(model, attr, None)
                    except Exception:
                        pass
            # inject LoRA into selected attention blocks of vision encoder
            if self.enable_lora and self.lora_layers > 0:
                self._inject_vision_lora(self.backbone, self.lora_layers, self.lora_r, self.lora_alpha)
            del model
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            msg = (
                f"[VisionMagnifier] Failed to load Qwen model for qwen_model_id={self.qwen_model_id}: {e}"
            )
            if self.allow_random_fallback:
                print(f"{msg}. Falling back to random features.")
                self.backbone = None
                self.image_processor = None
                self.use_real_model = False
                self._warn_no_qwen()
                return
            raise RuntimeError(msg) from e

    @staticmethod
    def _get_attention_modules(backbone):
        # Try to find the list of encoder blocks
        for attr in ["encoder", "vision_model", "vision_encoder", "clip_vision_model"]:
            if hasattr(backbone, attr):
                encoder = getattr(backbone, attr)
                if hasattr(encoder, "layers"):
                    return encoder.layers
                if hasattr(encoder, "blocks"):
                    return encoder.blocks
        if hasattr(backbone, "layers"):
            return backbone.layers
        if hasattr(backbone, "blocks"):
            return backbone.blocks
        return None

    @staticmethod
    def _inject_vision_lora(backbone, lora_layers: int, r: int, alpha: int):
        """
        Insert lightweight LoRA on q_proj / v_proj of the last `lora_layers` encoder blocks.
        """
        layers = VisionMagnifier._get_attention_modules(backbone)
        if layers is None:
            print("[VisionMagnifier] No encoder layers found for LoRA injection.")
            return
        target_layers = list(layers)[-lora_layers:] if lora_layers > 0 else []
        for layer in target_layers:
            # common attribute names for attention modules
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None) or getattr(layer, "attention", None)
            if attn is None:
                continue
            for proj_name in ["q_proj", "v_proj"]:
                if hasattr(attn, proj_name):
                    base_linear = getattr(attn, proj_name)
                    lora_module = LoRALinear.from_linear(base_linear, r=r, alpha=alpha)
                    setattr(attn, proj_name, lora_module)

    def _encode_real_frames(self, video_frames: torch.Tensor) -> torch.Tensor:
        feats_flat, _, _ = self._encode_real_frames_impl(video_frames, return_patch_tokens=False)
        return feats_flat

    def _encode_real_frames_with_patch_tokens(
        self,
        video_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        return self._encode_real_frames_impl(video_frames, return_patch_tokens=True)

    def _encode_real_frames_impl(
        self,
        video_frames: torch.Tensor,
        return_patch_tokens: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict[str, int]]]:
        """
        Encode frames via Qwen image processor/backbone. Falls back to random features if processor/backbone missing.
        Returns one feature per input frame, shape=[B*T_orig, vision_dim], in the order of video_frames.
        """
        debug_log = False
        compiling = False
        if hasattr(torch, "_dynamo"):
            try:
                compiling = torch._dynamo.is_compiling()
            except Exception:
                compiling = False
        debug_log = debug_log and (not compiling)

        if self.backbone is None:
            if not self.allow_random_fallback:
                raise RuntimeError(
                    "[VisionMagnifier] Qwen vision backbone is unavailable and random fallback is disabled."
                )
            self._warn_no_qwen()
            feats_flat = torch.randn(
                video_frames.size(0) * video_frames.size(1),
                self.vision_dim,
                device=video_frames.device,
                dtype=video_frames.dtype,
            )
            if return_patch_tokens:
                patch_tokens = feats_flat.unsqueeze(1)
                patch_meta = {
                    "tokens_per_frame": 1,
                    "patch_grid_h": 1,
                    "patch_grid_w": 1,
                    "patch_size": 1,
                    "temporal_patch_size": 1,
                    "spatial_merge_size": 1,
                }
                return feats_flat, patch_tokens, patch_meta
            return feats_flat, None, None

        B, T, C, H, W = video_frames.shape
        orig_T = T
        device = video_frames.device
        dtype = video_frames.dtype
        if debug_log:
            print(f"[VisionMagnifier][debug] encode start B={B} T={T} C={C} H={H} W={W} dtype={dtype}")

        patch_size, temporal_patch, merge_size = self._get_vision_params()
        if patch_size is None or temporal_patch is None or merge_size is None:
            if not self.allow_random_fallback:
                raise RuntimeError(
                    "[VisionMagnifier] Failed to resolve Qwen vision params and random fallback is disabled."
                )
            self._warn_no_qwen()
            feats_flat = torch.randn(B * T, self.vision_dim, device=device, dtype=dtype)
            if return_patch_tokens:
                patch_tokens = feats_flat.unsqueeze(1)
                patch_meta = {
                    "tokens_per_frame": 1,
                    "patch_grid_h": 1,
                    "patch_grid_w": 1,
                    "patch_size": 1,
                    "temporal_patch_size": 1,
                    "spatial_merge_size": 1,
                }
                return feats_flat, patch_tokens, patch_meta
            return feats_flat, None, None
        if debug_log:
            print(
                f"[VisionMagnifier][debug] vision params patch_size={patch_size} temporal_patch={temporal_patch} "
                f"merge_size={merge_size}"
            )

        align_factor = patch_size * merge_size
        aligned_hw = (H % align_factor == 0) and (W % align_factor == 0)
        if aligned_hw:
            resized_h, resized_w = H, W
        else:
            def smart_resize(num_frames, height, width, temporal_factor=2, factor=32, min_pixels=131072, max_pixels=786432):
                if num_frames < temporal_factor or height < factor or width < factor:
                    return height, width
                h_bar = round(height / factor) * factor
                w_bar = round(width / factor) * factor
                t_bar = round(num_frames / temporal_factor) * temporal_factor
                if t_bar * h_bar * w_bar > max_pixels:
                    beta = math.sqrt((num_frames * height * width) / max_pixels)
                    h_bar = max(factor, math.floor(height / beta / factor) * factor)
                    w_bar = max(factor, math.floor(width / beta / factor) * factor)
                elif t_bar * h_bar * w_bar < min_pixels:
                    beta = math.sqrt(min_pixels / (num_frames * height * width))
                    h_bar = math.ceil(height * beta / factor) * factor
                    w_bar = math.ceil(width * beta / factor) * factor
                return h_bar, w_bar

            resized_h, resized_w = smart_resize(
                num_frames=T,
                height=H,
                width=W,
                temporal_factor=temporal_patch,
                factor=align_factor,
            )
            if not self._warned_hw_misaligned:
                warnings.warn(
                    f"[VisionMagnifier] Input H/W ({H}x{W}) not divisible by align_factor={align_factor}, "
                    f"using smart_resize to ({resized_h}x{resized_w})."
                )
                self._warned_hw_misaligned = True

        frames = video_frames
        if resized_h != H or resized_w != W:
            frames_2d = frames.reshape(-1, C, H, W)
            frames_2d = F.interpolate(
                frames_2d,
                size=(resized_h, resized_w),
                mode="bilinear",
                align_corners=False,
            )
            frames = frames_2d.view(B, T, C, resized_h, resized_w)
            if debug_log:
                print(f"[VisionMagnifier][debug] resized frames to {resized_h}x{resized_w}")

        frames = frames / 255.0
        mean = torch.tensor([0.5, 0.5, 0.5], device=device, dtype=frames.dtype).view(1, 1, -1, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=device, dtype=frames.dtype).view(1, 1, -1, 1, 1)
        frames = (frames - mean) / std

        if T % temporal_patch != 0:
            pad_t = (-T) % temporal_patch
            frames_cf = frames.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]
            frames_cf = F.pad(frames_cf, (0, 0, 0, 0, 0, pad_t), mode="replicate")
            frames = frames_cf.permute(0, 2, 1, 3, 4)
            T = frames.shape[1]
            if debug_log:
                print(f"[VisionMagnifier][debug] temporal pad applied pad_t={pad_t} new_T={T}")

        grid_t = T // temporal_patch
        grid_h = resized_h // patch_size
        grid_w = resized_w // patch_size
        if debug_log:
            print(f"[VisionMagnifier][debug] grid sizes grid_t={grid_t} grid_h={grid_h} grid_w={grid_w}")
        frames_flat = frames.contiguous().view(
            B,
            grid_t,
            temporal_patch,
            C,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = frames_flat.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).contiguous()
        patches = patches.view(B, grid_t * grid_h * grid_w, C * temporal_patch * patch_size * patch_size)

        model_inputs = {
            "hidden_states": patches,
            "grid_thw": torch.tensor([grid_t, grid_h, grid_w], device=device, dtype=torch.long).unsqueeze(0).repeat(B, 1),
        }

        if debug_log:
            print(
                f"[VisionMagnifier][debug] backbone forward patches_shape={tuple(patches.shape)} "
                f"grid_thw=({grid_t},{grid_h},{grid_w})"
            )
        if self.enable_lora:
            outputs = self.backbone(**model_inputs)
        else:
            with torch.no_grad():
                outputs = self.backbone(**model_inputs)
        # Qwen3.5 vision returns unmerged tokens in `last_hidden_state` and
        # merged patch features in `pooler_output`; Qwen3-VL exposes only the
        # merged stream as the first output/last_hidden_state.
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feats = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            feats = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            feats = outputs[0]
        else:
            raise RuntimeError(f"Unexpected vision output type: {type(outputs)}")
        if debug_log:
            print(f"[VisionMagnifier][debug] backbone done feats_shape={tuple(feats.shape)}")

        hidden_dim = feats.shape[-1]
        tokens_per_patch = (grid_h * grid_w) // (merge_size * merge_size)
        tokens_per_video = grid_t * tokens_per_patch
        expected_tokens = B * tokens_per_video

        if feats.dim() != 2 or feats.shape[0] != expected_tokens:
            raise RuntimeError(
                f"Vision tokens mismatch: expected flat [N={expected_tokens}, D], got {feats.shape} "
                f"(grid_t={grid_t}, grid_h={grid_h}, grid_w={grid_w}, merge_size={merge_size})"
            )
        feats = feats.view(B, grid_t, tokens_per_patch, hidden_dim)

        patch_tokens = feats.unsqueeze(2).expand(-1, -1, temporal_patch, -1, -1)
        patch_tokens = patch_tokens.reshape(B, grid_t * temporal_patch, tokens_per_patch, hidden_dim)
        patch_tokens = patch_tokens[:, :orig_T]
        feats = patch_tokens.mean(dim=2)  # [B, T_orig, hidden_dim]
        feats = feats.reshape(-1, hidden_dim)
        if debug_log:
            print(f"[VisionMagnifier][debug] encode finish feats_flat_shape={tuple(feats.shape)} orig_T={orig_T}")

        if return_patch_tokens:
            patch_meta = {
                "tokens_per_frame": int(tokens_per_patch),
                "patch_grid_h": int(grid_h // merge_size),
                "patch_grid_w": int(grid_w // merge_size),
                "patch_size": int(patch_size),
                "temporal_patch_size": int(temporal_patch),
                "spatial_merge_size": int(merge_size),
            }
            return feats, patch_tokens.reshape(B * orig_T, tokens_per_patch, hidden_dim), patch_meta

        return feats, None, None

    def forward(
        self,
        video_frames: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
        alpha_weight: Optional[torch.Tensor] = None,
        stride: int = 1,
        vision_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            video_frames: [B, T, C, H, W] raw video frames
            alpha: [B, L, 1] gate on the downsampled time axis selecting the steps that use vision; None uses all
            alpha_weight: Optional [B, L, 1] gate used to weight the features (defaults to alpha)
            stride: int, step from downsampled time steps to raw frame indices
            vision_feats: Optional [B, T, D] precomputed Qwen visual features; skips vision encoding when given
        Returns:
            vis_dense: [B, L, project_dim] visual features aligned with motion_emb (0 at inactive steps)
            count_pred: [B, 1] count prediction
            active_mask: [B, L] bool, marks the downsampled time steps that use vision
        """
        B, T, C_in, H_in, W_in = video_frames.shape
        debug_log = False
        if debug_log:
            print(f"[VisionMagnifier][debug] forward start B={B} T={T} stride={stride} alpha_none={alpha is None}")
        if alpha is None:
            L = T
            raise NotImplementedError
        else:
            L = alpha.shape[1]
            alpha_flat = alpha.squeeze(-1)
            
            topk = 0
            if self.gate_topk > 0:
                topk = min(self.gate_topk, L)
            elif self.gate_topk_ratio > 0.0:
                topk = min(max(1, int(math.ceil(L * self.gate_topk_ratio))), L)

            if topk > 0:
                if self.gate_uniform_topk:
                    base_idx = (torch.arange(topk, device=alpha_flat.device) * L) // topk
                    topk_idx = base_idx.unsqueeze(0).expand(alpha_flat.size(0), -1)
                else:
                    _, topk_idx = torch.topk(alpha_flat, k=topk, dim=1, largest=True)

                active_mask = torch.zeros_like(alpha_flat, dtype=torch.bool)
                active_mask.scatter_(1, topk_idx, True)
            else:
                active_mask = torch.zeros_like(alpha_flat, dtype=torch.bool)
                warnings.warn("active_mask is zero with topk=0")
        if debug_log:
            print(f"[VisionMagnifier][debug] active_mask L={L} gate_thresh={self.gate_thresh}")

        idx_map = torch.clamp(self._get_idx_map(L, stride, video_frames.device), max=T - 1)
        param_dtype = vision_feats.dtype if vision_feats is not None else video_frames.dtype  # match runtime dtype (e.g., autocast bf16) to avoid scatter dtype mismatch
        vis_dense = torch.zeros(B, L, self.projector.out_features, device=video_frames.device, dtype=param_dtype)
        count_preds = torch.zeros(B, 1, device=video_frames.device, dtype=param_dtype)

        lengths = active_mask.sum(dim=1, dtype=torch.long)  # [B]
        max_len = lengths.max() if lengths.numel() > 0 else torch.tensor(0, device=video_frames.device)
        max_len_int = int(max_len.item()) if lengths.numel() > 0 else 0

        if self.training:
            if self._epoch_usage_epoch is None and self.current_epoch is not None:
                self._epoch_usage_epoch = int(self.current_epoch)
            compiling = False
            if hasattr(torch, "_dynamo"):
                try:
                    compiling = torch._dynamo.is_compiling()
                except Exception:
                    compiling = False
            if not compiling:
                total_len = int(lengths.sum().item())
                self._epoch_usage_frames += total_len
                self._epoch_usage_steps += int(B * L)
                self._epoch_usage_calls += 1

        def _dummy_path():
            dummy = self.projector(torch.zeros(B, 1, self.projector.in_features, device=video_frames.device, dtype=param_dtype))
            dummy_mask = torch.ones(B, 1, device=video_frames.device, dtype=torch.bool)
            dummy_agg = self._temporal_agg_varlen(dummy + self.agg_pos[:, :1, :].to(dtype=param_dtype), dummy_mask)
            return vis_dense, self.count_head(dummy_agg.mean(dim=1)), active_mask

        if max_len_int == 0:
            if debug_log:
                print("[VisionMagnifier][debug] no active frames, hitting dummy path (max_len_int=0)")
            return _dummy_path()

        valid_b, valid_l = active_mask.nonzero(as_tuple=True)
        if valid_b.numel() == 0:
            if debug_log:
                print("[VisionMagnifier][debug] active_mask has no True entries, hitting dummy path")
            return _dummy_path()
        frame_indices = idx_map[valid_l]
        seq_idx = torch.cumsum(active_mask, dim=1) - 1
        seq_idx = seq_idx.masked_fill(~active_mask, 0)
        pos = seq_idx[valid_b, valid_l]  # [sum_len]
        if vision_feats is None:
            stacked_frames = video_frames[valid_b, frame_indices]  # [sum_len, C, H, W]
            if debug_log:
                lengths_list = lengths.detach().cpu().tolist()
                print(
                    f"[VisionMagnifier][debug] collected frames={stacked_frames.shape[0]} "
                    f"lengths={lengths_list} max_len={max_len_int}"
                )
            padded_frames = self._get_padded_frame_buffer(
                (B, max_len_int, C_in, H_in, W_in),
                video_frames.device,
                video_frames.dtype,
            )
            padded_frames[valid_b, pos] = stacked_frames
            feats_padded = self._encode_real_frames(padded_frames)  # [B*max_len_int, vision_dim]
            if debug_log:
                print(f"[VisionMagnifier][debug] vision encode done feats_shape={tuple(feats_padded.shape)}")
            feats_padded = feats_padded.view(B, max_len_int, -1)
            feats_all = feats_padded[valid_b, pos]
        else:
            feats_src = vision_feats
            if feats_src.dim() == 2:
                feats_src = feats_src.unsqueeze(0)
            if feats_src.dim() != 3:
                raise RuntimeError(f"vision_feats must be [B,T,D], got shape {tuple(feats_src.shape)}")
            if feats_src.shape[0] == 1 and B > 1:
                feats_src = feats_src.expand(B, -1, -1)
            if feats_src.shape[0] != B:
                raise RuntimeError(
                    f"vision_feats batch mismatch: expected {B}, got {feats_src.shape[0]}"
                )
            feats_src = feats_src.to(device=video_frames.device, dtype=param_dtype)
            if feats_src.shape[1] != T:
                raise RuntimeError(
                    f"vision_feats length mismatch: expected {T}, got {feats_src.shape[1]}"
                )
            feats_all = feats_src[valid_b, frame_indices]

        total_len = lengths.sum()
        if (total_len == 0).all():
            if debug_log:
                print("[VisionMagnifier][debug] total_len=0 after encoding, hitting dummy path")
            return _dummy_path()
        if hasattr(torch, "_dynamo") and torch._dynamo.is_compiling():
            pass  # skip Python-side length assert during compile to avoid .item()
        else:
            if feats_all.size(0) < total_len.item():
                raise RuntimeError(
                    f"Vision feature length mismatch: expected at least {total_len.item()} frames, got {feats_all.size(0)} "
                    f"(lengths={lengths.tolist()})"
                )
        if debug_log:
            print(f"[VisionMagnifier][debug] selecting valid feats {tuple(feats_all.shape)}")

        if not self.gate_uniform_topk:
            alpha_selected = alpha_weight[valid_b, valid_l]
            feats_all = feats_all * alpha_selected

        feats_proj = self.projector(feats_all)  # [sum_len, project_dim]
        padded = torch.zeros(B, max_len_int, self.projector.out_features, device=video_frames.device, dtype=feats_proj.dtype)
        padded = padded.index_put((valid_b, pos), feats_proj)

        time_idx = self._get_arange(max_len_int, video_frames.device)
        pad_mask = time_idx.unsqueeze(0) >= lengths.unsqueeze(1)
        valid_mask = ~pad_mask

        agg_input = padded
        if self.use_temporal_conv and max_len_int > 1:
            conv_in = agg_input[:, :max_len_int, :] * valid_mask.unsqueeze(-1)
            conv_in = conv_in.transpose(1, 2)
            conv_out = self.temporal_pwconv(self.temporal_conv_act(self.temporal_dwconv(conv_in)))
            conv_out = conv_out.transpose(1, 2)
            conv_out = conv_out * valid_mask.unsqueeze(-1)
            agg_input = agg_input + conv_out

        if self.use_frame_diff and max_len_int > 1:
            delta = agg_input[:, 1:] - agg_input[:, :-1]
            if self.frame_diff_abs:
                delta = delta.abs()
            valid_pairs = valid_mask[:, 1:] & valid_mask[:, :-1]
            dt_feat = None
            if self.frame_diff_dt_mode != "none":
                frame_index_padded = torch.zeros(B, max_len_int, device=video_frames.device, dtype=torch.long)
                frame_index_padded[valid_b, pos] = frame_indices
                dt_min_int = max(1, int(self.frame_diff_dt_min))
                raw_dt = (frame_index_padded[:, 1:] - frame_index_padded[:, :-1]).clamp(min=dt_min_int)
                dt_map = torch.where(valid_pairs, raw_dt, torch.ones_like(raw_dt)).to(dtype=agg_input.dtype)
                dt_min = max(1.0, float(self.frame_diff_dt_min))
                if self.frame_diff_dt_mode in ("divide", "both"):
                    delta = delta / dt_map.unsqueeze(-1).clamp(min=dt_min)
                if self.frame_diff_dt_mode in ("embed", "both"):
                    dt_feat = self.dt_mlp(dt_map.unsqueeze(-1))
                    dt_feat = dt_feat * valid_pairs.unsqueeze(-1)
            delta = delta * valid_pairs.unsqueeze(-1)
            if dt_feat is not None:
                delta = delta + dt_feat
        if self.use_frame_diff and max_len_int > 1:
            delta_full = torch.cat([agg_input.new_zeros(B, 1, agg_input.shape[-1]), delta], dim=1)
            delta_feat = self.delta_mlp(delta_full)
            delta_support = torch.zeros_like(valid_mask)
            delta_support[:, 1:] = valid_pairs
            delta_feat = delta_feat * delta_support.unsqueeze(-1)
            if self.frame_diff_fuse == "concat":
                fused = self.delta_fuse(torch.cat([agg_input, delta_feat], dim=-1))
                agg_input = torch.where(delta_support.unsqueeze(-1), fused, agg_input)
            else:
                agg_input = agg_input + delta_feat

        if max_len_int > self.agg_pos.shape[1]:
            raise ValueError("Active frames exceed aggregator pos size")
        if self.disable_temporal_agg:
            agg_out_full = agg_input[:, :max_len_int, :]
        else:
            agg_in = agg_input[:, :max_len_int, :] + self.agg_pos[:, :max_len_int, :].to(dtype=agg_input.dtype)  # [B, max_len, D]
            agg_out_full = self._temporal_agg_varlen(agg_in, valid_mask)  # [B, max_len, D]

        agg_selected = agg_out_full[valid_b, pos]  # [sum_len, D]
        vis_dense = vis_dense.index_put((valid_b, valid_l), agg_selected)

        lengths_clamped = lengths.clamp(min=1).unsqueeze(-1)  # [B,1]
        mean_valid = (agg_out_full * valid_mask.unsqueeze(-1)).sum(dim=1) / lengths_clamped  # [B, D]
        count_preds = self.count_head(mean_valid)  # [B,1]
        if debug_log:
            print(
                f"[VisionMagnifier][debug] agg_out_full={tuple(agg_out_full.shape)} "
                f"vis_dense={tuple(vis_dense.shape)} count_preds_mean={float(count_preds.mean().item())}"
            )

        return vis_dense, count_preds, active_mask

    def _temporal_agg_varlen(self, agg_input: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        VarLen temporal aggregation using flash_attn varlen kernels to avoid padded attention.
        """
        B, S, _ = agg_input.shape
        if B == 0 or S == 0:
            return agg_input
        agg_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_input(agg_input, valid_mask)
        for block in self.temporal_blocks:
            agg_unpad = block(agg_unpad, cu_seqlens, int(max_seqlen))
        return pad_input(agg_unpad, indices, B, S)

    def _get_idx_map(self, L: int, stride: int, device: torch.device) -> torch.Tensor:
        key = (device, L, stride)
        cached = self._idx_map_cache.get(key)
        if cached is None:
            cached = torch.arange(L, device=device, dtype=torch.long) * int(stride)
            self._idx_map_cache[key] = cached
        return cached

    def _get_arange(self, max_len: int, device: torch.device) -> torch.Tensor:
        key = (device, max_len)
        cached = self._arange_cache.get(key)
        if cached is None:
            cached = torch.arange(max_len, device=device, dtype=torch.long)
            self._arange_cache[key] = cached
        return cached

    def _get_padded_frame_buffer(self, shape: tuple, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Reuse a zeroed frame buffer to avoid reallocating per forward.
        """
        key = (device, dtype, shape)
        buf = self._frame_pad_cache.get(key)
        if buf is None or tuple(buf.shape) != tuple(shape):
            buf = torch.zeros(shape, device=device, dtype=dtype)
            self._frame_pad_cache[key] = buf
        else:
            buf.zero_()
        return buf

    def _warn_no_qwen(self):
        if not self._warned_no_qwen:
            warnings.warn("[VisionMagnifier] use_real_model=False, using random visual features; check whether the Qwen vision model should be enabled.")
            self._warned_no_qwen = True

    def _patch_qwen35_vision_for_speed(self) -> None:
        """Speed up Qwen3_5VisionModel forward for large (e.g. 14x14) grids.

        Two monkey-patches, applied in-place on the loaded backbone:

        1. ``patch_embed``: the original uses ``nn.Conv3d`` with
           ``kernel_size == stride``, which is mathematically a per-patch
           linear projection but falls on a pathologically slow cuDNN
           kernel for large batched inputs (~100s/forward on 12544x3x2x16x16).
           We flatten the Conv3d weights and swap to ``F.linear``;
           output is bit-equivalent modulo bf16 accumulation order.

        2. ``backbone.forward``: we bypass the ``@capture_outputs`` /
           ``@merge_with_config_defaults`` decorators and cache the
           per-shape setup (``fast_pos_embed_interpolate``,
           ``rot_pos_emb``, ``cu_seqlens``) keyed by grid_thw. For
           Stage 3 precompute grid_thw is constant across calls so
           subsequent forwards skip the setup entirely.
        """
        import types
        import torch.nn.functional as F
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        backbone = self.backbone

        # 1. patch_embed: Conv3d -> equivalent Linear
        try:
            pe = backbone.patch_embed
            conv = pe.proj
            if isinstance(conv, nn.Conv3d):
                out_dim = conv.out_channels
                flat_weight = conv.weight.detach().reshape(out_dim, -1).clone()
                flat_bias = conv.bias.detach().clone() if conv.bias is not None else None
                new_linear = nn.Linear(
                    flat_weight.shape[1], out_dim, bias=flat_bias is not None
                ).to(device=conv.weight.device, dtype=conv.weight.dtype)
                with torch.no_grad():
                    new_linear.weight.copy_(flat_weight)
                    if flat_bias is not None:
                        new_linear.bias.copy_(flat_bias)
                new_linear.requires_grad_(False)
                # Register as a submodule so .to() / .half() / DDP
                # propagate. Original Conv3d stays in the module tree
                # (its forward is replaced) so save/load is unaffected.
                pe.add_module("_fast_linear", new_linear)
                feat_dim = flat_weight.shape[1]

                def fast_patch_embed_forward(self_pe, hidden_states):
                    flat = hidden_states.reshape(-1, feat_dim)
                    return self_pe._fast_linear(
                        flat.to(dtype=self_pe._fast_linear.weight.dtype)
                    )

                pe.forward = types.MethodType(fast_patch_embed_forward, pe)
                print(
                    "[VisionMagnifier] Patched Qwen3_5VisionPatchEmbed "
                    f"(Conv3d -> Linear, in={feat_dim}, out={out_dim})."
                )
        except Exception as exc:
            print(
                f"[VisionMagnifier] patch_embed fast-linear patch failed: {exc}; "
                "falling back to original Conv3d."
            )

        # 2. backbone.forward: per-shape pos-embed cache + decorator bypass
        pos_cache: dict = {}

        def fast_forward(self_bb, hidden_states, grid_thw, **kwargs):
            hidden_states = self_bb.patch_embed(hidden_states)

            shape_key = tuple(grid_thw.detach().cpu().flatten().tolist())
            cached = pos_cache.get(shape_key)
            if cached is None:
                pos_embeds = self_bb.fast_pos_embed_interpolate(grid_thw)
                rotary_pos_emb = self_bb.rot_pos_emb(grid_thw)
                seq_len = pos_embeds.shape[0]
                rotary = rotary_pos_emb.reshape(seq_len, -1)
                emb = torch.cat((rotary, rotary), dim=-1)
                position_embeddings = (emb.cos(), emb.sin())
                cu_seqlens = torch.repeat_interleave(
                    grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
                ).cumsum(dim=0, dtype=torch.int32)
                cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
                cached = (pos_embeds, position_embeddings, cu_seqlens)
                pos_cache[shape_key] = cached
            pos_embeds, position_embeddings, cu_seqlens = cached

            hidden_states = hidden_states + pos_embeds
            for blk in self_bb.blocks:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            merged_hidden_states = self_bb.merger(hidden_states)
            return BaseModelOutputWithPooling(
                last_hidden_state=hidden_states,
                pooler_output=merged_hidden_states,
            )

        backbone.forward = types.MethodType(fast_forward, backbone)
        print(
            "[VisionMagnifier] Patched Qwen3_5VisionModel.forward "
            "(per-shape pos cache, decorator bypass)."
        )

    def _is_main_process(self) -> bool:
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)
        self._epoch_usage_epoch = self.current_epoch
        self._epoch_usage_frames = 0
        self._epoch_usage_steps = 0
        self._epoch_usage_calls = 0

    def _log_epoch_usage(self):
        if self._epoch_usage_epoch is None:
            return
        if self._epoch_usage_logged_epoch == self._epoch_usage_epoch:
            return
        if not self._is_main_process():
            return
        active_steps = self._epoch_usage_frames
        total_steps = self._epoch_usage_steps
        active_ratio = active_steps / max(1, total_steps)
        called = self._epoch_usage_calls > 0
        print(
            f"[VisionMagnifier][Epoch {self._epoch_usage_epoch + 1}] called={called} "
            f"active_frames={active_steps} active_ratio={active_ratio:.6f}"
        )
        self._epoch_usage_logged_epoch = self._epoch_usage_epoch

    def flush_epoch_usage(self):
        self._log_epoch_usage()

    def _get_vision_params(self):
        """
        Try to fetch (patch_size, temporal_patch_size, merge_size) from backbone/config; fallback to defaults.
        """
        if hasattr(self, "_cached_vision_params"):
            return self._cached_vision_params

        patch_size = temporal_patch = merge_size = None
        cfg = getattr(self.backbone, "config", None)
        if cfg is not None:
            for candidate in (cfg, getattr(cfg, "vision_config", None)):
                if candidate is None:
                    continue
                patch_size = getattr(candidate, "patch_size", patch_size)
                temporal_patch = getattr(candidate, "temporal_patch_size", temporal_patch)
                merge_size = getattr(candidate, "spatial_merge_size", merge_size)
                if patch_size and temporal_patch and merge_size:
                    break
        if (patch_size is None or temporal_patch is None or merge_size is None) and self.qwen_model_id:
            try:
                acfg = AutoConfig.from_pretrained(self.qwen_model_id, trust_remote_code=True)
                candidate = getattr(acfg, "vision_config", None) or acfg
                patch_size = getattr(candidate, "patch_size", patch_size)
                temporal_patch = getattr(candidate, "temporal_patch_size", temporal_patch)
                merge_size = getattr(candidate, "spatial_merge_size", merge_size)
            except Exception:
                pass
        if patch_size is None or temporal_patch is None or merge_size is None:
            raise RuntimeError("Failed to resolve vision params (patch_size/temporal_patch_size/merge_size) for Qwen vision encoder.")

        self._cached_vision_params = (patch_size, temporal_patch, merge_size)
        return self._cached_vision_params

    def _assert_vision_dim(self):
        """
        Check that the configured vision_dim matches the backbone output dimension, and raise otherwise.
        """
        cfg = getattr(self.backbone, "config", None)
        resolved = None
        if cfg is not None:
            resolved = getattr(cfg, "out_hidden_size", None) or getattr(cfg, "hidden_size", None)
        if resolved is None:
            return
        if resolved != self.vision_dim:
            raise RuntimeError(
                f"[VisionMagnifier] vision_dim ({self.vision_dim}) does not match the backbone output dimension ({resolved}); "
                f"set --vision_dim to {resolved} or change the projector input dimension."
            )


class VarLenTransformerBlock(nn.Module):
    """
    Lightweight Transformer encoder block using flash_attn varlen for attention.
    """
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim * n_heads != d_model:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        hidden_dim = int(d_model * mlp_ratio)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x_unpad: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int):
        """
        x_unpad: [total_tokens, d_model] unpadded tokens
        cu_seqlens: cumulative sequence lengths for varlen attention
        max_seqlen: max sequence length in batch
        """
        x_norm = self.norm1(x_unpad)
        qkv = self.qkv(x_norm).view(x_norm.size(0), 3, self.n_heads, self.head_dim)
        attn_out = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=self.dropout.p if self.training else 0.0,
            softmax_scale=None,
            causal=False,
        )
        attn_out = attn_out.view(x_norm.size(0), self.d_model)
        x = x_unpad + self.dropout(self.proj(attn_out))
        x_norm2 = self.norm2(x)
        x = x + self.mlp(x_norm2)
        return x

class LoRALinear(nn.Module):
    """
    Minimal LoRA wrapper: y = W x + scale * B(Ax), base weight frozen.
    """
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self.scale = alpha / float(r)

        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @classmethod
    def from_linear(cls, base: nn.Linear, r: int = 8, alpha: int = 16):
        module = cls(base, r=r, alpha=alpha)
        return module

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))
