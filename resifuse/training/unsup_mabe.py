"""Unsupervised DEC (Deep Embedded Clustering) behavior segmentation for MABe.

Pipeline:
  1. Load pretrain ckpt into ResiFuseClassifier (same model as finetune).
  2. Compute global frame-select masks (residual or uniform) once on train/val/test.
  3. Freeze everything except vision_projector + motion_feat_norm + fused_feat_norm
     + vision_seq_encoder (the vision projection/fusion path) + cluster centers.
  4. K-means initialize K centers on pre-BiGRU fused features from train split.
  5. Train DEC loss (KL(P || Q) where Q = Student's-t(emb, centers), P = Q^2 / sum(Q^2)).
  6. After DEC: build cluster->class majority map on train split (GT available),
     apply it to train/val/test, compute single-label F1.

Designed to run on a single GPU (no DDP). Launch as:
    python -m resifuse.training.unsup_mabe --config ./config_mabe_finetune.yaml

Uses the same finetune config as the supervised path; reads extra DEC knobs:
  dec_num_clusters, dec_epochs, dec_lr, dec_centers_lr, dec_kmeans_samples,
  dec_kmeans_init_iters, dec_save_dir, unsup_run_name.
"""
from __future__ import annotations

import argparse
import os
import time
from types import SimpleNamespace
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from resifuse.data.datasets import MaBeMouseDataset
from resifuse.models.model import ResiFuseClassifier
from resifuse.config import load_config_with_defaults

from resifuse.training.mabe import (
    is_main_process,
    _get_base_model,
    _get_label_valid_mode,
    _select_frame_mask,
    _sanitize_frame_labels,
    _majority_aggregate_frame_labels,
    _downsample_frame_labels,
    _downsample_frame_mask,
    _normalize_metadata_list,
    _get_sample_key,
    _init_f1_counts,
    _counts_to_f1,
    maybe_load_pretrain_weights,
)


# ----------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------

def parse_args(config_path: str | None = None):
    cfg = load_config_with_defaults(config_path)
    # Always run unsup regardless of stage=pretrain/finetune in yaml.
    cfg["stage"] = "unsup"
    # Use dec_lr / dec_epochs rather than finetune_lr / finetune_epochs.
    cfg["lr"] = float(cfg.get("dec_lr", cfg.get("finetune_lr", 1e-4)) or 1e-4)
    cfg["epochs"] = int(cfg.get("dec_epochs", 20) or 20)
    return SimpleNamespace(**cfg)


# ----------------------------------------------------------------------
# Trainable-param control: keep only the vision projection path alive
# ----------------------------------------------------------------------

_UNSUP_TRAINABLE_PREFIXES = (
    "vision_projector.",
    "vision_seq_encoder.",
    "motion_feat_norm",
    "fused_feat_norm",
)


def _configure_unsup_trainables(model: nn.Module, args=None, is_main: bool = True) -> None:
    base_model = _get_base_model(model)
    for p in base_model.parameters():
        p.requires_grad = False
    prefixes = list(_UNSUP_TRAINABLE_PREFIXES)
    use_bigru = args is not None and _dec_use_bigru(args)
    if use_bigru:
        prefixes.append("motion_seq_encoder.")
    n_train = 0
    for name, p in base_model.named_parameters():
        if any(name.startswith(pf) or f".{pf}" in name for pf in prefixes):
            p.requires_grad = True
            n_train += p.numel()
    total = sum(p.numel() for p in base_model.parameters())
    if is_main:
        label = "projection+fusion+BiGRU" if use_bigru else "projection+fusion only"
        print(
            f"[Unsup] Trainables ({label}): "
            f"{n_train/1e6:.3f}M / {total/1e6:.3f}M"
        )


# ----------------------------------------------------------------------
# Global frame-select mask (residual / uniform only)
# ----------------------------------------------------------------------

@torch.no_grad()
def compute_global_masks(base_model, dataloader, device, budget_ratio, mode="residual", is_main=True):
    base_model.eval()
    all_scores: Dict[str, torch.Tensor] = {}
    n_batches = 0
    for batch in dataloader:
        n_batches += 1
        motion_feats = batch["motion_feats"].to(device)
        B, A, T = motion_feats.shape[:3]
        motion_flat = motion_feats.view(B * A, T, -1)
        scout_out = base_model.motion_scout(motion_flat)
        z_q = scout_out["z_q"]
        z_e = scout_out["z_e"]
        L = z_q.shape[1]
        if mode == "residual":
            r2 = (z_e - z_q).norm(dim=-1)
            scores = r2.view(B, A, L).mean(dim=1)
        elif mode == "uniform":
            scores = torch.zeros(B, L, device=device)
        else:
            raise ValueError(f"unsup only supports mode in {{residual, uniform}}, got {mode}")
        for b in range(B):
            key = _get_sample_key(batch, b)
            all_scores[key] = scores[b].cpu()

    keys = list(all_scores.keys())
    if not keys:
        return {}
    if mode == "uniform":
        # Evenly spaced selection per window.
        masks = {}
        sample_L = all_scores[keys[0]].numel()
        total_k = max(1, int(sample_L * budget_ratio))
        idx = torch.linspace(0, sample_L - 1, total_k).long()
        template = torch.zeros(sample_L, dtype=torch.bool)
        template[idx] = True
        for k in keys:
            masks[k] = template.clone()
    else:
        # === Per-slot bias correction (removes CNN boundary artifact) ===
        # See resifuse/training/mabe.py for rationale. Subtracts per-slot mean so
        # edge slots don't systematically win top-budget_ratio selection.
        if len(keys) > 1:
            score_stack = torch.stack([all_scores[k] for k in keys], dim=0)  # [W, L]
            slot_bias = score_stack.mean(dim=0)  # [L]
            for k in keys:
                all_scores[k] = all_scores[k] - slot_bias
            if is_main:
                print(
                    f"  [compute_global_masks] per-slot bias correction: "
                    f"mean={slot_bias.mean().item():.4f}  "
                    f"range=[{slot_bias.min().item():.4f}, {slot_bias.max().item():.4f}]",
                    flush=True,
                )

        vals = torch.cat([all_scores[k] for k in keys])
        total_frames = vals.numel()
        total_sel = max(1, int(total_frames * budget_ratio))
        threshold = vals.topk(total_sel).values[-1].item()
        masks = {k: (all_scores[k] >= threshold) for k in keys}

    if is_main:
        actual = sum(int(m.sum().item()) for m in masks.values())
        total = sum(int(m.numel()) for m in masks.values())
        print(
            f"  [compute_global_masks] mode={mode} batches={n_batches} windows={len(masks)} "
            f"selected {actual}/{total} ({actual/max(total,1)*100:.1f}%)",
            flush=True,
        )
    base_model.train()
    return masks


# ----------------------------------------------------------------------
# DEC primitives
# ----------------------------------------------------------------------

def kmeans_init(samples: torch.Tensor, k: int, iters: int = 50, seed: int | None = 0,
                batch_size: int = 4096) -> torch.Tensor:
    if samples.numel() == 0:
        raise RuntimeError("No samples for kmeans init.")
    rng = torch.Generator(device=samples.device)
    if seed is not None:
        rng.manual_seed(int(seed))
    perm = torch.randperm(samples.shape[0], generator=rng, device=samples.device)
    centers = samples[perm[:k]].clone()
    counts = torch.zeros(k, device=samples.device)
    for _ in range(int(iters)):
        new_sum = torch.zeros_like(centers)
        new_cnt = torch.zeros(k, device=samples.device)
        for start in range(0, samples.shape[0], batch_size):
            batch = samples[start:start + batch_size]
            d = torch.cdist(batch, centers)
            assign = d.argmin(dim=1)
            for j in range(k):
                sel = (assign == j)
                if sel.any():
                    new_sum[j] += batch[sel].sum(dim=0)
                    new_cnt[j] += sel.sum().float()
        alive = new_cnt > 0
        centers_new = centers.clone()
        centers_new[alive] = new_sum[alive] / new_cnt[alive].unsqueeze(-1)
        # Revive dead clusters with random samples.
        n_dead = int((~alive).sum().item())
        if n_dead > 0:
            resample_idx = torch.randint(0, samples.shape[0], (n_dead,), generator=rng, device=samples.device)
            centers_new[~alive] = samples[resample_idx]
        centers = centers_new
        counts = new_cnt
    return centers


def _soft_assign(emb: torch.Tensor, centers: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Student-t soft assignment q (DEC, alpha=1)."""
    dists = torch.cdist(emb, centers) ** 2
    q = 1.0 / (1.0 + dists)
    return q / (q.sum(dim=1, keepdim=True) + eps)


def _sharpen_target(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """DEC target p = q^2 / f_j, row-normalized; f_j = cluster frequency over q's rows."""
    p = (q ** 2) / (q.sum(dim=0, keepdim=True) + eps)
    return p / (p.sum(dim=1, keepdim=True) + eps)


def dec_loss(emb: torch.Tensor, centers: torch.Tensor, eps: float = 1e-8,
             target: torch.Tensor | None = None) -> torch.Tensor:
    """DEC KL loss between target distribution p (sharpened) and soft assignment q.

    ``target=None`` sharpens the current mini-batch's own q (per-step target).
    Otherwise ``target`` is a precomputed p for these rows (standard DEC:
    computed over the whole train set and held fixed between updates).
    """
    q = _soft_assign(emb, centers, eps)
    # Detach p to make it the target distribution (standard DEC).
    p = (_sharpen_target(q, eps) if target is None else target).detach()
    loss = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=1).mean()
    return loss


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Return flat off-diagonal entries of a square matrix. Used by VICReg."""
    n, m = x.shape
    assert n == m, f"expected square, got {(n, m)}"
    # Trick: flatten, drop last, reshape, drop first column, flatten.
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_losses(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    variance_target: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the three VICReg sub-losses between two batches of features
    that should represent "the same underlying thing" (e.g. two temporal
    neighbors of the same window).

    Both tensors are expected shape ``[N, D]``. Returns
    ``(invariance, variance, covariance)`` losses; caller weighs them.

    Semantics:
      * **invariance** — MSE between paired features. Pulls positive pairs
        together.
      * **variance** — ``mean(relu(gamma - std_d))`` over feature dims.
        Forces each feature dimension to retain non-trivial variance
        across the batch. This is VICReg's **anti-collapse** guarantee —
        any dim that squeezes below ``gamma`` gets a linear gradient
        pushing it back out. The main cure for DEC collapsing onto the
        majority class.
      * **covariance** — sum of squared off-diagonals of ``cov(F)``,
        divided by ``D``. Decorrelates feature dimensions so information
        is spread rather than piled into a single direction.

    Standard VICReg weights (see the paper) are λ_inv=25, λ_var=25,
    λ_cov=1. Those are tuned for images with strong augmentations where
    invariance is the dominant term. For DEC augmentation by "temporal
    neighbor" (a much weaker form of positive pair), a softer invariance
    weight is typical — see the CLI defaults / recommendations.
    """
    if features_a.shape != features_b.shape:
        raise RuntimeError(
            f"VICReg inputs must match shape; got {tuple(features_a.shape)} vs "
            f"{tuple(features_b.shape)}."
        )
    if features_a.dim() != 2:
        raise RuntimeError(f"VICReg expects [N, D]; got {tuple(features_a.shape)}.")

    N, D = features_a.shape
    # Invariance: simple MSE.
    inv_loss = torch.nn.functional.mse_loss(features_a, features_b)

    # Variance: compute per-dim std after centering.
    fa = features_a - features_a.mean(dim=0, keepdim=True)
    fb = features_b - features_b.mean(dim=0, keepdim=True)
    std_a = torch.sqrt(fa.var(dim=0, unbiased=False) + eps)
    std_b = torch.sqrt(fb.var(dim=0, unbiased=False) + eps)
    var_loss = (
        torch.mean(torch.nn.functional.relu(variance_target - std_a))
        + torch.mean(torch.nn.functional.relu(variance_target - std_b))
    ) / 2.0

    # Covariance: normalized sum-of-squares over off-diagonals.
    if N > 1:
        cov_a = (fa.T @ fa) / (N - 1)
        cov_b = (fb.T @ fb) / (N - 1)
        cov_loss = (
            _off_diagonal(cov_a).pow(2).sum() / D
            + _off_diagonal(cov_b).pow(2).sum() / D
        ) / 2.0
    else:
        cov_loss = inv_loss.new_zeros(())

    return inv_loss, var_loss, cov_loss


def assign_clusters(samples: torch.Tensor, centers: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
    out = []
    for start in range(0, samples.shape[0], batch_size):
        batch = samples[start:start + batch_size]
        d = torch.cdist(batch, centers)
        out.append(d.argmin(dim=1))
    return torch.cat(out, dim=0) if out else torch.zeros((0,), dtype=torch.long, device=samples.device)


def build_majority_map(cluster_ids: torch.Tensor, gt_class: torch.Tensor,
                       num_clusters: int, num_classes: int) -> torch.Tensor:
    """Count-based cluster -> class majority vote.

    For each cluster k, pick the class with the largest number of labeled
    frames landing in k. Simple and stable when class frequencies are
    balanced. **Breaks on heavily imbalanced data** (e.g. CalMS21 where
    investigation is 77% of train positives — every cluster's argmax
    degenerates to investigation because minority classes never reach a
    majority).

    Returns a LongTensor ``[num_clusters]``. Empty clusters get ``-1`` so
    callers can detect/report them.
    """
    counts = torch.zeros((num_clusters, num_classes), dtype=torch.long)
    cluster_ids = cluster_ids.long().cpu()
    gt_class = gt_class.long().cpu()
    valid = gt_class >= 0
    for c, t in zip(cluster_ids[valid].tolist(), gt_class[valid].tolist()):
        if 0 <= c < num_clusters and 0 <= t < num_classes:
            counts[c, t] += 1
    mapping = counts.argmax(dim=1)
    empty = counts.sum(dim=1) == 0
    mapping = mapping.masked_fill(empty, -1)
    return mapping


def build_majority_map_f1(cluster_ids: torch.Tensor, gt_class: torch.Tensor,
                          num_clusters: int, num_classes: int) -> torch.Tensor:
    """F1-based cluster -> class assignment (per-cluster argmax over
    per-class F1).

    For each cluster k and each class y, compute the binary F1 of
    "frame-in-cluster-k" as a predictor of "frame-is-class-y" (on the
    labeled train frames only). Cluster k is mapped to the class that
    maximises this F1.

    Semantically equivalent to ``scripts/unsup_eval.py``'s
    ``_argmax_cluster_to_class`` with ``threshold=0``. Unlike count-based
    majority, F1-based is **rebust to class imbalance** — precision and
    recall are both considered, so a minority-class-pure cluster can win
    its argmax even when its frame count is dwarfed by a majority-class
    cluster's incidental frames.

    Returns a LongTensor ``[num_clusters]``. Empty clusters get ``-1``.
    Uses the same GT read as ``build_majority_map`` — just a different
    statistic for picking the class.
    """
    cluster_ids = cluster_ids.long().cpu()
    gt_class = gt_class.long().cpu()
    valid = gt_class >= 0
    ci = cluster_ids[valid]
    gy = gt_class[valid]
    in_range = (ci >= 0) & (ci < num_clusters) & (gy >= 0) & (gy < num_classes)
    ci = ci[in_range].numpy()
    gy = gy[in_range].numpy()
    import numpy as _np
    tp = _np.zeros((num_clusters, num_classes), dtype=_np.int64)
    _np.add.at(tp, (ci, gy), 1)
    cluster_total = tp.sum(axis=1, keepdims=True)       # [K, 1]  frames in each cluster
    class_total = tp.sum(axis=0, keepdims=True)         # [1, C]  labeled frames per class
    fp = cluster_total - tp                             # [K, C]  in cluster k but not class y
    fn = class_total - tp                               # [K, C]  class y but not in cluster k
    denom = 2 * tp + fp + fn                            # [K, C]
    with _np.errstate(divide="ignore", invalid="ignore"):
        f1 = _np.where(denom > 0, (2 * tp) / denom, 0.0)
    mapping = _np.argmax(f1, axis=1).astype(_np.int64)   # [K]
    empty = cluster_total.squeeze(-1) == 0
    mapping[empty] = -1
    return torch.from_numpy(mapping).long()


# Mode dispatch table for run_unsup.
_MAJORITY_MAP_BUILDERS = {
    "count": build_majority_map,
    "f1":    build_majority_map_f1,
}


def _resolve_majority_map_builder(mode: str):
    mode = str(mode or "count").lower()
    if mode not in _MAJORITY_MAP_BUILDERS:
        raise ValueError(
            f"Unknown majority_map_mode={mode!r}. Valid: {sorted(_MAJORITY_MAP_BUILDERS)}."
        )
    return _MAJORITY_MAP_BUILDERS[mode], mode


# ----------------------------------------------------------------------
# Single-label F1 accumulator (shared by all three datasets)
# ----------------------------------------------------------------------

def accumulate_f1_counts_single_label(
    counts: torch.Tensor,
    pred_class: torch.Tensor,
    gt_class: torch.Tensor,
    valid: torch.Tensor,
    metadata,
    seen_by_video: dict,
    num_classes: int,
) -> None:
    """counts [num_classes, 4] columns = tp / fp / fn / total.

    Accumulates per-class single-label F1 by comparing pred_class == c vs gt_class == c.
    Dedups frames by (video_id, start + t*stride) using seen_by_video (same scheme as
    `_accumulate_f1_counts_frame_index`).
    """
    if pred_class.dim() == 1:
        pred_class = pred_class.unsqueeze(0)
    if gt_class.dim() == 1:
        gt_class = gt_class.unsqueeze(0)
    if valid.dim() == 1:
        valid = valid.unsqueeze(0)
    B, T = pred_class.shape
    meta_list = _normalize_metadata_list(metadata, batch_size=B)
    if meta_list is None or len(meta_list) != B:
        raise RuntimeError("Missing or invalid metadata for unsup single-label F1.")
    pred_class = pred_class.to(device=counts.device, dtype=torch.long)
    gt_class = gt_class.to(device=counts.device, dtype=torch.long)
    valid = valid.to(device=counts.device, dtype=torch.bool)
    for b, meta in enumerate(meta_list):
        video_id = meta.get("video_id")
        start = meta.get("start_frame")
        if start is None:
            start = meta.get("start")
        stride = int(meta.get("ds_stride", 1) or 1)
        if video_id is None or start is None:
            continue
        seen = seen_by_video.setdefault(str(video_id), set())
        valid_idx = torch.nonzero(valid[b], as_tuple=False).squeeze(-1)
        for t in valid_idx.tolist():
            frame_id = int(start) + int(t) * stride
            if frame_id in seen:
                continue
            seen.add(frame_id)
            p = int(pred_class[b, t].item())
            g = int(gt_class[b, t].item())
            if g < 0 or g >= num_classes:
                continue
            for c in range(num_classes):
                is_pred = (p == c)
                is_gt = (g == c)
                if is_pred and is_gt:
                    counts[c, 0] += 1.0
                elif is_pred and not is_gt:
                    counts[c, 1] += 1.0
                elif (not is_pred) and is_gt:
                    counts[c, 2] += 1.0
            counts[:, 3] += 1.0


# ----------------------------------------------------------------------
# Forward → fused features (per-frame, per-animal) + per-frame valid mask
# ----------------------------------------------------------------------

def _build_pad_and_valid(batch, device, T, A, B):
    """Returns (pad_mask [B, A, T] bool, frame_valid [B, T] bool).

    frame_valid marks frames that are not padding (used to filter DEC input /
    F1 eval). pad_mask is the per-(batch, animal, time) padding used by
    ResiFuseClassifier.forward (True = padded).
    """
    frame_mask = batch.get("frame_mask")
    if frame_mask is None:
        pad_mask = torch.zeros(B, A, T, dtype=torch.bool, device=device)
        frame_valid = torch.ones(B, T, dtype=torch.bool, device=device)
        return pad_mask, frame_valid
    frame_mask = frame_mask.to(device)
    if frame_mask.dim() == 1:
        frame_mask = frame_mask.unsqueeze(0)
    if frame_mask.dim() == 2:
        # [B, T]
        pad_mask = ~frame_mask.unsqueeze(1).expand(B, A, T)
        frame_valid = frame_mask
    elif frame_mask.dim() == 3:
        # [B, A, T]
        pad_mask = ~frame_mask
        frame_valid = frame_mask.any(dim=1)
    else:
        raise RuntimeError(f"Unexpected frame_mask shape: {tuple(frame_mask.shape)}")
    return pad_mask.bool(), frame_valid.bool()


def _forward_fused(model, batch, device, args, global_masks=None, grad_enabled=False):
    """Single forward call; returns (fused [B, A, L, D_z], frame_valid_ds [B, L] bool, logits_len L).

    When grad_enabled=True, keeps autograd on for the forward so DEC can
    backprop through the vision projection path. Callers that want a
    no-grad eval pass should either set grad_enabled=False and wrap the
    caller in torch.no_grad(), or rely on this function's inner
    set_grad_enabled.
    """
    kps = batch["keypoints"].to(device)
    motion_feats = batch["motion_feats"].to(device)
    B, A, T = motion_feats.shape[:3]
    pad_mask, frame_valid = _build_pad_and_valid(batch, device, T, A, B)

    frame_select_mask = None
    if global_masks is not None:
        masks_batch = []
        for b in range(B):
            key = _get_sample_key(batch, b)
            m = global_masks.get(key)
            if m is None:
                masks_batch.append(None)
            else:
                masks_batch.append(m.to(device))
        if all(m is not None for m in masks_batch):
            frame_select_mask = torch.stack(masks_batch, dim=0)

    vision_feats = batch.get("vision_feats")
    if vision_feats is not None:
        vision_feats = vision_feats.to(device)
    vision_patch_tokens = batch.get("vision_patch_tokens")
    if vision_patch_tokens is not None:
        vision_patch_tokens = vision_patch_tokens.to(device)

    with torch.set_grad_enabled(grad_enabled):
        out = model(
            motion_feats,
            kps,
            pad_mask=pad_mask,
            vision_patch_tokens=vision_patch_tokens,
            vision_feats=vision_feats,
            frame_select_mask=frame_select_mask,
        )
    # Pick pre- or post-BiGRU fused features based on --dec-use-bigru.
    if _dec_use_bigru(args):
        fused = out["fused_features_post_bigru"]  # [B, A, L, D_z]
    else:
        fused = out["fused_features"]  # [B, A, L, D_z]
    L = fused.shape[2]
    # Downsample frame_valid [B, T] -> [B, L] to match pre-BiGRU length.
    if L != T:
        stride_frame = max(1, T // L)
        frame_valid_ds = frame_valid[:, ::stride_frame][:, :L]
        if frame_valid_ds.shape[1] < L:
            pad = torch.zeros(B, L - frame_valid_ds.shape[1], dtype=torch.bool, device=device)
            frame_valid_ds = torch.cat([frame_valid_ds, pad], dim=1)
    else:
        frame_valid_ds = frame_valid
    return fused, frame_valid_ds, L


# ----------------------------------------------------------------------
# DEC training / feature collection
# ----------------------------------------------------------------------

def _train_agg_view(fused, frame_valid_ds, train_agg):
    """Per-window DEC rows in the train-agg layout.

    Returns ``(x [B, R, D], valid [B, R])``; ``x.reshape(-1, D)[valid.reshape(-1)]``
    is exactly the ``emb`` that ``train_one_epoch_dec`` feeds DEC.
    """
    B, A, L, D_z = fused.shape
    if train_agg == "per_animal":
        return (fused.reshape(B, A * L, D_z),
                frame_valid_ds.unsqueeze(1).expand(B, A, L).reshape(B, A * L))
    if train_agg == "mean":
        return fused.mean(dim=1), frame_valid_ds
    # "concat": [B, A, L, D_z] -> [B, L, A*D_z]
    return fused.permute(0, 2, 1, 3).reshape(B, L, A * D_z), frame_valid_ds


def _collect_fused(model, loader, device, args, global_masks, max_samples=None,
                   random_subsample=False, seed=0):
    """Collect pre-BiGRU fused features over a loader, filtering invalid frames.

    ``random_subsample=False`` keeps the first ``max_samples`` rows in loader
    order (legacy). ``True`` collects the whole loader and draws
    ``max_samples`` rows uniformly at random, so k-means sees every video.
    """
    model.eval()
    feats = []
    total = 0
    train_agg = _dec_train_agg(args)
    for batch in loader:
        fused, frame_valid_ds, L = _forward_fused(model, batch, device, args, global_masks=global_masks,
                                                   grad_enabled=False)
        # Feature pool must match the distribution DEC will train on,
        # otherwise k-means centers drift during the first DEC steps.
        x, valid = _train_agg_view(fused, frame_valid_ds, train_agg)
        sel = x.reshape(-1, x.shape[-1])[valid.reshape(-1)]
        if sel.numel() > 0:
            feats.append(sel.detach().to(torch.float32).cpu())
            total += sel.shape[0]
        if not random_subsample and max_samples is not None and total >= max_samples:
            break
    if not feats:
        raise RuntimeError("No fused features collected for DEC.")
    out = torch.cat(feats, dim=0)
    if random_subsample and max_samples is not None and out.shape[0] > max_samples:
        g = torch.Generator().manual_seed(int(seed))
        out = out[torch.randperm(out.shape[0], generator=g)[:int(max_samples)]]
    return out


@torch.no_grad()
def _calibrate_vision_center(model, loader, device, args, global_masks):
    """Set ``vision_feat_mean`` to the exact train-set mean of the visual term
    (selected frames only) and freeze it.

    Without this the mean buffer (absent from pretrain ckpts) stays 0 until the
    first train-mode forward, so k-means is fit on uncentered features and the
    first DEC step then shifts only the vision-selected frames by -mu — at
    0 < budget < 1 that strands roughly a budget-sized share of the centers.
    Returns the number of frames averaged (0 = no visual term, nothing done).
    """
    base = _get_base_model(model)
    if not getattr(base, "vision_center_mean", False):
        return 0
    model.eval()
    base._vision_center_accum = {}
    try:
        for batch in loader:
            _forward_fused(model, batch, device, args, global_masks=global_masks, grad_enabled=False)
        accum = base._vision_center_accum
    finally:
        base._vision_center_accum = None
    count = float(accum.get("count", 0.0))
    if count <= 0:
        return 0
    base.vision_feat_mean.copy_((accum["sum"] / count).to(base.vision_feat_mean.device))
    base.vision_feat_mean_inited.fill_(1.0)
    base.vision_center_frozen = True
    return int(count)


@torch.no_grad()
def _dec_epoch_pass(model, loader, centers, device, args, global_masks, *,
                    build_targets, reseed_empty, optimizer=None, seed=0, batch_size=8192):
    """One eval pass over train in the DEC train-agg layout, used to

      * reseed empty clusters (``reseed_empty``): each center with no train
        rows is moved onto a random member of the currently largest cluster
        (split-largest), and its AdamW moments are zeroed;
      * build the standard DEC target (``build_targets``): p computed over the
        whole train set with the current centers, held fixed for the epoch.

    Returns ``(targets, n_reseeded)``; ``targets`` maps sample key -> p
    ``[R, K]`` (zeros on invalid rows) or is None.
    """
    model.eval()
    train_agg = _dec_train_agg(args)
    keys, xs, valids = [], [], []
    for batch in loader:
        fused, frame_valid_ds, _ = _forward_fused(model, batch, device, args, global_masks=global_masks,
                                                  grad_enabled=False)
        x, valid = _train_agg_view(fused, frame_valid_ds, train_agg)
        keys.extend(_get_sample_key(batch, b) for b in range(x.shape[0]))
        xs.append(x.to(torch.float32).cpu())
        valids.append(valid.cpu())
    if len(set(keys)) != len(keys):
        raise RuntimeError("DEC epoch targets need unique sample keys; got duplicates.")
    X = torch.cat(xs, dim=0)          # [N, R, D]
    V = torch.cat(valids, dim=0)      # [N, R]
    flat = X[V]                       # [M, D]
    K = centers.shape[0]

    def _chunks():
        for s in range(0, flat.shape[0], batch_size):
            yield flat[s:s + batch_size].to(device)

    n_reseeded = 0
    if reseed_empty:
        assign = torch.cat([torch.cdist(x, centers).argmin(dim=1).cpu() for x in _chunks()])
        counts = torch.bincount(assign, minlength=K)
        dead = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        g = torch.Generator().manual_seed(int(seed))
        for j in dead:
            big = int(counts.argmax())
            members = torch.nonzero(assign == big, as_tuple=False).flatten()
            i = int(members[torch.randint(len(members), (1,), generator=g)])
            centers.data[j] = flat[i].to(device=centers.device, dtype=centers.dtype)
            # j now takes ~half of big; counts[j] stays 0 so a reseeded center
            # (no members in `assign`) is never picked as the one to split.
            counts[big] -= counts[big] // 2
        if dead and optimizer is not None:
            state = optimizer.state.get(centers, {})
            for k in ("exp_avg", "exp_avg_sq"):
                if k in state:
                    state[k][dead] = 0.0
        n_reseeded = len(dead)

    targets = None
    if build_targets:
        q = torch.cat([_soft_assign(x, centers).cpu() for x in _chunks()])
        P = torch.zeros(X.shape[0], X.shape[1], K)
        P[V] = _sharpen_target(q)
        targets = {k: P[i] for i, k in enumerate(keys)}
    return targets, n_reseeded


def train_one_epoch_dec(model, loader, centers, optimizer, device, args, global_masks, targets=None):
    """``targets``: optional sample key -> p ``[R, K]`` from ``_dec_epoch_pass``;
    when given, DEC uses that fixed per-epoch target instead of sharpening
    each mini-batch's own q."""
    model.train()
    base_model = _get_base_model(model)
    loss_meter = 0.0
    dec_loss_meter = 0.0
    vic_inv_meter = 0.0
    vic_var_meter = 0.0
    vic_cov_meter = 0.0
    n_batches = 0
    train_agg = _dec_train_agg(args)
    vic_cfg = _vicreg_config(args)
    for batch in loader:
        fused, frame_valid_ds, L = _forward_fused(model, batch, device, args, global_masks=global_masks,
                                                   grad_enabled=True)
        B, A, _, D_z = fused.shape

        # ---- Build per-frame embedding pool (same as the eval path). ----
        # We also keep a [B, L, D] tensor (T_dim preserved) for building
        # temporal-neighbor pairs to feed VICReg — so the VICReg and DEC
        # losses always see consistent features on every step.
        if train_agg == "per_animal":
            # Per-animal: each (b, a, t) is its own row. For VICReg we pair
            # (b, a, t) with (b, a, t+shift) — both animals' streams are
            # treated as independent sequences.
            fused_for_tv = fused.reshape(B * A, L, D_z)               # [B*A, L, D_z]
            frame_valid_tv = frame_valid_ds.unsqueeze(1).expand(B, A, L).reshape(B * A, L)
            valid_ba = frame_valid_ds.unsqueeze(1).expand(B, A, L).reshape(-1)
            emb = fused.reshape(-1, D_z)[valid_ba].to(torch.float32)
            tv_dim = D_z
        elif train_agg == "mean":
            fused_for_tv = fused.mean(dim=1)                          # [B, L, D_z]
            frame_valid_tv = frame_valid_ds
            valid_flat = frame_valid_ds.reshape(-1)
            emb = fused_for_tv.reshape(-1, D_z)[valid_flat].to(torch.float32)
            tv_dim = D_z
        else:  # "concat": [B, L, A*D_z], 1 sample/frame, per-animal info preserved
            fused_for_tv = fused.permute(0, 2, 1, 3).reshape(B, L, A * D_z)
            frame_valid_tv = frame_valid_ds
            valid_flat = frame_valid_ds.reshape(-1)
            emb = fused_for_tv.reshape(-1, A * D_z)[valid_flat].to(torch.float32)
            tv_dim = A * D_z
        if emb.numel() == 0:
            continue

        p_rows = None
        if targets is not None:
            _, valid_rows = _train_agg_view(fused, frame_valid_ds, train_agg)
            p_win = torch.stack([targets[_get_sample_key(batch, b)] for b in range(B)]).to(device)
            p_rows = p_win.reshape(-1, p_win.shape[-1])[valid_rows.reshape(-1)]
            if p_rows.shape[0] != emb.shape[0]:
                raise RuntimeError(
                    f"DEC target rows {p_rows.shape[0]} != embedding rows {emb.shape[0]}; "
                    f"frame validity differs between the target pass and training."
                )
        loss_dec = dec_loss(emb, centers, target=p_rows)

        # ---- Optional VICReg regularization (temporal-neighbor positives) ----
        loss_vic_inv = emb.new_zeros(())
        loss_vic_var = emb.new_zeros(())
        loss_vic_cov = emb.new_zeros(())
        if vic_cfg["active"]:
            shift = int(vic_cfg["shift"])
            if shift < L:
                # Pair frame t with frame t+shift in the same (b[*a]) window.
                f_a = fused_for_tv[:, :-shift, :]                     # [*, L-shift, D]
                f_b = fused_for_tv[:, shift:, :]                      # [*, L-shift, D]
                v_a = frame_valid_tv[:, :-shift]
                v_b = frame_valid_tv[:, shift:]
                pair_valid = (v_a & v_b).reshape(-1)                  # [*·(L-shift)]
                f_a = f_a.reshape(-1, tv_dim)[pair_valid].to(torch.float32)
                f_b = f_b.reshape(-1, tv_dim)[pair_valid].to(torch.float32)
                if f_a.shape[0] >= 2:  # need at least 2 samples for variance/covariance
                    loss_vic_inv, loss_vic_var, loss_vic_cov = vicreg_losses(
                        f_a, f_b,
                        variance_target=float(vic_cfg["variance_target"]),
                    )

        loss = loss_dec
        if vic_cfg["active"]:
            loss = (
                loss
                + float(vic_cfg["inv"]) * loss_vic_inv
                + float(vic_cfg["var"]) * loss_vic_var
                + float(vic_cfg["cov"]) * loss_vic_cov
            )

        optimizer.zero_grad()
        loss.backward()
        grad_clip = float(getattr(args, "grad_clip_norm", 0.0) or 0.0)
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] + [centers],
                max_norm=grad_clip,
            )
        optimizer.step()
        loss_meter += float(loss.item())
        dec_loss_meter += float(loss_dec.item())
        vic_inv_meter += float(loss_vic_inv.item())
        vic_var_meter += float(loss_vic_var.item())
        vic_cov_meter += float(loss_vic_cov.item())
        n_batches += 1

    denom = max(n_batches, 1)
    return {
        "total": loss_meter / denom,
        "dec": dec_loss_meter / denom,
        "vic_inv": vic_inv_meter / denom,
        "vic_var": vic_var_meter / denom,
        "vic_cov": vic_cov_meter / denom,
    }


# ----------------------------------------------------------------------
# GT label conversion + evaluation
# ----------------------------------------------------------------------

def _downsample_labels_unified(labels: torch.Tensor, target_len: int, stride: int) -> torch.Tensor:
    """Strided + padded downsample that handles both multi-hot [B, T, C] and
    single-label int [B, T] (freely).

    MaBe's `_downsample_frame_labels` assumes a trailing class axis and breaks
    for the freely 2-D path; this unified version dispatches on dim.
    """
    if labels.dim() == 3:
        return _downsample_frame_labels(labels, target_len, stride)
    if labels.dim() != 2:
        raise RuntimeError(f"Unexpected labels dim: {labels.dim()}")
    if stride > 1:
        labels = labels[:, ::stride]
    if labels.shape[1] < target_len:
        pad_len = target_len - labels.shape[1]
        pad = torch.full((labels.shape[0], pad_len), -1, dtype=labels.dtype, device=labels.device)
        labels = torch.cat([labels, pad], dim=1)
    return labels[:, :target_len]


def _labels_to_single_class(frame_labels: torch.Tensor) -> torch.Tensor:
    """Convert multi-hot [B, T, C] -> single-label [B, T] long via argmax on
    rows with any positive; returns -1 for rows whose labels are all < 0.
    For already-single-label [B, T] int input, returns it clamped to long.
    """
    if frame_labels.dim() == 3:
        valid_row = (frame_labels >= 0).any(dim=-1)
        has_pos = (frame_labels >= 0.5).any(dim=-1)
        gt = frame_labels.clamp(min=0).argmax(dim=-1).long()
        gt = torch.where(has_pos & valid_row, gt, torch.full_like(gt, -1))
        return gt
    if frame_labels.dim() == 2:
        return frame_labels.long()
    raise RuntimeError(f"Unexpected frame_labels dim: {frame_labels.dim()}")


def _compute_class_freq(gt_flat: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Per-class positive counts over the labeled train frames. Used as
    inverse-frequency tie-breaker during per-animal voting so minority
    classes win ties (the dyadic-label asymmetry we want to protect).
    """
    gt = gt_flat.long().cpu()
    valid = gt >= 0
    gt_valid = gt[valid]
    freq = torch.zeros(num_classes, dtype=torch.long)
    if gt_valid.numel() > 0:
        freq = torch.bincount(gt_valid, minlength=num_classes)
    return freq


_VALID_TIE_BREAKS = ("majority", "minority", "abstain")


def _vote_across_animals(
    pred_per_animal: torch.Tensor,
    class_freq: torch.Tensor,
    num_classes: int,
    tie_break: str = "majority",
) -> torch.Tensor:
    """Collapse per-animal class predictions to per-frame predictions by
    majority vote.

    ``pred_per_animal`` shape ``[B, A, L]`` long; ``-1`` means the cluster
    for that (batch, animal, time) was unmapped. Returns ``[B, L]`` long.

    ``tie_break`` controls what happens when two classes tie on vote count
    — this is the dominant case for A=2 (every disagreement is a tie):

      * ``"majority"`` (default, **safe**): common class wins ties. Minority
        class only wins when *all* animals vote for it. Conservative;
        minority detection ~= scene-mean behavior. Recommended on dyadic
        tasks where per-animal cluster assignments are noisy.
      * ``"minority"``: rare class wins ties (inverse frequency). Aggressive
        — any frame where even one animal lands in a minority-mapped
        cluster is labelled as that minority class. **Tends to collapse
        predictions to the rarest class when false-positives are common.**
        Only use when per-animal clusters are known to be highly clean.
      * ``"abstain"``: ties return ``-1``. Most conservative; F1 eval then
        counts these frames as FN for their true class. Gives the cleanest
        "both animals must agree" semantics.

    ``class_freq`` is needed for ``majority`` / ``minority`` modes; ignored
    by ``abstain``.
    """
    if tie_break not in _VALID_TIE_BREAKS:
        raise ValueError(f"tie_break={tie_break!r} not in {_VALID_TIE_BREAKS}")
    if pred_per_animal.dim() != 3:
        raise RuntimeError(
            f"_vote_across_animals expects pred_per_animal shape [B, A, L]; "
            f"got {tuple(pred_per_animal.shape)}."
        )
    B, A, L = pred_per_animal.shape
    # Map -1 (abstain) to sink index num_classes so one_hot never writes
    # into a real class bin.
    safe = torch.where(
        pred_per_animal < 0,
        torch.full_like(pred_per_animal, num_classes),
        pred_per_animal,
    )
    one_hot = torch.nn.functional.one_hot(safe, num_classes=num_classes + 1)  # [B, A, L, C+1]
    counts = one_hot[..., :num_classes].sum(dim=1)  # [B, L, C]  sum over A
    if counts.numel() == 0:
        return torch.full((B, L), -1, dtype=torch.long, device=pred_per_animal.device)

    max_count = int(counts.max().item())
    M = max_count + 1  # scale factor so count dominates the score

    if tie_break == "majority":
        # +freq so that common class wins ties (safe default).
        freq = class_freq.to(device=counts.device, dtype=torch.long)
        score = counts * M + freq.view(1, 1, -1)
        winner = score.argmax(dim=-1)
        no_votes = counts.sum(dim=-1) == 0
        return torch.where(no_votes, torch.full_like(winner, -1), winner)

    if tie_break == "minority":
        # -freq so that rare class wins ties (aggressive; can collapse).
        freq = class_freq.to(device=counts.device, dtype=torch.long)
        score = counts * M - freq.view(1, 1, -1)
        winner = score.argmax(dim=-1)
        no_votes = counts.sum(dim=-1) == 0
        return torch.where(no_votes, torch.full_like(winner, -1), winner)

    # "abstain": any non-unique argmax returns -1
    top = counts.max(dim=-1).values                              # [B, L]
    top_count = (counts == top.unsqueeze(-1)).sum(dim=-1)        # [B, L]
    is_tie = (top > 0) & (top_count > 1)
    no_votes = counts.sum(dim=-1) == 0
    winner = counts.argmax(dim=-1)
    return torch.where(is_tie | no_votes, torch.full_like(winner, -1), winner)


_VALID_TRAIN_AGG = ("per_animal", "mean", "concat")
_VALID_EVAL_AGG = ("mean", "per_animal_vote", "concat")


def _dec_train_agg(args) -> str:
    """How DEC's SGD step aggregates across animals.

    * ``per_animal`` (default, legacy): each (batch, animal, frame) is its
      own sample — DEC sees ``B*A*L`` embeddings per step. A× more
      samples; shares label across animals (noisy on dyadic tasks).
    * ``mean``: embeddings are averaged across A before DEC loss — each
      (batch, frame) is one ``D_z``-d sample. 1 sample/frame (same as
      BAMS/Beast), but per-animal information is lost.
    * ``concat``: per-animal embeddings are concatenated along the feature
      dim. Each (batch, frame) is one ``A*D_z``-d sample. 1 sample/frame
      AND preserves per-animal info. Matches the supervised finetune
      recipe (``animal_agg_mode=concat``). Requires ``dec_eval_agg=concat``
      too (feature dim must match between train and eval).
    """
    mode = str(getattr(args, "dec_train_agg", "per_animal") or "per_animal").lower()
    if mode not in _VALID_TRAIN_AGG:
        raise ValueError(
            f"Unknown dec_train_agg={mode!r}. Valid: {_VALID_TRAIN_AGG}."
        )
    return mode


def _dec_eval_agg(args) -> str:
    """How the cluster->class majority map is built and applied at eval.

    * ``mean`` (default, legacy): average fused features across A, cluster
      one ``D_z``-d embedding per frame.
    * ``per_animal_vote``: cluster each per-animal embedding independently,
      apply the map per-animal, then per-frame vote across A with the
      tie-break rule from ``_animal_vote_tiebreak``.
    * ``concat``: concatenate per-animal embeddings along the feature dim,
      cluster one ``A*D_z``-d embedding per frame. Requires
      ``dec_train_agg=concat`` too — centers trained in ``D_z`` space
      can't be used to cluster ``A*D_z``-d embeddings.

    ``per_animal_vote`` + ``concat`` is mutually exclusive by construction
    (concat already fuses animals; there's nothing to vote over).
    """
    mode = str(getattr(args, "dec_eval_agg", "mean") or "mean").lower()
    if mode not in _VALID_EVAL_AGG:
        raise ValueError(
            f"Unknown dec_eval_agg={mode!r}. Valid: {_VALID_EVAL_AGG}."
        )
    return mode


def _check_agg_compatibility(train_agg: str, eval_agg: str) -> None:
    """concat requires matching sides (feature dim constraint)."""
    if train_agg == "concat" and eval_agg != "concat":
        raise ValueError(
            f"dec_train_agg=concat requires dec_eval_agg=concat "
            f"(centers are trained in A*D_z space, can't cluster "
            f"D_z-d eval embeddings). Got dec_eval_agg={eval_agg!r}."
        )
    if eval_agg == "concat" and train_agg != "concat":
        raise ValueError(
            f"dec_eval_agg=concat requires dec_train_agg=concat "
            f"(eval needs centers in A*D_z space, but training built "
            f"{train_agg!r} centers in D_z space)."
        )


def _animal_vote_tiebreak(args) -> str:
    tb = str(getattr(args, "animal_vote_tiebreak", "majority") or "majority").lower()
    if tb not in _VALID_TIE_BREAKS:
        raise ValueError(
            f"Unknown animal_vote_tiebreak={tb!r}. Valid: {_VALID_TIE_BREAKS}."
        )
    return tb


def _vicreg_config(args) -> dict:
    """Resolve the VICReg auxiliary-loss hyperparameters.

    Returns a dict with:
      * ``inv`` / ``var`` / ``cov`` — loss weights (0 = disable that term)
      * ``variance_target`` — ``gamma`` in ``max(0, gamma - std)``
      * ``shift`` — temporal offset used to build positive pairs; pair frame
        t with frame t+shift. Must be >= 1 and strictly less than the
        per-window length (``L``) at runtime, or that batch is skipped.
      * ``active`` — convenience bool: any of inv/var/cov > 0.

    Default shift=1 (adjacent frames). Default weights all 0 → VICReg off.
    """
    inv = float(getattr(args, "vicreg_inv", 0.0) or 0.0)
    var = float(getattr(args, "vicreg_var", 0.0) or 0.0)
    cov = float(getattr(args, "vicreg_cov", 0.0) or 0.0)
    variance_target = float(getattr(args, "vicreg_variance_target", 1.0) or 1.0)
    shift = int(getattr(args, "vicreg_shift", 1) or 1)
    if shift < 1:
        raise ValueError(f"vicreg_shift must be >= 1, got {shift}.")
    return {
        "inv": inv,
        "var": var,
        "cov": cov,
        "variance_target": variance_target,
        "shift": shift,
        "active": (inv > 0.0) or (var > 0.0) or (cov > 0.0),
    }


# Opt-in DEC stability fixes; all off by default so earlier runs reproduce.
_DEC_FIX_KEYS = (
    "dec_freeze_vision_center",
    "dec_kmeans_random_sample",
    "dec_target_update",
    "dec_reseed_empty",
)


def _dec_fix_config(args) -> dict:
    target_update = str(getattr(args, "dec_target_update", "batch") or "batch").lower()
    if target_update not in ("batch", "epoch"):
        raise ValueError(f"dec_target_update must be 'batch' or 'epoch', got {target_update!r}.")
    return {
        "freeze_vision_center": bool(getattr(args, "dec_freeze_vision_center", False)),
        "kmeans_random_sample": bool(getattr(args, "dec_kmeans_random_sample", False)),
        "target_update": target_update,
        "reseed_empty": bool(getattr(args, "dec_reseed_empty", False)),
    }


def apply_dec_fix_overrides(args, ns) -> None:
    """Copy the DEC-fix CLI flags that were set (non-None) onto args."""
    for key in _DEC_FIX_KEYS:
        val = getattr(ns, key, None)
        if val is not None:
            setattr(args, key, val)


def _dec_use_bigru(args) -> bool:
    """Whether DEC adapts on top of the BiGRU-smoothed fused features.

    When True:
      * ``_forward_fused`` reads ``out["fused_features_post_bigru"]`` from
        the model, giving DEC access to per-frame representations that
        have been bidirectionally-integrated over the local temporal
        window (same BiGRU the supervised finetune uses before the cls
        heads).
      * ``motion_seq_encoder`` (BiGRU) params are added to the trainable
        set, so DEC backprop can shape them. BiGRU is a finetune-only
        module — absent from the pretrain ckpt — so it starts from the
        standard PyTorch random init unless you point
        ``--pretrain-ckpt`` at a supervised-finetune checkpoint.

    When False (default): DEC uses pre-BiGRU frame-level features (faster,
    but loses temporal context that helps disambiguate behaviors like
    CalMS21's attack / mount from the majority investigation).
    """
    val = getattr(args, "dec_use_bigru", False)
    if isinstance(val, str):
        return val.lower() in ("1", "true", "yes", "on")
    return bool(val)


@torch.no_grad()
def evaluate_dec_split(model, loader, centers, majority_map, device, args,
                       num_classes, global_masks=None,
                       class_freq: torch.Tensor | None = None):
    """Returns counts [num_classes, 4]. majority_map [K] long; maps cluster->class.

    Eval collapses per-animal embeddings to per-frame predictions based on
    ``args.dec_eval_agg``:

      * ``mean`` (default, backward compat): average ``fused`` across
        animals, cluster the scene-mean embedding, map once per frame.
        Works well when animals in the same scene do similar things
        (MaBe huddles / chases).
      * ``per_animal_vote``: cluster each (batch, animal, frame)
        independently, apply the mapping per-animal, then vote across
        animals with the tie-break in ``args.animal_vote_tiebreak``.
        Designed for dyadic tasks (CalMS21 attack/mount) where attacker
        and receiver have different motion but share the same frame
        label. ``class_freq`` must be supplied for the tie-break.
    """
    mode = _dec_eval_agg(args)
    if mode == "per_animal_vote" and class_freq is None:
        raise ValueError(
            "evaluate_dec_split(dec_eval_agg='per_animal_vote') requires class_freq "
            "(compute once on train single-label GT)."
        )
    model.eval()
    base_model = _get_base_model(model)
    counts = _init_f1_counts(num_classes, device=torch.device("cpu"))
    seen_by_video: Dict[str, set] = {}
    label_valid_mode = _get_label_valid_mode(args)

    for batch in loader:
        fused, frame_valid_ds, L = _forward_fused(model, batch, device, args,
                                                   global_masks=global_masks, grad_enabled=False)
        B, A, _, D_z = fused.shape
        mm = majority_map.to(device=fused.device)

        if mode == "mean":
            fused_mean = fused.mean(dim=1)                        # [B, L, D_z]
            flat = fused_mean.reshape(-1, D_z).to(torch.float32)
            cids = assign_clusters(flat, centers.detach().to(flat.device, dtype=torch.float32))
            cids = cids.view(B, L)
            pred_class = torch.where(
                (cids >= 0) & (cids < mm.shape[0]),
                mm[cids.clamp(min=0, max=mm.shape[0] - 1)],
                torch.full_like(cids, -1),
            )  # [B, L]
        elif mode == "concat":
            # [B, A, L, D_z] -> [B, L, A*D_z] — per-animal info preserved in
            # a fixed mouse-ordering concat, 1 cluster id per frame.
            fused_concat = fused.permute(0, 2, 1, 3).reshape(B, L, A * D_z)
            flat = fused_concat.reshape(-1, A * D_z).to(torch.float32)
            cids = assign_clusters(flat, centers.detach().to(flat.device, dtype=torch.float32))
            cids = cids.view(B, L)
            pred_class = torch.where(
                (cids >= 0) & (cids < mm.shape[0]),
                mm[cids.clamp(min=0, max=mm.shape[0] - 1)],
                torch.full_like(cids, -1),
            )  # [B, L]
        else:  # "per_animal_vote"
            flat = fused.reshape(-1, D_z).to(torch.float32)        # [B*A*L, D_z]
            cids_flat = assign_clusters(flat, centers.detach().to(flat.device, dtype=torch.float32))
            cids_ba = cids_flat.view(B, A, L)
            pred_per_animal = torch.where(
                (cids_ba >= 0) & (cids_ba < mm.shape[0]),
                mm[cids_ba.clamp(min=0, max=mm.shape[0] - 1)],
                torch.full_like(cids_ba, -1),
            )  # [B, A, L]
            pred_class = _vote_across_animals(
                pred_per_animal, class_freq, num_classes,
                tie_break=_animal_vote_tiebreak(args),
            )  # [B, L]

        # Prepare GT
        frame_labels = batch.get("frame_labels")
        frame_mask = batch.get("frame_mask")
        if frame_labels is None:
            continue
        frame_labels = frame_labels.to(device)
        if frame_mask is not None:
            frame_mask = frame_mask.to(device)
        frame_mask = _select_frame_mask(frame_mask, frame_labels, label_valid_mode)
        frame_labels, frame_mask = _majority_aggregate_frame_labels(frame_labels, frame_mask)
        frame_labels, frame_mask = _sanitize_frame_labels(frame_labels, frame_mask)
        T_full = frame_labels.shape[1] if frame_labels.dim() >= 2 else frame_labels.shape[0]
        stride = max(1, T_full // L)
        labels_ds = _downsample_labels_unified(frame_labels, L, stride)
        mask_ds = _downsample_frame_mask(frame_mask, L, stride) if frame_mask is not None else None
        gt_cls = _labels_to_single_class(labels_ds)
        if mask_ds is None:
            valid = (gt_cls >= 0) & frame_valid_ds & (pred_class >= 0)
        else:
            valid = mask_ds.bool() & (gt_cls >= 0) & frame_valid_ds & (pred_class >= 0)
        accumulate_f1_counts_single_label(
            counts,
            pred_class.cpu(), gt_cls.cpu(), valid.cpu(),
            batch.get("metadata"),
            seen_by_video,
            num_classes=num_classes,
        )
    return counts


@torch.no_grad()
def collect_cluster_gt_pairs(model, loader, centers, device, args, num_classes, global_masks=None):
    """Gather (cluster_id, gt_class) pairs on the train loader to build the
    majority map.

    Respects ``args.dec_eval_agg``:

      * ``mean``: cluster the scene-mean embedding per frame; emit one pair
        per frame.
      * ``per_animal_vote``: cluster each per-animal embedding; emit one
        pair per (frame, animal) — every animal contributes its own vote
        to the map. Aligns the map with the per-animal assignment done at
        eval.
    """
    mode = _dec_eval_agg(args)
    model.eval()
    base_model = _get_base_model(model)
    label_valid_mode = _get_label_valid_mode(args)
    all_cids = []
    all_gt = []
    for batch in loader:
        fused, frame_valid_ds, L = _forward_fused(model, batch, device, args,
                                                   global_masks=global_masks, grad_enabled=False)
        B, A, _, D_z = fused.shape

        if mode == "mean":
            fused_mean = fused.mean(dim=1)
            flat = fused_mean.reshape(-1, D_z).to(torch.float32)
            cids = assign_clusters(flat, centers.detach().to(flat.device, dtype=torch.float32)).view(B, L)
            cids_frame_or_ba = cids                                # [B, L]
        elif mode == "concat":
            fused_concat = fused.permute(0, 2, 1, 3).reshape(B, L, A * D_z)
            flat = fused_concat.reshape(-1, A * D_z).to(torch.float32)
            cids = assign_clusters(flat, centers.detach().to(flat.device, dtype=torch.float32)).view(B, L)
            cids_frame_or_ba = cids                                # [B, L]
        else:  # "per_animal_vote"
            flat = fused.reshape(-1, D_z).to(torch.float32)         # [B*A*L, D_z]
            cids_flat = assign_clusters(flat, centers.detach().to(flat.device, dtype=torch.float32))
            cids_frame_or_ba = cids_flat.view(B, A, L)              # [B, A, L]

        frame_labels = batch.get("frame_labels")
        frame_mask = batch.get("frame_mask")
        if frame_labels is None:
            continue
        frame_labels = frame_labels.to(device)
        if frame_mask is not None:
            frame_mask = frame_mask.to(device)
        frame_mask = _select_frame_mask(frame_mask, frame_labels, label_valid_mode)
        frame_labels, frame_mask = _majority_aggregate_frame_labels(frame_labels, frame_mask)
        frame_labels, frame_mask = _sanitize_frame_labels(frame_labels, frame_mask)
        T_full = frame_labels.shape[1] if frame_labels.dim() >= 2 else frame_labels.shape[0]
        stride = max(1, T_full // L)
        labels_ds = _downsample_labels_unified(frame_labels, L, stride)
        mask_ds = _downsample_frame_mask(frame_mask, L, stride) if frame_mask is not None else None
        gt_cls = _labels_to_single_class(labels_ds)                # [B, L]
        if mask_ds is None:
            valid_frame = (gt_cls >= 0) & frame_valid_ds           # [B, L]
        else:
            valid_frame = mask_ds.bool() & (gt_cls >= 0) & frame_valid_ds

        if mode in ("mean", "concat"):
            # cids shape [B, L]: 1 cluster id per frame.
            all_cids.append(cids_frame_or_ba[valid_frame].cpu())
            all_gt.append(gt_cls[valid_frame].cpu())
        else:  # "per_animal_vote"
            # Replicate GT and valid mask across animals so every per-animal
            # cluster assignment gets its own (cluster_id, gt_class) pair.
            valid_ba = valid_frame.unsqueeze(1).expand(B, A, L)     # [B, A, L]
            gt_ba = gt_cls.unsqueeze(1).expand(B, A, L)
            all_cids.append(cids_frame_or_ba[valid_ba].cpu())
            all_gt.append(gt_ba[valid_ba].cpu())
    if not all_cids:
        return torch.zeros((0,), dtype=torch.long), torch.zeros((0,), dtype=torch.long)
    return torch.cat(all_cids, dim=0), torch.cat(all_gt, dim=0)


# ----------------------------------------------------------------------
# F1 logging + summary
# ----------------------------------------------------------------------

def _log_split_f1(counts: torch.Tensor, label_names: List[str], split_name: str,
                  writer: SummaryWriter | None, global_step: int, tag_prefix: str = "unsup"):
    f1s = _counts_to_f1(counts)
    for idx, f1 in enumerate(f1s):
        label = label_names[idx] if idx < len(label_names) else f"class{idx}"
        tp = int(counts[idx, 0].item())
        fp = int(counts[idx, 1].item())
        fn = int(counts[idx, 2].item())
        total = int(counts[idx, 3].item())
        tn = max(total - tp - fp - fn, 0)
        print(
            f"[Unsup][{split_name}][F1][frame_index] label={label} f1={f1:.4f} "
            f"tp={tp} tn={tn} fp={fp} fn={fn}",
            flush=True,
        )
        if writer is not None:
            writer.add_scalar(f"{tag_prefix}/{split_name.lower()}_f1/{label}", f1, global_step)
    return f1s


def _print_best_val_summary(test_f1_history: List[List[float]], label_names: List[str],
                            best_epoch: int, best_val_macro: float):
    """Print the test F1 of the epoch with the best val macro F1."""
    if not (0 <= best_epoch < len(test_f1_history)):
        return
    f1s = test_f1_history[best_epoch]
    num_classes = len(f1s)
    print("\n" + "=" * 60)
    print(f"  Best-val epoch (epoch {best_epoch + 1}, val macro F1={best_val_macro:.4f}) (Test F1)")
    print("=" * 60)
    for c in range(num_classes):
        label = label_names[c] if c < len(label_names) else f"class{c}"
        print(f"  {label:25s} F1={f1s[c]:.4f}")
    print(f"  {'macro':25s} F1={sum(f1s) / max(num_classes, 1):.4f}")
    print("=" * 60)


# ----------------------------------------------------------------------
# Segbar visualization (paper-ready PDF). Style is byte-for-byte aligned
# with scripts/unsup_eval.py:_plot_segbar so MaBe / CalMS21 / Freely
# figures across methods use the SAME class->color table.
# ----------------------------------------------------------------------

# Palette shared with scripts/unsup_eval.py and kpms_unsup_common.py.
#   class 0 -> #F8B2A2  light coral
#   class 1 -> #E9687A  rose red
#   class 2 -> #B6B3D6  light purple
#   class 3 -> #FBD178  warm yellow
#   class 4 -> #D5D1D1  light gray
# pred = -1 (cluster didn't map to any class) -> _NO_CLASS_COLOR.
_CLASS_COLORS_HEX = ["#F8B2A2", "#E9687A", "#B6B3D6", "#FBD178", "#D5D1D1"]
_NO_CLASS_COLOR = "#EEEEEE"


def _class_palette(num_classes: int) -> List[str]:
    """Length-(num_classes + 1) palette. Index 0 is the -1 / no-class color;
    indices 1..num_classes are the class colors in the canonical order."""
    out = [_NO_CLASS_COLOR]
    for c in range(num_classes):
        out.append(_CLASS_COLORS_HEX[c] if c < len(_CLASS_COLORS_HEX) else "#888888")
    return out


@torch.no_grad()
def collect_segbar_data(model, loader, centers, majority_map, device, args,
                        num_classes, global_masks=None,
                        class_freq: torch.Tensor | None = None,
                        drop_unlabeled: bool = True) -> dict:
    """Single-pass collect of (gt_class, pred_class) per valid frame on the
    given loader, with the (video_id, start_frame, t_in_window) book-keeping
    needed by the segbar renderer.

    Mirrors :func:`evaluate_dec_split`'s forward pipeline (mode =
    args.dec_eval_agg) byte-for-byte; only the accumulator differs (per-frame
    arrays instead of F1 counts).
    """
    import numpy as _np
    mode = _dec_eval_agg(args)
    if mode == "per_animal_vote" and class_freq is None:
        raise ValueError(
            "collect_segbar_data(dec_eval_agg='per_animal_vote') requires class_freq."
        )
    model.eval()
    label_valid_mode = _get_label_valid_mode(args)

    pred_chunks: List[int] = []
    gt_chunks: List[int] = []
    vids: List[str] = []
    start_frames: List[int] = []
    t_in_window: List[int] = []

    for batch in loader:
        fused, frame_valid_ds, L = _forward_fused(
            model, batch, device, args, global_masks=global_masks, grad_enabled=False,
        )
        B, A, _, D_z = fused.shape
        mm = majority_map.to(device=fused.device)

        if mode == "mean":
            fused_mean = fused.mean(dim=1)
            flat = fused_mean.reshape(-1, D_z).to(torch.float32)
            cids = assign_clusters(
                flat, centers.detach().to(flat.device, dtype=torch.float32),
            ).view(B, L)
            pred_class = torch.where(
                (cids >= 0) & (cids < mm.shape[0]),
                mm[cids.clamp(min=0, max=mm.shape[0] - 1)],
                torch.full_like(cids, -1),
            )
        elif mode == "concat":
            fused_concat = fused.permute(0, 2, 1, 3).reshape(B, L, A * D_z)
            flat = fused_concat.reshape(-1, A * D_z).to(torch.float32)
            cids = assign_clusters(
                flat, centers.detach().to(flat.device, dtype=torch.float32),
            ).view(B, L)
            pred_class = torch.where(
                (cids >= 0) & (cids < mm.shape[0]),
                mm[cids.clamp(min=0, max=mm.shape[0] - 1)],
                torch.full_like(cids, -1),
            )
        else:  # per_animal_vote
            flat = fused.reshape(-1, D_z).to(torch.float32)
            cids_flat = assign_clusters(
                flat, centers.detach().to(flat.device, dtype=torch.float32),
            )
            cids_ba = cids_flat.view(B, A, L)
            pred_per_animal = torch.where(
                (cids_ba >= 0) & (cids_ba < mm.shape[0]),
                mm[cids_ba.clamp(min=0, max=mm.shape[0] - 1)],
                torch.full_like(cids_ba, -1),
            )
            pred_class = _vote_across_animals(
                pred_per_animal, class_freq, num_classes,
                tie_break=_animal_vote_tiebreak(args),
            )

        frame_labels = batch.get("frame_labels")
        frame_mask = batch.get("frame_mask")
        if frame_labels is None:
            continue
        frame_labels = frame_labels.to(device)
        if frame_mask is not None:
            frame_mask = frame_mask.to(device)
        frame_mask = _select_frame_mask(frame_mask, frame_labels, label_valid_mode)
        frame_labels, frame_mask = _majority_aggregate_frame_labels(frame_labels, frame_mask)
        frame_labels, frame_mask = _sanitize_frame_labels(frame_labels, frame_mask)
        T_full = frame_labels.shape[1] if frame_labels.dim() >= 2 else frame_labels.shape[0]
        stride = max(1, T_full // L)
        labels_ds = _downsample_labels_unified(frame_labels, L, stride)
        mask_ds = _downsample_frame_mask(frame_mask, L, stride) if frame_mask is not None else None
        gt_cls = _labels_to_single_class(labels_ds)
        if mask_ds is None:
            valid = (gt_cls >= 0) & frame_valid_ds & (pred_class >= 0)
        else:
            valid = mask_ds.bool() & (gt_cls >= 0) & frame_valid_ds & (pred_class >= 0)

        meta_list = _normalize_metadata_list(batch.get("metadata"), batch_size=B)
        if meta_list is None:
            continue

        valid_cpu = valid.detach().cpu()
        pred_cpu = pred_class.detach().cpu()
        gt_cpu = gt_cls.detach().cpu()
        for b in range(B):
            v = valid_cpu[b]
            if not bool(v.any().item()):
                continue
            idxs = torch.nonzero(v, as_tuple=False).squeeze(-1).tolist()
            meta = meta_list[b] if b < len(meta_list) else {}
            video_id = meta.get("video_id", "")
            sf_raw = meta.get("start_frame", 0)
            try:
                sf = int(sf_raw) if sf_raw is not None else 0
            except (TypeError, ValueError):
                sf = 0
            for t in idxs:
                pred_chunks.append(int(pred_cpu[b, t].item()))
                gt_chunks.append(int(gt_cpu[b, t].item()))
                vids.append(str(video_id) if video_id is not None else "")
                start_frames.append(sf)
                t_in_window.append(int(t))

    n = len(pred_chunks)
    gt_arr = _np.asarray(gt_chunks, dtype=_np.int64)
    pred_arr = _np.asarray(pred_chunks, dtype=_np.int64)
    if n == 0:
        return {
            "gt_class": gt_arr,
            "pred_class_raw": pred_arr,
            "pred_class_smoothed": None,
            "total_frames": 0,
            "video_boundaries": [],
        }

    # Stable sort by (video_id, start_frame, t_in_window) for natural per-video
    # chronological ordering. Same convention as
    # scripts/unsup_eval.py:_build_concat_sequences.
    sort_keys = list(zip(vids, start_frames, t_in_window))
    perm = sorted(range(n), key=lambda i: sort_keys[i])
    perm_arr = _np.asarray(perm, dtype=_np.int64)
    gt_arr = gt_arr[perm_arr]
    pred_arr = pred_arr[perm_arr]
    vids = [vids[i] for i in perm]

    if drop_unlabeled:
        keep = gt_arr >= 0
        n_dropped = int((~keep).sum())
        gt_arr = gt_arr[keep]
        pred_arr = pred_arr[keep]
        keep_idx = _np.nonzero(keep)[0].tolist()
        vids = [vids[i] for i in keep_idx]
        if n_dropped > 0:
            print(
                f"[segbar] dropped {n_dropped} unlabeled frames "
                f"(gt<0) from the concat sequence; keeping {len(gt_arr)} labeled frames.",
                flush=True,
            )

    boundaries: List[int] = []
    last_vid: str | None = None
    for i in range(len(vids)):
        if i > 0 and vids[i] != last_vid:
            boundaries.append(i)
        last_vid = vids[i]

    return {
        "gt_class": gt_arr,
        "pred_class_raw": pred_arr,
        "pred_class_smoothed": None,
        "total_frames": int(len(gt_arr)),
        "video_boundaries": boundaries,
    }


def _plot_segbar(data: dict, class_names: List[str], seed: int,
                 split_name: str, out_path) -> None:
    """Paper-ready class-coloured bars (GT + Pred). All rows are identical-size
    thin strips with no ticks / labels / spines — ready for Illustrator post-
    processing. Legend (class names + -1 / no class) sits above the figure for
    easy deletion. Saves as PDF regardless of the caller's ``out_path`` suffix.

    Byte-for-byte aligned with scripts/unsup_eval.py:_plot_segbar.
    """
    import numpy as _np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from pathlib import Path as _Path

    num_classes = len(class_names)
    palette = _class_palette(num_classes)

    rows: List[tuple] = [
        ("GT", data["gt_class"]),
        ("Pred", data["pred_class_raw"]),
    ]
    if data.get("pred_class_smoothed") is not None:
        rows.append(("Pred (smoothed)", data["pred_class_smoothed"]))

    total = int(data["total_frames"])
    if total <= 0:
        print("[segbar] nothing to plot (0 labeled frames).", flush=True)
        return

    row_h_inch = 0.35
    width = max(10.0, min(40.0, total / 1000.0 + 8.0))
    fig, axs = plt.subplots(
        len(rows), 1,
        figsize=(width, row_h_inch * len(rows) + 0.9),
        gridspec_kw={"hspace": 0.15},
        constrained_layout=True,
    )
    if len(rows) == 1:
        axs = [axs]

    def _strip_ax(ax):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_title("")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlim(0, max(0, total - 1))

    # broken_barh -> true vector paths (one rectangle per contiguous same-class
    # run). Stays crisp at any zoom and keeps the PDF compact for long runs.
    for ax, (_title, seq) in zip(axs, rows):
        seq_arr = _np.asarray(seq, dtype=_np.int64)
        if seq_arr.size == 0:
            _strip_ax(ax)
            continue
        change_pts = _np.nonzero(_np.diff(seq_arr))[0] + 1
        starts = _np.concatenate([[0], change_pts])
        ends = _np.concatenate([change_pts, [seq_arr.size]])
        buckets: Dict[int, list] = {}
        for s, e in zip(starts, ends):
            c = int(seq_arr[s])
            palette_idx = (c + 1) if 0 <= c < num_classes else 0
            buckets.setdefault(palette_idx, []).append((int(s), int(e - s)))
        for palette_idx, xranges in buckets.items():
            ax.broken_barh(xranges, (0, 1),
                           facecolors=palette[palette_idx], linewidth=0)
        ax.set_ylim(0, 1)
        _strip_ax(ax)

    handles = [mpatches.Patch(color=_NO_CLASS_COLOR, label="-1 / no class")]
    for c, name in enumerate(class_names):
        handles.append(mpatches.Patch(color=palette[c + 1], label=name))
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=min(len(handles), 6),
        fontsize=8,
        frameon=False,
    )

    out_path = _Path(out_path)
    if out_path.suffix.lower() != ".pdf":
        out_path = out_path.with_suffix(".pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Dataset builders (MABe)
# ----------------------------------------------------------------------

def _motion_input_dim(args, num_joints: int) -> int:
    include_velocity = bool(getattr(args, "motion_include_velocity", False))
    include_confidence = bool(getattr(args, "motion_include_confidence", True))
    max_neighbors = int(getattr(args, "max_neighbors", 2))
    use_neighbor = bool(getattr(args, "use_neighbor_feats", True))
    base_dim = num_joints * (3 if include_confidence else 2)
    vel_dim = num_joints * 2 if include_velocity else 0
    nbr_dim = (max_neighbors * 4) if use_neighbor else 0
    return base_dim + vel_dim + nbr_dim


def _build_mabe_dataset(args, split: str, seq_len: int | None = None) -> MaBeMouseDataset:
    """Build one split of MaBe. Pass ``seq_len=None`` for the train split so
    the dataset infers it from cache (``window_size / ds_stride``); then pass
    the inferred value to val/test. Same pattern as ``run_finetune``.
    """
    mabe_root = getattr(args, "mabe_data_dir", None) or getattr(args, "data_root", None)
    if not mabe_root:
        raise RuntimeError("mabe_data_dir must be set for MABe unsup.")
    stage_dir = os.path.join(mabe_root, "finetune")
    split_override = getattr(args, "mabe_finetune_split_json", None)
    split_json = os.path.expanduser(split_override) if split_override else os.path.join(stage_dir, "finetune_split.json")
    if not os.path.exists(split_json):
        raise RuntimeError(f"Split JSON not found: {split_json}")
    cache_dir = getattr(args, "mabe_data_cache_dir", None)
    if cache_dir is None:
        raise RuntimeError("mabe_data_cache_dir must be set for MABe unsup.")
    train_ratio = getattr(args, "finetune_train_ratio", None) or getattr(args, "train_ratio", 0.8)
    include_velocity = getattr(args, "motion_include_velocity", False)
    include_confidence = getattr(args, "motion_include_confidence", True)
    huddle_keep_ratio = getattr(args, "finetune_huddle_keep_ratio", 1.0) if split == "train" else 1.0
    return MaBeMouseDataset(
        data_dir=stage_dir,
        split=split,
        stage="finetune",
        train_ratio=train_ratio,
        split_json=split_json,
        ds_stride=getattr(args, "ds_stride", 1),
        seq_len=seq_len,
        multi_mouse=True,
        include_neighbor_feats=bool(getattr(args, "use_neighbor_feats", True)),
        max_neighbors=int(getattr(args, "max_neighbors", 2)),
        motion_include_velocity=include_velocity,
        motion_include_confidence=include_confidence,
        frame_resize=getattr(args, "frame_resize", None),
        frame_resize_align=getattr(args, "frame_resize_align", 32),
        dtype=args.torch_dtype,
        cache_dir=cache_dir,
        use_cache=bool(getattr(args, "use_cache", False)),
        use_vision_cache=bool(getattr(args, "use_vision_cache", False)),
        huddle_only_keep_ratio=huddle_keep_ratio,
    )


def _build_run_dir_unsup(args) -> str:
    base = getattr(args, "dec_save_dir", None) or "./checkpoints_unsup_mabe"
    os.makedirs(base, exist_ok=True)
    run_name = (getattr(args, "unsup_run_name", "") or "").strip()
    tag = time.strftime("%Y%m%d-%H%M%S")
    if not run_name:
        run_name = f"{tag}_dec_K{args.dec_num_clusters}_ep{args.epochs}_bs{args.batch_size}"
    else:
        run_name = f"{run_name}_{tag}"
    out = os.path.join(base, run_name)
    os.makedirs(out, exist_ok=True)
    return out


# ----------------------------------------------------------------------
# Main runner — consumed by the sibling calms21 / freely scripts too
# ----------------------------------------------------------------------

def run_unsup(args, build_datasets_fn, label_names_fallback: List[str] | None = None):
    """build_datasets_fn(args) -> (train_ds, val_ds, test_ds, label_names, num_classes).

    label_names may be empty; when so, we fall back to label_names_fallback
    or a synthetic [class0, class1, ...] list of length num_classes.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    is_main = True  # single-process
    args.rank = 0
    args.world_size = 1
    args.distributed = False

    # Global seeding so every random source (nn.Linear / BiGRU init, DataLoader
    # shuffle, SGD batch sampling, numpy RNGs) is reproducible for a given
    # --seed. Without this each run would still be only half-deterministic
    # (k-means init was seeded but network init + SGD trajectory were not).
    seed = int(getattr(args, "seed", 0) or 0)
    args.seed = seed
    import random as _pyrandom
    import numpy as _np
    _pyrandom.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[Unsup] global seed = {seed} (torch/numpy/random/cuda all seeded).", flush=True)

    train_ds, val_ds, test_ds, label_names, num_classes = build_datasets_fn(args)
    if not label_names:
        label_names = label_names_fallback or [f"class{i}" for i in range(num_classes)]
    args.head_a_classes = num_classes

    args.seq_len = getattr(train_ds, "seq_len", getattr(args, "seq_len", None))
    args.num_joints = getattr(train_ds, "num_joints", getattr(args, "num_joints", None))
    args.motion_input_dim = _motion_input_dim(args, args.num_joints)

    args.save_dir = _build_run_dir_unsup(args)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tb"))

    # Parallels the banner that train_entry_mabe.run_finetune prints so debug
    # output lines up side-by-side with supervised runs.
    frame_select_mode = str(getattr(args, "frame_select_mode", "residual")).lower()
    budget_ratio = float(getattr(args, "budget_ratio", 0.75))
    vision_alpha = float(getattr(args, "vision_alpha", 1.0))
    vision_fuse_mode = str(getattr(args, "vision_fuse_mode", "feature")).lower()
    use_social = bool(getattr(args, "use_social_pipeline", True))
    lambda_col = float(getattr(args, "lambda_col", 0.0) or 0.0)
    col_temp = float(getattr(args, "collision_temperature", 0.1) or 0.1)
    dec_K = int(args.dec_num_clusters)
    dec_lr = float(getattr(args, "dec_lr", args.lr) or args.lr)
    dec_centers_lr = float(getattr(args, "dec_centers_lr", dec_lr) or dec_lr)
    dec_km_samples = int(getattr(args, "dec_kmeans_samples", 50000) or 50000)
    dec_km_iters = int(getattr(args, "dec_kmeans_init_iters", 50) or 50)
    mm_mode_requested = str(getattr(args, "majority_map_mode", "count") or "count").lower()
    majority_map_fn, mm_mode = _resolve_majority_map_builder(mm_mode_requested)
    args.majority_map_mode = mm_mode
    train_agg = _dec_train_agg(args)
    args.dec_train_agg = train_agg
    eval_agg = _dec_eval_agg(args)
    args.dec_eval_agg = eval_agg
    _check_agg_compatibility(train_agg, eval_agg)
    animal_tie = _animal_vote_tiebreak(args)
    args.animal_vote_tiebreak = animal_tie
    use_bigru = _dec_use_bigru(args)
    args.dec_use_bigru = use_bigru
    vic_cfg = _vicreg_config(args)
    print("=" * 60)
    print(f"  Unsupervised DEC ({args.epochs} epochs, dataset={getattr(args, 'dataset', '?')})")
    print(f"  num_classes={num_classes}  labels={label_names}")
    print(f"  dec_num_clusters={dec_K} (> num_classes so majority-map can merge)")
    print(f"  dec_lr={dec_lr:.1e}, dec_centers_lr={dec_centers_lr:.1e}")
    print(f"  dec_kmeans_samples={dec_km_samples}, dec_kmeans_init_iters={dec_km_iters}")
    print(f"  frame_select_mode={frame_select_mode}")
    print(f"  budget_ratio={budget_ratio}, vision_alpha={vision_alpha}")
    print(f"  vision_fuse_mode={vision_fuse_mode}")
    print(f"  use_social_pipeline={use_social}")
    print(f"  lambda_col={lambda_col}, collision_temperature={col_temp} (unused in DEC, kept for parity)")
    print(f"  majority_map_mode={mm_mode} "
          f"({'count' if mm_mode == 'count' else 'F1'}-based cluster->class mapping on train labels)")
    if use_bigru:
        print(f"  dec_use_bigru=on (DEC sees post-BiGRU features; BiGRU params "
              f"are trained jointly with vision_projector/fused_feat_norm)")
    else:
        print(f"  dec_use_bigru=off (DEC sees pre-BiGRU per-frame features; "
              f"motion_seq_encoder frozen/ignored)")
    if train_agg == "per_animal":
        print(f"  dec_train_agg=per_animal (DEC sees [B*A*L, D_z] — A× samples per step)")
    elif train_agg == "mean":
        print(f"  dec_train_agg=mean (DEC sees [B*L, D_z] — 1 sample per frame; "
              f"same recipe as BAMS/Beast DEC; per-animal info averaged away)")
    else:  # concat
        print(f"  dec_train_agg=concat (DEC sees [B*L, A*D_z] — 1 sample per frame, "
              f"per-animal info preserved via fixed-order feature concat; "
              f"same animal agg as supervised finetune)")
    if eval_agg == "mean":
        print(f"  dec_eval_agg=mean (scene-mean embedding at eval; 1 pred/frame; "
              f"tie_break n/a)")
    elif eval_agg == "concat":
        print(f"  dec_eval_agg=concat (concatenated [A*D_z] embedding at eval; "
              f"1 pred/frame; tie_break n/a)")
    else:
        tb_desc = {
            "majority": "common class wins ties (safe; ~scene-mean on ties)",
            "minority": "rare class wins ties (aggressive; can collapse on dyadic A=2)",
            "abstain":  "ties -> -1 (F1 counts those frames as FN for true class)",
        }[animal_tie]
        print(f"  dec_eval_agg=per_animal_vote (cluster per-animal + vote) "
              f"tie_break={animal_tie} [{tb_desc}]")
    if vic_cfg["active"]:
        print(
            f"  vicreg=on (temporal positive pairs, shift={vic_cfg['shift']}): "
            f"inv={vic_cfg['inv']}, var={vic_cfg['var']} (γ={vic_cfg['variance_target']}), "
            f"cov={vic_cfg['cov']} — var>0 is the main anti-collapse term"
        )
    else:
        print(f"  vicreg=off (DEC loss only)")
    fix_cfg = _dec_fix_config(args)
    print(f"  dec_freeze_vision_center={fix_cfg['freeze_vision_center']} "
          f"(exact train mean of the visual term set before k-means, then frozen)")
    print(f"  dec_kmeans_random_sample={fix_cfg['kmeans_random_sample']} "
          f"(k-means pool drawn at random from all train frames, not the first N)")
    print(f"  dec_target_update={fix_cfg['target_update']} "
          f"({'p over the whole train set, fixed per epoch' if fix_cfg['target_update'] == 'epoch' else 'p re-sharpened on every mini-batch'})")
    print(f"  dec_reseed_empty={fix_cfg['reseed_empty']} "
          f"(empty clusters moved into the largest cluster at each epoch start)")
    print(f"  batch_size={args.batch_size}, num_workers={args.num_workers}")
    print(f"  seq_len={args.seq_len}, num_joints={args.num_joints}, "
          f"motion_input_dim={args.motion_input_dim}")
    print(f"  pretrain_ckpt={getattr(args, 'pretrain_ckpt', None)}")
    print(f"  save_dir={args.save_dir}")
    print("=" * 60, flush=True)

    dl_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    train_eval_loader = DataLoader(train_ds, shuffle=False, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **dl_kwargs)

    # ---- Build model, load pretrain, freeze supervised parts. ----
    model = ResiFuseClassifier(args).to(device).to(dtype=args.torch_dtype)
    # Load pretrain ckpt *before* freezing (maybe_load_pretrain_weights uses state_dict keys).
    maybe_load_pretrain_weights(model, getattr(args, "pretrain_ckpt", None), device)
    _configure_unsup_trainables(model, args=args, is_main=is_main)

    # ---- Compute global frame-select masks on all splits. ----
    base_model = _get_base_model(model)
    frame_select_mode = str(getattr(args, "frame_select_mode", "residual")).lower()
    budget_ratio = float(getattr(args, "budget_ratio", 0.75))
    global_masks = None
    if 0.0 < budget_ratio < 1.0 and frame_select_mode in ("residual", "uniform"):
        print(f"[Unsup] Computing global masks: mode={frame_select_mode}, budget={budget_ratio}")
        global_masks = {}
        for name, loader in [("train", train_eval_loader), ("val", val_loader), ("test", test_loader)]:
            ms = compute_global_masks(base_model, loader, device, budget_ratio, mode=frame_select_mode,
                                       is_main=is_main)
            global_masks.update(ms)
            n_sel = sum(int(m.sum().item()) for m in ms.values())
            n_tot = sum(int(m.numel()) for m in ms.values())
            print(f"  {name}: {len(ms)} windows, selected {n_sel}/{n_tot}")
    else:
        print(f"[Unsup] budget_ratio={budget_ratio} / mode={frame_select_mode} -> no global mask")

    # ---- Fix the vision centering before k-means sees the features ----
    if fix_cfg["freeze_vision_center"]:
        n_cal = _calibrate_vision_center(model, train_eval_loader, device, args, global_masks)
        if n_cal > 0:
            print(f"[Unsup] vision center calibrated on {n_cal} train frame-slots and frozen")
        else:
            print(f"[Unsup] vision center: no visual term (budget=0 or centering off), nothing to calibrate")

    # ---- K-means init on train fused features ----
    kmeans_samples = int(getattr(args, "dec_kmeans_samples", 50000) or 50000)
    kmeans_iters = int(getattr(args, "dec_kmeans_init_iters", 50) or 50)
    print(f"[Unsup] Collecting features for kmeans-init (target={kmeans_samples}, iters={kmeans_iters})")
    feats_for_init = _collect_fused(model, train_eval_loader, device, args,
                                     global_masks=global_masks, max_samples=kmeans_samples,
                                     random_subsample=fix_cfg["kmeans_random_sample"], seed=seed)
    feats_for_init = feats_for_init.to(device)
    K = int(getattr(args, "dec_num_clusters", num_classes * 3))
    print(f"[Unsup] kmeans init: n={feats_for_init.shape[0]} K={K}")
    centers_init = kmeans_init(feats_for_init, K, iters=kmeans_iters, seed=int(getattr(args, "seed", 0) or 0))
    centers = nn.Parameter(centers_init.detach().clone().to(device=device, dtype=torch.float32))
    del feats_for_init

    # ---- Build optimizer (vision projection params + centers with separate lr) ----
    dec_lr = float(getattr(args, "dec_lr", args.lr) or args.lr)
    centers_lr = float(getattr(args, "dec_centers_lr", dec_lr) or dec_lr)
    weight_decay = float(getattr(args, "weight_decay", 1e-2) or 1e-2)

    decay_params, nodecay_params = [], []
    for name, p in _get_base_model(model).named_parameters():
        if not p.requires_grad:
            continue
        if ("bias" in name.lower()) or ("norm" in name.lower()):
            nodecay_params.append(p)
        else:
            decay_params.append(p)
    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "lr": dec_lr, "weight_decay": weight_decay})
    if nodecay_params:
        param_groups.append({"params": nodecay_params, "lr": dec_lr, "weight_decay": 0.0})
    param_groups.append({"params": [centers], "lr": centers_lr, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(param_groups)

    # ---- DEC loop ----
    test_f1_history: List[List[float]] = []
    best_val_macro = -1.0
    best_epoch = -1
    total_epochs = args.epochs
    # Snapshot of the best-val epoch (model + centers + mmap) for the segbar
    # visualization. Kept in memory as CPU tensors so all three come from the
    # same epoch when rendered; mixing different epochs' state is incoherent.
    best_val_state: dict | None = None
    for epoch in range(total_epochs):
        print(f"\n=== [Unsup][MABe] Epoch {epoch + 1}/{total_epochs} ===")
        targets = None
        if fix_cfg["target_update"] == "epoch" or fix_cfg["reseed_empty"]:
            targets, n_reseeded = _dec_epoch_pass(
                model, train_eval_loader, centers, device, args, global_masks,
                build_targets=fix_cfg["target_update"] == "epoch",
                reseed_empty=fix_cfg["reseed_empty"],
                optimizer=optimizer, seed=seed * 1000 + epoch,
            )
            if fix_cfg["reseed_empty"]:
                print(f"[Unsup][Epoch {epoch + 1}] reseeded {n_reseeded}/{K} empty clusters (train-agg space)")
        train_losses = train_one_epoch_dec(
            model, train_loader, centers, optimizer, device, args, global_masks, targets=targets,
        )
        train_loss = train_losses["total"]
        writer.add_scalar("unsup/train_loss", train_loss, epoch + 1)
        writer.add_scalar("unsup/train_dec_loss", train_losses["dec"], epoch + 1)
        if vic_cfg["active"]:
            writer.add_scalar("unsup/train_vic_inv", train_losses["vic_inv"], epoch + 1)
            writer.add_scalar("unsup/train_vic_var", train_losses["vic_var"], epoch + 1)
            writer.add_scalar("unsup/train_vic_cov", train_losses["vic_cov"], epoch + 1)

        # Rebuild majority map from train split each epoch (features drift).
        cids_tr, gt_tr = collect_cluster_gt_pairs(
            model, train_eval_loader, centers, device, args, num_classes, global_masks=global_masks,
        )
        mmap = majority_map_fn(cids_tr, gt_tr, num_clusters=K, num_classes=num_classes)
        n_empty = int((mmap < 0).sum().item())
        mmap_clamped = mmap.clamp(min=0)  # replace -1 fallback -> class 0 (harmless for eval)
        if vic_cfg["active"]:
            print(
                f"[Unsup][Epoch {epoch + 1}] "
                f"train_loss={train_loss:.5f} "
                f"(dec={train_losses['dec']:.5f}, "
                f"vic_inv={train_losses['vic_inv']:.5f}, "
                f"vic_var={train_losses['vic_var']:.5f}, "
                f"vic_cov={train_losses['vic_cov']:.5f}) "
                f"| empty_clusters={n_empty}/{K}"
            )
        else:
            print(f"[Unsup][Epoch {epoch + 1}] train_loss={train_loss:.5f} | empty_clusters={n_empty}/{K}")

        # Update train class frequency from the current train pass — when in
        # 'majority' mode this is the tie-breaker for per-animal voting.
        # gt_tr already collected above; when mode='majority' it's replicated
        # across animals but that doesn't change relative class frequency.
        class_freq_now = _compute_class_freq(gt_tr, num_classes)

        train_counts = evaluate_dec_split(model, train_eval_loader, centers, mmap_clamped, device, args,
                                          num_classes, global_masks=global_masks,
                                          class_freq=class_freq_now)
        val_counts = evaluate_dec_split(model, val_loader, centers, mmap_clamped, device, args,
                                        num_classes, global_masks=global_masks,
                                        class_freq=class_freq_now)
        test_counts = evaluate_dec_split(model, test_loader, centers, mmap_clamped, device, args,
                                         num_classes, global_masks=global_masks,
                                         class_freq=class_freq_now)
        f1_tr = _log_split_f1(train_counts, label_names, "TrainEval", writer, epoch + 1)
        f1_val = _log_split_f1(val_counts, label_names, "Val", writer, epoch + 1)
        f1_te = _log_split_f1(test_counts, label_names, "Test", writer, epoch + 1)
        val_macro = sum(f1_val) / max(num_classes, 1)
        test_macro = sum(f1_te) / max(num_classes, 1)
        writer.add_scalar("unsup/val_macro_f1", val_macro, epoch + 1)
        writer.add_scalar("unsup/test_macro_f1", test_macro, epoch + 1)
        print(f"[Unsup][Epoch {epoch + 1}] val_macroF1={val_macro:.4f} test_macroF1={test_macro:.4f}")
        test_f1_history.append(f1_te)

        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_epoch = epoch
            ckpt_path = os.path.join(args.save_dir, "best_unsup.pth")
            torch.save({
                "epoch": epoch,
                "model": _get_base_model(model).state_dict(),
                "centers": centers.detach().cpu(),
                "majority_map": mmap_clamped.cpu(),
                "empty_clusters": n_empty,
                "num_classes": num_classes,
                "label_names": label_names,
                "dec_num_clusters": K,
                "val_macro_f1": val_macro,
            }, ckpt_path)

        # Snapshot the best-val epoch for the segbar.
        if epoch == best_epoch:
            best_val_state = {
                "model": {
                    k: v.detach().cpu().clone()
                    for k, v in _get_base_model(model).state_dict().items()
                },
                "centers": centers.detach().cpu().clone(),
                "majority_map": mmap_clamped.detach().cpu().clone(),
                "class_freq": (
                    class_freq_now.detach().cpu().clone()
                    if class_freq_now is not None else None
                ),
                "test_macro_f1": float(test_macro),
                "val_macro_f1": float(val_macro),
            }

    # Final summaries.
    _print_best_val_summary(test_f1_history, label_names, best_epoch, best_val_macro)
    print(f"\n[Unsup] best epoch={best_epoch + 1} val_macroF1={best_val_macro:.4f}")

    # Render the segbar (GT + prediction PDF) of the best-val epoch, restoring
    # its snapshotted model / centers / mmap together.
    if best_val_state is not None:
        try:
            t0 = time.perf_counter()
            _get_base_model(model).load_state_dict(best_val_state["model"])
            best_centers = best_val_state["centers"].to(device=device, dtype=torch.float32)
            best_mmap = best_val_state["majority_map"].to(device)
            best_class_freq = best_val_state.get("class_freq")
            if best_class_freq is not None:
                best_class_freq = best_class_freq.to(device)
            bar_data = collect_segbar_data(
                model, test_loader, best_centers, best_mmap, device, args,
                num_classes, global_masks=global_masks,
                class_freq=best_class_freq, drop_unlabeled=True,
            )
            segbar_path = os.path.join(args.save_dir, "segbar_concat.pdf")
            _plot_segbar(
                data=bar_data,
                class_names=list(label_names),
                seed=seed,
                split_name="test",
                out_path=segbar_path,
            )
            print(
                f"[Unsup][segbar] best-val epoch={best_epoch + 1} "
                f"(val_macroF1={best_val_state['val_macro_f1']:.4f}) | "
                f"wrote {bar_data['total_frames']} frames "
                f"({len(bar_data['video_boundaries']) + 1} videos) to "
                f"{segbar_path} in {time.perf_counter() - t0:.1f}s",
                flush=True,
            )
        except Exception as exc:
            print(f"[Unsup][segbar] failed to render: {exc}", flush=True)
    else:
        print("[Unsup][segbar] skipped (no epochs ran).", flush=True)

    writer.close()


# ----------------------------------------------------------------------
# MABe entrypoint
# ----------------------------------------------------------------------

def _mabe_build_datasets(args):
    # Train with seq_len=None so MaBeMouseDataset infers window_size/ds_stride
    # from cache; then reuse that length for val/test so all three splits
    # align with the cached windows.
    train_ds = _build_mabe_dataset(args, "train", seq_len=None)
    args.seq_len = train_ds.seq_len
    val_ds = _build_mabe_dataset(args, "val", seq_len=args.seq_len)
    test_ds = _build_mabe_dataset(args, "test", seq_len=args.seq_len)
    label_names = list(train_ds.label_names) if getattr(train_ds, "label_names", None) else []
    num_classes = len(label_names) if label_names else int(getattr(args, "head_a_classes", 4) or 4)
    return train_ds, val_ds, test_ds, label_names, num_classes


def main(config_path: str | None = None,
         majority_map_mode: str | None = None,
         dec_train_agg: str | None = None,
         dec_eval_agg: str | None = None,
         animal_vote_tiebreak: str | None = None,
         dec_use_bigru: bool | None = None,
         vicreg_inv: float | None = None,
         vicreg_var: float | None = None,
         vicreg_cov: float | None = None,
         vicreg_variance_target: float | None = None,
         vicreg_shift: int | None = None,
         seed: int | None = None,
         dec_fixes=None):
    args = parse_args(config_path)
    dtype_map = {
        "float32": torch.float32, "float": torch.float32,
        "float16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    args.torch_dtype = dtype_map.get(str(args.torch_dtype).lower(), torch.bfloat16)
    args.dataset = "mabe"
    if majority_map_mode is not None:
        args.majority_map_mode = majority_map_mode
    if dec_train_agg is not None:
        args.dec_train_agg = dec_train_agg
    if dec_eval_agg is not None:
        args.dec_eval_agg = dec_eval_agg
    if animal_vote_tiebreak is not None:
        args.animal_vote_tiebreak = animal_vote_tiebreak
    if dec_use_bigru is not None:
        args.dec_use_bigru = dec_use_bigru
    if vicreg_inv is not None:
        args.vicreg_inv = vicreg_inv
    if vicreg_var is not None:
        args.vicreg_var = vicreg_var
    if vicreg_cov is not None:
        args.vicreg_cov = vicreg_cov
    if vicreg_variance_target is not None:
        args.vicreg_variance_target = vicreg_variance_target
    if vicreg_shift is not None:
        args.vicreg_shift = vicreg_shift
    if seed is not None:
        args.seed = seed
    if dec_fixes is not None:
        apply_dec_fix_overrides(args, dec_fixes)
    run_unsup(args, _mabe_build_datasets)


def _add_shared_cli_args(parser: argparse.ArgumentParser, default_cfg: str) -> None:
    """CLI args shared by the mabe / calms21 / freely entrypoints."""
    parser.add_argument("-c", "--config", default=default_cfg)
    parser.add_argument(
        "--seed", type=int, default=None,
        help=(
            "Global seed for torch / numpy / python random / cuda. Controls "
            "nn module init, DataLoader shuffle, SGD batch sampling, and "
            "k-means init. Default 0 (or yaml's 'seed' key if set). Use "
            "different seeds across runs to estimate variance (like BAMS's "
            "--seeds flag in scripts/unsup_eval.py)."
        ),
    )
    parser.add_argument(
        "--majority-map-mode",
        choices=["count", "f1"],
        default=None,
        help=(
            "Cluster -> class assignment on train labels. 'count' (default when "
            "not set) uses per-cluster GT argmax and is simple but degenerates "
            "on class-imbalanced data (every cluster collapses to the dominant "
            "class). 'f1' picks the class that maximises per-class binary F1 "
            "and is robust to imbalance. Both read the same train GT; only "
            "which statistic we argmax differs. Overrides the yaml value."
        ),
    )
    parser.add_argument(
        "--dec-train-agg",
        choices=["per_animal", "mean", "concat"],
        default=None,
        help=(
            "How DEC's SGD step aggregates across animals. 'per_animal' "
            "(default) feeds [B*A*L, D_z] per step — A× more training "
            "samples. 'mean' averages fused features across A so each SGD "
            "step sees [B*L, D_z] (BAMS/Beast-style, 1 sample/frame, "
            "1/A× fewer samples, per-animal info lost). 'concat' "
            "concatenates per-animal features along the feature dim — "
            "each step sees [B*L, A*D_z], 1 sample/frame AND keeps "
            "per-animal info (matches the supervised finetune's "
            "animal_agg_mode=concat recipe). The k-means init pool uses "
            "the same aggregation so centers don't drift during the first "
            "DEC steps. When 'concat' is used, --dec-eval-agg must also "
            "be 'concat' (feature dim constraint). Overrides yaml."
        ),
    )
    parser.add_argument(
        "--dec-eval-agg",
        choices=["mean", "per_animal_vote", "concat"],
        default=None,
        help=(
            "How the cluster -> class majority map is built and applied at "
            "eval time. 'mean' (default) averages fused features across A "
            "first — one D_z-d embedding per frame. 'per_animal_vote' "
            "clusters each per-animal embedding independently, maps each "
            "to a class, then votes per frame (see --animal-vote-tiebreak). "
            "'concat' concatenates per-animal embeddings (one A*D_z-d "
            "embedding per frame) — requires --dec-train-agg=concat. "
            "Overrides yaml."
        ),
    )
    parser.add_argument(
        "--animal-vote-tiebreak",
        choices=["majority", "minority", "abstain"],
        default=None,
        help=(
            "Only used when --dec-eval-agg=per_animal_vote. For A=2 dyadic "
            "tasks every disagreement is a tie, so this knob decides almost "
            "every frame: 'majority' (safe default) makes the common class "
            "win ties; 'minority' makes the rare class win ties (very "
            "aggressive — CalMS21 testing showed this collapsed all "
            "predictions to the rarest class); 'abstain' returns -1 on ties "
            "so only unanimous minority votes produce minority predictions. "
            "Start with 'majority'; go to 'minority' only if per-animal "
            "clusters are very clean. Overrides yaml."
        ),
    )
    parser.add_argument(
        "--dec-use-bigru",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Include the supervised finetune's BiGRU (motion_seq_encoder) "
            "in the DEC adaptation. When on, DEC reads the post-BiGRU "
            "per-frame features (temporal context within the window), and "
            "BiGRU weights are added to the trainable set so DEC can shape "
            "them. The BiGRU is a finetune-only module (absent from the "
            "pretrain ckpt) so it starts from PyTorch's default init. "
            "Adds ~0.7M trainable params but can rescue imbalance-driven "
            "collapses on CalMS21 where per-frame fused features aren't "
            "enough to disambiguate minority classes. Default off. "
            "Overrides yaml (--no-dec-use-bigru forces off even if yaml "
            "sets it on)."
        ),
    )
    # VICReg auxiliary loss — stability regularizer that complements DEC.
    # Positive pairs are temporal neighbors (frame t vs frame t+shift in the
    # same window). Default all zero = VICReg off. See `vicreg_losses` for
    # details on each term. Recommended starter values:
    #   --vicreg-inv 0.1  --vicreg-var 1.0  --vicreg-cov 0.1
    # (invariance is softer than the VICReg paper's 25.0 because temporal
    # neighbors are a weaker positive-pair signal than strong image augs.)
    parser.add_argument(
        "--vicreg-inv", type=float, default=None,
        help=(
            "VICReg invariance loss weight (MSE between positive pairs). "
            "Default 0 = off. Temporal-neighbor pairs are weak augmentations "
            "— start at 0.1, NOT the paper's 25.0."
        ),
    )
    parser.add_argument(
        "--vicreg-var", type=float, default=None,
        help=(
            "VICReg variance loss weight (per-dim std floor). Default 0 = "
            "off. This is the main anti-collapse guarantee — directly "
            "prevents DEC from flattening per-frame features onto the "
            "majority-class attractor. Start at 1.0."
        ),
    )
    parser.add_argument(
        "--vicreg-cov", type=float, default=None,
        help=(
            "VICReg covariance loss weight (off-diagonal cov penalty). "
            "Default 0 = off. Decorrelates feature dimensions so info is "
            "spread. Start at 0.1."
        ),
    )
    parser.add_argument(
        "--vicreg-variance-target", type=float, default=None,
        help=(
            "VICReg variance target gamma in max(0, gamma - std_d). "
            "Default 1.0; the loss pulls every feature dim's std >= gamma. "
            "Larger gamma = stronger spread."
        ),
    )
    parser.add_argument(
        "--vicreg-shift", type=int, default=None,
        help=(
            "Temporal offset for VICReg positive pairs: frame t is paired "
            "with frame t+shift in the same window. Default 1 (adjacent "
            "frames). Must be < window length L."
        ),
    )
    # DEC stability fixes (all default off = legacy behavior).
    parser.add_argument(
        "--dec-freeze-vision-center",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "With vision_center_mean on: compute the exact train-set mean of "
            "the visual term (selected frames) before k-means and freeze it. "
            "Otherwise k-means is fit on uncentered features and the first "
            "DEC step shifts only vision-selected frames, emptying clusters "
            "at 0 < budget < 1."
        ),
    )
    parser.add_argument(
        "--dec-kmeans-random-sample",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Draw the k-means init pool (dec_kmeans_samples rows) at random "
            "from all train frames instead of taking the first N in loader "
            "order (matches scripts/unsup_eval.py)."
        ),
    )
    parser.add_argument(
        "--dec-target-update",
        choices=["batch", "epoch"],
        default=None,
        help=(
            "'batch' (legacy): DEC target p is re-sharpened from each "
            "mini-batch's own q. 'epoch': standard DEC — p is computed over "
            "the whole train set at each epoch start and held fixed."
        ),
    )
    parser.add_argument(
        "--dec-reseed-empty",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "At each epoch start, move every center with no train rows onto "
            "a random member of the largest cluster (split-largest)."
        ),
    )


def _parse_cli():
    parser = argparse.ArgumentParser(description="MABe unsupervised DEC segmentation entrypoint")
    _add_shared_cli_args(parser, default_cfg="./config_mabe_finetune.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    ns = _parse_cli()
    main(ns.config,
         majority_map_mode=ns.majority_map_mode,
         dec_train_agg=ns.dec_train_agg,
         dec_eval_agg=ns.dec_eval_agg,
         animal_vote_tiebreak=ns.animal_vote_tiebreak,
         dec_use_bigru=ns.dec_use_bigru,
         vicreg_inv=ns.vicreg_inv,
         vicreg_var=ns.vicreg_var,
         vicreg_cov=ns.vicreg_cov,
         vicreg_variance_target=ns.vicreg_variance_target,
         vicreg_shift=ns.vicreg_shift,
         seed=ns.seed,
         dec_fixes=ns)
