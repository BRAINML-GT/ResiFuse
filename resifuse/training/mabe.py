import os
import time
import shutil
import glob
from types import SimpleNamespace
from typing import Dict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from resifuse.data.datasets import MaBeMouseDataset
from resifuse.models.model import ResiFuseClassifier, ResiFusePretrainer, collision_contrastive_loss
from resifuse.config import load_config_with_defaults


def init_distributed():
    if not torch.cuda.is_available() or not dist.is_available():
        return False, 0, 0, 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return True, rank, local_rank, world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


def _set_seed(seed, rank: int = 0) -> None:
    """Seed all RNGs for reproducible / paired runs.

    No-op when seed is None (preserves the original unseeded behavior).
    Must be called *before* model construction so a given seed yields the
    same weight initialization across runs and across vision budgets, which
    is what makes a paired 75%-vs-100% comparison legitimate. The per-rank
    offset keeps DDP data shuffling decorrelated while model init stays
    identical (DDP broadcasts rank-0 weights anyway).
    """
    if seed is None:
        return
    import random
    import numpy as np
    s = int(seed) + int(rank)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    if is_main_process(rank):
        print(f"[seed] set base_seed={int(seed)} (rank offset -> {s})", flush=True)


def _is_main_checkpoint_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def parse_args(config_path: str | None = None):
    cfg = load_config_with_defaults(config_path)
    stage = cfg.get("stage", "finetune")
    if stage == "pretrain":
        cfg["lr"] = cfg["pretrain_lr"]
        cfg["epochs"] = cfg["pretrain_epochs"]
    elif stage == "finetune":
        cfg["lr"] = cfg.get("finetune_lr", 1e-4)
        cfg["epochs"] = cfg["finetune_epochs"]
    return SimpleNamespace(**cfg)


def _coerce_number(val):
    if isinstance(val, str):
        s = val.strip()
        try:
            num = float(s)
        except ValueError:
            return val
        return int(num) if num.is_integer() else num
    if isinstance(val, dict):
        return {k: _coerce_number(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_coerce_number(v) for v in val]
    return val


def _get_base_model(model: torch.nn.Module):
    base = model.module if isinstance(model, DDP) else model
    return getattr(base, "_orig_mod", base)


def _report_unused_params(model: torch.nn.Module, stage: str, batch_idx: int) -> None:
    if os.environ.get("RESIFUSE_DEBUG_UNUSED", "0") != "1":
        return
    base = _get_base_model(model)
    if getattr(base, "_reported_unused_params", False):
        return
    unused = []
    for idx, (name, param) in enumerate(model.named_parameters()):
        if not param.requires_grad:
            continue
        if param.grad is None:
            unused.append((idx, name, tuple(param.shape)))
    if not unused:
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    print(f"[ddp-unused][{stage}] rank={rank} batch={batch_idx + 1} unused={len(unused)}")
    for idx, name, shape in unused:
        print(f"[ddp-unused] idx={idx} name={name} shape={shape}")
    base._reported_unused_params = True



def _compute_social_head_ratios(attn_weights: torch.Tensor | None, key_padding_mask: torch.Tensor | None):
    if attn_weights is None:
        return None
    num_nodes = attn_weights.size(-1)
    if key_padding_mask is not None:
        valid_rows = ~key_padding_mask.all(dim=1)
        if not valid_rows.any():
            return None
        attn_weights = attn_weights[valid_rows]
        key_padding_mask = key_padding_mask[valid_rows]
    if key_padding_mask is None:
        valid = torch.ones(
            (attn_weights.size(0), num_nodes),
            device=attn_weights.device,
            dtype=attn_weights.dtype,
        )
    else:
        valid = (~key_padding_mask).to(dtype=attn_weights.dtype)
    valid_qk = (valid[:, :, None] * valid[:, None, :]).unsqueeze(1)
    attn_valid = attn_weights * valid_qk
    total = attn_valid.sum(dim=(0, 2, 3))
    eye = torch.eye(num_nodes, device=attn_weights.device, dtype=attn_weights.dtype).view(1, 1, num_nodes, num_nodes)
    self_sum = (attn_valid * eye).sum(dim=(0, 2, 3))
    denom = total.clamp(min=1e-6)
    r_self = self_sum / denom
    r_neighbor = (total - self_sum) / denom
    return r_self.detach().cpu(), r_neighbor.detach().cpu()


def _maybe_collect_social_attn_stats(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config,
    stage: str,
    epoch_idx: int,
    writer: SummaryWriter | None = None,
):
    if not bool(getattr(config, "social_attn_stat_enable", False)):
        return None
    num_batches = int(getattr(config, "social_attn_stat_batches", 0) or 0)
    if num_batches <= 0:
        return None
    base_model = _get_base_model(model)
    was_training = model.training
    model.eval()
    stats = None

    target_ratio = float(getattr(config, "mabe_mae_target_ratio", 0.5) or 0.5)
    context_ratio = float(getattr(config, "mabe_mae_context_ratio", 0.15) or 0.15)
    block_ratio = float(getattr(config, "mabe_mae_block_ratio", 0.5) or 0.5)
    min_block = int(getattr(config, "mabe_mae_block_min", 4) or 4)
    max_block = int(getattr(config, "mabe_mae_block_max", 16) or 16)
    label_valid_mode = _get_label_valid_mode(config)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            if stage == "pretrain":
                kps = batch["keypoints"].to(device)
                motion_feats = batch["motion_feats"].to(device)
                valid_mask = batch.get("mask")
                if valid_mask is None:
                    continue
                valid_mask = valid_mask.to(device)
                mae_mask = create_social_mae_mask(
                    valid_mask=valid_mask,
                    target_ratio=target_ratio,
                    context_ratio=context_ratio,
                    block_ratio=block_ratio,
                    min_block=min_block,
                    max_block=max_block,
                    device=device,
                    dtype=config.torch_dtype,
                )
                attn_w, key_padding_mask = base_model.get_social_attn_weights(
                    motion_feats,
                    kps,
                    mask=valid_mask,
                    mask_indices=mae_mask,
                )
            else:
                kps = batch["keypoints"].to(device)
                motion_feats = batch["motion_feats"].to(device)
                frame_labels = batch["frame_labels"].to(device)
                frame_mask = batch.get("frame_mask")
                if frame_mask is not None:
                    frame_mask = frame_mask.to(device)
                frame_mask = _select_frame_mask(frame_mask, frame_labels, label_valid_mode)
                B, A, T, _, _ = kps.shape
                if frame_mask.dim() == 3:
                    pad_mask = ~frame_mask
                else:
                    pad_mask = ~frame_mask.unsqueeze(1).expand(B, A, T)
                attn_w, key_padding_mask = base_model.get_social_attn_weights(
                    motion_feats,
                    kps,
                    pad_mask=pad_mask,
                )

            ratios = _compute_social_head_ratios(attn_w, key_padding_mask)
            if ratios is None:
                continue
            r_self, r_neighbor = ratios
            if stats is None:
                stats = {"self": r_self.clone(), "neighbor": r_neighbor.clone(), "count": 1}
            else:
                stats["self"] += r_self
                stats["neighbor"] += r_neighbor
                stats["count"] += 1

    if was_training:
        model.train()
    if stats is None:
        return None
    stats["self"] = stats["self"] / float(stats["count"])
    stats["neighbor"] = stats["neighbor"] / float(stats["count"])
    if writer is not None:
        for h, val in enumerate(stats["self"].tolist()):
            writer.add_scalar(f"{stage}/social_attn_r_self/h{h}", val, epoch_idx + 1)
        for h, val in enumerate(stats["neighbor"].tolist()):
            writer.add_scalar(f"{stage}/social_attn_r_neighbor/h{h}", val, epoch_idx + 1)
    return stats


def _get_scheduler_config(args) -> tuple[str, Dict]:
    raw = getattr(args, "lr_scheduler", "none")
    params = getattr(args, "lr_scheduler_params", {}) or {}
    if isinstance(raw, dict):
        name = raw.get("name") or raw.get("type") or raw.get("scheduler") or "none"
        params = raw.get("params", params)
    else:
        name = raw
    if not isinstance(params, dict):
        params = {}
    return str(name or "none"), _coerce_number(dict(params))


def _build_lr_scheduler(optimizer: torch.optim.Optimizer, args, stage: str | None = None):
    name_raw, params = _get_scheduler_config(args)
    name = name_raw.lower()
    total_epochs = max(1, int(getattr(args, "epochs", 1)))
    if name in ("", "none", "null", "off"):
        return None
    cfg = dict(params)
    scheduler = None
    resolved: Dict = {}

    if name in ("cosineannealinglr", "cosineannealing", "cosine"):
        t_max = int(cfg.pop("T_max", cfg.pop("t_max", total_epochs)))
        eta_min = cfg.pop("eta_min", cfg.pop("lr_min", 0.0))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min, **cfg)
        resolved = {"T_max": t_max, "eta_min": eta_min} | cfg
    elif name in ("steplr", "step"):
        step_size = int(cfg.pop("step_size", max(1, total_epochs // 3)))
        gamma = cfg.pop("gamma", 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, **cfg)
        resolved = {"step_size": step_size, "gamma": gamma} | cfg
    elif name in ("multisteplr", "multistep"):
        milestones_raw = cfg.pop("milestones", None)
        if milestones_raw is None:
            milestones = [int(total_epochs * 0.5), int(total_epochs * 0.75)]
        else:
            milestones = milestones_raw
        milestones = sorted({int(m) for m in milestones if int(m) > 0})
        if not milestones:
            milestones = [max(1, total_epochs // 2)]
        gamma = cfg.pop("gamma", 0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, **cfg)
        resolved = {"milestones": milestones, "gamma": gamma} | cfg
    elif name in ("exponentiallr", "exponential"):
        gamma = cfg.pop("gamma", 0.99)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, **cfg)
        resolved = {"gamma": gamma} | cfg
    elif name in ("linearlr", "linear"):
        total_iters = int(cfg.pop("total_iters", total_epochs))
        total_iters = max(1, total_iters)
        start_factor = cfg.pop("start_factor", 1.0)
        end_factor = cfg.pop("end_factor", None)
        if end_factor is None:
            end_factor = 0.0
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters, **cfg
        )
        resolved = {"start_factor": start_factor, "end_factor": end_factor, "total_iters": total_iters} | cfg
    else:
        raise ValueError(
            f"Unsupported lr_scheduler '{name_raw}'. Expected one of StepLR, MultiStepLR, ExponentialLR, "
            "LinearLR, CosineAnnealingLR, or none."
        )

    # ---- Wrap with linear warmup if configured ----
    warmup_ep = int(getattr(args, "lr_warmup_epochs", 0) or 0)
    if warmup_ep > 0 and scheduler is not None:
        start_factor = float(getattr(args, "lr_warmup_start_factor", 0.01) or 0.01)
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_ep,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, scheduler], milestones=[warmup_ep],
        )
        resolved["warmup_epochs"] = warmup_ep
        resolved["warmup_start_factor"] = start_factor

    if scheduler is not None and is_main_process(getattr(args, "rank", 0)):
        stage_prefix = f"[{stage}]" if stage else ""
        print(f"[LR Scheduler]{stage_prefix} Using {scheduler.__class__.__name__} with params: {resolved}")
    return scheduler


def _build_run_dir(args, stage: str) -> str:
    # A fixed `run_dir` (no timestamp) lets a requeued job find its own
    # latest.pth and resume; see _maybe_resume.
    fixed_dir = str(getattr(args, "run_dir", "") or "").strip()
    if fixed_dir:
        if is_main_process(getattr(args, "rank", 0)):
            os.makedirs(fixed_dir, exist_ok=True)
        return fixed_dir
    base_dir = args.pretrain_save_dir if stage == "pretrain" else args.finetune_save_dir
    if is_main_process(getattr(args, "rank", 0)):
        os.makedirs(base_dir, exist_ok=True)
    run_name = (getattr(args, "run_name", "") or "").strip()
    ts = time.strftime("%Y%m%d-%H%M%S")
    if not run_name:
        if stage == "pretrain":
            run_name = f"{ts}_pre_bs{args.batch_size}_lr{args.lr}_mask{args.mabe_mae_target_ratio}"
        else:
            run_name = f"{ts}_ft_bs{args.batch_size}_lr{args.lr}"
    else:
        run_name = f"{run_name}_{ts}"
    run_dir = os.path.join(base_dir, run_name)
    if getattr(args, "distributed", False):
        shared = [run_dir if is_main_process(args.rank) else None]
        dist.broadcast_object_list(shared, src=0)
        run_dir = shared[0]
    if is_main_process(getattr(args, "rank", 0)):
        os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _save_checkpoint(
    stage: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    best_val: float | None,
    best_epoch: int | None,
    epoch: int,
    run_dir: str,
    prefix: str,
    add_epoch_suffix: bool = True,
):
    if not _is_main_checkpoint_process():
        return None
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "epoch": epoch,
        "stage": stage,
    }
    ckpt_name = f"{prefix}_ep{epoch + 1}.pth" if add_epoch_suffix else f"{prefix}.pth"
    ckpt_path = os.path.join(run_dir, ckpt_name)
    torch.save(state, ckpt_path)
    latest = os.path.join(run_dir, "latest.pth")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(os.path.basename(ckpt_path), latest)
    except Exception:
        shutil.copyfile(ckpt_path, latest)
    return ckpt_path


def _cleanup_old_best(run_dir: str, prefix: str, keep_path: str | None = None):
    if not _is_main_checkpoint_process():
        return
    pattern = os.path.join(run_dir, f"{prefix}_ep*.pth")
    for path in glob.glob(pattern):
        if keep_path is not None and os.path.abspath(path) == os.path.abspath(keep_path):
            continue
        try:
            os.remove(path)
        except OSError:
            pass


def _maybe_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    run_dir: str,
    resume_path: str | None,
    device: torch.device,
    stage: str,
    is_main: bool,
) -> tuple[int, float | None, int | None]:
    start_epoch = 0
    best_val = None
    best_epoch = None
    candidate = resume_path or os.path.join(run_dir, "latest.pth")
    if candidate and os.path.exists(candidate):
        state = torch.load(candidate, map_location=device)
        model.load_state_dict(state.get("model", state))
        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        best_val = state.get("best_val", None)
        best_epoch = state.get("best_epoch", None)
        start_epoch = int(state.get("epoch", -1)) + 1
        if is_main:
            print(f"[Resume:{stage}] Loaded checkpoint from {candidate}, restarting at epoch {start_epoch + 1}")
    return start_epoch, best_val, best_epoch


def _configure_finetune_trainables(
    model: torch.nn.Module,
    args: None,
    is_main: bool = True,
):
    base_model = model.module if hasattr(model, "module") else model
    base_model = getattr(base_model, "_orig_mod", base_model)
    for _, p in base_model.named_parameters():
        p.requires_grad = True
    # Freeze motion_scout entirely (RVQ encoder is pretrained)
    for name, p in base_model.named_parameters():
        if name.startswith("motion_scout.") or ".motion_scout." in name:
            p.requires_grad = False

    # Optionally freeze pretrained intermediate modules (context_transformer, social_encoder, social_fuse)
    finetune_middle = bool(getattr(args, "finetune_middle_modules", False))
    if not finetune_middle:
        middle_prefixes = ("context_transformer.", "social_encoder.", "social_fuse.", "motion_norm.")
        frozen_middle = 0
        for name, p in base_model.named_parameters():
            if any(name.startswith(pf) or f".{pf}" in name for pf in middle_prefixes):
                p.requires_grad = False
                frozen_middle += p.numel()
        if is_main and frozen_middle > 0:
            print(f"[Finetune] Middle modules frozen: {frozen_middle/1e6:.3f}M params "
                  f"(context_transformer/social_encoder/social_fuse/motion_norm)")

    total = 0
    trainable = 0
    trainable_bytes = 0
    for p in base_model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            trainable_bytes += n * p.element_size()
    if is_main:
        print(
            f"[Finetune] Trainables: "
            f"{trainable/1e6:.3f}M / {total/1e6:.3f}M "
            f"({trainable_bytes/1e6:.1f} MB)"
        )


def _get_label_valid_mode(config) -> str:
    mode = str(getattr(config, "label_valid_mode", "any") or "any").lower()
    return "all" if mode in ("all", "full", "complete") else "any"


def _select_frame_mask(
    frame_mask: torch.Tensor | None,
    frame_labels: torch.Tensor,
    valid_mode: str = "any",
) -> torch.Tensor:
    # Cache-derived frame_mask_majority is authoritative when provided.
    if frame_mask is not None:
        return frame_mask.bool()
    return _coerce_frame_mask(frame_mask, frame_labels, valid_mode)


def _coerce_frame_mask(
    frame_mask: torch.Tensor | None,
    frame_labels: torch.Tensor,
    valid_mode: str = "any",
) -> torch.Tensor:
    valid_mode = "all" if str(valid_mode).lower() == "all" else "any"
    if frame_labels.dim() == 2:
        derived = (frame_labels >= 0) if valid_mode == "all" else (frame_labels != -1)
    else:
        if valid_mode == "all":
            derived = (frame_labels >= 0).all(dim=-1)
        else:
            derived = (frame_labels != -1).any(dim=-1)
    if frame_mask is None:
        return derived
    mask = frame_mask.bool()
    while mask.dim() < derived.dim():
        mask = mask.unsqueeze(0)
    return mask & derived


def _sanitize_frame_labels(
    frame_labels: torch.Tensor,
    frame_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_labels[frame_labels < 0] = 0
    return frame_labels, frame_mask


def _compute_pos_weight_from_loader(
    dataloader,
    config,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    label_valid_mode = _get_label_valid_mode(config)
    pos_sum = None
    neg_sum = None
    with torch.no_grad():
        for batch in dataloader:
            frame_labels = batch.get("frame_labels")
            if frame_labels is None:
                continue
            frame_labels = frame_labels.to(device or torch.device("cpu"))
            frame_mask = batch.get("frame_mask")
            if frame_mask is not None:
                frame_mask = frame_mask.to(frame_labels.device)
            frame_mask = _select_frame_mask(frame_mask, frame_labels, label_valid_mode)
            frame_labels, frame_mask = _majority_aggregate_frame_labels(frame_labels, frame_mask)
            frame_labels, frame_mask = _sanitize_frame_labels(frame_labels, frame_mask)

            if frame_labels.dim() == 2:
                frame_labels = frame_labels.unsqueeze(0)
            if frame_mask is None:
                valid = torch.ones(frame_labels.shape[:2], device=frame_labels.device, dtype=frame_labels.dtype)
            else:
                valid = frame_mask
                if valid.dim() == 3:
                    valid = valid.any(dim=1)
                valid = valid.to(dtype=frame_labels.dtype)
                if valid.dim() == 1:
                    valid = valid.unsqueeze(0)

            pos = (frame_labels * valid.unsqueeze(-1)).sum(dim=(0, 1))
            neg = ((1.0 - frame_labels) * valid.unsqueeze(-1)).sum(dim=(0, 1))
            pos_sum = pos if pos_sum is None else pos_sum + pos
            neg_sum = neg if neg_sum is None else neg_sum + neg

    if pos_sum is None or neg_sum is None:
        return None
    pos_weight = neg_sum / (pos_sum + 1e-6)
    pos_weight = torch.where(pos_sum > 0, pos_weight, torch.zeros_like(pos_weight))
    return pos_weight


def _majority_aggregate_frame_labels(
    frame_labels: torch.Tensor,
    frame_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if frame_labels.dim() != 4:
        return frame_labels, frame_mask
    labels = frame_labels
    valid = labels != -1
    if frame_mask is not None:
        mask = frame_mask
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(-1)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(-1)
        valid = valid & mask.bool()
    valid_count = valid.sum(dim=1)
    pos_count = ((labels == 1) & valid).sum(dim=1)
    majority = pos_count > (valid_count / 2.0)
    agg = labels.new_full((labels.shape[0], labels.shape[2], labels.shape[3]), -1)
    has_valid = valid_count > 0
    if has_valid.any():
        agg[has_valid] = majority[has_valid].to(dtype=labels.dtype)
    return agg, frame_mask


def _downsample_frame_labels(
    labels: torch.Tensor,
    target_len: int,
    stride: int,
) -> torch.Tensor:
    if labels.dim() == 2:
        labels = labels.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    if stride > 1:
        labels = labels[:, ::stride]
    if labels.shape[1] < target_len:
        pad_len = target_len - labels.shape[1]
        pad = torch.full(
            (labels.shape[0], pad_len, labels.shape[2]),
            -1,
            dtype=labels.dtype,
            device=labels.device,
        )
        labels = torch.cat([labels, pad], dim=1)
    labels = labels[:, :target_len]
    return labels.squeeze(0) if squeeze else labels


def _downsample_frame_mask(
    mask: torch.Tensor,
    target_len: int,
    stride: int,
) -> torch.Tensor:
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    if stride > 1:
        mask = mask[:, ::stride]
    if mask.shape[1] < target_len:
        pad_len = target_len - mask.shape[1]
        pad = torch.zeros((mask.shape[0], pad_len), dtype=mask.dtype, device=mask.device)
        mask = torch.cat([mask, pad], dim=1)
    mask = mask[:, :target_len]
    return mask.squeeze(0) if squeeze else mask


def _sample_block_mask_1d(
    length: int,
    ratio: float,
    min_block: int,
    max_block: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if length <= 0 or ratio <= 0.0:
        return torch.zeros((length,), device=device, dtype=dtype)
    target = int(round(ratio * length))
    if target <= 0:
        return torch.zeros((length,), device=device, dtype=dtype)
    min_block = max(1, min(int(min_block), length))
    max_block = max(min_block, min(int(max_block), length))
    mask = torch.zeros((length,), device=device, dtype=dtype)
    max_tries = max(4, target * 4)
    tries = 0
    while mask.sum().item() < target and tries < max_tries:
        block_len = int(torch.randint(min_block, max_block + 1, (1,), device=device).item())
        start = int(torch.randint(0, max(1, length - block_len + 1), (1,), device=device).item())
        mask[start:start + block_len] = 1
        tries += 1
    return mask


def create_social_mae_mask(
    valid_mask: torch.Tensor | None,
    target_ratio: float,
    context_ratio: float,
    block_ratio: float,
    min_block: int,
    max_block: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if valid_mask is None:
        raise ValueError("valid_mask is required for social M-MAE mask creation.")
    if valid_mask.dim() != 3:
        raise ValueError(f"Expected valid_mask [B, A, T], got {valid_mask.shape}")
    B, A, T = valid_mask.shape
    mask = torch.zeros((B, A, T), device=device, dtype=dtype)
    if A == 0 or T == 0:
        return mask
    target_idx = int(torch.randint(0, A, (1,), device=device).item())
    block_ratio = float(max(0.0, min(block_ratio, 1.0)))
    for a in range(A):
        ratio = float(target_ratio if a == target_idx else context_ratio)
        ratio = max(0.0, min(ratio, 1.0))
        point_ratio = ratio * (1.0 - block_ratio)
        block_ratio_use = ratio * block_ratio
        point_mask = (torch.rand((B, T), device=device) < point_ratio).to(dtype=dtype)
        block_mask = torch.zeros((B, T), device=device, dtype=dtype)
        if block_ratio_use > 0.0:
            for b in range(B):
                block_mask[b] = _sample_block_mask_1d(
                    T,
                    block_ratio_use,
                    min_block,
                    max_block,
                    device=device,
                    dtype=dtype,
                )
        cur = torch.clamp(point_mask + block_mask, 0.0, 1.0)
        mask[:, a] = cur
    mask = mask * valid_mask.to(dtype=dtype)
    return mask


def _normalize_vision_targets(base_model: torch.nn.Module, vision_feats: torch.Tensor) -> torch.Tensor:
    feats = F.normalize(vision_feats, dim=-1, eps=1e-6).detach()
    adapter = getattr(base_model, "target_adapter", None)
    if adapter is not None:
        feats = adapter(feats)
        feats = F.normalize(feats, dim=-1, eps=1e-6)
    return feats


def _broadcast_eval_stats(stats: dict, device: torch.device, src: int = 0) -> dict:
    keys = ("loss", "cls", "sparse", "pred", "col")
    values = [float(stats.get(k, 0.0) or 0.0) for k in keys]
    tensor = torch.tensor(values, device=device, dtype=torch.float32)
    dist.broadcast(tensor, src=src)
    for idx, key in enumerate(keys):
        stats[key] = tensor[idx].item()
    return stats


def compute_multihorizon_pred_loss(
    pred_vis: Dict[str, torch.Tensor],
    vision_feats: torch.Tensor | None,
    stride: int,
    horizons: list[int],
    weights: list[float],
    loss_type: str,
    base_model: torch.nn.Module,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    if vision_feats is None:
        return next(iter(pred_vis.values())).new_tensor(0.0), {}
    if len(weights) != len(horizons):
        weights = [1.0 for _ in horizons]
    L = next(iter(pred_vis.values())).shape[1]
    vision_ds = vision_feats[:, :: max(1, int(stride))] if stride > 1 else vision_feats
    assert vision_ds.shape[1] == L and stride == 1
    target_norm = _normalize_vision_targets(base_model, vision_ds)
    losses: Dict[int, torch.Tensor] = {}
    total = next(iter(pred_vis.values())).new_tensor(0.0)
    for h, w in zip(horizons, weights):
        if h <= 0 or h >= L:
            continue
        key = str(h)
        if key not in pred_vis:
            continue
        pred = pred_vis[key]
        pred_use = pred[:, :-h]
        target = target_norm[:, h:]
        pred_norm = F.normalize(pred_use, dim=-1, eps=1e-6)
        if loss_type == "mse":
            loss_per_step = (pred_norm - target).pow(2).mean(dim=-1)
        else:
            loss_per_step = 1.0 - (pred_norm * target).sum(dim=-1)
        if valid_mask is not None:
            mask_use = valid_mask[:, h:].to(dtype=loss_per_step.dtype)
            denom = mask_use.sum().clamp(min=1.0)
            loss = (loss_per_step * mask_use).sum() / denom
        else:
            loss = loss_per_step.mean()
        losses[h] = loss
        total = total + float(w) * loss
    return total, losses


def compute_multilabel_frame_loss(
    logits_seq: torch.Tensor,
    frame_labels: torch.Tensor,
    frame_mask: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pos_weight is not None:
        pos_weight = pos_weight.to(device=logits_seq.device, dtype=logits_seq.dtype)
    if logits_seq.dim() == 2:
        labels = frame_labels
        if labels.dim() == 3:
            labels = labels.clone()
            labels = labels.masked_fill(labels < 0, 0)
            label_vec = labels.any(dim=1).to(dtype=logits_seq.dtype)
        else:
            label_vec = labels.to(dtype=logits_seq.dtype)
        loss = F.binary_cross_entropy_with_logits(logits_seq, label_vec, pos_weight=pos_weight)
        return loss, logits_seq

    labels = frame_labels
    if labels.dim() == 2:
        labels = labels.unsqueeze(0)
    labels = labels.to(dtype=logits_seq.dtype)
    labels = labels.masked_fill(labels < 0, 0)
    loss_raw = F.binary_cross_entropy_with_logits(
        logits_seq,
        labels,
        reduction="none",
        pos_weight=pos_weight,
    )
    if frame_mask is not None:
        mask = frame_mask.to(dtype=loss_raw.dtype)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        denom = mask.sum().clamp(min=1.0)
        loss = (loss_raw * mask.unsqueeze(-1)).sum() / denom
    else:
        loss = loss_raw.mean()
    return loss, logits_seq


def _init_f1_counts(num_classes: int, device: torch.device) -> torch.Tensor:
    return torch.zeros((num_classes, 4), device=device, dtype=torch.float32)  # tp/fp/fn/total


def _normalize_metadata_list(metadata, batch_size: int | None = None) -> list[dict] | None:
    if metadata is None:
        return None
    if isinstance(metadata, list):
        if metadata and isinstance(metadata[0], dict):
            if batch_size is None:
                return metadata
            if len(metadata) >= batch_size:
                return metadata[:batch_size]
            return metadata + [{} for _ in range(batch_size - len(metadata))]
    if isinstance(metadata, dict):
        keys = list(metadata.keys())
        if not keys:
            return None
        # Build list of dicts from dict of lists
        first = metadata[keys[0]]
        if isinstance(first, list):
            size = len(first)
        elif torch.is_tensor(first):
            size = int(first.shape[0])
        else:
            return None
        if batch_size is not None:
            size = batch_size
        out = []
        for i in range(size):
            item = {}
            for k in keys:
                val = metadata[k]
                if isinstance(val, list):
                    if i < len(val):
                        item[k] = val[i]
                    elif i == 0:
                        cache_path = metadata.get("cache_path") or metadata.get("pt_file")
                        print(
                            f"[MetaWarn] key='{k}' list_len={len(val)} < batch_size={size}; "
                            f"cache_path={cache_path}",flush=True
                        )
                elif torch.is_tensor(val):
                    if i < val.shape[0]:
                        item[k] = val[i].item()
                    elif i == 0:
                        cache_path = metadata.get("cache_path") or metadata.get("pt_file")
                        print(
                            f"[MetaWarn] key='{k}' tensor_len={int(val.shape[0])} < batch_size={size}; "
                            f"cache_path={cache_path}",flush=True
                        )
                else:
                    item[k] = val
            out.append(item)
        return out
    return None


def _accumulate_f1_counts_frame_index(
    counts: torch.Tensor,
    logits: torch.Tensor,
    frame_labels: torch.Tensor,
    frame_mask: torch.Tensor | None,
    metadata,
    seen_by_video: dict,
) -> None:
    if logits.dim() != 3:
        raise RuntimeError("Expected per-frame logits [B,T,C] for frame-level F1.")
    labels = frame_labels
    if labels.dim() == 2:
        labels = labels.unsqueeze(0)
    labels = labels.masked_fill(labels < 0, 0)
    preds = torch.sigmoid(logits) >= 0.5
    gts = labels >= 0.5
    if frame_mask is not None:
        valid = frame_mask
        if valid.dim() == 1:
            valid = valid.unsqueeze(0)
        valid = valid.to(dtype=torch.bool)
    else:
        valid = torch.ones((logits.shape[0], logits.shape[1]), device=logits.device, dtype=torch.bool)
    meta_list = _normalize_metadata_list(metadata, batch_size=logits.shape[0])
    if meta_list is None or len(meta_list) != logits.shape[0]:
        raise RuntimeError("Missing or invalid metadata for frame-index F1.")
    for b, meta in enumerate(meta_list):
        video_id = meta.get("video_id")
        start = meta.get("start_frame")
        stride = meta.get("ds_stride", 1)
        if video_id is None or start is None:
            continue
        seen = seen_by_video.setdefault(video_id, set())
        valid_idx = torch.nonzero(valid[b], as_tuple=False).squeeze(-1)
        for t in valid_idx.tolist():
            frame_id = int(start) + int(t) * int(stride)
            if frame_id in seen:
                continue
            seen.add(frame_id)
            pred_t = preds[b, t]
            gt_t = gts[b, t]
            counts[:, 0] += (pred_t & gt_t).to(dtype=counts.dtype)
            counts[:, 1] += (pred_t & ~gt_t).to(dtype=counts.dtype)
            counts[:, 2] += ((~pred_t) & gt_t).to(dtype=counts.dtype)
            counts[:, 3] += 1.0


def _counts_to_f1(counts: torch.Tensor) -> list[float]:
    tp = counts[:, 0]
    fp = counts[:, 1]
    fn = counts[:, 2]
    denom = (2 * tp + fp + fn).clamp(min=1e-8)
    f1 = (2 * tp) / denom
    return f1.detach().cpu().tolist()


def _accumulate_f1_counts_frame_index_with_other(
    counts: torch.Tensor,
    logits: torch.Tensor,
    frame_labels: torch.Tensor,
    frame_mask: torch.Tensor | None,
    metadata,
    seen_by_video: dict,
) -> None:
    """Same frame-index-dedup F1 as ``_accumulate_f1_counts_frame_index`` but with
    a derived 'other' channel appended (``counts`` shape [C+1, 4]).

    'other' is defined purely from the SAME sigmoid>=0.5 decisions on the C real
    classes: pred_other = none of C fired; gt_other = none of C active. This lets
    CalMS21 report the 4-class macro-F1 (attack/investigation/mount/other) from
    exactly the model's existing 3-class predictions — no extra head, no retrain.
    Gated by ``config.report_other_class`` so mabe/openfield are unaffected.
    """
    if logits.dim() != 3:
        raise RuntimeError("Expected per-frame logits [B,T,C] for frame-level F1.")
    labels = frame_labels
    if labels.dim() == 2:
        labels = labels.unsqueeze(0)
    labels = labels.masked_fill(labels < 0, 0)
    preds = torch.sigmoid(logits) >= 0.5
    gts = labels >= 0.5
    preds = torch.cat([preds, ~preds.any(dim=-1, keepdim=True)], dim=-1)
    gts = torch.cat([gts, ~gts.any(dim=-1, keepdim=True)], dim=-1)
    if frame_mask is not None:
        valid = frame_mask
        if valid.dim() == 1:
            valid = valid.unsqueeze(0)
        valid = valid.to(dtype=torch.bool)
    else:
        valid = torch.ones((logits.shape[0], logits.shape[1]),
                           device=logits.device, dtype=torch.bool)
    meta_list = _normalize_metadata_list(metadata, batch_size=logits.shape[0])
    if meta_list is None or len(meta_list) != logits.shape[0]:
        raise RuntimeError("Missing or invalid metadata for frame-index F1.")
    for b, meta in enumerate(meta_list):
        video_id = meta.get("video_id")
        start = meta.get("start_frame")
        stride = meta.get("ds_stride", 1)
        if video_id is None or start is None:
            continue
        seen = seen_by_video.setdefault(video_id, set())
        valid_idx = torch.nonzero(valid[b], as_tuple=False).squeeze(-1)
        for t in valid_idx.tolist():
            frame_id = int(start) + int(t) * int(stride)
            if frame_id in seen:
                continue
            seen.add(frame_id)
            pred_t = preds[b, t]
            gt_t = gts[b, t]
            counts[:, 0] += (pred_t & gt_t).to(dtype=counts.dtype)
            counts[:, 1] += (pred_t & ~gt_t).to(dtype=counts.dtype)
            counts[:, 2] += ((~pred_t) & gt_t).to(dtype=counts.dtype)
            counts[:, 3] += 1.0


def train_one_epoch_pretrain(
    model: ResiFusePretrainer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config,
    epoch_idx: int,
):
    is_main = is_main_process(getattr(config, "rank", 0))
    if bool(getattr(config, "detect_anomaly", False)):
        torch.autograd.set_detect_anomaly(True)
    model.train()
    total_loss = 0.0
    total_mask = 0.0
    total_pred = 0.0
    total_recon = 0.0
    total_vq = 0.0
    num_batches = len(dataloader)
    grad_acc_steps = int(getattr(config, "pretrain_grad_acc_steps", None) or getattr(config, "grad_acc_steps", 1) or 1)
    grad_acc_steps = max(1, grad_acc_steps)
    progress_points = sorted({max(1, int(num_batches * frac)) for frac in (0.2, 0.4, 0.6, 0.8) if num_batches > 0})

    fetch_time = 0.0
    compute_time = 0.0
    end_time = time.perf_counter()
    vision_pred_weight = float(getattr(config, "vision_pred_weight", 0.0) or 0.0)
    vision_vres_weight = float(getattr(config, "vision_vres_weight", 0.0) or 0.0)
    pred_warmup = int(getattr(config, "vision_pred_warmup_epochs", 0) or 0)
    pred_scale = min(1.0, float(epoch_idx + 1) / pred_warmup) if pred_warmup > 0 else 1.0
    vision_pred_loss = str(getattr(config, "vision_pred_loss", "cosine") or "cosine").lower()
    if vision_pred_loss not in ("cosine", "mse"):
        vision_pred_loss = "cosine"
    horizons = getattr(config, "vision_pred_horizons", None) or []
    weights = getattr(config, "vision_pred_weights", None) or [1.0 for _ in horizons]
    horizons = [int(h) for h in horizons]
    weights = [float(w) for w in weights]
    per_horizon = {h: 0.0 for h in horizons}
    total_vres = 0.0

    target_ratio = float(getattr(config, "mabe_mae_target_ratio", 0.5) or 0.5)
    context_ratio = float(getattr(config, "mabe_mae_context_ratio", 0.15) or 0.15)
    block_ratio = float(getattr(config, "mabe_mae_block_ratio", 0.5) or 0.5)
    min_block = int(getattr(config, "mabe_mae_block_min", 4) or 4)
    max_block = int(getattr(config, "mabe_mae_block_max", 16) or 16)
    base_model = _get_base_model(model)
    optimizer.zero_grad()

    need_vision = (vision_pred_weight > 0.0) or (vision_vres_weight > 0.0)
    for batch_idx, batch in enumerate(dataloader):
        fetch_start = end_time
        kps = batch["keypoints"].to(device)       # [B, A, T, K, 3]
        motion_feats = batch["motion_feats"].to(device)  # [B, A, T, D]
        valid_mask = batch.get("mask")
        if valid_mask is None:
            raise RuntimeError("Pretrain batch missing 'mask' for M-MAE sampling.")
        valid_mask = valid_mask.to(device)
        vision_feats = batch.get("vision_feats") if need_vision else None
        if vision_feats is not None:
            vision_feats = vision_feats.to(device)
        if need_vision and vision_feats is None:
            raise RuntimeError(
                "Pretrain forward needs vision_feats (vision_pred_weight>0 or vision_vres_weight>0). "
                "Enable use_vision_cache in config."
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fetch_time += time.perf_counter() - fetch_start
        compute_start = time.perf_counter()

        B, A, T, K, _ = kps.shape
        kps_flat = kps.view(B * A, T, K, 3)
        motion_flat = motion_feats.view(B * A, T, -1)
        vision_flat = vision_feats.view(B * A, T, -1) if vision_feats is not None else None
        valid_flat = valid_mask.view(B * A, T)

        mae_mask = create_social_mae_mask(
            valid_mask=valid_mask,
            target_ratio=target_ratio,
            context_ratio=context_ratio,
            block_ratio=block_ratio,
            min_block=min_block,
            max_block=max_block,
            device=device,
            dtype=config.torch_dtype,
        )
        mae_flat = mae_mask.view(B * A, T)

        out = model(
            motion_feats,
            kps,
            mask=valid_mask,
            mask_indices=mae_mask,
            vision_feats=vision_flat,
        )

        pred_kps = out["pred_kps"]
        pred_vis = out["pred_vis"]
        vq_recon = out["vq_recon"]
        vq_loss = out["vq_loss"]
        kp_ds = out["kp_ds"]
        stride = out.get("stride", max(1, T // pred_kps.shape[1]))

        pred_len = pred_kps.shape[1]
        if vision_pred_weight > 0.0 and pred_len <= min(horizons or [1]):
            raise RuntimeError(
                f"vision_pred_horizon must be smaller than pred length ({pred_len})."
            )
        mask_ds = _downsample_frame_mask(mae_flat, pred_len, stride)
        valid_ds = _downsample_frame_mask(valid_flat.to(dtype=mask_ds.dtype), pred_len, stride)
        mask_ds = mask_ds * valid_ds

        mask_expanded = mask_ds.view(B * A, pred_len, 1, 1).expand_as(pred_kps)
        target_kps = kp_ds[..., :2]
        loss_mse = F.mse_loss(pred_kps * mask_expanded, target_kps * mask_expanded, reduction="sum")
        num_masked = mask_expanded.sum() + 1e-6
        loss_masked = loss_mse / num_masked

        target_motion = motion_flat[:, :: max(1, T // pred_len)][:, :pred_len, :]
        loss_recon = F.mse_loss(vq_recon, target_motion)
        if vision_pred_weight > 0.0:
            if vision_flat is None:
                raise RuntimeError(
                    "vision_pred_weight > 0 but no vision_feats found in batch. "
                    "Enable use_vision_cache."
                )
            loss_pred, loss_dict = compute_multihorizon_pred_loss(
                pred_vis,
                vision_flat,
                stride=stride,
                horizons=horizons,
                weights=weights,
                loss_type=vision_pred_loss,
                base_model=base_model,
                valid_mask=None,
            )
        else:
            loss_pred = motion_flat.new_tensor(0.0)
            loss_dict = {}

        warmup_epochs = getattr(config, "vq_loss_warmup_epochs", 0)
        vq_scale = min(1.0, float(epoch_idx + 1) / warmup_epochs) if warmup_epochs and warmup_epochs > 0 else 1.0
        vq_recon_term = config.vq_recon_weight * loss_recon
        vq_term = (config.vq_loss_weight * vq_scale) * vq_loss
        pred_term = (vision_pred_weight * pred_scale) * loss_pred
        vres_term = vision_vres_weight * out["vis_res_loss"]
        loss_full = loss_masked + pred_term + vq_recon_term + vq_term + vres_term
        loss_full_scaled = loss_full / grad_acc_steps
        loss_full_scaled.backward()
        _report_unused_params(model, "pretrain", batch_idx)
        if (batch_idx + 1) % grad_acc_steps == 0 or (batch_idx + 1) == num_batches:
            optimizer.step()
            base_model = _get_base_model(model)
            base_model.update_vq_ema()
            optimizer.zero_grad()

        total_loss += loss_full.item()
        total_mask += loss_masked.item()
        total_pred += loss_pred.item()
        total_recon += vq_recon_term.item()
        total_vq += vq_term.item()
        total_vres += vres_term.item()
        for h, val in loss_dict.items():
            per_horizon[h] = per_horizon.get(h, 0.0) + val.item()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_time += time.perf_counter() - compute_start
        end_time = time.perf_counter()
        if is_main and (batch_idx + 1 in progress_points):
            print(f"[Pretrain] Batch {batch_idx + 1}/{num_batches}")

    denom = max(1, num_batches)
    stats = {
        "loss": total_loss / denom,
        "mask": total_mask / denom,
        "pred": total_pred / denom,
        "vres": total_vres / denom,
        "vq_recon": total_recon / denom,
        "vq": total_vq / denom,
        "pred_h": {h: (per_horizon[h] / denom) for h in per_horizon},
        "avg_fetch": fetch_time / denom,
        "avg_compute": compute_time / denom,
        "total_fetch": fetch_time,
        "total_compute": compute_time,
        "num_batches": num_batches,
    }
    return stats


@torch.no_grad()
def evaluate_pretrain(
    model: ResiFusePretrainer,
    dataloader: DataLoader,
    device: torch.device,
    config,
    epoch_idx: int,
):
    model.eval()
    total_loss = 0.0
    total_mask = 0.0
    total_pred = 0.0
    total_recon = 0.0
    total_vq = 0.0
    num_batches = len(dataloader)
    vision_pred_weight = float(getattr(config, "vision_pred_weight", 0.0) or 0.0)
    pred_warmup = int(getattr(config, "vision_pred_warmup_epochs", 0) or 0)
    pred_scale = min(1.0, float(epoch_idx + 1) / pred_warmup) if pred_warmup > 0 else 1.0
    vision_pred_loss = str(getattr(config, "vision_pred_loss", "cosine") or "cosine").lower()
    if vision_pred_loss not in ("cosine", "mse"):
        vision_pred_loss = "cosine"
    horizons = getattr(config, "vision_pred_horizons", None) or []
    weights = getattr(config, "vision_pred_weights", None) or [1.0 for _ in horizons]
    horizons = [int(h) for h in horizons]
    weights = [float(w) for w in weights]
    per_horizon = {h: 0.0 for h in horizons}

    target_ratio = float(getattr(config, "mabe_mae_target_ratio", 0.5) or 0.5)
    context_ratio = float(getattr(config, "mabe_mae_context_ratio", 0.15) or 0.15)
    block_ratio = float(getattr(config, "mabe_mae_block_ratio", 0.5) or 0.5)
    min_block = int(getattr(config, "mabe_mae_block_min", 4) or 4)
    max_block = int(getattr(config, "mabe_mae_block_max", 16) or 16)
    base_model = _get_base_model(model)
    vision_vres_weight = float(getattr(config, "vision_vres_weight", 0.0) or 0.0)
    need_vision = (vision_pred_weight > 0.0) or (vision_vres_weight > 0.0)

    for batch in dataloader:
        kps = batch["keypoints"].to(device)
        motion_feats = batch["motion_feats"].to(device)
        valid_mask = batch.get("mask")
        if valid_mask is None:
            raise RuntimeError("Pretrain batch missing 'mask' for M-MAE sampling.")
        valid_mask = valid_mask.to(device)
        vision_feats = batch.get("vision_feats") if need_vision else None
        if vision_feats is not None:
            vision_feats = vision_feats.to(device)
        if need_vision and vision_feats is None:
            raise RuntimeError(
                "Pretrain eval needs vision_feats (vision_pred_weight>0 or vision_vres_weight>0). "
                "Enable use_vision_cache in config."
            )

        B, A, T, K, _ = kps.shape
        kps_flat = kps.view(B * A, T, K, 3)
        motion_flat = motion_feats.view(B * A, T, -1)
        vision_flat = vision_feats.view(B * A, T, -1) if vision_feats is not None else None
        valid_flat = valid_mask.view(B * A, T)

        mae_mask = create_social_mae_mask(
            valid_mask=valid_mask,
            target_ratio=target_ratio,
            context_ratio=context_ratio,
            block_ratio=block_ratio,
            min_block=min_block,
            max_block=max_block,
            device=device,
            dtype=config.torch_dtype,
        )
        mae_flat = mae_mask.view(B * A, T)

        out = model(
            motion_feats,
            kps,
            mask=valid_mask,
            mask_indices=mae_mask,
            vision_feats=vision_flat,
        )

        pred_kps = out["pred_kps"]
        pred_vis = out["pred_vis"]
        vq_recon = out["vq_recon"]
        vq_loss = out["vq_loss"]
        kp_ds = out["kp_ds"]
        stride = out.get("stride", max(1, T // pred_kps.shape[1]))

        if not torch.isfinite(pred_kps).all() or not torch.isfinite(kp_ds).all():
            continue
        if not torch.isfinite(vq_recon).all() or not torch.isfinite(vq_loss).all():
            continue

        pred_len = pred_kps.shape[1]
        mask_ds = _downsample_frame_mask(mae_flat, pred_len, stride)
        valid_ds = _downsample_frame_mask(valid_flat.to(dtype=mask_ds.dtype), pred_len, stride)
        mask_ds = mask_ds * valid_ds

        mask_expanded = mask_ds.view(B * A, pred_len, 1, 1).expand_as(pred_kps)
        target_kps = kp_ds[..., :2]
        loss_mse = F.mse_loss(pred_kps * mask_expanded, target_kps * mask_expanded, reduction="sum")
        num_masked = mask_expanded.sum() + 1e-6
        loss_masked = loss_mse / num_masked

        target_motion = motion_flat[:, :: max(1, T // pred_len)][:, :pred_len, :]
        loss_recon = F.mse_loss(vq_recon, target_motion)
        if vision_pred_weight > 0.0 and vision_flat is not None:
            loss_pred, loss_dict = compute_multihorizon_pred_loss(
                pred_vis,
                vision_flat,
                stride=stride,
                horizons=horizons,
                weights=weights,
                loss_type=vision_pred_loss,
                base_model=base_model,
                valid_mask=None,
            )
        else:
            loss_pred = motion_flat.new_tensor(0.0)
            loss_dict = {}

        warmup_epochs = getattr(config, "vq_loss_warmup_epochs", 0)
        vq_scale = min(1.0, float(epoch_idx + 1) / warmup_epochs) if warmup_epochs and warmup_epochs > 0 else 1.0
        vq_recon_term = config.vq_recon_weight * loss_recon
        vq_term = (config.vq_loss_weight * vq_scale) * vq_loss
        pred_term = (vision_pred_weight * pred_scale) * loss_pred
        loss_full = loss_masked + pred_term + vq_recon_term + vq_term

        if not torch.isfinite(loss_full):
            continue

        total_loss += loss_full.item()
        total_mask += loss_masked.item()
        total_pred += loss_pred.item()
        total_recon += vq_recon_term.item()
        total_vq += vq_term.item()
        for h, val in loss_dict.items():
            per_horizon[h] = per_horizon.get(h, 0.0) + val.item()

    denom = max(1, num_batches)
    stats = {
        "loss": total_loss / denom,
        "mask": total_mask / denom,
        "pred": total_pred / denom,
        "vq_recon": total_recon / denom,
        "vq": total_vq / denom,
        "pred_h": {h: (per_horizon[h] / denom) for h in per_horizon},
    }
    return stats


def maybe_load_pretrain_weights(model: ResiFuseClassifier, ckpt_path: str, device: torch.device):
    print(f"[Pretrain] Loading pretrain checkpoint from: {ckpt_path}", flush=True)
    if ckpt_path is None:
        return
    if not os.path.exists(ckpt_path):
        print(f"Pretrain checkpoint not found at {ckpt_path}, skipping warm-start.")
        return
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model_state = model.state_dict()
    model_keys = list(model_state.keys())

    def _strip_prefix(sd, prefix: str):
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    def _add_prefix(sd, prefix: str):
        return {f"{prefix}{k}": v for k, v in sd.items()}

    def _maybe_remap_prefix(sd):
        sd_keys = list(sd.keys())
        if not sd_keys:
            return sd
        remapped = sd
        # Handle DDP "module." prefix mismatch.
        if all(k.startswith("module.") for k in sd_keys) and not any(k.startswith("module.") for k in model_keys):
            remapped = _strip_prefix(remapped, "module.")
        elif not any(k.startswith("module.") for k in sd_keys) and any(k.startswith("module.") for k in model_keys):
            remapped = _add_prefix(remapped, "module.")

        sd_keys = list(remapped.keys())
        # Handle torch.compile "_orig_mod." prefix mismatch.
        if all(k.startswith("_orig_mod.") for k in sd_keys) and not any(k.startswith("_orig_mod.") for k in model_keys):
            remapped = _strip_prefix(remapped, "_orig_mod.")
        elif not any(k.startswith("_orig_mod.") for k in sd_keys) and any(k.startswith("_orig_mod.") for k in model_keys):
            remapped = _add_prefix(remapped, "_orig_mod.")
        return remapped

    orig_keys = list(state.keys())
    state = _maybe_remap_prefix(state)
    if orig_keys and list(state.keys()) != orig_keys:
        print("[Pretrain] Remapped checkpoint keys to match current model prefixes.")
    resized = {}
    for key, value in list(state.items()):
        if key not in model_state:
            continue
        target = model_state[key]
        if value.shape == target.shape:
            continue
        if value.ndim == 3 and target.ndim == 3 and value.shape[0] == target.shape[0] and value.shape[2] == target.shape[2]:
            src_len = value.shape[1]
            tgt_len = target.shape[1]
            # Resize temporal positional embeddings to match finetune seq_len.
            value_resized = F.interpolate(
                value.permute(0, 2, 1),
                size=tgt_len,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)
            state[key] = value_resized
            resized[key] = (src_len, tgt_len)
    if resized:
        for name, (src_len, tgt_len) in resized.items():
            print(f"Resized {name} from len={src_len} to len={tgt_len} for finetune.")
    # ---- Transfer visual_residual_predictor weights to cross_attn + mlp ----
    # Predictor structure: [0] Linear(D_v→D_z), [1] GELU, [2] Linear(D_z→D_z), [3] GELU, [4] Linear(D_z→D_z)
    # Maps to:  cross_attn.W_v ← [0],  mlp[0] ← [2],  mlp[2] ← [4]
    def _get_key(name):
        """Try with and without 'module.' prefix."""
        for prefix in ["", "module.", "_orig_mod.", "module._orig_mod."]:
            k = f"{prefix}{name}"
            if k in state:
                return k
        return None

    vrp_prefix = "visual_residual_predictor"
    vrp_map = {
        f"{vrp_prefix}.0.weight": [("cross_attn_global.W_v.weight", False), ("cross_attn_local.W_v.weight", False)],
        f"{vrp_prefix}.0.bias":   [("cross_attn_global.W_v.bias", True), ("cross_attn_local.W_v.bias", True)],
        f"{vrp_prefix}.2.weight": [("mlp_global.0.weight", False), ("mlp_local.0.weight", False)],
        f"{vrp_prefix}.2.bias":   [("mlp_global.0.bias", False), ("mlp_local.0.bias", False)],
        f"{vrp_prefix}.4.weight": [("mlp_global.2.weight", False), ("mlp_local.2.weight", False)],
        f"{vrp_prefix}.4.bias":   [("mlp_global.2.bias", False), ("mlp_local.2.bias", False)],
    }
    transferred = []
    for src_name, dst_list in vrp_map.items():
        src_key = _get_key(src_name)
        if src_key is None:
            continue
        src_tensor = state[src_key]
        for dst_name, is_optional_bias in dst_list:
            dst_key = _get_key(dst_name)
            if dst_key is None:
                # dst not in checkpoint yet (expected — it's a finetune-only module)
                # Find the right prefixed key in model_state
                for prefix in ["", "module.", "_orig_mod.", "module._orig_mod."]:
                    candidate = f"{prefix}{dst_name}"
                    if candidate in model_state:
                        if is_optional_bias and candidate not in state:
                            # W_v has no bias in CodebookCrossAttention (bias=False)
                            break
                        if model_state[candidate].shape == src_tensor.shape:
                            state[candidate] = src_tensor.clone()
                            transferred.append(f"{src_name} → {dst_name}")
                        break
            else:
                if model_state.get(dst_key) is not None and model_state[dst_key].shape == src_tensor.shape:
                    state[dst_key] = src_tensor.clone()
                    transferred.append(f"{src_name} → {dst_name}")
    if transferred:
        print(f"[Pretrain→Finetune] Transferred visual_residual_predictor weights:")
        for t in transferred:
            print(f"  {t}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded pretrain ckpt with missing keys: {missing}, unexpected keys: {unexpected}",flush=True)


def _get_sample_key(batch, b):
    """Generate a unique key for a sample in a batch (for uncertainty mask lookup)."""
    meta = batch.get("metadata")
    if meta is None:
        return str(b)

    def _get_field(meta, field, b, default=""):
        if isinstance(meta, list):
            m = meta[b] if b < len(meta) else {}
            return m.get(field, default)
        elif isinstance(meta, dict):
            val = meta.get(field)
            if isinstance(val, (list, tuple)):
                return val[b] if b < len(val) else default
            elif isinstance(val, torch.Tensor):
                return int(val[b])
            return val if val is not None else default
        return default

    vid = _get_field(meta, "video_id", b, "")
    wid = _get_field(meta, "window_idx", b, b)
    mid = _get_field(meta, "mouse_id", b, "")
    return f"{vid}_{wid}_{mid}"


def train_one_epoch_finetune(
    model: ResiFuseClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config,
    epoch_idx: int | None = None,
    pos_weight: torch.Tensor | None = None,
    lambda_col: float = 0.1,
    collision_temperature: float = 0.1,
    global_masks: dict | None = None,
):
    model.train()
    loss_meter = 0.0
    cls_meter = 0.0
    col_meter = 0.0
    label_valid_mode = _get_label_valid_mode(config)
    base_model = _get_base_model(model)
    grad_acc_steps = int(getattr(config, "finetune_grad_acc_steps", None) or getattr(config, "grad_acc_steps", 1) or 1)
    grad_acc_steps = max(1, grad_acc_steps)
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        kps = batch["keypoints"].to(device)
        motion_feats = batch["motion_feats"].to(device)
        frame_labels = batch["frame_labels"].to(device)
        frame_mask = batch.get("frame_mask")
        if frame_mask is not None:
            frame_mask = frame_mask.to(device)

        frame_mask = _select_frame_mask(frame_mask, frame_labels, label_valid_mode)
        frame_labels, frame_mask = _majority_aggregate_frame_labels(frame_labels, frame_mask)
        frame_labels, frame_mask = _sanitize_frame_labels(frame_labels, frame_mask)

        B, A, T, _, _ = kps.shape

        if frame_mask.dim() == 1:
            frame_mask = frame_mask.unsqueeze(0)
        if frame_mask.dim() == 3:
            frame_mask_base = frame_mask.any(dim=1)
            pad_mask = ~frame_mask
        else:
            frame_mask_base = frame_mask
            pad_mask = ~frame_mask.unsqueeze(1).expand(B, A, T)

        # Build per-sample uncertainty mask if available
        frame_select_mask = None
        if global_masks is not None:
            masks_batch = []
            n_miss = 0
            for b in range(B):
                key = _get_sample_key(batch, b)
                if key in global_masks:
                    masks_batch.append(global_masks[key].to(device))
                else:
                    masks_batch.append(None)
                    n_miss += 1
            if all(m is not None for m in masks_batch):
                frame_select_mask = torch.stack(masks_batch, dim=0)  # [B, L]
            if batch_idx == 0 and is_main_process(getattr(config, "rank", 0)):
                n_hit = B - n_miss
                print(f"[Uncertainty] batch 0: {n_hit}/{B} samples matched, "
                      f"mask={'active' if frame_select_mask is not None else 'FALLBACK'}",
                      flush=True)

        vision_patch_tokens = batch.get("vision_patch_tokens")
        if vision_patch_tokens is not None:
            vision_patch_tokens = vision_patch_tokens.to(device)
        vision_feats_raw = batch.get("vision_feats")
        if vision_feats_raw is not None:
            vision_feats_raw = vision_feats_raw.to(device)
        out = model(
            motion_feats,
            kps,
            pad_mask=pad_mask,
            vision_patch_tokens=vision_patch_tokens,
            vision_feats=vision_feats_raw,
            frame_select_mask=frame_select_mask,
        )
        logits = out["logits"]  # [B, T, C]

        logits_len = logits.shape[1] if logits.dim() == 3 else 1
        frame_labels_ds = _downsample_frame_labels(frame_labels, logits_len, 1)
        frame_mask_ds = _downsample_frame_mask(frame_mask_base, logits_len, 1)
        cls_loss, _ = compute_multilabel_frame_loss(
            logits,
            frame_labels_ds,
            frame_mask_ds,
            pos_weight=pos_weight,
        )

        col_loss = cls_loss.new_tensor(0.0)
        if lambda_col > 0.0:
            code_indices = out.get("active_indices_flat")  # [N_active]
            active_v = out.get("active_v_flat")            # [N_active, D]
            active_mask = out.get("mask_global", None)
            if active_mask is not None:
                active_mask = active_mask | out.get("mask_local", active_mask)
            if code_indices is not None and active_v is not None and active_mask is not None:
                # Build per-frame labels aligned with active_mask [B*A, L]
                labels_col = frame_labels_ds
                if labels_col.dim() == 3:
                    labels_col = labels_col.argmax(dim=-1)  # [B, L]
                # Expand to [B*A, L] to match active_mask
                L = active_mask.shape[1]
                labels_expanded = labels_col[:, :L].unsqueeze(1).expand(B, A, L).reshape(B * A, L)
                active_labels = labels_expanded[active_mask]  # [N_active]
                col_loss = collision_contrastive_loss(
                    code_indices,
                    active_v,
                    active_labels,
                    temperature=collision_temperature,
                )

        loss_full = cls_loss + lambda_col * col_loss
        loss_scaled = loss_full / grad_acc_steps
        loss_scaled.backward()
        _report_unused_params(model, "finetune", batch_idx)
        if (batch_idx + 1) % grad_acc_steps == 0 or (batch_idx + 1) == len(dataloader):
            grad_clip = float(getattr(config, "grad_clip_norm", 0.0) or 0.0)
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=grad_clip,
                )
            optimizer.step()
            if bool(getattr(config, "finetune_update_vq_ema", False)):
                base_model.update_vq_ema()
            optimizer.zero_grad()

        loss_meter += loss_full.item()
        cls_meter += cls_loss.item()
        col_meter += col_loss.item()

    denom = max(len(dataloader), 1)
    return {
        "loss": loss_meter / denom,
        "cls": cls_meter / denom,
        "col": col_meter / denom,
    }


def evaluate_finetune(
    model: ResiFuseClassifier,
    dataloader: DataLoader,
    device: torch.device,
    config,
    epoch_idx: int | None = None,
    pos_weight: torch.Tensor | None = None,
    lambda_col: float = 0.1,
    collision_temperature: float = 0.1,
    global_masks: dict | None = None,
):
    is_main = is_main_process(getattr(config, "rank", 0))
    model.eval()
    loss_meter = 0.0
    cls_meter = 0.0
    col_meter = 0.0
    label_valid_mode = _get_label_valid_mode(config)
    base_model = _get_base_model(model)
    f1_counts = None
    seen_by_video = {}
    report_other = bool(getattr(config, "report_other_class", False))
    f1_counts_other = None
    seen_by_video_other = {}
    debug_logged = False
    # Optional inference-latency measurement (forward-only, this eval loop IS
    # inference). Enable with BENCH_INFER=1; zero overhead otherwise.
    _bench_infer = bool(os.environ.get("BENCH_INFER"))
    _infer_ms_total = 0.0
    _infer_windows = 0
    with torch.no_grad():
        for batch in dataloader:
            kps = batch["keypoints"].to(device)
            motion_feats = batch["motion_feats"].to(device)
            frame_labels = batch["frame_labels"].to(device)
            frame_mask = batch.get("frame_mask")
            if frame_mask is not None:
                frame_mask = frame_mask.to(device)
            frame_mask = _select_frame_mask(frame_mask, frame_labels, label_valid_mode)
            frame_labels, frame_mask = _majority_aggregate_frame_labels(frame_labels, frame_mask)
            frame_labels, frame_mask = _sanitize_frame_labels(frame_labels, frame_mask)

            B, A, T, _, _ = kps.shape
            if frame_mask.dim() == 1:
                frame_mask = frame_mask.unsqueeze(0)
            if frame_mask.dim() == 3:
                frame_mask_base = frame_mask.any(dim=1)
                pad_mask = ~frame_mask
            else:
                frame_mask_base = frame_mask
                pad_mask = ~frame_mask.unsqueeze(1).expand(B, A, T)

            # Build per-sample uncertainty mask if available
            frame_select_mask = None
            if global_masks is not None:
                masks_batch = []
                for b in range(B):
                    key = _get_sample_key(batch, b)
                    if key in global_masks:
                        masks_batch.append(global_masks[key].to(device))
                    else:
                        masks_batch.append(None)
                if all(m is not None for m in masks_batch):
                    frame_select_mask = torch.stack(masks_batch, dim=0)

            vision_patch_tokens = batch.get("vision_patch_tokens")
            if vision_patch_tokens is not None:
                vision_patch_tokens = vision_patch_tokens.to(device)
            vision_feats_raw = batch.get("vision_feats")
            if vision_feats_raw is not None:
                vision_feats_raw = vision_feats_raw.to(device)
            if _bench_infer and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            _fwd_t0 = time.perf_counter() if _bench_infer else 0.0
            out = model(
                motion_feats,
                kps,
                pad_mask=pad_mask,
                vision_patch_tokens=vision_patch_tokens,
                vision_feats=vision_feats_raw,
                frame_select_mask=frame_select_mask,
            )
            if _bench_infer:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                _infer_ms_total += (time.perf_counter() - _fwd_t0) * 1e3
                _infer_windows += int(motion_feats.shape[0])
            logits = out["logits"]  # [B, T, C]

            logits_len = logits.shape[1] if logits.dim() == 3 else 1
            frame_labels_ds = _downsample_frame_labels(frame_labels, logits_len, 1)
            frame_mask_ds = _downsample_frame_mask(frame_mask_base, logits_len, 1)
            if is_main and not debug_logged:
                valid_mask = frame_mask_ds if frame_mask_ds is not None else torch.ones(
                    (frame_labels_ds.shape[0], frame_labels_ds.shape[1]),
                    device=frame_labels_ds.device,
                    dtype=torch.bool,
                )
                labels_pos = ((frame_labels_ds == 1) & valid_mask.unsqueeze(-1)).sum(dim=(0, 1))
                preds = torch.sigmoid(logits) > 0.5
                preds_pos = (preds & valid_mask.unsqueeze(-1)).sum(dim=(0, 1))
                logits_mask = valid_mask.unsqueeze(-1).expand_as(logits)
                logits_valid = logits[logits_mask]
                if logits_valid.numel() > 0:
                    logits_flat = logits[logits_mask].view(-1)
                    logits_min = float(logits_flat.min().item())
                    logits_max = float(logits_flat.max().item())
                else:
                    logits_min = float("nan")
                    logits_max = float("nan")
                logits_mean = (logits * logits_mask).sum(dim=(0, 1)) / logits_mask.sum(dim=(0, 1)).clamp(min=1)
                logits_var = ((logits - logits_mean) ** 2 * logits_mask).sum(dim=(0, 1)) / logits_mask.sum(dim=(0, 1)).clamp(min=1)
                logits_std = logits_var.sqrt()
                print(
                    f"[Finetune][Debug][F1] label_pos_per_class={labels_pos.tolist()} "
                    f"pred_pos_per_class={preds_pos.tolist()}",
                    flush=True,
                )
                print(
                    "[Finetune][Debug][Logits] "
                    f"mean={logits_mean.tolist()} std={logits_std.tolist()} "
                    f"min={logits_min:.4f} max={logits_max:.4f}",
                    flush=True,
                )
                debug_logged = True
            cls_loss, _ = compute_multilabel_frame_loss(
                logits,
                frame_labels_ds,
                frame_mask_ds,
                pos_weight=pos_weight,
            )

            col_loss = cls_loss.new_tensor(0.0)
            if lambda_col > 0.0:
                code_indices = out.get("active_indices_flat")
                active_v = out.get("active_v_flat")
                active_mask = out.get("mask_global", None)
                if active_mask is not None:
                    active_mask = active_mask | out.get("mask_local", active_mask)
                if code_indices is not None and active_v is not None and active_mask is not None:
                    labels_col = frame_labels_ds
                    if labels_col.dim() == 3:
                        labels_col = labels_col.argmax(dim=-1)
                    L = active_mask.shape[1]
                    labels_expanded = labels_col[:, :L].unsqueeze(1).expand(B, A, L).reshape(B * A, L)
                    active_labels = labels_expanded[active_mask]
                    col_loss = collision_contrastive_loss(
                        code_indices,
                        active_v,
                        active_labels,
                        temperature=collision_temperature,
                    )

            loss = cls_loss + lambda_col * col_loss

            loss_meter += loss.item()
            cls_meter += cls_loss.item()
            col_meter += col_loss.item()

            if f1_counts is None:
                num_classes = logits.shape[-1] if logits.dim() >= 2 else 1
                f1_counts = _init_f1_counts(num_classes, device)
            _accumulate_f1_counts_frame_index(
                f1_counts,
                logits,
                frame_labels_ds,
                frame_mask_ds,
                batch.get("metadata"),
                seen_by_video,
            )
            if report_other and logits.dim() == 3:
                if f1_counts_other is None:
                    f1_counts_other = _init_f1_counts(logits.shape[-1] + 1, device)
                _accumulate_f1_counts_frame_index_with_other(
                    f1_counts_other,
                    logits,
                    frame_labels_ds,
                    frame_mask_ds,
                    batch.get("metadata"),
                    seen_by_video_other,
                )

    if _bench_infer and is_main and _infer_windows > 0:
        # window = 200 raw frames / 30 fps = 6.667 s of video (ds_stride=4 -> 50
        # processed frames). Model-only forward (runs on cached vision feats;
        # excludes VideoPrism encode + video decode).
        ms_per_window = _infer_ms_total / _infer_windows
        video_sec_per_window = 200.0 / 30.0
        rtf = (ms_per_window / 1e3) / video_sec_per_window
        print(f"[InferTiming] model_forward_only: {_infer_windows} windows, "
              f"{ms_per_window:.2f} ms/window, budget={getattr(config, 'budget_ratio', '?')}, "
              f"video_sec/window={video_sec_per_window:.2f}, model_only_RTF={rtf:.4f}", flush=True)

    denom = max(len(dataloader), 1)
    return {
        "loss": loss_meter / denom,
        "cls": cls_meter / denom,
        "col": col_meter / denom,
        "f1_counts": f1_counts,
        "f1_counts_4": f1_counts_other,
    }


@torch.no_grad()
def precompute_H_t(
    model: ResiFuseClassifier,
    dataloader: DataLoader,
    device: torch.device,
    config,
) -> dict:
    """After warm-up, iterate over the dataloader, get logits, compute entropy H_t per sample."""
    model.eval()
    label_valid_mode = _get_label_valid_mode(config)
    H_t_cache = {}
    for batch in dataloader:
        kps = batch["keypoints"].to(device)
        motion_feats = batch["motion_feats"].to(device)
        frame_labels = batch["frame_labels"].to(device)
        frame_mask = batch.get("frame_mask")
        if frame_mask is not None:
            frame_mask = frame_mask.to(device)
        frame_mask = _select_frame_mask(frame_mask, frame_labels, label_valid_mode)
        frame_labels, frame_mask = _majority_aggregate_frame_labels(frame_labels, frame_mask)

        B, A, T, _, _ = kps.shape
        if frame_mask.dim() == 1:
            frame_mask = frame_mask.unsqueeze(0)
        if frame_mask.dim() == 3:
            pad_mask = ~frame_mask
        else:
            pad_mask = ~frame_mask.unsqueeze(1).expand(B, A, T)

        out = model(
            motion_feats,
            kps,
            pad_mask=pad_mask,
            warmup_only=True,
        )
        logits = out["logits"]  # [B, T, C]
        # Compute entropy: H_t = -sum(p * log(p)) where p = softmax(logits)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        H_t = -(probs * log_probs).sum(dim=-1)  # [B, T]

        meta_list = _normalize_metadata_list(batch.get("metadata"), batch_size=B)
        if meta_list is not None:
            for b, meta in enumerate(meta_list):
                key = (meta.get("video_id"), meta.get("start_frame"))
                H_t_cache[key] = H_t[b].cpu()

    return H_t_cache


def run_pretrain(args):
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    include_velocity = getattr(args, "motion_include_velocity", True)
    include_confidence = getattr(args, "motion_include_confidence", True)

    _vp_w = float(getattr(args, "vision_pred_weight", 0.0) or 0.0)
    _vr_w = float(getattr(args, "vision_vres_weight", 0.0) or 0.0)
    if _vp_w <= 0.0 and _vr_w <= 0.0:
        args.use_vision_cache = False

    device = torch.device(args.device)
    is_main = is_main_process(args.rank)
    if is_main:
        print(
            f"========== [MABe][Pretrain] vision_pred_weight={getattr(args, 'vision_pred_weight', None)} ==========",
            flush=True,
        )
        if _vp_w <= 0.0 and _vr_w <= 0.0:
            print("[MABe][Pretrain] No vision losses enabled -> use_vision_cache=False (skip vision IO).", flush=True)
        print(f"[MABe][Pretrain] device={device} | mask_ratio=roulette")
    args.save_dir = _build_run_dir(args, stage="pretrain")
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tb")) if is_main else None

    mabe_root = getattr(args, "mabe_data_dir", None) or getattr(args, "data_root", None)
    if not mabe_root:
        raise RuntimeError("mabe_data_dir must be set for MABe pretrain.")
    stage_dir = os.path.join(mabe_root, "pretrain")
    cache_dir = getattr(args, "mabe_data_cache_dir", None)
    if cache_dir is None:
        raise RuntimeError("mabe_data_cache_dir must be set for MABe pretrain.")
    split_json = os.path.join(stage_dir, "pretrain_split.json")
    if not os.path.exists(split_json):
        raise RuntimeError(f"Split JSON not found: {split_json}")

    train_ds = MaBeMouseDataset(
        data_dir=stage_dir,
        split="train",
        stage="pretrain",
        train_ratio=train_ratio,
        ds_stride=getattr(args, "ds_stride", 1),
        seq_len=None,
        multi_mouse=True,
        include_neighbor_feats=args.use_neighbor_feats,
        max_neighbors=args.max_neighbors,
        motion_include_velocity=include_velocity,
        motion_include_confidence=include_confidence,
        frame_resize=getattr(args, "frame_resize", None),
        frame_resize_align=getattr(args, "frame_resize_align", 32),
        dtype=args.torch_dtype,
        cache_dir=cache_dir,
        use_cache=bool(getattr(args, "use_cache", True)),
        use_vision_cache=bool(getattr(args, "use_vision_cache", True)),
        split_json=split_json
    )
    args.seq_len = train_ds.seq_len
    val_ds = MaBeMouseDataset(
        data_dir=stage_dir,
        split="val",
        stage="pretrain",
        train_ratio=train_ratio,
        ds_stride=getattr(args, "ds_stride", 1),
        seq_len=args.seq_len,
        multi_mouse=True,
        include_neighbor_feats=args.use_neighbor_feats,
        max_neighbors=args.max_neighbors,
        motion_include_velocity=include_velocity,
        motion_include_confidence=include_confidence,
        frame_resize=getattr(args, "frame_resize", None),
        frame_resize_align=getattr(args, "frame_resize_align", 32),
        dtype=args.torch_dtype,
        cache_dir=cache_dir,
        use_cache=bool(getattr(args, "use_cache", True)),
        use_vision_cache=bool(getattr(args, "use_vision_cache", True)),
        split_json=split_json
    )
    test_ds = MaBeMouseDataset(
        data_dir=stage_dir,
        split="test",
        stage="pretrain",
        train_ratio=train_ratio,
        ds_stride=getattr(args, "ds_stride", 1),
        seq_len=args.seq_len,
        multi_mouse=True,
        include_neighbor_feats=args.use_neighbor_feats,
        max_neighbors=args.max_neighbors,
        motion_include_velocity=include_velocity,
        motion_include_confidence=include_confidence,
        frame_resize=getattr(args, "frame_resize", None),
        frame_resize_align=getattr(args, "frame_resize_align", 32),
        dtype=args.torch_dtype,
        cache_dir=cache_dir,
        use_cache=bool(getattr(args, "use_cache", True)),
        use_vision_cache=bool(getattr(args, "use_vision_cache", True)),
        split_json=split_json
    )

    args.num_joints = train_ds.num_joints

    neighbor_dim = (args.max_neighbors * 4) if args.use_neighbor_feats else 0
    vel_dim = args.num_joints * 2 if include_velocity else 0
    base_dim = args.num_joints * (3 if include_confidence else 2)
    args.motion_input_dim = base_dim + vel_dim + neighbor_dim

    eval_distributed = bool(getattr(args, "eval_distributed", True))
    train_sampler = DistributedSampler(train_ds, num_replicas=args.world_size, rank=args.rank, shuffle=True) if args.distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=args.world_size, rank=args.rank, shuffle=False) if args.distributed and eval_distributed else None
    test_sampler = DistributedSampler(test_ds, num_replicas=args.world_size, rank=args.rank, shuffle=False) if args.distributed and eval_distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = ResiFusePretrainer(args).to(device).to(dtype=args.torch_dtype)

    if getattr(args, "torch_compile", True):
        print("enable torch compile!")
        torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(model, mode="default", backend="inductor")

    unused_flag = os.environ.get("RESIFUSE_DEBUG_UNUSED", "0") == "1"

    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank] if device.type == "cuda" else None,
            output_device=args.local_rank if device.type == "cuda" else None,
            find_unused_parameters=unused_flag,
        )

    base_model = _get_base_model(model)
                
    weight_decay = getattr(args, "weight_decay", 1e-2)
    vq_decay, vq_nodecay, other_decay, other_nodecay = [], [], [], []
    for name, p in base_model.named_parameters():
        if not p.requires_grad:
            continue
        is_vq = "vqvae" in name
        is_no_decay = ("bias" in name.lower()) or ("norm" in name.lower())
        if is_vq:
            (vq_nodecay if is_no_decay else vq_decay).append(p)
        else:
            (other_nodecay if is_no_decay else other_decay).append(p)

    param_groups = []
    vq_lr = args.lr * getattr(args, "vq_lr_scale", 0.5)
    if vq_decay:
        param_groups.append({"params": vq_decay, "lr": vq_lr, "weight_decay": weight_decay})
    if vq_nodecay:
        param_groups.append({"params": vq_nodecay, "lr": vq_lr, "weight_decay": 0.0})
    if other_decay:
        param_groups.append({"params": other_decay, "lr": args.lr, "weight_decay": weight_decay})
    if other_nodecay:
        param_groups.append({"params": other_nodecay, "lr": args.lr, "weight_decay": 0.0})
    
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = _build_lr_scheduler(optimizer, args, stage="pretrain")

    start_epoch, best_val, best_epoch = _maybe_resume(
        model,
        optimizer,
        scheduler,
        args.save_dir,
        getattr(args, "pretrain_resume_from", None),
        device,
        "pretrain",
        is_main,
    )

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main:
            print(f"\n=== [MABe][Pretrain] Epoch {epoch + 1}/{args.epochs} ===")
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        _pt_t0 = time.perf_counter()
        train_stats = train_one_epoch_pretrain(model, train_loader, optimizer, device, args, epoch_idx=epoch)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        if is_main:
            print(f"[Timing] pretrain_epoch={epoch + 1} train_step_sec={time.perf_counter() - _pt_t0:.2f}",
                  flush=True)
        val_stats = evaluate_pretrain(model, val_loader, device, args, epoch_idx=epoch)
        val_loss = val_stats["loss"]

        if scheduler is not None:
            scheduler.step()

        if args.distributed:
            horizons = getattr(args, "vision_pred_horizons", None) or []
            horizons = [int(h) for h in horizons]
            train_tensor = torch.tensor(
                [
                    train_stats["loss"],
                    train_stats["mask"],
                    train_stats["pred"],
                    train_stats["vq_recon"],
                    train_stats["vq"],
                    train_stats.get("vres", 0.0),
                    *[train_stats.get("pred_h", {}).get(h, 0.0) for h in horizons],
                ],
                device=device,
            )
            dist.all_reduce(train_tensor, op=dist.ReduceOp.SUM)
            train_tensor = train_tensor / args.world_size
            pred_h = {h: train_tensor[6 + idx].item() for idx, h in enumerate(horizons)}
            train_stats = {
                "loss": train_tensor[0].item(),
                "mask": train_tensor[1].item(),
                "pred": train_tensor[2].item(),
                "vq_recon": train_tensor[3].item(),
                "vq": train_tensor[4].item(),
                "vres": train_tensor[5].item(),
                "pred_h": pred_h,
                "avg_fetch": train_stats.get("avg_fetch", 0.0),
                "avg_compute": train_stats.get("avg_compute", 0.0),
                "total_fetch": train_stats.get("total_fetch", 0.0),
                "total_compute": train_stats.get("total_compute", 0.0),
                "num_batches": train_stats.get("num_batches", 0),
            }
            val_tensor = torch.tensor(
                [
                    val_stats["loss"],
                    val_stats["mask"],
                    val_stats["pred"],
                    val_stats["vq_recon"],
                    val_stats["vq"],
                    *[val_stats.get("pred_h", {}).get(h, 0.0) for h in horizons],
                ],
                device=device,
            )
            dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
            val_pred_h = {h: (val_tensor[5 + idx].item() / args.world_size) for idx, h in enumerate(horizons)}
            val_stats = {
                "loss": (val_tensor[0].item() / args.world_size),
                "mask": (val_tensor[1].item() / args.world_size),
                "pred": (val_tensor[2].item() / args.world_size),
                "vq_recon": (val_tensor[3].item() / args.world_size),
                "vq": (val_tensor[4].item() / args.world_size),
                "pred_h": val_pred_h,
            }
            val_loss = val_stats["loss"]

        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_path = _save_checkpoint(
                "pretrain",
                model,
                optimizer,
                scheduler,
                best_val,
                best_epoch,
                epoch,
                args.save_dir,
                "best_pretrain",
            )
            _cleanup_old_best(args.save_dir, "best_pretrain", keep_path=best_path)

        if is_main:
            # Codebook usage snapshot from EMA buffers (already DDP-synced).
            # Reads buffers maintained by training; no extra forward passes.
            base_model_for_stats = _get_base_model(model)
            cb_stats = base_model_for_stats.motion_scout.usage_stats()
            cb_q1, cb_q2 = cb_stats["q1"], cb_stats["q2"]
            # eK = effective number of clusters (perplexity exp(H)). More
            # informative than `used` when threshold is small: it tells you
            # how many entries are doing real probability-mass work, not
            # just "non-dead" status.
            cb_summary = (
                f" | cb1[used={cb_q1['used']}/{cb_q1['K']} eK={cb_q1['effective_K']:.0f} "
                f"H={cb_q1['norm_entropy']:.2f} top10={cb_q1['top10_frac']*100:.0f}% "
                f"|c|={cb_q1['mean_norm']:.2f}] "
                f"cb2[used={cb_q2['used']}/{cb_q2['K']} eK={cb_q2['effective_K']:.0f} "
                f"H={cb_q2['norm_entropy']:.2f} top10={cb_q2['top10_frac']*100:.0f}% "
                f"|c|={cb_q2['mean_norm']:.2f}]"
            )
            print(
                f"[Pretrain] Epoch {epoch + 1} | "
                f"train_loss={train_stats['loss']:.6f} "
                f"(mask={train_stats['mask']:.6f}, pred={train_stats['pred']:.6f}, "
                f"vres={train_stats.get('vres', 0.0):.6f}, "
                f"vq_recon={train_stats['vq_recon']:.6f}, vq={train_stats['vq']:.6f}) | "
                f"val_loss={val_stats['loss']:.6f} "
                f"(mask={val_stats['mask']:.6f}, pred={val_stats['pred']:.6f}, "
                f"vq_recon={val_stats['vq_recon']:.6f}, vq={val_stats['vq']:.6f})"
                f"{cb_summary}"
            )
            if writer is not None:
                writer.add_scalar("pretrain/train_loss", train_stats["loss"], epoch + 1)
                writer.add_scalar("pretrain/val_loss", val_stats["loss"], epoch + 1)
                writer.add_scalar("pretrain/train_mask", train_stats["mask"], epoch + 1)
                writer.add_scalar("pretrain/val_mask", val_stats["mask"], epoch + 1)
                writer.add_scalar("pretrain/train_pred", train_stats["pred"], epoch + 1)
                writer.add_scalar("pretrain/val_pred", val_stats["pred"], epoch + 1)
                writer.add_scalar("pretrain/train_vres", train_stats.get("vres", 0.0), epoch + 1)
                writer.add_scalar("pretrain/train_vq_recon", train_stats["vq_recon"], epoch + 1)
                writer.add_scalar("pretrain/val_vq_recon", val_stats["vq_recon"], epoch + 1)
                writer.add_scalar("pretrain/train_vq", train_stats["vq"], epoch + 1)
                writer.add_scalar("pretrain/val_vq", val_stats["vq"], epoch + 1)
                for level, st in (("q1", cb_q1), ("q2", cb_q2)):
                    writer.add_scalar(f"pretrain/cb_{level}_used_frac", st["used"] / st["K"], epoch + 1)
                    writer.add_scalar(f"pretrain/cb_{level}_effective_K", st["effective_K"], epoch + 1)
                    writer.add_scalar(f"pretrain/cb_{level}_dead", st["dead"], epoch + 1)
                    writer.add_scalar(f"pretrain/cb_{level}_norm_entropy", st["norm_entropy"], epoch + 1)
                    writer.add_scalar(f"pretrain/cb_{level}_top10_frac", st["top10_frac"], epoch + 1)
                    writer.add_scalar(f"pretrain/cb_{level}_top50_frac", st["top50_frac"], epoch + 1)
                    writer.add_scalar(f"pretrain/cb_{level}_mean_norm", st["mean_norm"], epoch + 1)
                    writer.add_scalar(f"pretrain/cb_{level}_avg_min_inter_dist", st["avg_min_inter_dist"], epoch + 1)
                    writer.add_scalar(f"pretrain/cb_{level}_min_inter_dist", st["min_inter_dist"], epoch + 1)
                for h, val in train_stats.get("pred_h", {}).items():
                    writer.add_scalar(f"pretrain/train_pred_h{h}", val, epoch + 1)
                for h, val in val_stats.get("pred_h", {}).items():
                    writer.add_scalar(f"pretrain/val_pred_h{h}", val, epoch + 1)
            social_stats = _maybe_collect_social_attn_stats(
                model,
                train_loader,
                device,
                args,
                stage="pretrain",
                epoch_idx=epoch,
                writer=writer,
            )
            if social_stats is not None:
                mean_self = float(social_stats["self"].mean().item())
                mean_neighbor = float(social_stats["neighbor"].mean().item())
                per_self = ", ".join(f"h{idx}={val:.4f}" for idx, val in enumerate(social_stats["self"].tolist()))
                per_neighbor = ", ".join(f"h{idx}={val:.4f}" for idx, val in enumerate(social_stats["neighbor"].tolist()))
                print(
                    f"[Pretrain][SocialAttn] mean_R_self={mean_self:.4f} "
                    f"mean_R_neighbor={mean_neighbor:.4f}"
                )
                print(f"[Pretrain][SocialAttn] R_self: {per_self}")
                print(f"[Pretrain][SocialAttn] R_neighbor: {per_neighbor}")

        if (epoch + 1) % getattr(args, "pretrain_save_interval", 1) == 0 or (epoch + 1) == args.epochs:
            _save_checkpoint(
                "pretrain",
                model,
                optimizer,
                scheduler,
                best_val,
                best_epoch,
                epoch,
                args.save_dir,
                "resifuse_pretrain",
            )

    if test_loader is not None:
        test_stats = evaluate_pretrain(model, test_loader, device, args, epoch_idx=max(args.epochs - 1, 0))
        if args.distributed:
            horizons = getattr(args, "vision_pred_horizons", None) or []
            horizons = [int(h) for h in horizons]
            test_tensor = torch.tensor(
                [
                    test_stats["loss"],
                    test_stats["mask"],
                    test_stats["pred"],
                    test_stats["vq_recon"],
                    test_stats["vq"],
                    *[test_stats.get("pred_h", {}).get(h, 0.0) for h in horizons],
                ],
                device=device,
            )
            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
            test_pred_h = {h: (test_tensor[5 + idx].item() / args.world_size) for idx, h in enumerate(horizons)}
            test_stats = {
                "loss": (test_tensor[0].item() / args.world_size),
                "mask": (test_tensor[1].item() / args.world_size),
                "pred": (test_tensor[2].item() / args.world_size),
                "vq_recon": (test_tensor[3].item() / args.world_size),
                "vq": (test_tensor[4].item() / args.world_size),
                "pred_h": test_pred_h,
            }
        if is_main:
            print(
                f"[Pretrain] Test | loss={test_stats['loss']:.6f} "
                f"(mask={test_stats['mask']:.6f}, pred={test_stats['pred']:.6f}, "
                f"vq_recon={test_stats['vq_recon']:.6f}, vq={test_stats['vq']:.6f})"
            )
            if writer is not None:
                writer.add_scalar("pretrain/test_loss", test_stats["loss"], args.epochs)
                writer.add_scalar("pretrain/test_mask", test_stats["mask"], args.epochs)
                writer.add_scalar("pretrain/test_pred", test_stats["pred"], args.epochs)
                writer.add_scalar("pretrain/test_vq_recon", test_stats["vq_recon"], args.epochs)
                writer.add_scalar("pretrain/test_vq", test_stats["vq"], args.epochs)


def run_finetune(args):
    train_ratio = args.finetune_train_ratio if args.finetune_train_ratio is not None else args.train_ratio
    val_ratio = args.finetune_val_ratio if args.finetune_val_ratio is not None else args.val_ratio
    test_ratio = args.finetune_test_ratio if args.finetune_test_ratio is not None else args.test_ratio
    include_velocity = getattr(args, "motion_include_velocity", True)
    include_confidence = getattr(args, "motion_include_confidence", True)
    huddle_keep_ratio = getattr(args, "finetune_huddle_keep_ratio", 1.0)

    device = torch.device(args.device)
    is_main = is_main_process(args.rank)
    if is_main:
        print(f"[MABe][Finetune] device={device}")
    args.save_dir = _build_run_dir(args, stage="finetune")
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tb")) if is_main else None

    mabe_root = getattr(args, "mabe_data_dir", None) or getattr(args, "data_root", None)
    if not mabe_root:
        raise RuntimeError("mabe_data_dir must be set for MABe finetune.")
    stage_dir = os.path.join(mabe_root, "finetune")
    split_override = getattr(args, "mabe_finetune_split_json", None)
    if split_override:
        split_json = os.path.expanduser(split_override)
    else:
        split_json = os.path.join(stage_dir, "finetune_split.json")
    if not os.path.exists(split_json):
        raise RuntimeError(f"Split JSON not found: {split_json}")
    cache_dir = getattr(args, "mabe_data_cache_dir", None)
    if cache_dir is None:
        raise RuntimeError("mabe_data_cache_dir must be set for MABe finetune.")
    train_ds = MaBeMouseDataset(
        data_dir=stage_dir,
        split="train",
        stage="finetune",
        train_ratio=train_ratio,
        split_json=split_json,
        ds_stride=getattr(args, "ds_stride", 1),
        seq_len=None,
        multi_mouse=True,
        include_neighbor_feats=args.use_neighbor_feats,
        max_neighbors=args.max_neighbors,
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
    args.seq_len = train_ds.seq_len
    val_ds = MaBeMouseDataset(
        data_dir=stage_dir,
        split="val",
        stage="finetune",
        train_ratio=train_ratio,
        split_json=split_json,
        ds_stride=getattr(args, "ds_stride", 1),
        seq_len=args.seq_len,
        multi_mouse=True,
        include_neighbor_feats=args.use_neighbor_feats,
        max_neighbors=args.max_neighbors,
        motion_include_velocity=include_velocity,
        motion_include_confidence=include_confidence,
        frame_resize=getattr(args, "frame_resize", None),
        frame_resize_align=getattr(args, "frame_resize_align", 32),
        dtype=args.torch_dtype,
        cache_dir=cache_dir,
        use_cache=bool(getattr(args, "use_cache", False)),
        use_vision_cache=bool(getattr(args, "use_vision_cache", False))
    )
    test_ds = MaBeMouseDataset(
        data_dir=stage_dir,
        split="test",
        stage="finetune",
        train_ratio=train_ratio,
        split_json=split_json,
        ds_stride=getattr(args, "ds_stride", 1),
        seq_len=args.seq_len,
        multi_mouse=True,
        include_neighbor_feats=args.use_neighbor_feats,
        max_neighbors=args.max_neighbors,
        motion_include_velocity=include_velocity,
        motion_include_confidence=include_confidence,
        frame_resize=getattr(args, "frame_resize", None),
        frame_resize_align=getattr(args, "frame_resize_align", 32),
        dtype=args.torch_dtype,
        cache_dir=cache_dir,
        use_cache=bool(getattr(args, "use_cache", False)),
        use_vision_cache=bool(getattr(args, "use_vision_cache", False))
    )

    args.num_joints = train_ds.num_joints
    neighbor_dim = (args.max_neighbors * 4) if args.use_neighbor_feats else 0
    vel_dim = args.num_joints * 2 if include_velocity else 0
    base_dim = args.num_joints * (3 if include_confidence else 2)
    args.motion_input_dim = base_dim + vel_dim + neighbor_dim
    label_names = train_ds.label_names
    if label_names:
        args.head_a_classes = len(label_names)
        print(f"[Finetune] Override head_a_classes to {args.head_a_classes} from label_names.")

    eval_distributed = False
    train_sampler = DistributedSampler(train_ds, num_replicas=args.world_size, rank=args.rank, shuffle=True) if args.distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=args.world_size, rank=args.rank, shuffle=False) if args.distributed and eval_distributed else None
    test_sampler = DistributedSampler(test_ds, num_replicas=args.world_size, rank=args.rank, shuffle=False) if args.distributed and eval_distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_eval_sampler = DistributedSampler(train_ds, num_replicas=args.world_size, rank=args.rank, shuffle=False) if args.distributed and eval_distributed else None
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_eval_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    pos_weight_mode = str(getattr(args, "use_pos_weight", "true")).lower()
    if pos_weight_mode in ("false", "0", "none"):
        pos_weight = None
        if is_main:
            print("[Finetune] pos_weight disabled")
    else:
        pos_weight = _compute_pos_weight_from_loader(train_eval_loader, args, device=torch.device("cpu"))
        if pos_weight is not None:
            if pos_weight_mode == "sqrt":
                pos_weight = pos_weight.sqrt()
            pos_weight = pos_weight.to(device)
            if is_main:
                print(f"[Finetune] pos_weight({pos_weight_mode})={pos_weight.tolist()}")

    model = ResiFuseClassifier(args).to(device).to(dtype=args.torch_dtype)
    _configure_finetune_trainables(model, args, is_main=is_main)
    if getattr(args, "torch_compile", True):
        print("enable torch compile!")
        torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(model, mode="default", backend="inductor")
    
    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank] if device.type == "cuda" else None,
            output_device=args.local_rank if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    maybe_load_pretrain_weights(model, args.pretrain_ckpt, device)

    def _build_finetune_optimizer(mod, lr):
        # All trainable modules use finetune_lr (the pretrained RVQ encoder and
        # middle modules are frozen by _configure_finetune_trainables).
        base_model = mod.module if hasattr(mod, "module") else mod
        base_model = getattr(base_model, "_orig_mod", base_model)
        weight_decay = getattr(args, "weight_decay", 1e-2)
        decay, nodecay = [], []
        selector_params = []  # frame_select_mode == "learnable" only
        for name, p in base_model.named_parameters():
            if not p.requires_grad:
                continue
            # Frame selector: tiny MLP, separate group with its own lr.
            if name.startswith("frame_selector.") or ".frame_selector." in name:
                selector_params.append(p)
            elif ("bias" in name.lower()) or ("norm" in name.lower()):
                nodecay.append(p)
            else:
                decay.append(p)
        param_groups = []
        if decay:
            param_groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
        if nodecay:
            param_groups.append({"params": nodecay, "lr": lr, "weight_decay": 0.0})
        if selector_params:
            selector_lr = float(getattr(args, "learnable_selector_lr", lr))
            param_groups.append({"params": selector_params, "lr": selector_lr, "weight_decay": 0.0})
        return torch.optim.AdamW(param_groups)

    optimizer = _build_finetune_optimizer(model, args.lr)
    ft_scheduler = _build_lr_scheduler(optimizer, args, stage="finetune")

    start_epoch, best_val, best_epoch = _maybe_resume(
        model,
        optimizer,
        ft_scheduler,
        args.save_dir,
        getattr(args, "finetune_resume_from", None),
        device,
        "finetune",
        is_main,
    )

    # ---- v2 training config params ----
    _lc = getattr(args, "lambda_col", None)
    lambda_col = float(_lc) if _lc is not None else 0.1
    _ct = getattr(args, "collision_temperature", None)
    collision_temperature = float(_ct) if _ct is not None else 0.1
    total_epochs = args.epochs

    # Helper to run eval on all splits and log results for a given epoch / phase
    def _run_eval_and_log(epoch, phase_label, global_masks=None):
        eval_kwargs = dict(
            device=device,
            config=args,
            epoch_idx=epoch,
            pos_weight=pos_weight,
            lambda_col=lambda_col,
            collision_temperature=collision_temperature,
            global_masks=global_masks,
        )
        if args.distributed and not eval_distributed:
            if is_main:
                eval_model = model.module if hasattr(model, "module") else model
                train_eval_stats = evaluate_finetune(eval_model, train_eval_loader, **eval_kwargs)
                val_stats = evaluate_finetune(eval_model, val_loader, **eval_kwargs)
                test_stats = evaluate_finetune(eval_model, test_loader, **eval_kwargs)
            else:
                train_eval_stats = {"loss": 0.0, "cls": 0.0, "col": 0.0, "f1_counts": None}
                val_stats = {"loss": 0.0, "cls": 0.0, "col": 0.0, "f1_counts": None}
                test_stats = {"loss": 0.0, "cls": 0.0, "col": 0.0, "f1_counts": None}
            train_eval_stats = _broadcast_eval_stats(train_eval_stats, device, src=0)
            val_stats = _broadcast_eval_stats(val_stats, device, src=0)
            test_stats = _broadcast_eval_stats(test_stats, device, src=0)
        else:
            train_eval_stats = evaluate_finetune(model, train_eval_loader, **eval_kwargs)
            val_stats = evaluate_finetune(model, val_loader, **eval_kwargs)
            test_stats = evaluate_finetune(model, test_loader, **eval_kwargs)
        if args.distributed and eval_distributed:
            for stats in (train_eval_stats, val_stats, test_stats):
                st = torch.tensor([stats["loss"], stats["cls"], stats.get("col", 0.0)], device=device)
                dist.all_reduce(st, op=dist.ReduceOp.SUM)
                stats.update({"loss": st[0].item() / args.world_size, "cls": st[1].item() / args.world_size, "col": st[2].item() / args.world_size})
                counts = stats.get("f1_counts")
                if counts is not None:
                    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        return train_eval_stats, val_stats, test_stats

    def _log_epoch(epoch, phase_label, train_stats, train_eval_stats, val_stats, test_stats):
        nonlocal best_val, best_epoch
        updated_best = False
        val_loss = val_stats["loss"]
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            updated_best = True
        if is_main:
            print(
                f"[MABe][Finetune][{phase_label}] Epoch {epoch + 1} | "
                f"train_loss={train_stats['loss']:.5f} (cls={train_stats['cls']:.5f}, col={train_stats.get('col', 0.0):.5f}) "
                f"| val_loss={val_stats['loss']:.5f} (cls={val_stats['cls']:.5f}, col={val_stats.get('col', 0.0):.5f}) "
                f"| test_loss={test_stats['loss']:.5f} (cls={test_stats['cls']:.5f}, col={test_stats.get('col', 0.0):.5f})",
                flush=True,
            )
            writer.add_scalar("finetune/train_loss", train_stats["loss"], epoch + 1)
            writer.add_scalar("finetune/train_cls", train_stats["cls"], epoch + 1)
            writer.add_scalar("finetune/train_col", train_stats.get("col", 0.0), epoch + 1)
            writer.add_scalar("finetune/val_loss", val_stats["loss"], epoch + 1)
            writer.add_scalar("finetune/val_cls", val_stats["cls"], epoch + 1)
            writer.add_scalar("finetune/val_col", val_stats.get("col", 0.0), epoch + 1)
            writer.add_scalar("finetune/test_loss", test_stats["loss"], epoch + 1)
            writer.add_scalar("finetune/test_cls", test_stats["cls"], epoch + 1)
            writer.add_scalar("finetune/test_col", test_stats.get("col", 0.0), epoch + 1)
            writer.add_scalar("finetune/best_val_loss", best_val if best_val is not None else float("nan"), epoch + 1)
            for split_name, stats in (("TrainEval", train_eval_stats), ("Val", val_stats), ("Test", test_stats)):
                counts = stats.get("f1_counts")
                if counts is None:
                    continue
                f1_scores = _counts_to_f1(counts)
                for idx, f1 in enumerate(f1_scores):
                    label = label_names[idx] if idx < len(label_names) else f"class{idx}"
                    tp = int(counts[idx, 0].item())
                    fp = int(counts[idx, 1].item())
                    fn = int(counts[idx, 2].item())
                    total = int(counts[idx, 3].item())
                    tn = max(total - tp - fp - fn, 0)
                    print(
                        f"[Finetune][{split_name}][F1][frame_index] label={label} f1={f1:.4f} "
                        f"tp={tp} tn={tn} fp={fp} fn={fn}",
                        flush=True,
                    )
                    if writer:
                        writer.add_scalar(f"finetune/{split_name.lower()}_f1/{label}", f1, epoch + 1)
            if ((epoch + 1) % getattr(args, "finetune_save_interval", 1) == 0) or (epoch + 1 == total_epochs):
                _save_checkpoint("finetune", model, optimizer, ft_scheduler, best_val, best_epoch, epoch, args.save_dir, "resifuse_finetune")
            if updated_best:
                best_path = _save_checkpoint(
                    "finetune", model, optimizer, ft_scheduler, best_val, best_epoch, epoch,
                    args.save_dir, "best_finetune", add_epoch_suffix=True,
                )
                _cleanup_old_best(args.save_dir, "best_finetune", keep_path=best_path)
        if args.distributed:
            dist.barrier()

    # ========== Compute per-sample global frame selection masks ==========
    @torch.no_grad()
    def _compute_global_masks(base_model, dataloader, device, budget_ratio, mode="uncertainty"):
        """Compute global frame selection masks for all windows.

        Modes:
          - uncertainty: select frames with lowest motion |logit| (needs trained motion head)
          - residual: select frames with highest RVQ quantization error ||z_e - z_q||
          - residual_inverted: same score, sign flipped -> selects the LOWEST
            residual frames. Mechanism control: at budget b it selects exactly
            the frames that budget (1-b) residual selection discards.
          - recon_error: select frames with highest VQ-VAE reconstruction error
          - surprise: select frames with highest temporal z_q change
          - online: select frames with highest per-class binary-entropy (needs trained motion head)
          - confidence: select frames with LOWEST mean keypoint confidence (DLC
            pose-tracking uncertainty; independent of trained model, baseline
            ablation for residual selection)

        Returns: dict[sample_key -> bool tensor [L]] for each window.
        """
        saved_alpha = base_model.vision_alpha
        base_model.vision_alpha = 0.0
        base_model.eval()

        # Pass 1: collect per-frame scores for all windows
        all_scores = {}  # key -> [L] float tensor
        n_batches = 0
        for batch in dataloader:
            n_batches += 1
            kps = batch["keypoints"].to(device)
            motion_feats = batch["motion_feats"].to(device)
            B, A, T = motion_feats.shape[:3]
            motion_flat = motion_feats.view(B * A, T, -1)

            # Run motion scout for all modes
            scout_out = base_model.motion_scout(motion_flat)
            z_q = scout_out["z_q"]   # [B*A, L, D]
            z_e = scout_out["z_e"]   # [B*A, L, D]
            L = z_q.shape[1]

            if mode == "uncertainty":
                # Need full forward to get motion_logits
                frame_mask = batch.get("frame_mask")
                if frame_mask is None:
                    frame_mask = batch.get("frame_labels")
                    if frame_mask is not None:
                        frame_mask = (frame_mask.sum(dim=-1) >= 0).to(dtype=torch.bool)
                if frame_mask is not None:
                    frame_mask = frame_mask.to(device)
                    if frame_mask.dim() == 1:
                        frame_mask = frame_mask.unsqueeze(0)
                    if frame_mask.dim() == 3:
                        pad_mask = ~frame_mask
                    else:
                        pad_mask = ~frame_mask.unsqueeze(1).expand(B, A, T)
                else:
                    pad_mask = torch.zeros(B, A, T, dtype=torch.bool, device=device)
                out = base_model(motion_feats, kps, pad_mask=pad_mask)
                # Lower |logit| = more uncertain = higher score (we want to select these)
                scores = -out["logits"].abs().mean(dim=-1)  # [B, L], negate so topk selects lowest |logit|
            elif mode == "residual":
                r2 = (z_e - z_q).norm(dim=-1)  # [B*A, L]
                scores = r2.view(B, A, L).mean(dim=1)  # [B, L], average over animals
            elif mode == "residual_inverted":
                # Mechanism control for reviewer ZgJn: identical residual score,
                # sign flipped, so the same global-threshold machinery below
                # keeps the LOWEST-residual frames instead of the highest.
                # The per-slot bias correction is linear, so negating here is
                # equivalent to negating the corrected score: the selected set is
                # exactly the complement of the top-(1-budget) set at the same
                # score ordering (up to int() truncation and >= ties).
                r2 = (z_e - z_q).norm(dim=-1)  # [B*A, L]
                scores = -r2.view(B, A, L).mean(dim=1)  # [B, L]
            elif mode == "recon_error":
                recon = base_model.motion_scout.vqvae.decode(z_q, target_len=T)
                err = (recon - motion_flat).pow(2).mean(dim=-1)  # [B*A, T]
                if err.shape[1] != L:
                    stride_re = max(1, err.shape[1] // L)
                    err = err[:, ::stride_re][:, :L]
                scores = err.view(B, A, L).mean(dim=1)  # [B, L]
            elif mode == "surprise":
                diff = (z_q[:, 1:] - z_q[:, :-1]).norm(dim=-1)  # [B*A, L-1]
                surprise = torch.cat([diff[:, :1] * 0, diff], dim=1)  # [B*A, L]
                scores = surprise.view(B, A, L).mean(dim=1)  # [B, L]
            elif mode == "confidence":
                # Baseline ablation: select frames by DLC keypoint confidence.
                # Low mean confidence = uncertain pose tracking → vision
                # injection most useful. Independent of any trained model —
                # a pure dataset prior.
                conf = batch.get("confidence")
                if conf is None:
                    raise RuntimeError(
                        "frame_select_mode='confidence' requires batch['confidence']; "
                        "dataset did not provide per-frame per-keypoint confidences."
                    )
                conf = conf.to(device).float()  # expected [B, A, T, K]
                if conf.dim() == 4:
                    conf_pf = conf.mean(dim=-1)  # [B, A, T] average over keypoints
                elif conf.dim() == 3:
                    conf_pf = conf             # already [B, A, T]
                else:
                    raise RuntimeError(f"Unexpected confidence shape: {tuple(conf.shape)}")
                # Mean across animals (one score per window per frame).
                conf_avg = conf_pf.mean(dim=1)  # [B, T]
                # Downsample to the VQ latent length L if the motion encoder
                # strides down (mirrors recon_error / surprise handling).
                if conf_avg.shape[1] != L:
                    stride_re = max(1, conf_avg.shape[1] // L)
                    conf_avg = conf_avg[:, ::stride_re][:, :L]
                # Low confidence → high selection priority → negate so
                # topk() picks the lowest-confidence frames.
                # Keep any preprocessing-zeroed frames out of the selection by
                # treating zero confidence as "artificial" (they were zeroed in
                # preprocessing when the raw value was non-finite, not an
                # honest low-confidence signal); push their score very low so
                # topk skips them.
                scores_raw = -conf_avg
                scores_raw = torch.where(
                    conf_avg <= 0.0,
                    torch.full_like(scores_raw, -1e4),
                    scores_raw,
                )
                scores = scores_raw  # [B, L]
            elif mode == "online":
                # Same as uncertainty but using binary entropy instead of |logit|
                frame_mask = batch.get("frame_mask")
                if frame_mask is None:
                    frame_mask = batch.get("frame_labels")
                    if frame_mask is not None:
                        frame_mask = (frame_mask.sum(dim=-1) >= 0).to(dtype=torch.bool)
                if frame_mask is not None:
                    frame_mask = frame_mask.to(device)
                    if frame_mask.dim() == 1:
                        frame_mask = frame_mask.unsqueeze(0)
                    if frame_mask.dim() == 3:
                        pad_mask = ~frame_mask
                    else:
                        pad_mask = ~frame_mask.unsqueeze(1).expand(B, A, T)
                else:
                    pad_mask = torch.zeros(B, A, T, dtype=torch.bool, device=device)
                out = base_model(motion_feats, kps, pad_mask=pad_mask)
                probs = torch.sigmoid(out["logits"])  # [B, L, C]
                H = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
                scores = H.mean(dim=-1)  # [B, L], higher entropy = more uncertain = select
            else:
                raise ValueError(f"Unknown global mask mode: {mode}")

            for b in range(B):
                key = _get_sample_key(batch, b)
                all_scores[key] = scores[b].cpu()

        # === Per-slot bias correction (removes CNN boundary artifact) ===
        # The conv encoder has zero-padding boundary effects: scores at edge
        # slots (near t=0 or t=L-1) are systematically inflated regardless of
        # behavior content. Since the bias depends only on slot position, we
        # estimate it as the per-slot mean across all windows (which averages
        # out the window-specific behavior signal) and subtract. This prevents
        # edge slots from disproportionately winning the top-budget_ratio cut.
        all_keys = list(all_scores.keys())
        if len(all_keys) > 1:
            score_stack = torch.stack([all_scores[k] for k in all_keys], dim=0)  # [W, L]
            slot_bias = score_stack.mean(dim=0)  # [L]
            for key in all_keys:
                all_scores[key] = all_scores[key] - slot_bias
            if is_main_process(0):
                print(
                    f"  [_compute_global_masks] per-slot bias correction: "
                    f"mean={slot_bias.mean().item():.4f}  "
                    f"range=[{slot_bias.min().item():.4f}, {slot_bias.max().item():.4f}]",
                    flush=True,
                )

        # Pass 2: global threshold — collect all scores, find top budget% globally
        all_vals = torch.cat([all_scores[k] for k in all_keys])  # [total_frames]
        total_frames = all_vals.numel()
        total_select = max(1, int(total_frames * budget_ratio))
        threshold = all_vals.topk(total_select).values[-1].item()  # score threshold

        # Pass 3: apply global threshold to each window
        masks = {}
        for key in all_keys:
            masks[key] = (all_scores[key] >= threshold)

        base_model.vision_alpha = saved_alpha
        base_model.train()

        actual_selected = sum(m.sum().item() for m in masks.values())
        if is_main_process(0):
            print(f"  [_compute_global_masks] mode={mode}, {n_batches} batches, {len(masks)} windows, "
                  f"global threshold={threshold:.4f}, selected {int(actual_selected)}/{total_frames} frames "
                  f"({actual_selected/total_frames*100:.1f}%)", flush=True)
        return masks

    # ========== Phased training ==========
    test_f1_history = []
    val_macro_history = []  # per-epoch val macro F1, for best-val epoch selection
    base_model = _get_base_model(model)
    vision_alpha_cfg = float(getattr(args, "vision_alpha", 0.5))
    budget_ratio = float(getattr(args, "budget_ratio", 0.25))
    vision_fuse_mode = str(getattr(args, "vision_fuse_mode", "add")).lower()
    # budget_ratio=0 → pure motion baseline, skip all vision phases
    if budget_ratio <= 0:
        motion_epochs = 0
        vision_epochs = 0
        vision_alpha_cfg = 0.0
    elif vision_fuse_mode == "feature":
        # Feature fusion must be jointly trained; ignore phase config
        motion_epochs = 0
        vision_epochs = 0
        if is_main:
            print("[Finetune] vision_fuse_mode=feature → joint training (ignoring motion_only_epochs/vision_only_epochs)")
    else:
        motion_epochs = int(getattr(args, "motion_only_epochs", 0) or 0)
        vision_epochs = int(getattr(args, "vision_only_epochs", 0) or 0)
    joint_start = motion_epochs + vision_epochs

    if is_main:
        print(f"\n{'='*60}")
        if budget_ratio <= 0:
            print(f"  Fine-tuning ({total_epochs} epochs, pure motion baseline)")
        else:
            print(f"  Fine-tuning ({total_epochs} epochs)")
            print(f"  Phase 1: motion-only epochs 1-{motion_epochs}" if motion_epochs > 0 else "  Phase 1: skipped")
            print(f"  Phase 2: vision-only epochs {motion_epochs+1}-{joint_start}" if vision_epochs > 0 else "  Phase 2: skipped")
            print(f"  Phase 3: joint epochs {joint_start+1}-{total_epochs}")
        print(f"  frame_select_mode={getattr(args, 'frame_select_mode', 'residual')}")
        print(f"  budget_ratio={budget_ratio}, vision_alpha={vision_alpha_cfg}")
        print(f"  lambda_col={lambda_col}, collision_temperature={collision_temperature}")
        print(f"{'='*60}")

    global_masks = None  # will be populated if frame_select_mode needs global computation
    frame_select_mode = str(getattr(args, "frame_select_mode", "residual")).lower()
    global_modes = {"uncertainty", "residual", "residual_inverted", "recon_error", "surprise", "online", "confidence"}
    use_global_masks = (
        frame_select_mode in global_modes
        and 0 < budget_ratio < 1.0
    )
    # "uncertainty" and "online" need Phase 1 to train motion head first; others can run at epoch 0
    needs_trained_head = {"uncertainty", "online"}
    global_mask_epoch = motion_epochs if frame_select_mode in needs_trained_head else 0

    for epoch in range(start_epoch, total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ---- Global mask computation ----
        if use_global_masks and epoch == global_mask_epoch and global_masks is None:
            if is_main:
                print(f"\n[Global Masks] Computing {frame_select_mode} masks (budget={budget_ratio})...")
            # One-time frame-selection overhead: a single pass of the frozen
            # motion scout over the dataset (no vision). Timed once so the
            # rebuttal can report it as a fixed cost, not per-epoch.
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            _sel_t0 = time.perf_counter()
            global_masks = {}
            for loader_name, loader in [("train", train_eval_loader), ("val", val_loader), ("test", test_loader)]:
                masks = _compute_global_masks(base_model, loader, device, budget_ratio, mode=frame_select_mode)
                global_masks.update(masks)
                if is_main:
                    n_masks = len(masks)
                    avg_sel = sum(m.sum().item() for m in masks.values()) / max(n_masks, 1)
                    print(f"  {loader_name}: {n_masks} windows, avg selected frames={avg_sel:.1f}")
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            if is_main:
                print(f"[Timing] frame_selection_overhead_sec={time.perf_counter() - _sel_t0:.3f} "
                      f"(one-time, budget={budget_ratio}, mode={frame_select_mode})", flush=True)

        # ---- Phase switching ----
        if epoch < motion_epochs:
            # Phase 1: motion only — disable vision head
            phase = "motion-only"
            base_model.vision_alpha = 0.0
            for p in base_model.vision_cls_head.parameters():
                p.requires_grad = False
            for p in base_model.social_cls_head.parameters():
                p.requires_grad = True
        elif epoch < joint_start:
            # Phase 2: vision only — freeze motion head, enable vision head
            phase = "vision-only"
            base_model.vision_alpha = vision_alpha_cfg
            for p in base_model.social_cls_head.parameters():
                p.requires_grad = False
            for p in base_model.vision_cls_head.parameters():
                p.requires_grad = True
            # Reset the lr at Phase 2 start (vision head gets full lr)
            if epoch == motion_epochs:
                for pg in optimizer.param_groups:
                    pg["lr"] = pg.get("initial_lr", pg["lr"])
                if ft_scheduler is not None:
                    remaining = total_epochs - motion_epochs
                    ft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=remaining,
                        eta_min=float(getattr(args, "lr_scheduler_params", {}).get("eta_min", 1e-5)),
                    )
                if is_main:
                    print(f"  [Phase 2] Reset LR to {args.lr}"
                          f"{f', scheduler T_max={remaining}' if ft_scheduler is not None else ', no scheduler'}")
        else:
            # Phase 3: joint — both heads trainable
            phase = "joint"
            base_model.vision_alpha = vision_alpha_cfg
            for p in base_model.social_cls_head.parameters():
                p.requires_grad = True
            for p in base_model.vision_cls_head.parameters():
                p.requires_grad = True

        if is_main:
            print(f"\n=== [MABe][Finetune] Epoch {epoch + 1}/{total_epochs} [{phase}] ===")

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        _ep_t0 = time.perf_counter()
        train_stats = train_one_epoch_finetune(
            model, train_loader, optimizer, device, args,
            epoch_idx=epoch, pos_weight=pos_weight,
            lambda_col=lambda_col, collision_temperature=collision_temperature,
            global_masks=global_masks,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        _train_sec = time.perf_counter() - _ep_t0
        if is_main:
            # Train-step wall-clock per epoch (excludes eval passes below), so the
            # rebuttal can report per-budget training compute and extrapolate.
            print(f"[Timing] epoch={epoch + 1} phase={phase} train_step_sec={_train_sec:.2f} "
                  f"budget={budget_ratio}", flush=True)
        train_eval_stats, val_stats, test_stats = _run_eval_and_log(epoch, "Train",
                                                                     global_masks=global_masks)
        if ft_scheduler is not None:
            ft_scheduler.step()
        _log_epoch(epoch, "Train", train_stats, train_eval_stats, val_stats, test_stats)
        # Collect test F1 for summary
        test_counts = test_stats.get("f1_counts")
        if test_counts is not None:
            test_f1_history.append(_counts_to_f1(test_counts))
        val_counts = val_stats.get("f1_counts")
        if val_counts is not None:
            val_f1 = _counts_to_f1(val_counts)
            val_macro_history.append(sum(val_f1) / max(len(val_f1), 1))

    # ========== Summary: test F1 at the epoch with the best val macro F1 ==========
    if is_main and test_f1_history and len(val_macro_history) == len(test_f1_history):
        num_classes = len(test_f1_history[0])
        best_ep = max(range(len(val_macro_history)), key=lambda e: val_macro_history[e])
        best_f1 = test_f1_history[best_ep]
        print(f"\n{'='*60}")
        print(f"  Best-val epoch (epoch {best_ep + 1}, val macro F1={val_macro_history[best_ep]:.4f}) (Test F1)")
        print(f"{'='*60}")
        for c in range(num_classes):
            label = label_names[c] if c < len(label_names) else f"class{c}"
            print(f"  {label:25s} F1={best_f1[c]:.4f}")
        print(f"  {'macro':25s} F1={sum(best_f1) / max(num_classes, 1):.4f}")
        print(f"{'='*60}")

    # ========== Final test evaluation ==========
    final_test = evaluate_finetune(
        model, test_loader, device, args,
        epoch_idx=total_epochs - 1, pos_weight=pos_weight,
        lambda_col=lambda_col, collision_temperature=collision_temperature,
    )
    if args.distributed:
        test_tensor = torch.tensor(
            [final_test["loss"], final_test["cls"], final_test.get("col", 0.0)],
            device=device,
        )
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        final_test.update({
            "loss": test_tensor[0].item() / args.world_size,
            "cls": test_tensor[1].item() / args.world_size,
            "col": test_tensor[2].item() / args.world_size,
        })
        counts = final_test.get("f1_counts")
        if counts is not None:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    if is_main:
        print(
            f"[MABe][Finetune] Test | loss={final_test['loss']:.5f} "
            f"(cls={final_test['cls']:.5f}, col={final_test.get('col', 0.0):.5f})"
        )
        counts = final_test.get("f1_counts")
        if counts is not None:
            f1_scores = _counts_to_f1(counts)
            for idx, f1 in enumerate(f1_scores):
                label = label_names[idx] if idx < len(label_names) else f"class{idx}"
                tp = int(counts[idx, 0].item())
                fp = int(counts[idx, 1].item())
                fn = int(counts[idx, 2].item())
                total = int(counts[idx, 3].item())
                tn = max(total - tp - fp - fn, 0)
                print(
                    f"[Finetune][Test][F1][frame_index] label={label} f1={f1:.4f} "
                    f"tp={tp} tn={tn} fp={fp} fn={fn}",
                    flush=True,
                )
        if writer:
            writer.add_scalar("finetune/test_loss", final_test["loss"], total_epochs)
            writer.add_scalar("finetune/test_cls", final_test["cls"], total_epochs)
            writer.add_scalar("finetune/test_col", final_test.get("col", 0.0), total_epochs)
            if counts is not None:
                for idx, f1 in enumerate(f1_scores):
                    label = label_names[idx] if idx < len(label_names) else f"class{idx}"
                    writer.add_scalar(f"finetune/test_f1/{label}", f1, total_epochs)
    if writer:
        writer.close()


def main(config_path: str | None = None, seed: int | None = None):
    args = parse_args(config_path)
    if seed is not None:
        args.seed = seed
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    if env_world > 1:
        _main_worker(args)
    else:
        if args.distributed and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            world_size = torch.cuda.device_count()
            _set_default_dist_env(args)
            mp.spawn(_spawn_worker, nprocs=world_size, args=(world_size, args))
        else:
            _main_worker(args)


def _spawn_worker(rank: int, world_size: int, args):
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    _set_default_dist_env(args)
    _main_worker(args)


def _main_worker(args):
    if args.num_joints is None:
        args.num_joints = 12
    _set_default_dist_env(args)
    is_dist, rank, local_rank, world_size = init_distributed()
    args.distributed = is_dist
    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size
    if is_dist:
        args.device = f"cuda:{local_rank}"
    if is_main_process(rank):
        print(f"[DDP] world_size={world_size}, rank={rank}, local_rank={local_rank}")
        if torch.cuda.is_available() and is_dist:
            print(f"[DDP] device={torch.cuda.get_device_name(local_rank)}")
    dtype_map = {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    args.torch_dtype = dtype_map.get(args.torch_dtype.lower(), torch.bfloat16)
    args.dataset = "mabe"
    _set_seed(getattr(args, "seed", None), rank)
    if args.stage == "pretrain":
        run_pretrain(args)
    else:
        run_finetune(args)


def _set_default_dist_env(args):
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(getattr(args, "master_port", 29510))


if __name__ == "__main__":
    main()
