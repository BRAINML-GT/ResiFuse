import os
import math
import time
import shutil
import glob
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, Tuple
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from resifuse.data.datasets import FreelyMovingMouseDataset
from resifuse.models.model import ResiFuseClassifier, ResiFusePretrainer
from torch.utils.tensorboard import SummaryWriter
from resifuse.config import load_config_with_defaults


def init_distributed():
    if not torch.cuda.is_available() or not dist.is_available():
        return False, 0, 0, 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
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


def log_gpu_memory(prefix: str, device: torch.device, rank: int):
    if device.type != "cuda":
        return
    torch.cuda.synchronize(device)
    alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    if is_main_process(rank):
        print(f"{prefix} | max_alloc={alloc:.2f} GB, max_reserved={reserved:.2f} GB")
    torch.cuda.reset_peak_memory_stats(device)

def parse_args(config_path: str | None = None):
    cfg = load_config_with_defaults(config_path)
    stage = cfg.get("stage", "pretrain")
    if stage == "pretrain":
        cfg["lr"] = cfg["pretrain_lr"]
        cfg["epochs"] = cfg["pretrain_epochs"]
    elif stage == "finetune":
        cfg["lr"] = cfg["finetune_lr"]
        cfg["epochs"] = cfg["finetune_epochs"]
    return SimpleNamespace(**cfg)


def _coerce_number(val):
    """Convert numeric strings (including scientific notation) into ints/floats; recurse for containers."""
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


def _get_sparse_lambda(config, epoch_idx: int | None = None) -> float:
    """
    Linearly warm up lambda_sparse from 0 -> target over sparse_warmup_epochs.
    """
    base = float(getattr(config, "lambda_sparse", 0.0) or 0.0)
    warmup = int(getattr(config, "sparse_warmup_epochs", 0) or 0)
    if epoch_idx is None or warmup <= 0:
        return base
    progress = min(max(epoch_idx, 0) / float(warmup), 1.0)
    return base * progress


def _get_active_ratio_label(config) -> str:
    gate_topk = int(getattr(config, "gate_topk", 0) or 0)
    gate_topk_ratio = float(getattr(config, "gate_topk_ratio", 0.0) or 0.0)
    if gate_topk > 0:
        return f"active_ratio_topk={gate_topk}"
    if gate_topk_ratio > 0.0:
        return f"active_ratio_topk_ratio={gate_topk_ratio}"
    gate_thresh = getattr(config, "gate_thresh", 0.5)
    return f"active_ratio>{gate_thresh}"


def _compute_active_ratio(alpha: torch.Tensor, config) -> float:
    L = alpha.shape[1]
    gate_topk = int(getattr(config, "gate_topk", 0) or 0)
    gate_topk_ratio = float(getattr(config, "gate_topk_ratio", 0.0) or 0.0)
    if gate_topk > 0 or gate_topk_ratio > 0.0:
        if gate_topk > 0:
            topk = min(gate_topk, L)
        else:
            topk = min(max(1, int(math.ceil(L * gate_topk_ratio))), L)
        return topk / float(L)
    gate_thresh = getattr(config, "gate_thresh", 0.5)
    return (alpha > gate_thresh).float().mean().item()


def _get_base_model(model: torch.nn.Module):
    base = model.module if isinstance(model, DDP) else model
    return getattr(base, "_orig_mod", base)


def _maybe_set_visual_epoch(model: torch.nn.Module, epoch: int):
    base = _get_base_model(model)
    if hasattr(base, "visual_magnifier"):
        base.visual_magnifier.set_epoch(epoch)


def _maybe_flush_visual_epoch(model: torch.nn.Module):
    base = _get_base_model(model)
    if hasattr(base, "visual_magnifier"):
        base.visual_magnifier.flush_epoch_usage()


def _init_gate_bias(model: torch.nn.Module, target_alpha: float) -> None:
    base = _get_base_model(model)
    scout = getattr(base, "motion_scout", None)
    if scout is None:
        return
    alpha = float(target_alpha)
    alpha = min(max(alpha, 1e-4), 1.0 - 1e-4)
    logit = math.log(alpha / (1.0 - alpha))
    with torch.no_grad():
        for layer in (getattr(scout, "gate_mlp", None), getattr(scout, "gate_pos", None), getattr(scout, "gate_neg", None)):
            if layer is None or getattr(layer, "bias", None) is None:
                continue
            layer.bias.zero_()
        if getattr(scout, "gate_mlp", None) is not None and scout.gate_mlp.bias is not None:
            scout.gate_mlp.bias.fill_(logit)


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
    # copy to avoid mutating config
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
    # Timestamp with millisecond precision + PID tail to avoid run_dir collision
    # when two jobs launch in the same second (e.g. batched job submission).
    now = time.time()
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}-{os.getpid() % 10000:04d}"
    if not run_name:
        if stage == "pretrain":
            run_name = f"{ts}_pre_bs{args.batch_size}_lr{args.lr}_mask{args.mask_ratio}"
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
    # Atomic symlink update to avoid TOCTOU race when two jobs share a run_dir:
    # create symlink at a unique temp path, then os.replace() atomically swaps
    # it into place (kernel-level atomic on POSIX filesystems).
    try:
        tmp_link = f"{latest}.tmp.{os.getpid()}"
        if os.path.islink(tmp_link) or os.path.exists(tmp_link):
            os.remove(tmp_link)
        os.symlink(os.path.basename(ckpt_path), tmp_link)
        os.replace(tmp_link, latest)
    except Exception:
        try:
            if os.path.islink(latest) or os.path.exists(latest):
                os.remove(latest)
        except OSError:
            pass
        shutil.copyfile(ckpt_path, latest)
    return ckpt_path


def _cleanup_old_best(run_dir: str, prefix: str, keep_path: str | None = None):
    """Remove stale best checkpoints that still carry epoch suffixes, keeping only the latest."""
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
    args=None,
    is_main: bool = True,
):
    """Single-stage freeze policy mirroring resifuse/training/mabe.py:

    - motion_scout always frozen (RVQ codebook is pretrained)
    - If finetune_middle_modules=False (default), also freeze the intermediate
      pretrained modules so finetune only trains cls_heads + seq_encoder + the
      vision/pred heads on top.
    - No staged unfreezing, no epoch-dependent reconfiguration.
    """
    base_model = model.module if hasattr(model, "module") else model
    base_model = getattr(base_model, "_orig_mod", base_model)
    for _, p in base_model.named_parameters():
        p.requires_grad = True
    # Freeze motion_scout entirely (RVQ encoder is pretrained)
    for name, p in base_model.named_parameters():
        if name.startswith("motion_scout.") or ".motion_scout." in name:
            p.requires_grad = False

    finetune_middle = bool(getattr(args, "finetune_middle_modules", False))
    if not finetune_middle:
        # Same prefixes as resifuse/training/mabe.py. social_encoder/social_fuse are
        # dead compute with A=1 but we still freeze them to stay byte-for-byte
        # aligned with mabe's freezing policy.
        middle_prefixes = ("context_transformer.", "social_encoder.", "social_fuse.", "motion_norm.")
        frozen_middle = 0
        for name, p in base_model.named_parameters():
            if any(name.startswith(pf) or f".{pf}" in name for pf in middle_prefixes):
                p.requires_grad = False
                frozen_middle += p.numel()
        if is_main and frozen_middle > 0:
            print(
                f"[Finetune] Middle modules frozen: {frozen_middle/1e6:.3f}M params "
                f"(context_transformer/social_encoder/social_fuse/motion_norm)"
            )

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

def create_mask(batch: int, t: int, ratio: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rand = torch.rand(batch, t, device=device, dtype=dtype)
    return (rand < ratio).to(dtype)

def _normalize_vision_targets(base_model: torch.nn.Module, vision_feats: torch.Tensor) -> torch.Tensor:
    feats = F.normalize(vision_feats, dim=-1, eps=1e-6).detach()
    adapter = getattr(base_model, "target_adapter", None)
    if adapter is not None:
        feats = adapter(feats)
        feats = F.normalize(feats, dim=-1, eps=1e-6)
    return feats


def compute_multihorizon_pred_loss(
    pred_vis: Dict[str, torch.Tensor],
    vision_feats: torch.Tensor | None,
    stride: int,
    horizons: list[int],
    weights: list[float],
    loss_type: str,
    base_model: torch.nn.Module,
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
            loss = F.mse_loss(pred_norm, target)
        else:
            loss = 1.0 - (pred_norm * target).sum(dim=-1)
            loss = loss.mean()
        losses[h] = loss
        total = total + float(w) * loss
    return total, losses


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
    total_timing = {}
    num_batches = len(dataloader)
    dataset_len = len(dataloader.dataset)
    global_total_batches = math.ceil(dataset_len / config.batch_size) if dataset_len > 0 else 0
    grad_acc_steps = int(getattr(config, "pretrain_grad_acc_steps", None) or getattr(config, "grad_acc_steps", 1) or 1)
    grad_acc_steps = max(1, grad_acc_steps)
    progress_points = sorted({max(1, int(num_batches * frac)) for frac in (0.2, 0.4, 0.6, 0.8) if num_batches > 0})

    fetch_time = 0.0
    compute_time = 0.0
    end_time = time.perf_counter()
    vision_pred_weight = float(getattr(config, "vision_pred_weight", 0.0) or 0.0)
    pred_warmup = int(getattr(config, "vision_pred_warmup_epochs", 0) or 0)
    pred_scale = min(1.0, float(epoch_idx + 1) / pred_warmup) if pred_warmup > 0 else 1.0
    vision_pred_loss = str(getattr(config, "vision_pred_loss", "cosine") or "cosine").lower()
    if vision_pred_loss not in ("cosine", "mse"):
        vision_pred_loss = "cosine"
    horizons = getattr(config, "vision_pred_horizons", None)
    weights = getattr(config, "vision_pred_weights", None) or [1.0 for _ in horizons]
    horizons = [int(h) for h in horizons]
    weights = [float(w) for w in weights]
    per_horizon = {h: 0.0 for h in horizons}
    base_model = _get_base_model(model)
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        fetch_start = end_time
        kps = batch["keypoints"].to(device)       # [B, A, T, K, 3]
        motion_feats = batch["motion_feats"].to(device)  # [B, A, T, D]
        vision_feats = batch.get("vision_feats") if vision_pred_weight > 0.0 else None
        if vision_feats is not None:
            vision_feats = vision_feats.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fetch_time += time.perf_counter() - fetch_start
        compute_start = time.perf_counter()

        B, A, T, K, _ = kps.shape  # A=1 for freely single-mouse
        kps_flat = kps.view(B * A, T, K, 3)
        motion_flat = motion_feats.view(B * A, T, -1)
        vision_flat = vision_feats.view(B * A, T, -1) if vision_feats is not None else None

        # M-MAE mask kept Bernoulli (freely) but reshaped to [B, A, T] for mabe forward.
        mask_ba = (torch.rand(B, A, T, device=device) < config.mask_ratio).to(dtype=config.torch_dtype)
        mask = mask_ba.view(B * A, T)  # keep [B*A, T] alias for downstream loss code
        # Freely pretrain has no per-frame validity concept → all-valid mask.
        valid_mask = torch.ones(B, A, T, dtype=torch.bool, device=device)

        out = model(
            motion_feats,
            kps,
            None,
            mask=valid_mask,
            mask_indices=mask_ba,
            vision_feats=vision_feats,
        )

        mask_ds = out["mask_ds"] if out["mask_ds"] is not None else mask
        kp_ds = out["kp_ds"]
        pred_kps = out["pred_kps"]
        pred_vis = out["pred_vis"]
        vq_recon = out["vq_recon"]
        vq_loss = out["vq_loss"]
        stride = out.get("stride", max(1, T // pred_kps.shape[1]))
        if vision_pred_weight > 0.0 and pred_kps.shape[1] <= min(horizons):
            raise RuntimeError(
                f"vision_pred_horizon must be smaller than pred length ({pred_kps.shape[1]})."
            )
        # downsample mask to match pred length
        if mask_ds is None:
            mask_ds = create_mask(B * A, pred_kps.shape[1], config.mask_ratio, device, dtype=config.torch_dtype)
        mask_expanded = mask_ds.view(B * A, pred_kps.shape[1], 1, 1).expand_as(pred_kps)
        target_kps = kp_ds[..., :2]  # mask only xy
        loss_mse = F.mse_loss(pred_kps * mask_expanded, target_kps * mask_expanded, reduction="sum")
        num_masked = mask_expanded.sum() + 1e-6
        loss_masked = loss_mse / num_masked

        target_motion = motion_flat[:, :: max(1, T // pred_kps.shape[1])][:, :pred_kps.shape[1], :]
        loss_recon = F.mse_loss(vq_recon, target_motion)
        if vision_pred_weight > 0.0 and vision_flat is None:
            raise RuntimeError(
                "vision_pred_weight > 0 but no vision_feats found in batch. "
                "Enable use_vision_cache and set vision_cache_dir."
            )
        loss_pred, loss_dict = compute_multihorizon_pred_loss(
            pred_vis,
            vision_flat,
            stride=stride,
            horizons=horizons,
            weights=weights,
            loss_type=vision_pred_loss,
            base_model=base_model,
        )
        warmup_epochs = getattr(config, "vq_loss_warmup_epochs", 0)
        vq_scale = min(1.0, float(epoch_idx + 1) / warmup_epochs) if warmup_epochs and warmup_epochs > 0 else 1.0
        vq_recon_term = config.vq_recon_weight * loss_recon
        vq_term = (config.vq_loss_weight * vq_scale) * vq_loss
        pred_term = (vision_pred_weight * pred_scale) * loss_pred
        loss_full = loss_masked + pred_term + vq_recon_term + vq_term
        loss_full_scaled = loss_full / grad_acc_steps

        loss_full_scaled.backward()
        if (batch_idx + 1) % grad_acc_steps == 0 or (batch_idx + 1) == num_batches:
            optimizer.step()
            base_model.update_vq_ema()
            optimizer.zero_grad()

        total_loss += loss_full.item()
        total_mask += loss_masked.item()
        total_pred += loss_pred.item()
        total_recon += vq_recon_term.item()
        total_vq += vq_term.item()
        for h, val in loss_dict.items():
            per_horizon[h] = per_horizon.get(h, 0.0) + val.item()
        if out.get("timing") is not None:
            for k, v in out["timing"].items():
                total_timing[k] = total_timing.get(k, 0.0) + v
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_time += time.perf_counter() - compute_start
        end_time = time.perf_counter()
        if is_main and (batch_idx + 1 in progress_points):
            processed = batch_idx + 1
            progress_frac = processed / max(num_batches, 1)
            processed_global = min(global_total_batches, math.ceil(progress_frac * max(global_total_batches, 1)))
            remaining_global = max(0, global_total_batches - processed_global)
            print(
                f"[Pretrain] Progress: global {processed_global}/{global_total_batches} batches "
                f"(rank-local {processed}/{num_batches}), remaining_global {remaining_global}"
            )

    denom = max(num_batches, 1)
    return {
        "loss": total_loss / denom,
        "mask": total_mask / denom,
        "pred": total_pred / denom,
        "vq_recon": total_recon / denom,
        "vq": total_vq / denom,
        "pred_h": {h: per_horizon[h] / denom for h in horizons},
        "avg_fetch": fetch_time / max(num_batches, 1),
        "avg_compute": compute_time / max(num_batches, 1),
        "total_fetch": fetch_time,
        "total_compute": compute_time,
        "timing": {k: v / denom for k, v in total_timing.items()} if total_timing else {},
        "timing_total": total_timing,
        "num_batches": num_batches,
    }


def evaluate_pretrain(
    model: ResiFusePretrainer,
    dataloader: DataLoader,
    device: torch.device,
    config,
    epoch_idx: int | None = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mask = 0.0
    total_pred = 0.0
    total_recon = 0.0
    total_vq = 0.0
    vision_pred_weight = float(getattr(config, "vision_pred_weight", 0.0) or 0.0)
    pred_warmup = int(getattr(config, "vision_pred_warmup_epochs", 0) or 0)
    pred_scale = min(1.0, float((epoch_idx or 0) + 1) / pred_warmup) if pred_warmup > 0 else 1.0
    vision_pred_loss = str(getattr(config, "vision_pred_loss", "cosine") or "cosine").lower()
    if vision_pred_loss not in ("cosine", "mse"):
        vision_pred_loss = "cosine"
    horizons = getattr(config, "vision_pred_horizons", None)
    weights = getattr(config, "vision_pred_weights", None) or [1.0 for _ in horizons]
    horizons = [int(h) for h in horizons]
    weights = [float(w) for w in weights]
    per_horizon = {h: 0.0 for h in horizons}
    base_model = _get_base_model(model)
    with torch.no_grad():
        for batch in dataloader:
            kps = batch["keypoints"].to(device)
            motion_feats = batch["motion_feats"].to(device)
            vision_feats = batch.get("vision_feats") if vision_pred_weight > 0.0 else None
            if vision_feats is not None:
                vision_feats = vision_feats.to(device)
            B, A, T, K, _ = kps.shape  # A=1 for freely
            kps_flat = kps.view(B * A, T, K, 3)
            motion_flat = motion_feats.view(B * A, T, -1)
            vision_flat = vision_feats.view(B * A, T, -1) if vision_feats is not None else None

            mask_ba = (torch.rand(B, A, T, device=device) < config.mask_ratio).to(dtype=config.torch_dtype)
            mask = mask_ba.view(B * A, T)
            valid_mask = torch.ones(B, A, T, dtype=torch.bool, device=device)
            out = model(
                motion_feats,
                kps,
                None,
                mask=valid_mask,
                mask_indices=mask_ba,
                vision_feats=vision_feats,
            )

            mask_ds = out["mask_ds"] if out["mask_ds"] is not None else mask
            kp_ds = out["kp_ds"]
            pred_kps = out["pred_kps"]
            pred_vis = out["pred_vis"]
            vq_recon = out["vq_recon"]
            vq_loss = out["vq_loss"]
            if vision_pred_weight > 0.0 and pred_kps.shape[1] <= min(horizons):
                raise RuntimeError(
                    f"vision_pred_horizon must be smaller than pred length ({pred_kps.shape[1]})."
                )

            mask_expanded = mask_ds.view(B * A, pred_kps.shape[1], 1, 1).expand_as(pred_kps)
            target_kps = kp_ds[..., :2]
            loss_mse = F.mse_loss(pred_kps * mask_expanded, target_kps * mask_expanded, reduction="sum")
            num_masked = mask_expanded.sum() + 1e-6
            loss_masked = loss_mse / num_masked
            target_motion = motion_flat[:, :: max(1, T // pred_kps.shape[1])][:, :pred_kps.shape[1], :]
            loss_recon = F.mse_loss(vq_recon, target_motion)
            vq_recon_term = config.vq_recon_weight * loss_recon
            vq_term = config.vq_loss_weight * vq_loss
            if vision_pred_weight > 0.0 and vision_flat is None:
                raise RuntimeError(
                    "vision_pred_weight > 0 but no vision_feats found in batch. "
                    "Enable use_vision_cache and set vision_cache_dir."
                )
            stride = out.get("stride", max(1, T // pred_kps.shape[1]))
            loss_pred, loss_dict = compute_multihorizon_pred_loss(
                pred_vis,
                vision_flat,
                stride=stride,
                horizons=horizons,
                weights=weights,
                loss_type=vision_pred_loss,
                base_model=base_model,
            )
            pred_term = (vision_pred_weight * pred_scale) * loss_pred
            loss = loss_masked + pred_term + vq_recon_term + vq_term
            total_loss += loss.item()
            total_mask += loss_masked.item()
            total_pred += loss_pred.item()
            total_recon += vq_recon_term.item()
            total_vq += vq_term.item()
            for h, val in loss_dict.items():
                per_horizon[h] = per_horizon.get(h, 0.0) + val.item()
    denom = max(len(dataloader), 1)
    return {
        "loss": total_loss / denom,
        "mask": total_mask / denom,
        "pred": total_pred / denom,
        "vq_recon": total_recon / denom,
        "vq": total_vq / denom,
        "pred_h": {h: per_horizon[h] / denom for h in horizons},
    }


def compute_seq_cls_loss(
    logits: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    mask: torch.Tensor,
    scenario: torch.Tensor,
    head_classes: Dict[str, int],
    class_weights: Dict[str, torch.Tensor] | None = None,
    loss_cfg=None,
) -> Tuple[torch.Tensor, int, int, Dict[str, torch.Tensor]]:
    """
    logits: dict head -> [B, L, C]
    labels: [B, L] with -1 as ignore
    mask: [B, L] bool (valid positions)
    scenario: [B] (per sample)
    """
    device = labels.device
    total_loss = torch.tensor(0.0, device=device)
    total_correct = 0
    total_samples = 0
    counts = {k: torch.zeros((v, 3), device=device) for k, v in head_classes.items()}  # tp/fp/fn
    head_order = ["A", "B", "C"]
    if len(head_classes) == 1 and "A" in head_classes:
        head_order = ["A"]
    for idx, head_key in enumerate(head_order):
        if head_key not in head_classes or head_key not in logits:
            continue
        head_mask = (scenario == idx).unsqueeze(1) & mask if head_order != ["A"] else mask
        if not head_mask.any():
            continue
        head_logits = logits[head_key][head_mask]  # [N, C]
        head_labels = labels[head_mask]  # [N]
        valid = head_labels >= 0
        if not valid.any():
            continue
        head_logits = head_logits[valid]
        head_labels = head_labels[valid]

        loss_type = (getattr(loss_cfg, "loss_type", "ce") or "ce").lower()
        weight = None
        if loss_type != "focal" and class_weights is not None:
            weight = class_weights.get(head_key)
            weight = weight.to(device=head_logits.device, dtype=head_logits.dtype)
        # if weight is None and loss_type != "focal":
        #     raise NotImplementedError
        gamma = float(getattr(loss_cfg, "focal_gamma", 2.0) or 2.0)
        alpha_cfg = getattr(loss_cfg, "focal_alpha", None)
        alpha_tensor = None
        if alpha_cfg is not None:
            try:
                alpha_tensor = torch.as_tensor(alpha_cfg, device=head_logits.device, dtype=head_logits.dtype)
                if alpha_tensor.numel() == 1:
                    alpha_tensor = alpha_tensor.expand(head_classes[head_key])
                if alpha_tensor.numel() != head_classes[head_key]:
                    alpha_tensor = None
            except Exception:
                alpha_tensor = None

        label_smoothing = float(getattr(loss_cfg, "label_smoothing", 0.0) or 0.0)
        if loss_type == "focal":
            log_prob = F.log_softmax(head_logits, dim=-1)
            prob = log_prob.exp()
            target_log_prob = log_prob.gather(1, head_labels.unsqueeze(1)).squeeze(1)
            target_prob = prob.gather(1, head_labels.unsqueeze(1)).squeeze(1)
            focal_term = (1 - target_prob).pow(gamma)
            loss_vec = -focal_term * target_log_prob
            if alpha_tensor is not None:
                loss_vec = loss_vec * alpha_tensor[head_labels]
            loss_sum = loss_vec.sum()
        else:
            loss_sum = F.cross_entropy(
                head_logits,
                head_labels,
                reduction="sum",
                weight=weight,
                label_smoothing=label_smoothing,
            )
        total_loss = total_loss + loss_sum
        preds = head_logits.argmax(dim=-1)
        total_correct += (preds == head_labels).sum().item()
        total_samples += head_labels.numel()
        for c in range(head_classes[head_key]):
            tp = ((preds == c) & (head_labels == c)).sum()
            fp = ((preds == c) & (head_labels != c)).sum()
            fn = ((preds != c) & (head_labels == c)).sum()
            counts[head_key][c, 0] += tp
            counts[head_key][c, 1] += fp
            counts[head_key][c, 2] += fn
    return total_loss, total_correct, total_samples, counts


def _compute_f1_from_counts(counts: Dict[str, torch.Tensor]) -> Dict[str, list]:
    f1_scores: Dict[str, list] = {}
    for hk, tens in counts.items():
        tp, fp, fn = tens[:, 0], tens[:, 1], tens[:, 2]
        denom = 2 * tp + fp + fn
        f1 = torch.where(denom > 0, 2 * tp / denom.clamp(min=1e-8), torch.zeros_like(tp))
        f1_scores[hk] = f1.cpu().tolist()
    return f1_scores


def _mean_f1(scores: Dict[str, list], active_heads: list[str]) -> float:
    values = []
    for hk in active_heads:
        vals = scores.get(hk, [])
        if vals:
            values.extend(vals)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _counts_to_stats(counts: Dict[str, torch.Tensor]) -> Dict[str, list]:
    stats: Dict[str, list] = {}
    for hk, tens in counts.items():
        tp, fp, fn = tens[:, 0], tens[:, 1], tens[:, 2]
        support = tp + fn
        pred = tp + fp
        stacked = torch.stack([tp, fp, fn, support, pred], dim=1)
        stats[hk] = stacked.cpu().tolist()
    return stats


def _init_per_video_counts(head_classes: Dict[str, int], device: torch.device):
    return {
        hk: defaultdict(lambda: torch.zeros((num_classes, 3), device=device))
        for hk, num_classes in head_classes.items()
        if num_classes > 0
    }


def _gather_per_video_counts(
    per_video_counts: Dict[str, dict],
    distributed: bool,
    world_size: int,
):
    """
    Merge per-video counts across ranks; always return cpu tensors.
    """
    local_cpu = {hk: {stem: tens.detach().cpu() for stem, tens in stems.items()} for hk, stems in per_video_counts.items()}
    if not distributed or world_size <= 1:
        return local_cpu
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_cpu)
    merged = {hk: {} for hk in local_cpu.keys()}
    for obj in gathered:
        if obj is None:
            continue
        for hk, stems in obj.items():
            if hk not in merged:
                merged[hk] = {}
            for stem, tens in stems.items():
                if stem not in merged[hk]:
                    merged[hk][stem] = tens.clone()
                else:
                    merged[hk][stem] += tens
    return merged


def _log_per_video_f1(
    split_name: str,
    per_video_counts: Dict[str, Dict[str, torch.Tensor]],
    head_classes: Dict[str, int],
    label_names: Dict[str, list] | None,
    is_main: bool = True,
):
    if not is_main or not per_video_counts:
        return
    label_names = label_names or {}
    for hk, stems in per_video_counts.items():
        num_classes = head_classes.get(hk, 0)
        if num_classes <= 0 or not stems:
            continue
        for stem, tens in sorted(stems.items()):
            tp, fp, fn = tens[:, 0], tens[:, 1], tens[:, 2]
            denom = 2 * tp + fp + fn
            f1 = torch.where(denom > 0, 2 * tp / denom.clamp(min=1e-8), torch.zeros_like(tp))
            for idx in range(num_classes):
                lname_list = label_names.get(hk, [])
                lname = lname_list[idx] if idx < len(lname_list) else f"{hk}{idx}"
                print(
                    f"[PerVideo][{split_name}] head={hk} video={stem} label={lname} "
                    f"f1={f1[idx]:.4f} tp={tp[idx]:.0f} fp={fp[idx]:.0f} "
                    f"fn={fn[idx]:.0f} support={(tp[idx] + fn[idx]):.0f} pred={(tp[idx] + fp[idx]):.0f}"
                )


def _get_split_stems(dataset) -> list[str]:
    stems = []
    seen = set()
    entries = getattr(dataset, "cache_entries", None)
    if entries:
        for e in entries:
            if not isinstance(e, dict):
                continue
            stem = e.get("stem")
            if stem and stem not in seen:
                stems.append(stem)
                seen.add(stem)
        if stems:
            return stems
    for s in getattr(dataset, "samples", []):
        if not isinstance(s, dict):
            continue
        stem = s.get("stem")
        if stem and stem not in seen:
                stems.append(stem)
                seen.add(stem)
    return stems


def _map_labels_for_head_a(labels: torch.Tensor, label_map: dict | None = None) -> torch.Tensor:
    """
    Map raw fused labels to head A indices: Rearing->0, Grooming->1, others->-1.
    Falls back to default mapping if label_map is not provided.
    """
    mapped = torch.full_like(labels, -1)
    if label_map:
        for src, dst in label_map.items():
            try:
                src_val = int(src)
                dst_val = int(dst)
            except Exception:
                continue
            mapped[labels == src_val] = dst_val
    else:
        mapped[labels == 1] = 0
        mapped[labels == 2] = 1
    return mapped


def _read_labels_for_dataset(dataset, label_path: str):
    """Read per-frame labels with the granularity the dataset was configured for.

    3-class (rebuttal diagnostic) keeps Supported/Unsupported/Grooming separate;
    2-class (default) fuses the rearing classes. Used by the class-weight and
    dataset-stat helpers so their counts match the labels the model is trained on.
    """
    if int(getattr(dataset, "freely_num_classes", 2)) == 3:
        return FreelyMovingMouseDataset._read_labels_3class(label_path)
    return FreelyMovingMouseDataset._read_labels(label_path)


def _compute_class_weights(dataset, head_classes: Dict[str, int], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Compute class weights for head A using the same logic as train_tcn:
    weight_i = N_total / (num_classes * count_i + 1e-6)
    """
    weights: Dict[str, torch.Tensor] = {}
    num_classes_a = head_classes.get("A", 0)
    if num_classes_a <= 0:
        return weights
    label_dir = getattr(dataset, "label_dir", None)
    if not label_dir:
        return weights
    ds_stride = max(1, int(getattr(dataset, "ds_stride", 1)))
    label_map = getattr(dataset, "label_map", None)
    stems = _get_split_stems(dataset)
    if not stems:
        return weights
    counts = torch.zeros(num_classes_a, dtype=torch.float32)
    for stem in stems:
        label_path = os.path.join(label_dir, f"{stem}_labels.csv")
        if not os.path.exists(label_path):
            continue
        try:
            labels = _read_labels_for_dataset(dataset, label_path)
        except Exception:
            continue
        labels_t = torch.as_tensor(labels, dtype=torch.long)
        labels_t = labels_t[:: ds_stride]
        mapped = _map_labels_for_head_a(labels_t, label_map)
        valid = mapped >= 0
        if valid.any():
            binc = torch.bincount(mapped[valid], minlength=num_classes_a).float()
            counts += binc
    total = counts.sum()
    if total > 0:
        weights_a = total / (counts * num_classes_a + 1e-6)
        weights["A"] = weights_a.to(device=device, dtype=torch.float32)
    return weights


def _print_dataset_stats(name: str, dataset, weights_a: torch.Tensor | None = None) -> None:
    """
    Mimic train_tcn: print video count and per-class labeled frame counts for head A.
    """
    stems = _get_split_stems(dataset)
    label_dir = getattr(dataset, "label_dir", None)
    if not stems or not label_dir:
        print(f"{name} -> stems/label_dir unavailable")
        return
    ds_stride = max(1, int(getattr(dataset, "ds_stride", 1)))
    label_map = getattr(dataset, "label_map", None)
    n3 = int(getattr(dataset, "freely_num_classes", 2)) == 3
    class_names = ["supported", "unsupported", "grooming"] if n3 else ["rearing", "grooming"]
    total_labeled = 0
    counts = {nm: 0 for nm in class_names}
    for stem in stems:
        label_path = os.path.join(label_dir, f"{stem}_labels.csv")
        if not os.path.exists(label_path):
            continue
        try:
            labels = _read_labels_for_dataset(dataset, label_path)
        except Exception:
            continue
        labels_t = torch.as_tensor(labels, dtype=torch.long)[:: ds_stride]
        mapped = _map_labels_for_head_a(labels_t, label_map)
        valid = mapped >= 0
        total_labeled += int(valid.sum())
        for ci, nm in enumerate(class_names):
            counts[nm] += int((mapped == ci).sum())
    print(f"{name} -> videos: {stems}, labeled_frames: {total_labeled}")
    print(f"  per-class labeled frames: {counts}")
    if weights_a is not None:
        print(f"  class weights (head A): {weights_a}")


def _get_sample_key(batch, b):
    """Unique per-window key for global_masks lookup in freely finetune.

    Mirrors the mabe (video_id, window_idx, mouse_id) composite key, but for
    freely single-mouse data we use (stem, start, end) — stem identifies the
    video and (start, end) identify the window within it.
    """
    def _scalar(val, b, default=0):
        if isinstance(val, (list, tuple)):
            return val[b] if b < len(val) else default
        if torch.is_tensor(val):
            return int(val[b].item()) if val.dim() > 0 else int(val.item())
        return val if val is not None else default

    stem = _scalar(batch.get("stem"), b, "")
    start = _scalar(batch.get("start"), b, 0)
    end = _scalar(batch.get("end"), b, 0)
    return f"{stem}_{int(start)}_{int(end)}"


def train_one_epoch_finetune(
    model: ResiFuseClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config,
    class_weights: Dict[str, torch.Tensor] | None = None,
    epoch_idx: int | None = None,
    global_masks: dict | None = None,
):
    is_main = is_main_process(getattr(config, "rank", 0))
    model.train()
    loss_meter, correct_meter, sample_meter = 0.0, 0, 0
    cls_meter, sparse_meter, count_meter = 0.0, 0.0, 0.0
    pred_meter = 0.0
    alpha_meter, alpha_min_meter, alpha_max_meter, alpha_active_meter = 0.0, 0.0, 0.0, 0.0
    gate_mean_meter, gate_min_meter, gate_max_meter = 0.0, 0.0, 0.0
    alpha_tail_batches = int(getattr(config, "alpha_tail_batches", 5) or 5)
    alpha_tail_batches = max(1, alpha_tail_batches)
    alpha_tail_vals = []
    head_classes = {"A": config.head_a_classes}
    counts = {k: torch.zeros((v, 3), device=device) for k, v in head_classes.items()}
    lambda_sparse = _get_sparse_lambda(config, epoch_idx)
    finetune_pred_weight = float(getattr(config, "finetune_pred_weight", 0.0) or 0.0)
    vision_pred_loss = str(getattr(config, "vision_pred_loss", "cosine") or "cosine").lower()
    if vision_pred_loss not in ("cosine", "mse"):
        vision_pred_loss = "cosine"
    horizons = getattr(config, "vision_pred_horizons", None) or [getattr(config, "vision_pred_horizon", 1)]
    weights = getattr(config, "vision_pred_weights", None) or [1.0 for _ in horizons]
    horizons = [int(h) for h in horizons]
    weights = [float(w) for w in weights]
    active_heads_cfg = getattr(config, "finetune_active_heads", ["A"]) or ["A"]
    active_heads = [h for h in active_heads_cfg if h in head_classes]
    if not active_heads:
        active_heads = ["A"]
    base_model = _get_base_model(model)
    grad_acc_steps = int(getattr(config, "finetune_grad_acc_steps", None) or getattr(config, "grad_acc_steps", 1) or 1)
    grad_acc_steps = max(1, grad_acc_steps)
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        kps = batch["keypoints"].to(device)          # [B, A, T, K, 3]
        motion_feats = batch["motion_feats"].to(device)  # [B, A, T, D]
        # frames skipped when dataset.load_frames=False (vision cache path).
        vision_feats = batch.get("vision_feats")
        if vision_feats is not None:
            vision_feats = vision_feats.to(device)
        scenario = batch["scenario"].to(device)
        count = batch["count"].to(device)
        frame_labels = batch["frame_labels"].to(device)  # [B, T]
        frame_mask = batch["frame_mask"].to(device)      # [B, T]
        valid_samples = frame_mask.sum(dim=1) > 0
        if not valid_samples.any():
            continue
        kps = kps[valid_samples]
        motion_feats = motion_feats[valid_samples]
        scenario = scenario[valid_samples]
        count = count[valid_samples]
        frame_labels = frame_labels[valid_samples]
        frame_mask = frame_mask[valid_samples]
        if vision_feats is not None:
            vision_feats = vision_feats[valid_samples]
        B, A, T, K, _ = kps.shape
        kps_flat = kps.view(B * A, T, K, 3)
        motion_flat = motion_feats.view(B * A, T, -1)
        vision_flat = vision_feats.view(B * A, T, -1) if vision_feats is not None else None

        scenario_flat = scenario.unsqueeze(1).repeat(1, A).reshape(-1)
        count_flat = count.unsqueeze(1).repeat(1, A).reshape(-1)
        count = batch["count"].to(device)
        frame_labels_flat = frame_labels.unsqueeze(1).repeat(1, A, 1).reshape(-1, T)
        frame_mask_flat = frame_mask.unsqueeze(1).repeat(1, A, 1).reshape(-1, T)

        # mabe forward expects pad_mask [B, A, T] (True = padded).
        pad_mask_ba = (~frame_mask).unsqueeze(1)  # [B, 1, T]

        # Build per-sample frame_select_mask from global_masks dict.
        # Keyed by (stem, start, end) — populated once per finetune run via
        # _compute_global_masks on train/val/test loaders.
        # Strict: if any sample key is missing, raise immediately — never fall
        # back to per-batch selection (that would silently diverge from mabe
        # protocol and produce irreproducible results).
        frame_select_mask = None
        if global_masks is not None:
            masks_batch = []
            missing_keys = []
            for b in range(B):
                key = _get_sample_key(batch, b)
                m = global_masks.get(key)
                if m is None:
                    missing_keys.append(key)
                else:
                    masks_batch.append(m.to(device))
            if missing_keys:
                raise RuntimeError(
                    f"[Finetune] global_masks missing {len(missing_keys)}/{B} sample keys "
                    f"in batch {batch_idx}. First missing keys: {missing_keys[:3]}. "
                    f"This should never happen — all train/val/test windows must be "
                    f"registered during _compute_global_masks."
                )
            frame_select_mask = torch.stack(masks_batch, dim=0)  # [B, L]

        out = model(
            motion_feats,
            kps,
            pad_mask=pad_mask_ba,
            vision_feats=vision_feats,
            frame_select_mask=frame_select_mask,
        )
        # Wrap mabe's single-logits output into freely's multi-head dict form
        # so compute_seq_cls_loss keeps working verbatim.
        logits_seq = {"A": out["logits"]}
        L = logits_seq["A"].shape[1]
        cur_stride = max(1, T // L)
        # If the model applies internal temporal compression (MotionScout stride
        # blocks), downsample labels/mask to match L.
        if L != frame_labels_flat.shape[1]:
            labels_ds = frame_labels_flat[:, ::cur_stride][:, :L]
            mask_ds = frame_mask_flat[:, ::cur_stride][:, :L]
        else:
            labels_ds, mask_ds = frame_labels_flat, frame_mask_flat
        loss_sum, correct, num_samples, cts = compute_seq_cls_loss(
            logits_seq, labels_ds, mask_ds, scenario_flat, head_classes, class_weights, loss_cfg=config
        )
        cls_loss = loss_sum / max(num_samples, 1)
        for k in counts:
            counts[k] += cts[k]
        # if is_main and batch_idx == 0:
        #     was_training = model.training
        #     model.eval()
        #     with torch.no_grad():
        #         eval_logits_seq, eval_alpha, _, _, _ = model(
        #             motion_flat, kps_flat, frames_flat, pad_mask=pad_mask, vision_feats=vision_flat
        #         )
        #         eval_alpha_mean = eval_alpha.mean().item()
        #         eval_alpha_min = eval_alpha.min().item()
        #         eval_alpha_max = eval_alpha.max().item()
        #         eval_alpha_active = (eval_alpha > getattr(config, "gate_thresh", 0.5)).float().mean().item()
        #     if was_training:
        #         model.train()
        #     active_label = _get_active_ratio_label(config)
        #     print(
        #         "          [Alpha][Compare] "
        #         f"train_mean={alpha_mean:.6f} train_min={alpha_min:.6f} train_max={alpha_max:.6f} "
        #         f"train_{active_label}: {alpha_active:.6f} | "
        #         f"eval_mean={eval_alpha_mean:.6f} eval_min={eval_alpha_min:.6f} eval_max={eval_alpha_max:.6f} "
        #         f"eval_{active_label}: {eval_alpha_active:.6f}",flush=True
        #     )
        loss_full = cls_loss
        loss_scaled = loss_full / grad_acc_steps

        loss_scaled.backward()
        if (batch_idx + 1) % grad_acc_steps == 0 or (batch_idx + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        loss_meter += loss_full.item()
        cls_meter += cls_loss.item()
        correct_meter += correct
        sample_meter += num_samples

        # if batch_idx % config.log_interval == 0 and is_main:
        #     acc = correct / max(num_samples, 1) if num_samples > 0 else 0.0
        #     print(
        #         f"[Finetune] Iter {batch_idx}: loss={loss.item():.5f}, "
        #         f"cls_loss={cls_loss.item():.5f}, sparsity={sparsity_loss.item():.5f}, "
        #         f"count={count_loss.item():.5f}, acc={acc:.3f}, alpha_mean={alpha.mean().item():.4f}"
        #     )

    denom = max(len(dataloader), 1)
    avg_loss = loss_meter / denom
    acc = correct_meter / max(sample_meter, 1)
    return {
        "loss": avg_loss,
        "cls": cls_meter / denom,
        "acc": acc,
        "counts": counts,
        # Kept for log-string back-compat; these training signals were removed
        # when aligning finetune with resifuse/training/mabe.py (CE-only head-A loss).
        "sparse": 0.0,
        "count": 0.0,
        "pred": 0.0,
    }


def evaluate_finetune(
    model: ResiFuseClassifier,
    dataloader: DataLoader,
    device: torch.device,
    config,
    epoch_idx: int | None = None,
    global_masks: dict | None = None,
):
    model.eval()
    loss_meter, correct_meter, sample_meter = 0.0, 0, 0
    cls_meter, sparse_meter, count_meter = 0.0, 0.0, 0.0
    pred_meter = 0.0
    alpha_active_meter = 0.0
    alpha_mean_meter, alpha_min_meter, alpha_max_meter = 0.0, 0.0, 0.0
    gate_mean_meter, gate_min_meter, gate_max_meter = 0.0, 0.0, 0.0
    head_classes = {"A": config.head_a_classes}
    counts = {k: torch.zeros((v, 3), device=device) for k, v in head_classes.items()}  # cols: tp, fp, fn
    per_video_counts = _init_per_video_counts(head_classes, device)
    head_keys = list(head_classes.keys())
    # Independent NxN confusion matrix for head A (diagnostic only; NEVER feeds
    # into the tp/fp/fn `counts` above). conf_mat[gt, pred]. Used to inspect the
    # Supported<->Unsupported confusion at r=0% vs r=100% in the 3-class run.
    n_classes_A = int(head_classes.get("A", 0))
    conf_mat = torch.zeros((n_classes_A, n_classes_A), device=device, dtype=torch.float32)
    class_weights = getattr(config, "_head_class_weights", None)
    lambda_sparse = _get_sparse_lambda(config, epoch_idx)
    finetune_pred_weight = float(getattr(config, "finetune_pred_weight", 0.0) or 0.0)
    vision_pred_loss = str(getattr(config, "vision_pred_loss", "cosine") or "cosine").lower()
    if vision_pred_loss not in ("cosine", "mse"):
        vision_pred_loss = "cosine"
    horizons = getattr(config, "vision_pred_horizons", None) or [getattr(config, "vision_pred_horizon", 1)]
    weights = getattr(config, "vision_pred_weights", None) or [1.0 for _ in horizons]
    horizons = [int(h) for h in horizons]
    weights = [float(w) for w in weights]
    base_model = _get_base_model(model)
    # Optional inference-latency measurement (forward-only). Enable with
    # BENCH_INFER=1; zero overhead otherwise.
    _bench_infer = bool(os.environ.get("BENCH_INFER"))
    _infer_ms_total = 0.0
    _infer_windows = 0
    with torch.no_grad():
        for batch in dataloader:
            kps = batch["keypoints"].to(device)
            motion_feats = batch["motion_feats"].to(device)
            # frames skipped when dataset.load_frames=False (vision cache path).
            vision_feats = batch.get("vision_feats")
            if vision_feats is not None:
                vision_feats = vision_feats.to(device)
            labels = batch["label"].to(device)
            scenario = batch["scenario"].to(device)
            count = batch["count"].to(device)
            frame_labels = batch["frame_labels"].to(device)
            frame_mask = batch["frame_mask"].to(device)
            stems = batch.get("stem", None)

            valid_samples = frame_mask.sum(dim=1) > 0
            if not valid_samples.any():
                continue
            kps = kps[valid_samples]
            motion_feats = motion_feats[valid_samples]
            labels = labels[valid_samples]
            scenario = scenario[valid_samples]
            count = count[valid_samples]
            frame_labels = frame_labels[valid_samples]
            frame_mask = frame_mask[valid_samples]
            if vision_feats is not None:
                vision_feats = vision_feats[valid_samples]
            if stems is not None and isinstance(stems, (list, tuple)):
                mask_list = valid_samples.detach().cpu().tolist()
                stems = [s for s, keep in zip(stems, mask_list) if keep]

            B, A, T, K, _ = kps.shape
            kps_flat = kps.view(B * A, T, K, 3)
            motion_flat = motion_feats.view(B * A, T, -1)
            vision_flat = vision_feats.view(B * A, T, -1) if vision_feats is not None else None
            labels_flat = labels.unsqueeze(1).repeat(1, A).reshape(-1)
            scenario_flat = scenario.unsqueeze(1).repeat(1, A).reshape(-1)
            count_flat = count.unsqueeze(1).repeat(1, A).reshape(-1)
            frame_labels_flat = frame_labels.unsqueeze(1).repeat(1, A, 1).reshape(-1, T)
            frame_mask_flat = frame_mask.unsqueeze(1).repeat(1, A, 1).reshape(-1, T)
            stems_flat = None
            if stems is not None:
                stems_flat = []
                for stem in stems:
                    s_val = stem if isinstance(stem, str) else str(stem)
                    stems_flat.extend([s_val] * A)

            pad_mask_ba = (~frame_mask).unsqueeze(1)  # [B, 1, T]

            frame_select_mask = None
            if global_masks is not None:
                masks_batch = []
                missing_keys = []
                for b in range(B):
                    key = _get_sample_key(batch, b)
                    m = global_masks.get(key)
                    if m is None:
                        missing_keys.append(key)
                    else:
                        masks_batch.append(m.to(device))
                if missing_keys:
                    raise RuntimeError(
                        f"[Eval] global_masks missing {len(missing_keys)}/{B} sample keys. "
                        f"First missing keys: {missing_keys[:3]}. This should never happen — "
                        f"all eval windows must be registered during _compute_global_masks."
                    )
                frame_select_mask = torch.stack(masks_batch, dim=0)

            if _bench_infer and torch.cuda.is_available():
                torch.cuda.synchronize()
            _fwd_t0 = time.perf_counter() if _bench_infer else 0.0
            out = model(
                motion_feats,
                kps,
                pad_mask=pad_mask_ba,
                vision_feats=vision_feats,
                frame_select_mask=frame_select_mask,
            )
            if _bench_infer:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _infer_ms_total += (time.perf_counter() - _fwd_t0) * 1e3
                _infer_windows += int(motion_feats.shape[0])
            logits_seq = {"A": out["logits"]}
            L = logits_seq["A"].shape[1]
            cur_stride = max(1, T // L)
            if L != frame_labels_flat.shape[1]:
                labels_ds = frame_labels_flat[:, ::cur_stride][:, :L]
                mask_ds = frame_mask_flat[:, ::cur_stride][:, :L]
            else:
                labels_ds, mask_ds = frame_labels_flat, frame_mask_flat
            loss_sum, correct, num_samples, cts = compute_seq_cls_loss(
                logits_seq, labels_ds, mask_ds, scenario_flat, head_classes, class_weights, loss_cfg=config
            )
            cls_loss = loss_sum / max(num_samples, 1)
            loss = cls_loss

            loss_meter += loss.item()
            cls_meter += cls_loss.item()
            correct_meter += correct
            sample_meter += num_samples
            for k in counts:
                counts[k] += cts[k]
            # Confusion matrix accumulation (independent of the F1 counts above).
            if n_classes_A > 0 and "A" in logits_seq:
                preds_A = logits_seq["A"].argmax(dim=-1)          # [B, L]
                vmask = mask_ds.bool() & (labels_ds >= 0)
                gt = labels_ds[vmask].long()
                pr = preds_A[vmask].long()
                if gt.numel() > 0:
                    flat = gt * n_classes_A + pr
                    conf_mat += torch.bincount(
                        flat, minlength=n_classes_A * n_classes_A
                    ).reshape(n_classes_A, n_classes_A).float()
            if stems_flat is not None:
                for idx in range(labels_ds.shape[0]):
                    head_idx = int(scenario_flat[idx].item())
                    if head_idx < 0 or head_idx >= len(head_keys):
                        continue
                    hk = head_keys[head_idx]
                    if hk not in per_video_counts:
                        continue
                    stem_name = stems_flat[idx] if idx < len(stems_flat) else "unknown"
                    logits_b = logits_seq[hk][idx]
                    labels_b = labels_ds[idx]
                    mask_b = mask_ds[idx]
                    valid = mask_b & (labels_b >= 0)
                    if not valid.any():
                        continue
                    preds_b = logits_b[valid].argmax(dim=-1)
                    labels_valid = labels_b[valid]
                    counts_tens = per_video_counts[hk][stem_name]
                    for c in range(head_classes[hk]):
                        tp = ((preds_b == c) & (labels_valid == c)).sum()
                        fp = ((preds_b == c) & (labels_valid != c)).sum()
                        fn = ((preds_b != c) & (labels_valid == c)).sum()
                        counts_tens[c, 0] += tp
                        counts_tens[c, 1] += fp
                        counts_tens[c, 2] += fn
    denom = max(len(dataloader), 1)
    avg_loss = loss_meter / denom
    if _bench_infer and is_main_process(getattr(config, "rank", 0)) and _infer_windows > 0:
        ms_per_window = _infer_ms_total / _infer_windows
        print(f"[InferTiming] model_forward_only: {_infer_windows} windows, "
              f"{ms_per_window:.2f} ms/window, budget={getattr(config, 'budget_ratio', '?')}", flush=True)

    acc = correct_meter / max(sample_meter, 1)
    per_video_counts = _gather_per_video_counts(per_video_counts, getattr(config, "distributed", False), getattr(config, "world_size", 1))
    if getattr(config, "distributed", False) and int(getattr(config, "world_size", 1)) > 1:
        dist.all_reduce(conf_mat, op=dist.ReduceOp.SUM)
    return {
        "loss": avg_loss,
        "cls": cls_meter / denom,
        "acc": acc,
        "counts": counts,
        "confusion_matrix": conf_mat,
        "per_video_counts": per_video_counts,
        # Back-compat zero keys (features dropped when aligning with mabe).
        "sparse": 0.0,
        "count": 0.0,
        "pred": 0.0,
    }


def run_pretrain(args):
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    include_velocity = getattr(args, "motion_include_velocity", True)
    include_confidence = getattr(args, "motion_include_confidence", True)

    device = torch.device(args.device)
    is_main = is_main_process(args.rank)
    if is_main:
        print(f"🚀 Step1 Pre-train | device={device} | mask_ratio={args.mask_ratio}")
    if float(getattr(args, "vision_pred_weight", 0.0) or 0.0) > 0.0:
        if not getattr(args, "use_vision_cache", False):
            raise RuntimeError("vision_pred_weight > 0 requires use_vision_cache=true.")
        if not getattr(args, "vision_cache_dir", None):
            raise RuntimeError("vision_pred_weight > 0 requires vision_cache_dir.")
    args.save_dir = _build_run_dir(args, stage="pretrain")
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tb")) if is_main else None

    vision_feat_dir = getattr(args, "vision_cache_dir", None)
    if is_main:
        print(f"[Pretrain] vision_feat_dir={vision_feat_dir}")
    train_ds = FreelyMovingMouseDataset(
        root_dir=args.data_root,
        seq_len=args.seq_len,
        ds_stride=getattr(args, "ds_stride", 1),
        frame_resize=getattr(args, "frame_resize", None),
        frame_resize_align=getattr(args, "frame_resize_align", 32),
        pretrain_stride=args.pretrain_stride,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stage="pretrain",
        dtype=args.torch_dtype,
        max_neighbors=args.max_neighbors,
        use_neighbor_feats=args.use_neighbor_feats,
        num_joints=args.num_joints,
        conf_mask_thresh=args.conf_mask_thresh,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        oversample_grooming_factor=1,
        vision_feat_dir=vision_feat_dir,
        use_vision_cache=getattr(args, "use_vision_cache", False),
        use_motion_scaler=getattr(args, "use_motion_scaler", False),
        marker_subject_centric=getattr(args, "marker_subject_centric", True),
        motion_include_velocity=include_velocity,
        motion_include_confidence=include_confidence,
    )
    motion_scaler = train_ds.motion_scaler if getattr(args, "use_motion_scaler", False) else None
    val_ds = FreelyMovingMouseDataset(
        root_dir=args.data_root,
        seq_len=args.seq_len,
        ds_stride=getattr(args, "ds_stride", 1),
        frame_resize=getattr(args, "frame_resize", None),
        frame_resize_align=getattr(args, "frame_resize_align", 32),
        pretrain_stride=args.pretrain_stride,
        split="val",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stage="pretrain",
        dtype=args.torch_dtype,
        max_neighbors=args.max_neighbors,
        use_neighbor_feats=args.use_neighbor_feats,
        num_joints=args.num_joints,
        conf_mask_thresh=args.conf_mask_thresh,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        oversample_grooming_factor=1,
        vision_feat_dir=vision_feat_dir,
        use_vision_cache=getattr(args, "use_vision_cache", False),
        use_motion_scaler=getattr(args, "use_motion_scaler", False),
        motion_scaler=motion_scaler,
        marker_subject_centric=getattr(args, "marker_subject_centric", True),
        motion_include_velocity=include_velocity,
        motion_include_confidence=include_confidence,
    )
    test_ds = FreelyMovingMouseDataset(
        root_dir=args.data_root,
        seq_len=args.seq_len,
        ds_stride=getattr(args, "ds_stride", 1),
        frame_resize=getattr(args, "frame_resize", None),
        frame_resize_align=getattr(args, "frame_resize_align", 32),
        pretrain_stride=args.pretrain_stride,
        split="test",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stage="pretrain",
        dtype=args.torch_dtype,
        max_neighbors=args.max_neighbors,
        use_neighbor_feats=args.use_neighbor_feats,
        num_joints=args.num_joints,
        conf_mask_thresh=args.conf_mask_thresh,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        oversample_grooming_factor=1,
        vision_feat_dir=vision_feat_dir,
        use_vision_cache=getattr(args, "use_vision_cache", False),
        use_motion_scaler=getattr(args, "use_motion_scaler", False),
        motion_scaler=motion_scaler,
        marker_subject_centric=getattr(args, "marker_subject_centric", True),
        motion_include_velocity=include_velocity,
        motion_include_confidence=include_confidence,
    )
    args.num_joints = train_ds.num_joints
    neighbor_dim = (args.max_neighbors * 4) if args.use_neighbor_feats else 0
    vel_dim = args.num_joints * 2 if include_velocity else 0
    base_dim = args.num_joints * (3 if include_confidence else 2)
    args.motion_input_dim = base_dim + vel_dim + neighbor_dim

    train_sampler = DistributedSampler(train_ds, num_replicas=args.world_size, rank=args.rank, shuffle=True) if args.distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=args.world_size, rank=args.rank, shuffle=False) if args.distributed else None

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

    if is_main:
        def _print_pretrain_split_stats(name: str, ds):
            num_videos = len(ds.samples) if hasattr(ds, "samples") else 0
            print(
                f"[Pretrain] Split={name} | videos={num_videos} | windows={len(ds)} | "
                f"seq_len={getattr(ds, 'seq_len', 'na')} | ds_stride={getattr(ds, 'ds_stride', 'na')} | "
                f"pretrain_stride={getattr(ds, 'pretrain_stride', 'na')}"
            )
            stems = []
            if hasattr(ds, "samples"):
                for s in ds.samples:
                    stem = s.get("stem") if isinstance(s, dict) else None
                    if stem:
                        stems.append(stem)
            if stems:
                stems_sorted = sorted(stems)
                print(f"[Pretrain] Split={name} video stems: {', '.join(stems_sorted)}")
            else:
                print(f"[Pretrain] Split={name} video stems: none")
        _print_pretrain_split_stats("train", train_ds)
        _print_pretrain_split_stats("val", val_ds)
        _print_pretrain_split_stats("test", test_ds)

    args.motion_input_dim = base_dim + vel_dim + neighbor_dim
    model = ResiFusePretrainer(args).to(device).to(dtype=args.torch_dtype)
    if getattr(args, "torch_compile", True):
        print("enable torch compile!")
        torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(model, mode="default", backend="inductor")
        # if not torch._dynamo.config.capture_scalar_outputs:
        #     torch._dynamo.config.capture_scalar_outputs = True
    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank] if device.type == "cuda" else None,
            output_device=args.local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    base_model = model.module if isinstance(model, DDP) else model
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
        epoch_start = time.perf_counter()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        _maybe_set_visual_epoch(model, epoch)
        if is_main:
            print(f"\n=== [Pretrain] Epoch {epoch + 1}/{args.epochs} ===")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _pt_t0 = time.perf_counter()
        train_stats = train_one_epoch_pretrain(model, train_loader, optimizer, device, args, epoch_idx=epoch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if is_main:
            print(f"[Timing] pretrain_epoch={epoch + 1} train_step_sec={time.perf_counter() - _pt_t0:.2f}", flush=True)
        val_start = time.perf_counter()
        val_stats = evaluate_pretrain(model, val_loader, device, args, epoch_idx=epoch)
        val_loss = val_stats["loss"]
        val_time = time.perf_counter() - val_start

        scheduler.step()
        if args.distributed:
            horizons = getattr(args, "vision_pred_horizons", None) or [getattr(args, "vision_pred_horizon", 1)]
            horizons = [int(h) for h in horizons]
            train_tensor = torch.tensor(
                [
                    train_stats["loss"],
                    train_stats["mask"],
                    train_stats["pred"],
                    train_stats["vq_recon"],
                    train_stats["vq"],
                    *[train_stats.get("pred_h", {}).get(h, 0.0) for h in horizons],
                ],
                device=device,
            )
            dist.all_reduce(train_tensor, op=dist.ReduceOp.SUM)
            train_tensor = train_tensor / args.world_size
            pred_h = {h: train_tensor[5 + idx].item() for idx, h in enumerate(horizons)}
            train_stats = {
                "loss": train_tensor[0].item(),
                "mask": train_tensor[1].item(),
                "pred": train_tensor[2].item(),
                "vq_recon": train_tensor[3].item(),
                "vq": train_tensor[4].item(),
                "pred_h": pred_h,
                "avg_fetch": train_stats.get("avg_fetch", 0.0),
                "avg_compute": train_stats.get("avg_compute", 0.0),
                "timing": train_stats.get("timing", {}),
                "timing_total": train_stats.get("timing_total", {}),
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
        # track best validation and save best checkpoint (use aggregated val loss when distributed)
        updated_best = False
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            updated_best = True
        epoch_time = time.perf_counter() - epoch_start
        if is_main:
            lines = [
                "[Pretrain] Epoch "
                f"{epoch + 1} | train_loss={train_stats['loss']:.6f} "
                f"(mask={train_stats['mask']:.6f}, pred={train_stats['pred']:.6f}, "
                f"vq_recon={train_stats['vq_recon']:.6f}, vq={train_stats['vq']:.6f}) | "
                f"val_loss={val_loss:.6f} "
                f"(mask={val_stats['mask']:.6f}, pred={val_stats['pred']:.6f}, "
                f"vq_recon={val_stats['vq_recon']:.6f}, vq={val_stats['vq']:.6f})",
                f"          avg_fetch={train_stats.get('avg_fetch', 0.0):.4f}s | avg_compute={train_stats.get('avg_compute', 0.0):.4f}s | "
                f"total_fetch={train_stats.get('total_fetch', 0.0):.1f}s | total_compute={train_stats.get('total_compute', 0.0):.1f}s | "
                f"val_time={val_time:.1f}s | epoch_time={epoch_time:.1f}s",
            ]
            pred_h_train = train_stats.get("pred_h", {})
            pred_h_val = val_stats.get("pred_h", {})
            if pred_h_train:
                train_parts = " | ".join(f"h{h}={pred_h_train[h]:.6f}" for h in sorted(pred_h_train.keys()))
                val_parts = " | ".join(f"h{h}={pred_h_val.get(h, 0.0):.6f}" for h in sorted(pred_h_train.keys()))
                lines.append(f"          pred_h: train[{train_parts}] val[{val_parts}]")
            
            if train_stats.get("timing_total"):
                total_str = ", ".join(f"{k}={v:.4f}s" for k, v in sorted(train_stats["timing_total"].items()))
                lines.append(f"          Module total (epoch): {total_str}")
            if train_stats.get("timing"):
                timing_str = ", ".join(f"{k}={v:.4f}s" for k, v in sorted(train_stats["timing"].items()))
                lines.append(f"          Module avg/batch: {timing_str}")
            # Codebook health snapshot (EMA-smoothed, from DDP-aggregated buffers).
            # Watches for RVQ collapse: `used` << K or `effective_K` near 1 means
            # only a tiny fraction of entries carry probability mass. `top10_frac`
            # close to 1.0 means ~10 entries dominate. `norm_entropy` close to 0
            # is equally bad.
            try:
                base_pt = _get_base_model(model)
                if hasattr(base_pt, "motion_scout") and hasattr(base_pt.motion_scout, "usage_stats"):
                    cb_stats = base_pt.motion_scout.usage_stats()
                    for qname in ("q1", "q2"):
                        s = cb_stats.get(qname)
                        if not s:
                            continue
                        lines.append(
                            f"          [Codebook/{qname}] "
                            f"used={s['used']}/{s['K']} "
                            f"eff_K={s['effective_K']:.1f} "
                            f"norm_H={s['norm_entropy']:.3f} "
                            f"top10_frac={s['top10_frac']:.3f} "
                            f"top50_frac={s['top50_frac']:.3f} "
                            f"mean_norm={s['mean_norm']:.3f} "
                            f"min_inter_d={s['min_inter_dist']:.3f}"
                        )
                        writer.add_scalar(f"pretrain/codebook_{qname}_used", s["used"], epoch + 1)
                        writer.add_scalar(f"pretrain/codebook_{qname}_effective_K", s["effective_K"], epoch + 1)
                        writer.add_scalar(f"pretrain/codebook_{qname}_norm_entropy", s["norm_entropy"], epoch + 1)
                        writer.add_scalar(f"pretrain/codebook_{qname}_top10_frac", s["top10_frac"], epoch + 1)
            except Exception as _cb_err:
                lines.append(f"          [Codebook] usage_stats unavailable: {_cb_err}")
            print("\n".join(lines))
            writer.add_scalar("pretrain/train_loss", train_stats["loss"], epoch + 1)
            writer.add_scalar("pretrain/train_mask", train_stats["mask"], epoch + 1)
            writer.add_scalar("pretrain/train_pred", train_stats["pred"], epoch + 1)
            writer.add_scalar("pretrain/train_vq_recon", train_stats["vq_recon"], epoch + 1)
            writer.add_scalar("pretrain/train_vq", train_stats["vq"], epoch + 1)
            for h, val in train_stats.get("pred_h", {}).items():
                writer.add_scalar(f"pretrain/train_pred_h{h}", val, epoch + 1)
            writer.add_scalar("pretrain/avg_fetch", train_stats.get("avg_fetch", 0.0), epoch + 1)
            writer.add_scalar("pretrain/avg_compute", train_stats.get("avg_compute", 0.0), epoch + 1)
            for name, tval in train_stats.get("timing", {}).items():
                writer.add_scalar(f"pretrain/timing_{name}", tval, epoch + 1)
            writer.add_scalar("pretrain/val_loss", val_loss, epoch + 1)
            writer.add_scalar("pretrain/val_mask", val_stats["mask"], epoch + 1)
            writer.add_scalar("pretrain/val_pred", val_stats["pred"], epoch + 1)
            writer.add_scalar("pretrain/val_vq_recon", val_stats["vq_recon"], epoch + 1)
            writer.add_scalar("pretrain/val_vq", val_stats["vq"], epoch + 1)
            for h, val in val_stats.get("pred_h", {}).items():
                writer.add_scalar(f"pretrain/val_pred_h{h}", val, epoch + 1)
            writer.add_scalar("pretrain/best_val_loss", best_val if best_val is not None else float("nan"), epoch + 1)
            if ((epoch + 1) % getattr(args, "pretrain_save_interval", 1) == 0) or (epoch + 1 == args.epochs):
                _save_checkpoint("pretrain", model, optimizer, scheduler, best_val, best_epoch, epoch, args.save_dir, "resifuse_pretrain")
            if updated_best:
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
                    add_epoch_suffix=True,
                )
                _cleanup_old_best(args.save_dir, "best_pretrain", keep_path=best_path)
        _maybe_flush_visual_epoch(model)
        if args.distributed:
            dist.barrier()
    if writer:
        writer.close()


# def maybe_load_pretrain_weights(model: ResiFuseClassifier, ckpt_path: str, device: torch.device):
#     if ckpt_path is None:
#         return
#     if not os.path.exists(ckpt_path):
#         print(f"Pretrain checkpoint not found at {ckpt_path}, skipping warm-start.")
#         return
#     state = torch.load(ckpt_path, map_location=device)
#     if isinstance(state, dict) and "model" in state:
#         state = state["model"]
#     missing, unexpected = model.load_state_dict(state, strict=False)
#     print(f"Loaded pretrain ckpt with missing keys: {missing}, unexpected keys: {unexpected}")

def maybe_load_pretrain_weights(model: ResiFuseClassifier, ckpt_path: str, device: torch.device):
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
    # resized = {}
    # for key, value in list(state.items()):
    #     if key not in model_state:
    #         continue
    #     target = model_state[key]
    #     if value.shape == target.shape:
    #         continue
    #     if value.ndim == 3 and target.ndim == 3 and value.shape[0] == target.shape[0] and value.shape[2] == target.shape[2]:
    #         src_len = value.shape[1]
    #         tgt_len = target.shape[1]
    #         # Resize temporal positional embeddings to match finetune seq_len.
    #         value_resized = F.interpolate(
    #             value.permute(0, 2, 1),
    #             size=tgt_len,
    #             mode="linear",
    #             align_corners=False,
    #         ).permute(0, 2, 1)
    #         state[key] = value_resized
    #         resized[key] = (src_len, tgt_len)
    # if resized:
    #     for name, (src_len, tgt_len) in resized.items():
    #         print(f"Resized {name} from len={src_len} to len={tgt_len} for finetune.")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded pretrain ckpt with missing keys: {missing}, unexpected keys: {unexpected}")

def run_finetune(args):
    # 3-class Open-Field diagnostic: head A must output freely_num_classes logits.
    # Enforce head_a_classes == freely_num_classes so the model head, loss, and F1
    # all agree (the model builds social/vision cls heads from config.head_a_classes).
    _fn_classes = int(getattr(args, "freely_num_classes", 2))
    if _fn_classes >= 2 and int(getattr(args, "head_a_classes", 2)) != _fn_classes:
        print(f"[Finetune] freely_num_classes={_fn_classes} -> forcing head_a_classes "
              f"{getattr(args, 'head_a_classes', None)} -> {_fn_classes}")
        args.head_a_classes = _fn_classes
    train_ratio = args.finetune_train_ratio if args.finetune_train_ratio is not None else args.train_ratio
    val_ratio = args.finetune_val_ratio if args.finetune_val_ratio is not None else args.val_ratio
    test_ratio = args.finetune_test_ratio if args.finetune_test_ratio is not None else args.test_ratio
    include_velocity = getattr(args, "motion_include_velocity", True)
    include_confidence = getattr(args, "motion_include_confidence", True)

    device = torch.device(args.device)
    is_main = is_main_process(args.rank)
    if float(getattr(args, "finetune_pred_weight", 0.0) or 0.0) > 0.0:
        if not getattr(args, "use_vision_cache", False):
            raise RuntimeError("finetune_pred_weight > 0 requires use_vision_cache=true.")
        if not getattr(args, "vision_cache_dir", None):
            raise RuntimeError("finetune_pred_weight > 0 requires vision_cache_dir.")
    if is_main:
        print(f"🚀 Step2 Finetuning | device={device} | lambda_sparse={args.lambda_sparse}")
    args.save_dir = _build_run_dir(args, stage="finetune")
    loss_type = (getattr(args, "loss_type", "ce") or "ce").lower()
    if is_main:
        loss_parts = [f"type={loss_type}"]
        if loss_type == "focal":
            loss_parts.append(f"gamma={getattr(args, 'focal_gamma', None)}")
            loss_parts.append(f"alpha={getattr(args, 'focal_alpha', None)}")
        print(f"[Finetune] Loss config: " + " | ".join(loss_parts))
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tb")) if is_main else None

    head_classes = {"A": args.head_a_classes}
    active_heads_cfg = getattr(args, "finetune_active_heads", ["A"]) or ["A"]
    active_heads = [h for h in active_heads_cfg if h in head_classes]
    if not active_heads:
        active_heads = ["A"]
    if is_main:
        print(f"[Finetune] Active heads for logging: {active_heads}")
    train_ds = FreelyMovingMouseDataset(
        root_dir=args.data_root,
        seq_len=args.seq_len,
        ds_stride=getattr(args, "ds_stride", 1),
        frame_resize=getattr(args, "frame_resize", None),
        frame_resize_align=getattr(args, "frame_resize_align", 32),
        pretrain_stride=args.pretrain_stride,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stage="finetune",
        freely_num_classes=int(getattr(args, "freely_num_classes", 2)),
        dtype=args.torch_dtype,
        max_neighbors=args.max_neighbors,
        use_neighbor_feats=args.use_neighbor_feats,
        num_joints=args.num_joints,
        conf_mask_thresh=args.conf_mask_thresh,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        oversample_grooming_factor=getattr(args, "oversample_grooming_factor", 1),
        vision_feat_dir=getattr(args, "vision_cache_dir", None),
        use_vision_cache=getattr(args, "use_vision_cache", False),
        use_motion_scaler=getattr(args, "use_motion_scaler", False),
        drop_empty_frame_mask=False,
        marker_subject_centric=getattr(args, "marker_subject_centric", True),
        motion_include_velocity=include_velocity,
        motion_include_confidence=include_confidence,
        load_frames=not bool(getattr(args, "use_vision_cache", False)),
    )
    motion_scaler = train_ds.motion_scaler if getattr(args, "use_motion_scaler", False) else None
    val_ds = FreelyMovingMouseDataset(
        root_dir=args.data_root,
        seq_len=args.seq_len,
        ds_stride=getattr(args, "ds_stride", 1),
        frame_resize=getattr(args, "frame_resize", None),
        frame_resize_align=getattr(args, "frame_resize_align", 32),
        pretrain_stride=args.pretrain_stride,
        split="val",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stage="finetune",
        freely_num_classes=int(getattr(args, "freely_num_classes", 2)),
        dtype=args.torch_dtype,
        max_neighbors=args.max_neighbors,
        use_neighbor_feats=args.use_neighbor_feats,
        num_joints=args.num_joints,
        conf_mask_thresh=args.conf_mask_thresh,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        oversample_grooming_factor=1,
        vision_feat_dir=getattr(args, "vision_cache_dir", None),
        use_vision_cache=getattr(args, "use_vision_cache", False),
        use_motion_scaler=getattr(args, "use_motion_scaler", False),
        motion_scaler=motion_scaler,
        drop_empty_frame_mask=False,
        marker_subject_centric=getattr(args, "marker_subject_centric", True),
        motion_include_velocity=include_velocity,
        motion_include_confidence=include_confidence,
        load_frames=not bool(getattr(args, "use_vision_cache", False)),
    )
    test_ds = FreelyMovingMouseDataset(
        root_dir=args.data_root,
        seq_len=args.seq_len,
        ds_stride=getattr(args, "ds_stride", 1),
        frame_resize=getattr(args, "frame_resize", None),
        frame_resize_align=getattr(args, "frame_resize_align", 32),
        pretrain_stride=args.pretrain_stride,
        split="test",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stage="finetune",
        freely_num_classes=int(getattr(args, "freely_num_classes", 2)),
        dtype=args.torch_dtype,
        max_neighbors=args.max_neighbors,
        use_neighbor_feats=args.use_neighbor_feats,
        num_joints=args.num_joints,
        conf_mask_thresh=args.conf_mask_thresh,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        oversample_grooming_factor=1,
        vision_feat_dir=getattr(args, "vision_cache_dir", None),
        use_vision_cache=getattr(args, "use_vision_cache", False),
        use_motion_scaler=getattr(args, "use_motion_scaler", False),
        motion_scaler=motion_scaler,
        drop_empty_frame_mask=False,
        marker_subject_centric=getattr(args, "marker_subject_centric", True),
        motion_include_velocity=include_velocity,
        motion_include_confidence=include_confidence,  
        load_frames=not bool(getattr(args, "use_vision_cache", False)),
    )
    args._head_class_weights = _compute_class_weights(train_ds, head_classes, device) if getattr(args, "use_class_weights", False) else None
    args.num_joints = train_ds.num_joints
    neighbor_dim = (args.max_neighbors * 4) if args.use_neighbor_feats else 0
    vel_dim = args.num_joints * 2 if include_velocity else 0
    base_dim = args.num_joints * (3 if include_confidence else 2)
    args.motion_input_dim = base_dim + vel_dim + neighbor_dim

    train_sampler = DistributedSampler(train_ds, num_replicas=args.world_size, rank=args.rank, shuffle=True) if args.distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=args.world_size, rank=args.rank, shuffle=False) if args.distributed else None
    test_sampler = DistributedSampler(test_ds, num_replicas=args.world_size, rank=args.rank, shuffle=False) if args.distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_eval_sampler = DistributedSampler(train_ds, num_replicas=args.world_size, rank=args.rank, shuffle=False) if args.distributed else None
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
    if is_main:
        weights_a = getattr(args, "_head_class_weights", None)
        weights_a = weights_a.get("A") if weights_a else None
        _print_dataset_stats("Train Split", train_ds, weights_a=weights_a)
        if val_loader:
            _print_dataset_stats("Val Split", val_ds)
        else:
            print("Val Split -> no labeled frames (val_fraction=0 or disabled)")
        _print_dataset_stats("Test Split", test_ds)
    model = ResiFuseClassifier(args).to(device).to(dtype=args.torch_dtype)
    _configure_finetune_trainables(model, args=args, is_main=is_main)
    if getattr(args, "torch_compile", True):
        print("enable torch compile!")
        torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(model, mode="default", backend="inductor")
        # if not torch._dynamo.config.capture_scalar_outputs:
        #     torch._dynamo.config.capture_scalar_outputs = True
    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank] if device.type == "cuda" else None,
            output_device=args.local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    maybe_load_pretrain_weights(model, args.pretrain_ckpt, device)

    def _build_finetune_optimizer(mod, lr):
        # All trainable modules use finetune_lr (the pretrained RVQ encoder and
        # middle modules are frozen by _configure_finetune_trainables).
        base_model = mod.module if hasattr(mod, "module") else mod
        base_model = getattr(base_model, "_orig_mod", base_model)
        weight_decay = getattr(args, "weight_decay", 1e-2)
        decay, nodecay = [], []
        for name, p in base_model.named_parameters():
            if not p.requires_grad:
                continue
            is_no_decay = ("bias" in name.lower()) or ("norm" in name.lower())
            (nodecay if is_no_decay else decay).append(p)
        param_groups = []
        if decay:
            param_groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
        if nodecay:
            param_groups.append({"params": nodecay, "lr": lr, "weight_decay": 0.0})
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

    total_epochs = int(args.epochs)
    vision_alpha_cfg = float(getattr(args, "vision_alpha", 1.0))
    budget_ratio = float(getattr(args, "budget_ratio", 0.0) or 0.0)
    lambda_col = float(getattr(args, "lambda_col", 0.0) or 0.0)
    collision_temperature = float(getattr(args, "collision_temperature", 0.1) or 0.1)
    if is_main:
        print(f"\n{'=' * 60}")
        if budget_ratio <= 0:
            print(f"  Fine-tuning ({total_epochs} epochs, pure motion baseline)")
        else:
            print(f"  Fine-tuning ({total_epochs} epochs)")
        print("  Phase 1: skipped")
        print("  Phase 2: skipped")
        print(f"  Phase 3: joint epochs {start_epoch + 1}-{total_epochs}")
        print(f"  frame_select_mode={getattr(args, 'frame_select_mode', 'residual')}")
        print(f"  budget_ratio={budget_ratio}, vision_alpha={vision_alpha_cfg}")
        print(f"  lambda_col={lambda_col}, collision_temperature={collision_temperature}")
        print(f"{'=' * 60}")

    # Track per-epoch test F1 for head A (list of [rearing_f1, grooming_f1]).
    test_f1_history_A: "list[list[float]]" = []
    val_macro_history_A: "list[float]" = []  # head-A val macro F1, for best-val epoch selection

    # ========================================================================
    # Global frame-selection masks (ported verbatim from resifuse/training/mabe.py).
    # With budget_ratio=0.25 + frame_select_mode=residual, we compute a
    # dataset-level threshold once at epoch 0 and reuse it every epoch.
    # Freely has no Phase 1/Phase 2, so all modes can run at epoch 0.
    # ========================================================================
    base_model = _get_base_model(model)
    frame_select_mode = str(getattr(args, "frame_select_mode", "residual")).lower()
    global_modes = {"uncertainty", "residual", "recon_error", "surprise", "online"}
    use_global_masks = (
        frame_select_mode in global_modes
        and 0 < budget_ratio < 1.0
    )

    def _compute_global_masks(base_model, dataloader, device, budget_ratio, mode="residual"):
        """Compute global frame selection masks for all windows.

        Port of resifuse/training/mabe.py::_compute_global_masks. The only adaptation
        for freely is the per-sample key — we use (stem, start, end) instead of
        mabe's (video_id, mouse_id, window_idx) since freely is single-mouse.

        Modes:
          - uncertainty: select frames with lowest motion |logit|
          - residual:    select frames with highest RVQ quant error ‖z_e - z_q‖
          - recon_error: select frames with highest VQ-VAE reconstruction error
          - surprise:    select frames with highest temporal z_q change
          - online:      binary entropy (needs trained motion head)
        """
        saved_alpha = getattr(base_model, "vision_alpha", 0.0)
        base_model.vision_alpha = 0.0
        base_model.eval()

        all_scores = {}
        n_batches = 0
        for batch in dataloader:
            n_batches += 1
            kps = batch["keypoints"].to(device)
            motion_feats = batch["motion_feats"].to(device)
            B, A, T = motion_feats.shape[:3]
            motion_flat = motion_feats.view(B * A, T, -1)

            scout_out = base_model.motion_scout(motion_flat)
            z_q = scout_out["z_q"]
            z_e = scout_out["z_e"]
            L = z_q.shape[1]

            if mode == "uncertainty":
                frame_mask = batch.get("frame_mask")
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
                vfeats = batch.get("vision_feats")
                if vfeats is not None:
                    vfeats = vfeats.to(device)
                out = base_model(motion_feats, kps, pad_mask=pad_mask, vision_feats=vfeats)
                scores = -out["logits"].abs().mean(dim=-1)  # [B, L]
            elif mode == "residual":
                r2 = (z_e - z_q).norm(dim=-1)           # [B*A, L]
                scores = r2.view(B, A, L).mean(dim=1)   # [B, L]
            elif mode == "recon_error":
                recon = base_model.motion_scout.vqvae.decode(z_q, target_len=T)
                err = (recon - motion_flat).pow(2).mean(dim=-1)
                if err.shape[1] != L:
                    stride_re = max(1, err.shape[1] // L)
                    err = err[:, ::stride_re][:, :L]
                scores = err.view(B, A, L).mean(dim=1)
            elif mode == "surprise":
                diff = (z_q[:, 1:] - z_q[:, :-1]).norm(dim=-1)
                surprise = torch.cat([diff[:, :1] * 0, diff], dim=1)
                scores = surprise.view(B, A, L).mean(dim=1)
            elif mode == "online":
                frame_mask = batch.get("frame_mask")
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
                vfeats = batch.get("vision_feats")
                if vfeats is not None:
                    vfeats = vfeats.to(device)
                out = base_model(motion_feats, kps, pad_mask=pad_mask, vision_feats=vfeats)
                probs = torch.sigmoid(out["logits"])
                H = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
                scores = H.mean(dim=-1)
            else:
                raise ValueError(f"Unknown global mask mode: {mode}")

            for b in range(B):
                key = _get_sample_key(batch, b)
                all_scores[key] = scores[b].cpu()

        all_keys = list(all_scores.keys())

        # === Per-slot bias correction (removes CNN boundary artifact) ===
        # See resifuse/training/mabe.py for rationale. Subtracts per-slot mean so
        # edge slots don't systematically win top-budget_ratio selection.
        if len(all_keys) > 1:
            score_stack = torch.stack([all_scores[k] for k in all_keys], dim=0)  # [W, L]
            slot_bias = score_stack.mean(dim=0)  # [L]
            for key in all_keys:
                all_scores[key] = all_scores[key] - slot_bias
            if is_main:
                print(
                    f"  [_compute_global_masks] per-slot bias correction: "
                    f"mean={slot_bias.mean().item():.4f}  "
                    f"range=[{slot_bias.min().item():.4f}, {slot_bias.max().item():.4f}]",
                    flush=True,
                )

        all_vals = torch.cat([all_scores[k] for k in all_keys])
        total_frames = all_vals.numel()
        total_select = max(1, int(total_frames * budget_ratio))
        threshold = all_vals.topk(total_select).values[-1].item()

        masks = {}
        for key in all_keys:
            masks[key] = (all_scores[key] >= threshold)

        base_model.vision_alpha = saved_alpha
        base_model.train()

        actual_selected = sum(m.sum().item() for m in masks.values())
        if is_main:
            print(
                f"  [_compute_global_masks] mode={mode}, {n_batches} batches, "
                f"{len(masks)} windows, global threshold={threshold:.4f}, "
                f"selected {int(actual_selected)}/{total_frames} frames "
                f"({actual_selected / total_frames * 100:.1f}%)",
                flush=True,
            )
        return masks

    # One-time global masks computation (freely runs all modes at epoch 0).
    global_masks = None
    if use_global_masks:
        if is_main:
            print(f"\n[Global Masks] Computing {frame_select_mode} masks (budget={budget_ratio})...")
        global_masks = {}
        # Note: train_eval_loader is used for the train-split mask since it
        # iterates the train set in eval mode (shuffle=False) for reproducibility.
        for loader_name, loader in [("train", train_eval_loader), ("val", val_loader), ("test", test_loader)]:
            masks = _compute_global_masks(base_model, loader, device, budget_ratio, mode=frame_select_mode)
            global_masks.update(masks)
            if is_main:
                n_masks = len(masks)
                avg_sel = sum(m.sum().item() for m in masks.values()) / max(n_masks, 1)
                print(f"  {loader_name}: {n_masks} windows, avg selected frames={avg_sel:.1f}")

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main:
            print(f"\n=== [Finetune] Epoch {epoch + 1}/{args.epochs} ===")
        cw = getattr(args, "_head_class_weights", None)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _ep_t0 = time.perf_counter()
        train_stats = train_one_epoch_finetune(
            model, train_loader, optimizer, device, args,
            class_weights=cw, epoch_idx=epoch, global_masks=global_masks,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if is_main:
            print(f"[Timing] epoch={epoch + 1} train_step_sec={time.perf_counter() - _ep_t0:.2f} "
                  f"budget={getattr(args, 'budget_ratio', 0.0)}", flush=True)
        # Skip the expensive train-eval forward pass: it's a full eval-mode forward
        # over the entire train set each epoch (+1 pass == ~25% slower finetune).
        # The in-train metrics from train_one_epoch_finetune are what we track;
        # eval-mode train metrics are stubbed with dummy zeros so downstream log
        # code stays untouched.
        head_classes_stub = {"A": int(getattr(args, "head_a_classes", 2))}
        train_eval_stats = {
            "loss": 0.0,
            "acc": 0.0,
            "cls": 0.0,
            "sparse": 0.0,
            "count": 0.0,
            "pred": 0.0,
            "counts": {k: torch.zeros((v, 3), device=device) for k, v in head_classes_stub.items()},
            "per_video_counts": {},
        }
        val_stats = evaluate_finetune(model, val_loader, device, args, epoch_idx=epoch, global_masks=global_masks)
        test_stats = evaluate_finetune(model, test_loader, device, args, epoch_idx=epoch, global_masks=global_masks)
        train_eval_counts = train_eval_stats["counts"]
        val_counts = val_stats["counts"]
        test_counts = test_stats["counts"]
        if ft_scheduler is not None:
            ft_scheduler.step()
        if args.distributed:
            train_tensor = torch.tensor(
                [
                    train_eval_stats["loss"],
                    train_eval_stats["acc"],
                    train_eval_stats["cls"],
                    train_eval_stats["sparse"],
                    train_eval_stats["count"],
                    train_eval_stats.get("pred", 0.0),
                ],
                device=device,
            )
            dist.all_reduce(train_tensor, op=dist.ReduceOp.SUM)
            train_eval_stats.update(
                {
                    "loss": train_tensor[0].item() / args.world_size,
                    "acc": train_tensor[1].item() / args.world_size,
                    "cls": train_tensor[2].item() / args.world_size,
                    "sparse": train_tensor[3].item() / args.world_size,
                    "count": train_tensor[4].item() / args.world_size,
                    "pred": train_tensor[5].item() / args.world_size,
                }
            )
            for hk, tens in train_eval_counts.items():
                dist.all_reduce(tens, op=dist.ReduceOp.SUM)
            val_tensor = torch.tensor(
                [
                    val_stats["loss"],
                    val_stats["acc"],
                    val_stats["cls"],
                    val_stats["sparse"],
                    val_stats["count"],
                    val_stats.get("pred", 0.0),
                ],
                device=device,
            )
            dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
            val_stats.update(
                {
                    "loss": val_tensor[0].item() / args.world_size,
                    "acc": val_tensor[1].item() / args.world_size,
                    "cls": val_tensor[2].item() / args.world_size,
                    "sparse": val_tensor[3].item() / args.world_size,
                    "count": val_tensor[4].item() / args.world_size,
                    "pred": val_tensor[5].item() / args.world_size,
                }
            )
            for hk, tens in val_counts.items():
                dist.all_reduce(tens, op=dist.ReduceOp.SUM)
            test_tensor = torch.tensor(
                [
                    test_stats["loss"],
                    test_stats["acc"],
                    test_stats["cls"],
                    test_stats["sparse"],
                    test_stats["count"],
                    test_stats.get("pred", 0.0),
                ],
                device=device,
            )
            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
            test_stats.update(
                {
                    "loss": test_tensor[0].item() / args.world_size,
                    "acc": test_tensor[1].item() / args.world_size,
                    "cls": test_tensor[2].item() / args.world_size,
                    "sparse": test_tensor[3].item() / args.world_size,
                    "count": test_tensor[4].item() / args.world_size,
                    "pred": test_tensor[5].item() / args.world_size,
                }
            )
            for hk, tens in test_counts.items():
                dist.all_reduce(tens, op=dist.ReduceOp.SUM)
        train_eval_f1_scores = _compute_f1_from_counts(train_eval_counts)
        val_f1_scores = _compute_f1_from_counts(val_counts)
        test_f1_scores = _compute_f1_from_counts(test_counts)
        # Record head-A test F1 per epoch for end-of-run summary.
        if "A" in test_f1_scores:
            test_f1_history_A.append(list(test_f1_scores["A"]))
        if "A" in val_f1_scores and val_f1_scores["A"]:
            val_macro_history_A.append(sum(val_f1_scores["A"]) / len(val_f1_scores["A"]))
        # track best validation (mean F1) and save best checkpoint
        updated_best = False
        val_f1_mean = _mean_f1(val_f1_scores, active_heads)
        if best_val is None or val_f1_mean > best_val:
            best_val = val_f1_mean
            best_epoch = epoch
            updated_best = True
        # compute per-class F1 from counts (tp, fp, fn)
        train_eval_count_stats = _counts_to_stats(train_eval_counts)
        val_count_stats = _counts_to_stats(val_counts)
        test_count_stats = _counts_to_stats(test_counts)
        if is_main:
            pred_tag = ""
            if float(getattr(args, "finetune_pred_weight", 0.0) or 0.0) > 0.0:
                pred_tag = (
                    f", pred={train_stats.get('pred', 0.0):.5f}"
                    f", val_pred={val_stats.get('pred', 0.0):.5f}"
                    f", test_pred={test_stats.get('pred', 0.0):.5f}"
                )
            print(
                f"[Finetune] Epoch {epoch + 1} | "
                f"train_loss={train_stats['loss']:.5f} (cls={train_stats['cls']:.5f}, sparse={train_stats['sparse']:.5f}, count={train_stats['count']:.5f}, pred={train_stats['pred']:.5f}) "
                f"| train_acc={train_stats['acc']:.4f} | "
                f"val_loss={val_stats['loss']:.5f} (cls={val_stats['cls']:.5f}, sparse={val_stats['sparse']:.5f}, count={val_stats['count']:.5f}, pred={val_stats['pred']:.5f}) "
                f"| val_acc={val_stats['acc']:.4f} | "
                f"test_loss={test_stats['loss']:.5f} (cls={test_stats['cls']:.5f}, sparse={test_stats['sparse']:.5f}, count={test_stats['count']:.5f}, pred={test_stats['pred']:.5f}) "
                f"| test_acc={test_stats['acc']:.4f}{pred_tag}",flush=True
            )
            active_label = _get_active_ratio_label(args)
            alpha_parts = []
            if train_stats.get("alpha") is not None:
                alpha_parts.append(f"mean={train_stats['alpha']:.6f}")
            if train_stats.get("alpha_min") is not None:
                alpha_parts.append(f"min={train_stats['alpha_min']:.6f}")
            if train_stats.get("alpha_max") is not None:
                alpha_parts.append(f"max={train_stats['alpha_max']:.6f}")
            if train_stats.get("alpha_active_ratio") is not None:
                alpha_parts.append(f"{active_label}: {train_stats['alpha_active_ratio']:.6f}")
            if alpha_parts:
                print("          [Alpha] " + " | ".join(alpha_parts),flush=True)
            if train_stats.get("alpha_tail_mean") is not None:
                print(
                    "          [Alpha][Tail] "
                    f"mean_last_{getattr(args, 'alpha_tail_batches', 5)}={train_stats['alpha_tail_mean']:.6f}",flush=True
                )
            gate_parts = []
            if train_stats.get("gate_mean") is not None:
                gate_parts.append(f"mean={train_stats['gate_mean']:.6f}")
            if train_stats.get("gate_min") is not None:
                gate_parts.append(f"min={train_stats['gate_min']:.6f}")
            if train_stats.get("gate_max") is not None:
                gate_parts.append(f"max={train_stats['gate_max']:.6f}")
            if gate_parts:
                print("          [VisionGate] " + " | ".join(gate_parts),flush=True)
            if train_eval_stats.get("alpha_active_ratio") is not None:
                eval_alpha_parts = []
                if train_eval_stats.get("alpha") is not None:
                    eval_alpha_parts.append(f"mean={train_eval_stats['alpha']:.6f}")
                if train_eval_stats.get("alpha_min") is not None:
                    eval_alpha_parts.append(f"min={train_eval_stats['alpha_min']:.6f}")
                if train_eval_stats.get("alpha_max") is not None:
                    eval_alpha_parts.append(f"max={train_eval_stats['alpha_max']:.6f}")
                eval_alpha_parts.append(f"{active_label}: {train_eval_stats['alpha_active_ratio']:.6f}")
                print("          [TrainEval][Alpha] " + " | ".join(eval_alpha_parts),flush=True)
            gate_eval_parts = []
            if train_eval_stats.get("gate_mean") is not None:
                gate_eval_parts.append(f"mean={train_eval_stats['gate_mean']:.6f}")
            if train_eval_stats.get("gate_min") is not None:
                gate_eval_parts.append(f"min={train_eval_stats['gate_min']:.6f}")
            if train_eval_stats.get("gate_max") is not None:
                gate_eval_parts.append(f"max={train_eval_stats['gate_max']:.6f}")
            if gate_eval_parts:
                print("          [TrainEval][VisionGate] " + " | ".join(gate_eval_parts),flush=True)
            if val_stats.get("alpha_active_ratio") is not None:
                val_alpha_parts = []
                if val_stats.get("alpha") is not None:
                    val_alpha_parts.append(f"mean={val_stats['alpha']:.6f}")
                if val_stats.get("alpha_min") is not None:
                    val_alpha_parts.append(f"min={val_stats['alpha_min']:.6f}")
                if val_stats.get("alpha_max") is not None:
                    val_alpha_parts.append(f"max={val_stats['alpha_max']:.6f}")
                val_alpha_parts.append(f"{active_label}: {val_stats['alpha_active_ratio']:.6f}")
                print("          [Val][Alpha] " + " | ".join(val_alpha_parts),flush=True)
            gate_val_parts = []
            if val_stats.get("gate_mean") is not None:
                gate_val_parts.append(f"mean={val_stats['gate_mean']:.6f}")
            if val_stats.get("gate_min") is not None:
                gate_val_parts.append(f"min={val_stats['gate_min']:.6f}")
            if val_stats.get("gate_max") is not None:
                gate_val_parts.append(f"max={val_stats['gate_max']:.6f}")
            if gate_val_parts:
                print("          [Val][VisionGate] " + " | ".join(gate_val_parts),flush=True)
            if test_stats.get("alpha_active_ratio") is not None:
                test_alpha_parts = []
                if test_stats.get("alpha") is not None:
                    test_alpha_parts.append(f"mean={test_stats['alpha']:.6f}")
                if test_stats.get("alpha_min") is not None:
                    test_alpha_parts.append(f"min={test_stats['alpha_min']:.6f}")
                if test_stats.get("alpha_max") is not None:
                    test_alpha_parts.append(f"max={test_stats['alpha_max']:.6f}")
                test_alpha_parts.append(f"{active_label}: {test_stats['alpha_active_ratio']:.6f}")
                print("          [Test][Alpha] " + " | ".join(test_alpha_parts),flush=True)
            gate_test_parts = []
            if test_stats.get("gate_mean") is not None:
                gate_test_parts.append(f"mean={test_stats['gate_mean']:.6f}")
            if test_stats.get("gate_min") is not None:
                gate_test_parts.append(f"min={test_stats['gate_min']:.6f}")
            if test_stats.get("gate_max") is not None:
                gate_test_parts.append(f"max={test_stats['gate_max']:.6f}")
            if gate_test_parts:
                print("          [Test][VisionGate] " + " | ".join(gate_test_parts),flush=True)
            head_a_names = ["Supported", "Unsupported", "Grooming"]
            if getattr(args, "head_a_classes", 3) == 2:
                head_a_names = ["Rearing", "Grooming"]
            elif getattr(args, "head_a_classes", 3) == 1:
                head_a_names = ["Rearing"]
            label_names = {
                "A": head_a_names if "A" in active_heads else [],
                "B": [f"B{i}" for i in range(len(val_f1_scores.get("B", [])))] if "B" in active_heads else [],
                "C": [f"C{i}" for i in range(len(val_f1_scores.get("C", [])))] if "C" in active_heads else [],
            }
            for hk in active_heads:
                scores = train_eval_f1_scores.get(hk, [])
                names = label_names.get(hk, [])
                for idx, f1 in enumerate(scores):
                    lname = names[idx] if idx < len(names) else f"{hk}{idx}"
                    tp, fp, fn, sup, pred = train_eval_count_stats[hk][idx]
                    print(f"[Finetune][TrainEval][F1] head={hk} label={lname} f1={f1:.4f}")
                    print(f"  [Counts][TrainEval] head={hk} label={lname} tp={tp:.0f} fp={fp:.0f} fn={fn:.0f} support={sup:.0f} pred={pred:.0f}")
            for hk in active_heads:
                scores = val_f1_scores.get(hk, [])
                names = label_names.get(hk, [])
                for idx, f1 in enumerate(scores):
                    lname = names[idx] if idx < len(names) else f"{hk}{idx}"
                    print(f"[Finetune][Val][F1] head={hk} label={lname} f1={f1:.4f}")
                    tp, fp, fn, sup, pred = val_count_stats[hk][idx]
                    print(f"  [Counts][Val] head={hk} label={lname} tp={tp:.0f} fp={fp:.0f} fn={fn:.0f} support={sup:.0f} pred={pred:.0f}")
            for hk in active_heads:
                scores = test_f1_scores.get(hk, [])
                names = label_names.get(hk, [])
                for idx, f1 in enumerate(scores):
                    lname = names[idx] if idx < len(names) else f"{hk}{idx}"
                    print(f"[Finetune][Test][F1] head={hk} label={lname} f1={f1:.4f}",flush=True)
                    tp, fp, fn, sup, pred = test_count_stats[hk][idx]
                    print(f"  [Counts][Test] head={hk} label={lname} tp={tp:.0f} fp={fp:.0f} fn={fn:.0f} support={sup:.0f} pred={pred:.0f}",flush=True)
            # Head-A confusion matrix (diagnostic, esp. Supported<->Unsupported at r=0%).
            cm = test_stats.get("confusion_matrix")
            if cm is not None and cm.numel() > 0 and "A" in active_heads:
                cm_rows = cm.long().cpu().tolist()
                cnames = head_a_names
                hdr = " ".join(f"{n[:11]:>11s}" for n in cnames)
                print(f"[Finetune][Test][ConfMat] ep={epoch + 1} rows=gt cols=pred | {hdr}", flush=True)
                for gi, row in enumerate(cm_rows):
                    gname = cnames[gi] if gi < len(cnames) else f"A{gi}"
                    rowstr = " ".join(f"{v:>11d}" for v in row)
                    recall = row[gi] / max(sum(row), 1)
                    print(f"  [ConfMat][Test] gt={gname[:12]:<12s} {rowstr}  recall={recall:.4f}", flush=True)
            # per_video_train = {hk: stems for hk, stems in train_eval_stats.get("per_video_counts", {}).items() if hk in active_heads}
            # per_video_val = {hk: stems for hk, stems in val_stats.get("per_video_counts", {}).items() if hk in active_heads}
            # per_video_test = {hk: stems for hk, stems in test_stats.get("per_video_counts", {}).items() if hk in active_heads}
            # _log_per_video_f1("Train", per_video_train, head_classes, label_names, is_main=is_main)
            # _log_per_video_f1("Val", per_video_val, head_classes, label_names, is_main=is_main)
            # _log_per_video_f1("Test", per_video_test, head_classes, label_names, is_main=is_main)
            writer.add_scalar("finetune/train_loss", train_stats["loss"], epoch + 1)
            writer.add_scalar("finetune/train_cls", train_stats["cls"], epoch + 1)
            writer.add_scalar("finetune/train_sparse", train_stats["sparse"], epoch + 1)
            writer.add_scalar("finetune/train_count", train_stats["count"], epoch + 1)
            if "pred" in train_stats:
                writer.add_scalar("finetune/train_pred", train_stats["pred"], epoch + 1)
            writer.add_scalar("finetune/train_acc", train_stats["acc"], epoch + 1)
            if train_stats.get("alpha") is not None:
                writer.add_scalar("finetune/train_alpha", train_stats["alpha"], epoch + 1)
            if train_stats.get("alpha_min") is not None:
                writer.add_scalar("finetune/alpha_min", train_stats["alpha_min"], epoch + 1)
            if train_stats.get("alpha_max") is not None:
                writer.add_scalar("finetune/alpha_max", train_stats["alpha_max"], epoch + 1)
            if train_stats.get("alpha_active_ratio") is not None:
                writer.add_scalar("finetune/alpha_active_ratio", train_stats["alpha_active_ratio"], epoch + 1)
            if train_stats.get("gate_mean") is not None:
                writer.add_scalar("finetune/gate_mean", train_stats["gate_mean"], epoch + 1)
            if train_stats.get("gate_min") is not None:
                writer.add_scalar("finetune/gate_min", train_stats["gate_min"], epoch + 1)
            if train_stats.get("gate_max") is not None:
                writer.add_scalar("finetune/gate_max", train_stats["gate_max"], epoch + 1)

            writer.add_scalar("finetune/val_loss", val_stats["loss"], epoch + 1)
            writer.add_scalar("finetune/val_cls", val_stats["cls"], epoch + 1)
            writer.add_scalar("finetune/val_sparse", val_stats["sparse"], epoch + 1)
            writer.add_scalar("finetune/val_count", val_stats["count"], epoch + 1)
            if "pred" in val_stats:
                writer.add_scalar("finetune/val_pred", val_stats["pred"], epoch + 1)
            writer.add_scalar("finetune/val_acc", val_stats["acc"], epoch + 1)

            writer.add_scalar("finetune/test_loss", test_stats["loss"], epoch + 1)
            writer.add_scalar("finetune/test_cls", test_stats["cls"], epoch + 1)
            writer.add_scalar("finetune/test_sparse", test_stats["sparse"], epoch + 1)
            writer.add_scalar("finetune/test_count", test_stats["count"], epoch + 1)
            if "pred" in test_stats:
                writer.add_scalar("finetune/test_pred", test_stats["pred"], epoch + 1)
            writer.add_scalar("finetune/test_acc", test_stats["acc"], epoch + 1)
            writer.add_scalar("finetune/best_val_f1", best_val if best_val is not None else float("nan"), epoch + 1)
            if ((epoch + 1) % getattr(args, "finetune_save_interval", 1) == 0) or (epoch + 1 == args.epochs):
                _save_checkpoint("finetune", model, optimizer, ft_scheduler, best_val, best_epoch, epoch, args.save_dir, "resifuse_finetune")
            if updated_best:
                best_path = _save_checkpoint(
                    "finetune",
                    model,
                    optimizer,
                    ft_scheduler,
                    best_val,
                    best_epoch,
                    epoch,
                    args.save_dir,
                    "best_finetune",
                    add_epoch_suffix=True,
                )
                _cleanup_old_best(args.save_dir, "best_finetune", keep_path=best_path)
        _maybe_flush_visual_epoch(model)
        if args.distributed:
            dist.barrier()

    # ========== Summary: test F1 at the epoch with the best head-A val macro F1 ==========
    if is_main and test_f1_history_A and len(val_macro_history_A) == len(test_f1_history_A):
        num_classes_A = len(test_f1_history_A[0])
        # Freely head A: fused Rearing (Supported+Unsupported) vs Grooming.
        summary_label_names = ["Rearing", "Grooming"][:num_classes_A]
        best_ep = max(range(len(val_macro_history_A)), key=lambda e: val_macro_history_A[e])
        best_f1 = test_f1_history_A[best_ep]
        print(f"\n{'=' * 60}")
        print(f"  Best-val epoch (epoch {best_ep + 1}, val macro F1={val_macro_history_A[best_ep]:.4f}) (Test F1)")
        print(f"{'=' * 60}")
        for c in range(num_classes_A):
            label = summary_label_names[c] if c < len(summary_label_names) else f"A{c}"
            print(f"  {label:25s} F1={best_f1[c]:.4f}")
        print(f"  {'macro':25s} F1={sum(best_f1) / max(num_classes_A, 1):.4f}")
        print(f"{'=' * 60}")

    final_test = evaluate_finetune(model, test_loader, device, args, epoch_idx=args.epochs - 1, global_masks=global_masks)
    test_counts = final_test["counts"]
    if args.distributed:
        test_tensor = torch.tensor(
            [
                final_test["loss"],
                final_test["acc"],
                final_test["cls"],
                final_test["sparse"],
                final_test["count"],
                final_test.get("pred", 0.0),
            ],
            device=device,
        )
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        final_test.update(
            {
                "loss": test_tensor[0].item() / args.world_size,
                "acc": test_tensor[1].item() / args.world_size,
                "cls": test_tensor[2].item() / args.world_size,
                "sparse": test_tensor[3].item() / args.world_size,
                "count": test_tensor[4].item() / args.world_size,
                "pred": test_tensor[5].item() / args.world_size,
            }
        )
        for hk, tens in test_counts.items():
            dist.all_reduce(tens, op=dist.ReduceOp.SUM)
    test_f1 = {}
    for hk, tens in test_counts.items():
        tp, fp, fn = tens[:, 0], tens[:, 1], tens[:, 2]
        denom = 2 * tp + fp + fn
        f1 = torch.where(denom > 0, 2 * tp / denom.clamp(min=1e-8), torch.zeros_like(tp))
        test_f1[hk] = f1.cpu().tolist()
    if is_main:
        pred_tag = ""
        if float(getattr(args, "finetune_pred_weight", 0.0) or 0.0) > 0.0:
            pred_tag = f", pred={final_test.get('pred', 0.0):.5f}"
        print(
            f"[Finetune] Test | loss={final_test['loss']:.5f} "
            f"(cls={final_test['cls']:.5f}, sparse={final_test['sparse']:.5f}, count={final_test['count']:.5f}) "
            f"| acc={final_test['acc']:.4f}{pred_tag}"
        )
        label_names = {
            "A": ["Supported", "Unsupported", "Grooming"] if "A" in active_heads else [],
            "B": [f"B{i}" for i in range(len(test_f1.get("B", [])))] if "B" in active_heads else [],
            "C": [f"C{i}" for i in range(len(test_f1.get("C", [])))] if "C" in active_heads else [],
        }
        for hk in active_heads:
            scores = test_f1.get(hk, [])
            names = label_names.get(hk, [])
            for idx, f1 in enumerate(scores):
                lname = names[idx] if idx < len(names) else f"{hk}{idx}"
                print(f"[Finetune][Test][F1] head={hk} label={lname} f1={f1:.4f}")
        writer.add_scalar("finetune/test_loss", final_test["loss"], args.epochs)
        writer.add_scalar("finetune/test_cls", final_test["cls"], args.epochs)
        writer.add_scalar("finetune/test_sparse", final_test["sparse"], args.epochs)
        writer.add_scalar("finetune/test_count", final_test["count"], args.epochs)
        if "pred" in final_test:
            writer.add_scalar("finetune/test_pred", final_test["pred"], args.epochs)
        writer.add_scalar("finetune/test_acc", final_test["acc"], args.epochs)
    if writer:
        writer.close()


def main(config_path: str | None = None, seed: int | None = None):
    args = parse_args(config_path)
    if seed is not None:
        args.seed = seed
    # Detect if launched under torchrun (env WORLD_SIZE)
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    if env_world > 1:
        _main_worker(args)
    else:
        # If not under torchrun but distributed requested and multiple GPUs exist, spawn
        if args.distributed and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            world_size = torch.cuda.device_count()
            _set_default_dist_env(args)
            mp.spawn(_spawn_worker, nprocs=world_size, args=(world_size, args))
        else:
            # fallback to single process (no DDP)
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
    # init distributed
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
    # map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    args.torch_dtype = dtype_map.get(args.torch_dtype.lower(), torch.bfloat16)
    include_velocity = getattr(args, "motion_include_velocity", True)
    include_confidence = getattr(args, "motion_include_confidence", True)
    neighbor_dim = (args.max_neighbors * 4) if args.use_neighbor_feats else 0
    vel_dim = args.num_joints * 2 if include_velocity else 0
    base_dim = args.num_joints * (3 if include_confidence else 2)
    args.motion_input_dim = base_dim + vel_dim + neighbor_dim

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
