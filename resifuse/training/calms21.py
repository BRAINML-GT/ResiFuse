"""CalMS21 Task 1 pretrain + finetune entrypoint.

Mirrors resifuse/training/mabe.py but uses CalMS21Dataset plus a thin wrapper
that translates the single-class frame label [T] into beast/bams-style
multi-hot [T, 3] over {attack, investigation, mount} (drops the 'other'
source class via frame_mask). Everything that is dataset-agnostic —
M-MAE masking, VQ-EMA updates, VRP / multi-horizon prediction, global
frame-select masks, joint fine-tuning, F1 frame-index dedup — is
imported as-is from train_entry_mabe to avoid drift.

Usage mirrors MaBe:
    torchrun --nproc_per_node=N --master_port $MP -m resifuse.training.calms21 \
        --config ./config_calms21_pretrain.yaml
or via the dispatcher resifuse/train.py when dataset=calms21.
"""
from __future__ import annotations

import argparse
import os
import time
import traceback
from types import SimpleNamespace
from typing import Dict, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from resifuse.data.datasets import CalMS21Dataset
from resifuse.models.model import (
    ResiFuseClassifier,
    ResiFusePretrainer,
    collision_contrastive_loss,
)
from resifuse.config import load_config_with_defaults

from resifuse.training.mabe import (
    # dist + IO scaffolding
    init_distributed,
    is_main_process,
    _is_main_checkpoint_process,
    _get_base_model,
    _set_default_dist_env,
    # LR scheduler + checkpoint helpers
    _build_lr_scheduler,
    _build_run_dir,
    _save_checkpoint,
    _cleanup_old_best,
    _maybe_resume,
    # label / mask helpers (multi-hot + majority-agg are no-ops for [T, C] input)
    _get_label_valid_mode,
    _select_frame_mask,
    _sanitize_frame_labels,
    _compute_pos_weight_from_loader,
    _majority_aggregate_frame_labels,
    _downsample_frame_labels,
    _downsample_frame_mask,
    # M-MAE mask
    create_social_mae_mask,
    # broadcast / losses / metrics
    _broadcast_eval_stats,
    compute_multilabel_frame_loss,
    _init_f1_counts,
    _accumulate_f1_counts_frame_index,
    _counts_to_f1,
    # loops
    train_one_epoch_pretrain,
    evaluate_pretrain,
    train_one_epoch_finetune,
    evaluate_finetune,
    # warm-start (cross_attn transfer is a no-op when those keys aren't in
    # the model state dict; VRP transfer still runs)
    maybe_load_pretrain_weights,
    # finetune helpers
    _configure_finetune_trainables,
    _get_sample_key,
    # social attn stats
    _maybe_collect_social_attn_stats,
)


# ----------------------------------------------------------------------
# Beast/bams label protocol: drop the "other" source class
# ----------------------------------------------------------------------
CALMS21_SOURCE_CLASSES = ["attack", "investigation", "mount", "other"]
CALMS21_TARGET_CLASSES = ["attack", "investigation", "mount"]


def init_distributed():
    if not torch.cuda.is_available() or not dist.is_available():
        return False, 0, 0, 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    return True, rank, local_rank, world_size


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
    # All ranks derive the same launch tag from environment variables so
    # CalMS21 does not depend on broadcast_object_list from the MABe helper.
    job_id = str(os.environ.get("SLURM_JOB_ID", "") or "").strip()
    step_id = str(os.environ.get("SLURM_STEP_ID", "") or "").strip()
    master_port = str(os.environ.get("MASTER_PORT", "") or "").strip()
    launch_bits = [x for x in (job_id, step_id, master_port) if x]
    if not launch_bits or not getattr(args, "distributed", False):
        # Runs started one after another in the same job (e.g. a seed loop)
        # get separate directories; otherwise each would resume from the
        # previous run's latest.pth. Under DDP the tag must match across
        # ranks, so no timestamp is added there.
        launch_bits.append(time.strftime("%Y%m%d-%H%M%S"))
    launch_tag = "-".join(launch_bits)
    if not run_name:
        if stage == "pretrain":
            run_name = (
                f"{launch_tag}_pre_bs{args.batch_size}_lr{args.lr}_"
                f"mask{args.mabe_mae_target_ratio}"
            )
        else:
            run_name = f"{launch_tag}_ft_bs{args.batch_size}_lr{args.lr}"
    else:
        run_name = f"{run_name}_{launch_tag}"
    run_dir = os.path.join(base_dir, run_name)
    if is_main_process(getattr(args, "rank", 0)):
        os.makedirs(run_dir, exist_ok=True)
    if getattr(args, "distributed", False) and dist.is_available() and dist.is_initialized():
        local_rank = int(getattr(args, "local_rank", 0) or 0)
        dist.barrier(device_ids=[local_rank])
    return run_dir


def parse_args(config_path: str | None = None):
    cfg = load_config_with_defaults(config_path)
    stage = cfg.get("stage", "finetune")
    if stage == "pretrain":
        cfg["lr"] = cfg["pretrain_lr"]
        cfg["epochs"] = cfg["pretrain_epochs"]
    else:
        cfg["lr"] = cfg.get("finetune_lr", 1e-4)
        cfg["epochs"] = cfg["finetune_epochs"]
    return SimpleNamespace(**cfg)


# ----------------------------------------------------------------------
# CalMS21 → MaBe batch shape wrapper
# ----------------------------------------------------------------------

class CalMS21TrainWrapper(Dataset):
    """Adapt :class:`CalMS21Dataset` output to the batch format expected by
    the MaBe training loops.

    Per-sample transforms:
      * ``frame_labels``  int64 [T] (0=attack, 1=investigation, 2=mount,
        3=other)  →  int64 [T, 3] multi-hot over the target classes.
      * ``frame_mask``    bool [T]  = ``mask.any(A) & (label NOT IN {-1, 3})``.
        i.e. drop padded frames AND frames whose source label is ignored.
      * ``metadata``      alias ``start`` → ``start_frame`` so the F1
        frame-index dedup in ``_accumulate_f1_counts_frame_index`` (which
        expects the MaBe key) works unchanged.
      * dtype: cast ``keypoints`` / ``motion_feats`` / ``vision_feats`` /
        ``vision_patch_tokens`` to ``dtype`` (bf16 for training).
    """

    def __init__(self, base: CalMS21Dataset, stage: str, dtype: torch.dtype,
                 include_other: bool = False):
        self.base = base
        self.stage = stage
        self.dtype = dtype
        # include_other=False (default): 3-class convention (a) — 'other' frames
        # are masked out (unchanged published behavior).
        # include_other=True: 4-class convention (b) — keep 'other' frames as
        # all-negative multi-hot targets so training/eval span the full 4-class
        # frame set. Reads the SAME cache read-only; enabled only by the separate
        # 4-class config, so the 3-class dataset/checkpoints are never touched.
        self.include_other = bool(include_other)
        # Expose shape info so run_pretrain / run_finetune can size the
        # model without dipping into the underlying CalMS21Dataset.
        self.num_joints = int(base.num_joints)
        self.num_mice = int(base.num_mice)
        self.seq_len = int(base.window_size)
        self.label_names = list(CALMS21_TARGET_CLASSES)
        self._keep_src = [CALMS21_SOURCE_CLASSES.index(n) for n in CALMS21_TARGET_CLASSES]
        self._ignore_src = [
            i for i, n in enumerate(CALMS21_SOURCE_CLASSES) if n not in CALMS21_TARGET_CLASSES
        ]
        # CalMS21 keypoints come from cache in raw pixel coordinates. The
        # shared pretrain loss regresses keypoints with a direct MSE term, so
        # keep the target scale bounded and dataset-agnostic by normalizing xy
        # to [0, 1] using the original video dimensions.
        self._kp_xy_scale = torch.tensor(
            [float(base.video_width), float(base.video_height)],
            dtype=torch.float32,
        ).view(1, 1, 1, 2)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.base[idx]
        dt = self.dtype

        raw = s["frame_labels"].long()                         # [T] int64
        T = int(raw.shape[0])
        multi = torch.zeros((T, len(CALMS21_TARGET_CLASSES)), dtype=torch.int64)
        for c, src in enumerate(self._keep_src):
            multi[:, c] = (raw == src).long()
        valid = raw != -1
        if not self.include_other:
            for src in self._ignore_src:
                valid = valid & (raw != src)
        scene_valid = s["mask"].any(dim=0)                      # [T]
        frame_mask = valid & scene_valid

        meta = dict(s.get("metadata", {}))
        # The F1 tracker (shared with MaBe) expects 'start_frame'; the
        # CalMS21 cache writes the absolute window start under 'start'.
        if "start_frame" not in meta:
            meta["start_frame"] = int(meta.get("start", 0))

        keypoints = s["keypoints"].to(torch.float32).clone()
        keypoints[..., :2] = keypoints[..., :2] / self._kp_xy_scale

        out: Dict[str, torch.Tensor] = {
            "keypoints":    keypoints.to(dt),
            "motion_feats": s["motion_feats"].to(dt),
            "mask":         s["mask"],
            "frame_labels": multi,
            "frame_mask":   frame_mask,
            "metadata":     meta,
        }
        if "vision_feats" in s:
            out["vision_feats"] = s["vision_feats"].to(dt)
        if "vision_patch_tokens" in s:
            out["vision_patch_tokens"] = s["vision_patch_tokens"].to(dt)
        return out


def _build_wrapped_dataset(args, split: str) -> CalMS21TrainWrapper:
    data_dir = getattr(args, "calms21_data_dir", None)
    cache_root = getattr(args, "calms21_cache_dir", None)
    split_json = getattr(args, "calms21_split_json", None)
    if not data_dir or not cache_root:
        raise RuntimeError(
            "calms21_data_dir and calms21_cache_dir must be set in the config."
        )
    cache_dir = os.path.join(cache_root, split)
    base = CalMS21Dataset(
        data_dir=data_dir,
        split=split,
        split_json=split_json,
        ds_stride=int(getattr(args, "ds_stride", 4)),
        crop_size=int(getattr(args, "calms21_crop_size", 448)),
        frame_resize=getattr(args, "frame_resize", None),
        use_vision_cache=bool(getattr(args, "use_vision_cache", True)),
        cache_dir=cache_dir,
        return_frame_labels=True,
        dtype=args.torch_dtype,
    )
    stage = getattr(args, "stage", "finetune")
    include_other = bool(getattr(args, "calms21_include_other", False))
    return CalMS21TrainWrapper(base, stage=stage, dtype=args.torch_dtype,
                               include_other=include_other)


def _motion_input_dim(args, num_joints: int) -> int:
    include_velocity = bool(getattr(args, "motion_include_velocity", False))
    include_confidence = bool(getattr(args, "motion_include_confidence", True))
    max_neighbors = int(getattr(args, "max_neighbors", 1))
    use_neighbor = bool(getattr(args, "use_neighbor_feats", True))
    base_dim = num_joints * (3 if include_confidence else 2)
    vel_dim = num_joints * 2 if include_velocity else 0
    nbr_dim = (max_neighbors * 4) if use_neighbor else 0
    return base_dim + vel_dim + nbr_dim


# ----------------------------------------------------------------------
# Pretrain
# ----------------------------------------------------------------------

def run_pretrain(args):
    device = torch.device(args.device)
    is_main = is_main_process(args.rank)

    # Bypass any torchrun log-forwarding flakiness: each rank also writes
    # traces to a per-rank file under /tmp so we can post-mortem inspect
    # rank != 0 progress even if its stdout is dropped.
    trace_path = f"/tmp/calms21_pretrain_trace_rank{args.rank}_{os.getpid()}.log"
    trace_fh = open(trace_path, "w", buffering=1)  # line-buffered

    def _trace(msg: str) -> None:
        line = f"[trace][rank{args.rank}] {msg}"
        print(line, flush=True)
        trace_fh.write(line + "\n")
        trace_fh.flush()

    _trace(f"trace file: {trace_path}")

    if is_main:
        print(
            f"========== [CalMS21][Pretrain] "
            f"vision_pred_weight={getattr(args, 'vision_pred_weight', None)} ==========",
            flush=True,
        )
        print(f"[CalMS21][Pretrain] device={device} | M-MAE target={args.mabe_mae_target_ratio}")
    _trace("enter run_pretrain")
    args.save_dir = _build_run_dir(args, stage="pretrain")
    _trace(f"build_run_dir done: save_dir={args.save_dir}")
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tb")) if is_main else None

    _trace("building train_ds...")
    train_ds = _build_wrapped_dataset(args, "train")
    _trace("building val_ds...")
    val_ds = _build_wrapped_dataset(args, "val")
    _trace("building test_ds...")
    test_ds = _build_wrapped_dataset(args, "test")
    _trace(f"datasets done: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    args.seq_len = train_ds.seq_len
    args.num_joints = train_ds.num_joints
    args.motion_input_dim = _motion_input_dim(args, args.num_joints)
    _trace(f"dims: seq_len={args.seq_len} num_joints={args.num_joints} motion_dim={args.motion_input_dim}")

    eval_distributed = bool(getattr(args, "eval_distributed", True))
    train_sampler = DistributedSampler(train_ds, num_replicas=args.world_size, rank=args.rank, shuffle=True) if args.distributed else None
    val_sampler   = DistributedSampler(val_ds,   num_replicas=args.world_size, rank=args.rank, shuffle=False) if args.distributed and eval_distributed else None
    test_sampler  = DistributedSampler(test_ds,  num_replicas=args.world_size, rank=args.rank, shuffle=False) if args.distributed and eval_distributed else None

    dl_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=train_sampler is None, sampler=train_sampler, **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, sampler=val_sampler,   **dl_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, sampler=test_sampler,  **dl_kwargs)
    _trace("loaders done; constructing model...")

    model = ResiFusePretrainer(args).to(device).to(dtype=args.torch_dtype)
    num_param_tensors = sum(1 for _ in model.parameters())
    num_params = sum(p.numel() for p in model.parameters())
    _trace(
        f"model built: params={num_params/1e6:.2f}M "
        f"param_tensors={num_param_tensors} on {device}"
    )
    if getattr(args, "torch_compile", False):
        if is_main:
            print("enable torch compile!")
        torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(model, mode="default", backend="inductor")

    unused_flag = os.environ.get("RESIFUSE_DEBUG_UNUSED", "0") == "1"
    if args.distributed:
        _trace("wrapping model in DDP...")
        model = DDP(
            model,
            device_ids=[args.local_rank] if device.type == "cuda" else None,
            output_device=args.local_rank if device.type == "cuda" else None,
            find_unused_parameters=unused_flag,
        )
    _trace("DDP wrap done")
    base_model = _get_base_model(model)

    # Split params: VQ vs others, decay vs no-decay.
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

    vq_lr = args.lr * getattr(args, "vq_lr_scale", 0.5)
    param_groups = []
    if vq_decay:     param_groups.append({"params": vq_decay,     "lr": vq_lr,   "weight_decay": weight_decay})
    if vq_nodecay:   param_groups.append({"params": vq_nodecay,   "lr": vq_lr,   "weight_decay": 0.0})
    if other_decay:  param_groups.append({"params": other_decay,  "lr": args.lr, "weight_decay": weight_decay})
    if other_nodecay:param_groups.append({"params": other_nodecay,"lr": args.lr, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = _build_lr_scheduler(optimizer, args, stage="pretrain")

    start_epoch, best_val, best_epoch = _maybe_resume(
        model, optimizer, scheduler, args.save_dir,
        getattr(args, "pretrain_resume_from", None), device, "pretrain", is_main,
    )

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main:
            print(f"\n=== [CalMS21][Pretrain] Epoch {epoch + 1}/{args.epochs} ===")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _pt_t0 = time.perf_counter()
        train_stats = train_one_epoch_pretrain(model, train_loader, optimizer, device, args, epoch_idx=epoch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if is_main:
            print(f"[Timing] pretrain_epoch={epoch + 1} train_step_sec={time.perf_counter() - _pt_t0:.2f}", flush=True)
        val_stats = evaluate_pretrain(model, val_loader, device, args, epoch_idx=epoch)

        if scheduler is not None:
            scheduler.step()

        if args.distributed:
            horizons = [int(h) for h in (getattr(args, "vision_pred_horizons", None) or [])]
            train_tensor = torch.tensor(
                [
                    train_stats["loss"], train_stats["mask"], train_stats["pred"],
                    train_stats["vq_recon"], train_stats["vq"], train_stats.get("vres", 0.0),
                    *[train_stats.get("pred_h", {}).get(h, 0.0) for h in horizons],
                ],
                device=device,
            )
            dist.all_reduce(train_tensor, op=dist.ReduceOp.SUM)
            train_tensor = train_tensor / args.world_size
            pred_h_tr = {h: train_tensor[6 + idx].item() for idx, h in enumerate(horizons)}
            train_stats = {
                "loss": train_tensor[0].item(), "mask": train_tensor[1].item(),
                "pred": train_tensor[2].item(), "vq_recon": train_tensor[3].item(),
                "vq": train_tensor[4].item(),   "vres": train_tensor[5].item(),
                "pred_h": pred_h_tr,
                "avg_fetch":   train_stats.get("avg_fetch", 0.0),
                "avg_compute": train_stats.get("avg_compute", 0.0),
                "total_fetch":   train_stats.get("total_fetch", 0.0),
                "total_compute": train_stats.get("total_compute", 0.0),
                "num_batches":   train_stats.get("num_batches", 0),
            }
            val_tensor = torch.tensor(
                [
                    val_stats["loss"], val_stats["mask"], val_stats["pred"],
                    val_stats["vq_recon"], val_stats["vq"],
                    *[val_stats.get("pred_h", {}).get(h, 0.0) for h in horizons],
                ],
                device=device,
            )
            dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
            pred_h_val = {h: (val_tensor[5 + idx].item() / args.world_size) for idx, h in enumerate(horizons)}
            val_stats = {
                "loss": val_tensor[0].item() / args.world_size,
                "mask": val_tensor[1].item() / args.world_size,
                "pred": val_tensor[2].item() / args.world_size,
                "vq_recon": val_tensor[3].item() / args.world_size,
                "vq":       val_tensor[4].item() / args.world_size,
                "pred_h":   pred_h_val,
            }
        val_loss = val_stats["loss"]

        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_path = _save_checkpoint(
                "pretrain", model, optimizer, scheduler, best_val, best_epoch, epoch,
                args.save_dir, "best_pretrain",
            )
            _cleanup_old_best(args.save_dir, "best_pretrain", keep_path=best_path)

        if is_main:
            base_for_stats = _get_base_model(model)
            cb_stats = base_for_stats.motion_scout.usage_stats()
            cb_q1, cb_q2 = cb_stats["q1"], cb_stats["q2"]
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
                writer.add_scalar("pretrain/val_loss",   val_stats["loss"],   epoch + 1)
                writer.add_scalar("pretrain/train_mask", train_stats["mask"], epoch + 1)
                writer.add_scalar("pretrain/val_mask",   val_stats["mask"],   epoch + 1)
                writer.add_scalar("pretrain/train_pred", train_stats["pred"], epoch + 1)
                writer.add_scalar("pretrain/val_pred",   val_stats["pred"],   epoch + 1)
                writer.add_scalar("pretrain/train_vres", train_stats.get("vres", 0.0), epoch + 1)
                writer.add_scalar("pretrain/train_vq_recon", train_stats["vq_recon"], epoch + 1)
                writer.add_scalar("pretrain/val_vq_recon",   val_stats["vq_recon"],   epoch + 1)
                writer.add_scalar("pretrain/train_vq", train_stats["vq"], epoch + 1)
                writer.add_scalar("pretrain/val_vq",   val_stats["vq"],   epoch + 1)
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
                model, train_loader, device, args,
                stage="pretrain", epoch_idx=epoch, writer=writer,
            )
            if social_stats is not None:
                mean_self = float(social_stats["self"].mean().item())
                mean_neighbor = float(social_stats["neighbor"].mean().item())
                per_self = ", ".join(
                    f"h{idx}={val:.4f}" for idx, val in enumerate(social_stats["self"].tolist())
                )
                per_neighbor = ", ".join(
                    f"h{idx}={val:.4f}" for idx, val in enumerate(social_stats["neighbor"].tolist())
                )
                print(
                    f"[Pretrain][SocialAttn] mean_R_self={mean_self:.4f} "
                    f"mean_R_neighbor={mean_neighbor:.4f}"
                )
                print(f"[Pretrain][SocialAttn] R_self: {per_self}")
                print(f"[Pretrain][SocialAttn] R_neighbor: {per_neighbor}")

        if (epoch + 1) % getattr(args, "pretrain_save_interval", 1) == 0 or (epoch + 1) == args.epochs:
            _save_checkpoint(
                "pretrain", model, optimizer, scheduler, best_val, best_epoch, epoch,
                args.save_dir, "resifuse_pretrain",
            )

    # Final test eval.
    test_stats = evaluate_pretrain(model, test_loader, device, args, epoch_idx=max(args.epochs - 1, 0))
    if args.distributed:
        horizons = [int(h) for h in (getattr(args, "vision_pred_horizons", None) or [])]
        test_tensor = torch.tensor(
            [
                test_stats["loss"], test_stats["mask"], test_stats["pred"],
                test_stats["vq_recon"], test_stats["vq"],
                *[test_stats.get("pred_h", {}).get(h, 0.0) for h in horizons],
            ],
            device=device,
        )
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        pred_h_te = {h: (test_tensor[5 + idx].item() / args.world_size) for idx, h in enumerate(horizons)}
        test_stats = {
            "loss": test_tensor[0].item() / args.world_size,
            "mask": test_tensor[1].item() / args.world_size,
            "pred": test_tensor[2].item() / args.world_size,
            "vq_recon": test_tensor[3].item() / args.world_size,
            "vq":       test_tensor[4].item() / args.world_size,
            "pred_h":   pred_h_te,
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
    if writer is not None:
        writer.close()


# ----------------------------------------------------------------------
# Finetune
# ----------------------------------------------------------------------

def run_finetune(args):
    device = torch.device(args.device)
    is_main = is_main_process(args.rank)
    if is_main:
        print(f"[CalMS21][Finetune] device={device}")
    args.save_dir = _build_run_dir(args, stage="finetune")
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tb")) if is_main else None

    train_ds = _build_wrapped_dataset(args, "train")
    val_ds = _build_wrapped_dataset(args, "val")
    test_ds = _build_wrapped_dataset(args, "test")

    args.seq_len = train_ds.seq_len
    args.num_joints = train_ds.num_joints
    args.motion_input_dim = _motion_input_dim(args, args.num_joints)
    label_names = train_ds.label_names
    args.head_a_classes = len(label_names)
    # Report the CalMS21-original 4-class macro-F1 (attack/investigation/mount/
    # other) — only when the 4-class dataset toggle is on (else gt_other==0 and
    # the metric is a degenerate constant). 'other' is derived post-hoc from the
    # same 3-class sigmoid decisions in evaluate_finetune: a pure eval-time metric
    # that does NOT change ResiFuse's architecture, loss, or training.
    args.report_other_class = bool(getattr(args, "calms21_include_other", False))
    label_names_4 = list(label_names) + ["other"]
    if is_main:
        print(f"[Finetune] Override head_a_classes to {args.head_a_classes} from label_names.")
        print(f"[Finetune] calms21_include_other={getattr(args, 'calms21_include_other', False)} "
              f"report_other_class={args.report_other_class} -> 4-class labels: {label_names_4}")

    eval_distributed = False
    train_sampler = DistributedSampler(train_ds, num_replicas=args.world_size, rank=args.rank, shuffle=True) if args.distributed else None
    val_sampler   = DistributedSampler(val_ds,   num_replicas=args.world_size, rank=args.rank, shuffle=False) if args.distributed and eval_distributed else None
    test_sampler  = DistributedSampler(test_ds,  num_replicas=args.world_size, rank=args.rank, shuffle=False) if args.distributed and eval_distributed else None
    dl_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader      = DataLoader(train_ds, shuffle=train_sampler is None, sampler=train_sampler, **dl_kwargs)
    train_eval_sampler= DistributedSampler(train_ds, num_replicas=args.world_size, rank=args.rank, shuffle=False) if args.distributed and eval_distributed else None
    train_eval_loader = DataLoader(train_ds, shuffle=False, sampler=train_eval_sampler, **dl_kwargs)
    val_loader        = DataLoader(val_ds,   shuffle=False, sampler=val_sampler,   **dl_kwargs)
    test_loader       = DataLoader(test_ds,  shuffle=False, sampler=test_sampler,  **dl_kwargs)

    # pos_weight
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
    if getattr(args, "torch_compile", False):
        if is_main:
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
    # Warm-start from pretrain. Shared helper handles VRP → cross_attn
    # transfer; on CalMS21 the cross_attn keys aren't in the classifier
    # state_dict so that branch naturally no-ops. `pretrain_ckpt` is the
    # warm-start slot (distinct from `finetune_resume_from`, which resumes
    # an in-progress finetune run).
    maybe_load_pretrain_weights(model, getattr(args, "pretrain_ckpt", None), device)

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
        groups = []
        if decay:   groups.append({"params": decay,   "lr": lr, "weight_decay": weight_decay})
        if nodecay: groups.append({"params": nodecay, "lr": lr, "weight_decay": 0.0})
        return torch.optim.AdamW(groups)

    optimizer = _build_finetune_optimizer(model, args.lr)
    ft_scheduler = _build_lr_scheduler(optimizer, args, stage="finetune")

    start_epoch, best_val, best_epoch = _maybe_resume(
        model, optimizer, ft_scheduler, args.save_dir,
        getattr(args, "finetune_resume_from", None), device, "finetune", is_main,
    )

    _lc = getattr(args, "lambda_col", None)
    lambda_col = float(_lc) if _lc is not None else 0.0
    _ct = getattr(args, "collision_temperature", None)
    collision_temperature = float(_ct) if _ct is not None else 0.1
    total_epochs = args.epochs

    base_model = _get_base_model(model)
    vision_alpha_cfg = float(getattr(args, "vision_alpha", 1.0))
    budget_ratio = float(getattr(args, "budget_ratio", 0.25))
    vision_fuse_mode = str(getattr(args, "vision_fuse_mode", "feature")).lower()
    # feature-fusion must be trained jointly; phases 1/2 are disabled.
    if budget_ratio <= 0:
        motion_epochs = 0; vision_epochs = 0; vision_alpha_cfg = 0.0
    elif vision_fuse_mode == "feature":
        motion_epochs = 0; vision_epochs = 0
        if is_main:
            print("[Finetune] vision_fuse_mode=feature -> joint training (phase config ignored)")
    else:
        motion_epochs = int(getattr(args, "motion_only_epochs", 0) or 0)
        vision_epochs = int(getattr(args, "vision_only_epochs", 0) or 0)
    joint_start = motion_epochs + vision_epochs

    if is_main:
        print("=" * 60)
        if budget_ratio <= 0:
            print(f"  Fine-tuning ({total_epochs} epochs, pure motion baseline)")
        else:
            print(f"  Fine-tuning ({total_epochs} epochs)")
            print(f"  Phase 1: motion-only 1-{motion_epochs}" if motion_epochs > 0 else "  Phase 1: skipped")
            print(f"  Phase 2: vision-only {motion_epochs+1}-{joint_start}" if vision_epochs > 0 else "  Phase 2: skipped")
            print(f"  Phase 3: joint {joint_start+1}-{total_epochs}")
        print(f"  frame_select_mode={getattr(args, 'frame_select_mode', 'residual')}")
        print(f"  budget_ratio={budget_ratio}, vision_alpha={vision_alpha_cfg}")
        print(f"  vision_fuse_mode={vision_fuse_mode}")
        print(f"  lambda_col={lambda_col}, collision_temperature={collision_temperature}")
        print("=" * 60)

    # ---- Global frame-select masks (shared with MaBe) ------------------
    @torch.no_grad()
    def _compute_global_masks(base_model, dataloader, device, budget_ratio, mode="residual"):
        saved_alpha = base_model.vision_alpha
        base_model.vision_alpha = 0.0
        base_model.eval()
        all_scores: Dict[str, torch.Tensor] = {}
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

            if mode in ("uncertainty", "online"):
                frame_mask = batch.get("frame_mask")
                if frame_mask is not None:
                    frame_mask = frame_mask.to(device)
                    if frame_mask.dim() == 1:
                        frame_mask = frame_mask.unsqueeze(0)
                    pad_mask = ~frame_mask.unsqueeze(1).expand(B, A, T) if frame_mask.dim() == 2 else ~frame_mask
                else:
                    pad_mask = torch.zeros(B, A, T, dtype=torch.bool, device=device)
                out = base_model(motion_feats, kps, pad_mask=pad_mask)
                if mode == "uncertainty":
                    scores = -out["logits"].abs().mean(dim=-1)
                else:  # online: binary entropy
                    probs = torch.sigmoid(out["logits"])
                    Hs = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
                    scores = Hs.mean(dim=-1)
            elif mode == "residual":
                r2 = (z_e - z_q).norm(dim=-1)
                scores = r2.view(B, A, L).mean(dim=1)
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
            else:
                raise ValueError(f"Unknown global mask mode: {mode}")

            for b in range(B):
                key = _get_sample_key(batch, b)
                all_scores[key] = scores[b].cpu()

        keys = list(all_scores.keys())
        if not keys:
            base_model.vision_alpha = saved_alpha
            base_model.train()
            return {}

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
                    f"  [_compute_global_masks] per-slot bias correction: "
                    f"mean={slot_bias.mean().item():.4f}  "
                    f"range=[{slot_bias.min().item():.4f}, {slot_bias.max().item():.4f}]",
                    flush=True,
                )

        vals = torch.cat([all_scores[k] for k in keys])
        total_frames = vals.numel()
        total_sel = max(1, int(total_frames * budget_ratio))
        threshold = vals.topk(total_sel).values[-1].item()
        masks = {k: (all_scores[k] >= threshold) for k in keys}
        base_model.vision_alpha = saved_alpha
        base_model.train()
        actual_selected = sum(int(m.sum().item()) for m in masks.values())
        if is_main:
            print(
                f"  [_compute_global_masks] mode={mode}, {n_batches} batches, "
                f"{len(masks)} windows, threshold={threshold:.4f}, "
                f"selected {actual_selected}/{total_frames} "
                f"({actual_selected/total_frames*100:.1f}%)",
                flush=True,
            )
        return masks

    frame_select_mode = str(getattr(args, "frame_select_mode", "residual")).lower()
    global_modes = {"uncertainty", "residual", "recon_error", "surprise", "online"}
    use_global_masks = frame_select_mode in global_modes and 0 < budget_ratio < 1.0
    needs_trained_head = {"uncertainty", "online"}
    global_mask_epoch = motion_epochs if frame_select_mode in needs_trained_head else 0
    global_masks = None

    test_f1_history: List[List[float]] = []
    test_f1_history_4: List[List[float]] = []   # 4-class (incl. 'other'), CalMS21-original
    val_macro_history: List[float] = []          # 3-class val macro F1, for best-val epoch selection

    def _run_eval_and_log(epoch, phase_label, global_masks=None):
        eval_kwargs = dict(
            device=device, config=args, epoch_idx=epoch,
            pos_weight=pos_weight,
            lambda_col=lambda_col, collision_temperature=collision_temperature,
            global_masks=global_masks,
        )
        if args.distributed and not eval_distributed:
            if is_main:
                eval_model = model.module if hasattr(model, "module") else model
                train_eval_stats = evaluate_finetune(eval_model, train_eval_loader, **eval_kwargs)
                val_stats        = evaluate_finetune(eval_model, val_loader,        **eval_kwargs)
                test_stats       = evaluate_finetune(eval_model, test_loader,       **eval_kwargs)
            else:
                empty = {"loss": 0.0, "cls": 0.0, "col": 0.0, "f1_counts": None, "f1_counts_4": None}
                train_eval_stats, val_stats, test_stats = dict(empty), dict(empty), dict(empty)
            train_eval_stats = _broadcast_eval_stats(train_eval_stats, device, src=0)
            val_stats        = _broadcast_eval_stats(val_stats,        device, src=0)
            test_stats       = _broadcast_eval_stats(test_stats,       device, src=0)
        else:
            train_eval_stats = evaluate_finetune(model, train_eval_loader, **eval_kwargs)
            val_stats        = evaluate_finetune(model, val_loader,        **eval_kwargs)
            test_stats       = evaluate_finetune(model, test_loader,       **eval_kwargs)
        if args.distributed and eval_distributed:
            for stats in (train_eval_stats, val_stats, test_stats):
                st = torch.tensor([stats["loss"], stats["cls"], stats.get("col", 0.0)], device=device)
                dist.all_reduce(st, op=dist.ReduceOp.SUM)
                stats.update({
                    "loss": st[0].item() / args.world_size,
                    "cls":  st[1].item() / args.world_size,
                    "col":  st[2].item() / args.world_size,
                })
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
                f"[CalMS21][Finetune][{phase_label}] Epoch {epoch + 1} | "
                f"train_loss={train_stats['loss']:.5f} "
                f"(cls={train_stats['cls']:.5f}, col={train_stats.get('col', 0.0):.5f}) "
                f"| val_loss={val_stats['loss']:.5f} "
                f"(cls={val_stats['cls']:.5f}, col={val_stats.get('col', 0.0):.5f}) "
                f"| test_loss={test_stats['loss']:.5f} "
                f"(cls={test_stats['cls']:.5f}, col={test_stats.get('col', 0.0):.5f})",
                flush=True,
            )
            writer.add_scalar("finetune/train_loss", train_stats["loss"], epoch + 1)
            writer.add_scalar("finetune/train_cls",  train_stats["cls"],  epoch + 1)
            writer.add_scalar("finetune/train_col",  train_stats.get("col", 0.0), epoch + 1)
            writer.add_scalar("finetune/val_loss",   val_stats["loss"],   epoch + 1)
            writer.add_scalar("finetune/val_cls",    val_stats["cls"],    epoch + 1)
            writer.add_scalar("finetune/val_col",    val_stats.get("col", 0.0),   epoch + 1)
            writer.add_scalar("finetune/test_loss",  test_stats["loss"],  epoch + 1)
            writer.add_scalar("finetune/test_cls",   test_stats["cls"],   epoch + 1)
            writer.add_scalar("finetune/test_col",   test_stats.get("col", 0.0),  epoch + 1)
            writer.add_scalar("finetune/best_val_loss", best_val if best_val is not None else float("nan"), epoch + 1)
            for split_name, stats in (("TrainEval", train_eval_stats), ("Val", val_stats), ("Test", test_stats)):
                counts = stats.get("f1_counts")
                if counts is None:
                    continue
                f1_scores = _counts_to_f1(counts)
                for idx, f1 in enumerate(f1_scores):
                    label = label_names[idx] if idx < len(label_names) else f"class{idx}"
                    tp = int(counts[idx, 0].item()); fp = int(counts[idx, 1].item())
                    fn = int(counts[idx, 2].item()); total = int(counts[idx, 3].item())
                    tn = max(total - tp - fp - fn, 0)
                    print(
                        f"[Finetune][{split_name}][F1][frame_index] label={label} f1={f1:.4f} "
                        f"tp={tp} tn={tn} fp={fp} fn={fn}",
                        flush=True,
                    )
                    if writer:
                        writer.add_scalar(f"finetune/{split_name.lower()}_f1/{label}", f1, epoch + 1)
            if ((epoch + 1) % getattr(args, "finetune_save_interval", 1) == 0) or (epoch + 1 == total_epochs):
                _save_checkpoint(
                    "finetune", model, optimizer, ft_scheduler, best_val, best_epoch, epoch,
                    args.save_dir, "resifuse_finetune",
                )
            if updated_best:
                best_path = _save_checkpoint(
                    "finetune", model, optimizer, ft_scheduler, best_val, best_epoch, epoch,
                    args.save_dir, "best_finetune", add_epoch_suffix=True,
                )
                _cleanup_old_best(args.save_dir, "best_finetune", keep_path=best_path)
        if args.distributed:
            dist.barrier()

    for epoch in range(start_epoch, total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if use_global_masks and epoch == global_mask_epoch and global_masks is None:
            if is_main:
                print(f"\n[Global Masks] Computing {frame_select_mode} masks (budget={budget_ratio})...")
            global_masks = {}
            for loader_name, loader in [("train", train_eval_loader), ("val", val_loader), ("test", test_loader)]:
                masks = _compute_global_masks(base_model, loader, device, budget_ratio, mode=frame_select_mode)
                global_masks.update(masks)
                if is_main:
                    n_masks = len(masks)
                    avg_sel = sum(int(m.sum().item()) for m in masks.values()) / max(n_masks, 1)
                    print(f"  {loader_name}: {n_masks} windows, avg selected frames={avg_sel:.1f}")

        # Phase switching (no-op under vision_fuse_mode='feature').
        if epoch < motion_epochs:
            phase = "motion-only"
            base_model.vision_alpha = 0.0
            for p in base_model.vision_cls_head.parameters(): p.requires_grad = False
            for p in base_model.social_cls_head.parameters(): p.requires_grad = True
        elif epoch < joint_start:
            phase = "vision-only"
            base_model.vision_alpha = vision_alpha_cfg
            for p in base_model.social_cls_head.parameters(): p.requires_grad = False
            for p in base_model.vision_cls_head.parameters(): p.requires_grad = True
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
                    print(
                        f"  [Phase 2] Reset LR to {args.lr}"
                        f"{f', scheduler T_max={remaining}' if ft_scheduler is not None else ''}"
                    )
        else:
            phase = "joint"
            base_model.vision_alpha = vision_alpha_cfg
            for p in base_model.social_cls_head.parameters(): p.requires_grad = True
            for p in base_model.vision_cls_head.parameters(): p.requires_grad = True

        if is_main:
            print(f"\n=== [CalMS21][Finetune] Epoch {epoch + 1}/{total_epochs} [{phase}] ===")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _ep_t0 = time.perf_counter()
        train_stats = train_one_epoch_finetune(
            model, train_loader, optimizer, device, args,
            epoch_idx=epoch, pos_weight=pos_weight,
            lambda_col=lambda_col, collision_temperature=collision_temperature,
            global_masks=global_masks,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if is_main:
            print(f"[Timing] epoch={epoch + 1} train_step_sec={time.perf_counter() - _ep_t0:.2f} "
                  f"budget={getattr(args, 'budget_ratio', 0.0)}", flush=True)
        train_eval_stats, val_stats, test_stats = _run_eval_and_log(epoch, "Train", global_masks=global_masks)
        if ft_scheduler is not None:
            ft_scheduler.step()
        _log_epoch(epoch, "Train", train_stats, train_eval_stats, val_stats, test_stats)

        test_counts = test_stats.get("f1_counts")
        if test_counts is not None:
            test_f1_history.append(_counts_to_f1(test_counts))
        test_counts_4 = test_stats.get("f1_counts_4")
        if test_counts_4 is not None:
            test_f1_history_4.append(_counts_to_f1(test_counts_4))
        val_counts = val_stats.get("f1_counts")
        if val_counts is not None:
            val_f1 = _counts_to_f1(val_counts)
            val_macro_history.append(sum(val_f1) / max(len(val_f1), 1))

    # ----- Summary: test F1 at the epoch with the best (3-class) val macro F1 -----
    def _print_summary(title, f1s, names):
        nc = len(f1s)
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)
        for c in range(nc):
            label = names[c] if c < len(names) else f"class{c}"
            print(f"  {label:25s} F1={f1s[c]:.4f}")
        print(f"  {'macro':25s} F1={sum(f1s) / max(nc, 1):.4f}")
        print("=" * 60)

    if is_main and test_f1_history and len(val_macro_history) == len(test_f1_history):
        best_ep = max(range(len(val_macro_history)), key=lambda e: val_macro_history[e])
        _print_summary(
            f"[3-class] Best-val epoch (epoch {best_ep + 1}, val macro F1={val_macro_history[best_ep]:.4f}) (Test F1)",
            test_f1_history[best_ep], label_names)
        # 4-class (incl. 'other') at the same epoch, the CalMS21-original convention.
        if test_f1_history_4 and len(test_f1_history_4) == len(test_f1_history):
            _print_summary(f"[4-class] Best-val epoch (same epoch: {best_ep + 1}) (Test F1)",
                           test_f1_history_4[best_ep], label_names_4)

    # ----- Final test evaluation -----
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
            "cls":  test_tensor[1].item() / args.world_size,
            "col":  test_tensor[2].item() / args.world_size,
        })
        counts = final_test.get("f1_counts")
        if counts is not None:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        counts_4 = final_test.get("f1_counts_4")
        if counts_4 is not None:
            dist.all_reduce(counts_4, op=dist.ReduceOp.SUM)
    if is_main:
        print(
            f"[CalMS21][Finetune] Test | loss={final_test['loss']:.5f} "
            f"(cls={final_test['cls']:.5f}, col={final_test.get('col', 0.0):.5f})"
        )
        counts = final_test.get("f1_counts")
        if counts is not None:
            f1_scores = _counts_to_f1(counts)
            for idx, f1 in enumerate(f1_scores):
                label = label_names[idx] if idx < len(label_names) else f"class{idx}"
                tp = int(counts[idx, 0].item()); fp = int(counts[idx, 1].item())
                fn = int(counts[idx, 2].item()); total = int(counts[idx, 3].item())
                tn = max(total - tp - fp - fn, 0)
                print(
                    f"[Finetune][Test][F1][frame_index] label={label} f1={f1:.4f} "
                    f"tp={tp} tn={tn} fp={fp} fn={fn}",
                    flush=True,
                )
            macro3 = sum(f1_scores) / max(len(f1_scores), 1)
            print(f"[Finetune][Test][F1][frame_index] 3class_macro={macro3:.4f}", flush=True)
        counts_4 = final_test.get("f1_counts_4")
        if counts_4 is not None:
            f1_4 = _counts_to_f1(counts_4)
            for idx, f1 in enumerate(f1_4):
                label = label_names_4[idx] if idx < len(label_names_4) else f"class{idx}"
                print(f"[Finetune][Test][F1][4class] label={label} f1={f1:.4f}", flush=True)
            print(f"[Finetune][Test][F1][4class] 4class_macro={sum(f1_4)/max(len(f1_4),1):.4f}", flush=True)
        if writer:
            writer.add_scalar("finetune/test_loss", final_test["loss"], total_epochs)
            writer.add_scalar("finetune/test_cls",  final_test["cls"],  total_epochs)
            writer.add_scalar("finetune/test_col",  final_test.get("col", 0.0), total_epochs)
            if counts is not None:
                for idx, f1 in enumerate(f1_scores):
                    label = label_names[idx] if idx < len(label_names) else f"class{idx}"
                    writer.add_scalar(f"finetune/test_f1/{label}", f1, total_epochs)
    if writer:
        writer.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _set_seed(seed, rank: int = 0) -> None:
    """Seed all RNGs for reproducible / paired runs.

    No-op when seed is None (preserves the original unseeded behavior).
    Must be called *before* model construction so a given seed yields the
    same weight initialization across runs and across vision budgets. The
    per-rank offset keeps DDP data shuffling decorrelated while model init
    stays identical (DDP broadcasts rank-0 weights anyway).
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
    rank = int(os.environ.get("RANK", "0"))
    try:
        if getattr(args, "num_joints", None) is None:
            args.num_joints = 7
        _set_default_dist_env(args)
        is_dist, rank, local_rank, world_size = init_distributed()
        args.distributed = is_dist
        args.rank = rank
        args.local_rank = local_rank
        args.world_size = world_size
        if is_dist:
            args.device = f"cuda:{local_rank}"
        # All-rank trace so we can see if any rank silently never makes it past
        # init_distributed (rank 0 is the only one that prints the banner below).
        print(f"[trace][rank{rank}] init_distributed done: local_rank={local_rank} "
              f"world_size={world_size} device={args.device}", flush=True)
        if is_main_process(rank):
            print(f"[DDP] world_size={world_size}, rank={rank}, local_rank={local_rank}")
            if torch.cuda.is_available() and is_dist:
                print(f"[DDP] device={torch.cuda.get_device_name(local_rank)}")
        dtype_map = {
            "float32":  torch.float32, "float":    torch.float32,
            "float16":  torch.float16, "half":     torch.float16,
            "bfloat16": torch.bfloat16,
        }
        args.torch_dtype = dtype_map.get(str(args.torch_dtype).lower(), torch.bfloat16)
        args.dataset = "calms21"
        _set_seed(getattr(args, "seed", None), rank)
        if args.stage == "pretrain":
            run_pretrain(args)
        else:
            run_finetune(args)
    except Exception:
        debug_root = os.environ.get("RESIFUSE_RANK_DEBUG_DIR", "/tmp")
        os.makedirs(debug_root, exist_ok=True)
        crash_path = os.path.join(
            debug_root,
            f"calms21_rank{rank}_crash_{os.getpid()}.log",
        )
        with open(crash_path, "w") as fh:
            traceback.print_exc(file=fh)
        print(f"[trace][rank{rank}] exception log written to {crash_path}", flush=True)
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def _parse_cli() -> str:
    parser = argparse.ArgumentParser(description="CalMS21 training entrypoint")
    parser.add_argument("-c", "--config", default="./config_calms21_finetune.yaml")
    ns = parser.parse_args()
    return ns.config


if __name__ == "__main__":
    cfg_path = _parse_cli()
    main(cfg_path)
