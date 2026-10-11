"""Standalone JAX/Flax VideoPrism feature extractor for MABe / Freely / CalMS21.

VideoPrism (Zhao et al., ICML 2024) is distributed as JAX/Flax weights only —
its custom temporal-attention layers cannot be directly loaded into HF
transformers. Rather than mixing JAX and PyTorch inside the main precompute
pipeline, this script runs VideoPrism in its native JAX runtime and writes
cache files that the downstream pretrain / finetune pipeline consumes with
zero changes.

Supports three datasets:
  mabe     — per-video windows laid out under {cache}/{stage}/{split}/
             (pretrain | finetune stage)
  calms21  — per-video windows laid out under {cache}/{split}/
             (no stage dimension)
  freely   — per-window sidecars keyed by ds.cache_entries[i]["relpath"]
             (same layout resifuse/features/hf_freely.py produces)

Output formats match the existing torch-based precompute scripts exactly so
the downstream pipeline doesn't need to know which runtime produced them.

Setup
-----
1. Install videoprism + JAX with CUDA:
     pip install git+https://github.com/google-deepmind/videoprism.git
     pip install -U "jax[cuda12]" "flax"

2. Set env vars (recommended) so JAX shares the GPU nicely:
     export XLA_PYTHON_CLIENT_PREALLOCATE=false

Usage
-----
  # MABe (same CLI style as resifuse/features/hf_mabe.py)
  python3 -m resifuse.features.videoprism --dataset mabe \
      --config ./config_mabe_pretrain.yaml \
      --stage pretrain --splits train val test \
      --devices 0,1,2,3 --overwrite

  # CalMS21 (no stage)
  python3 -m resifuse.features.videoprism --dataset calms21 \
      --config ./config_calms21_pretrain.yaml \
      --splits train val test \
      --devices 0 --overwrite

  # Freely moving mouse
  python3 -m resifuse.features.videoprism --dataset freely \
      --config ./config_free_pretrain.yaml \
      --stage pretrain --splits train val test \
      --devices 0,1 --overwrite
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from resifuse.features.videoprism_datasets import (
    build_dataset_mabe,
    build_dataset_calms21,
    build_dataset_freely,
    build_video_index,
    freely_cache_entries,
    load_cfg,
    _extract_sample_payload,
    _render_progress,
)


# ----------------------------------------------------------------------
# JAX setup (must run BEFORE jax is imported anywhere)
# ----------------------------------------------------------------------

def _configure_jax_env_before_import():
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")


def _load_videoprism(variant: str):
    try:
        from videoprism import models as vp_models
    except ImportError as e:
        raise ImportError(
            "`videoprism` package not installed. Install with:\n"
            "  pip install git+https://github.com/google-deepmind/videoprism.git\n"
        ) from e
    available = getattr(vp_models, "CHECKPOINTS", None)
    if available is not None and variant not in available:
        raise ValueError(
            f"Variant '{variant}' not in videoprism.models.CHECKPOINTS. "
            f"Available: {sorted(available.keys())}"
        )
    return vp_models.get_model(variant), vp_models.load_pretrained_weights(variant)


def _make_jit_forward(flax_model):
    """JIT-compile a fixed-signature apply call.

    Per the official example:
        outputs, _ = flax_model.apply(loaded_state, inputs, train=False)
    """
    import jax

    @jax.jit
    def _apply(params, video):
        outputs, _aux = flax_model.apply(params, video, train=False)
        return outputs
    return _apply


# ----------------------------------------------------------------------
# Per-window encoding (shared across datasets)
# ----------------------------------------------------------------------

def _encode_window(
    frames_torch: torch.Tensor,  # [A, T, C, H, W] float or uint8 in [0, 255]
    jit_apply,
    params,
    clip_length: int,
    image_size: int,
) -> np.ndarray:
    """Run VideoPrism on one (A animals, T frames) window → [A, T, D] numpy."""
    import jax.numpy as jnp

    if frames_torch.dim() == 4:
        frames_torch = frames_torch.unsqueeze(0)  # [1, T, C, H, W]
    A, T, C, H, W = frames_torch.shape

    if H != image_size or W != image_size:
        flat = frames_torch.reshape(A * T, C, H, W).float()
        flat = F.interpolate(
            flat, size=(image_size, image_size),
            mode="bicubic", align_corners=False, antialias=True,
        ).clamp_(0.0, 255.0)
        frames = flat.reshape(A, T, C, image_size, image_size)
    else:
        frames = frames_torch.float()

    frames = frames / 255.0
    frames = frames.permute(0, 1, 3, 4, 2).contiguous()   # NHWC

    pad_t = (clip_length - T % clip_length) % clip_length
    if pad_t > 0:
        last = frames[:, -1:].expand(A, pad_t, image_size, image_size, C)
        frames = torch.cat([frames, last], dim=1)
    T_padded = frames.shape[1]
    n_clips = T_padded // clip_length

    frames = frames.reshape(A, n_clips, clip_length, image_size, image_size, C)
    frames_np = frames.cpu().numpy().astype(np.float32, copy=False)

    per_animal = []
    for a in range(A):
        video_jax = jnp.asarray(frames_np[a])       # [n_clips, clip_length, 288, 288, 3]
        out = jit_apply(params, video_jax)          # [n_clips, clip_length*16*16, D]
        per_animal.append(np.asarray(out))

    feats = np.stack(per_animal, axis=0)
    D = feats.shape[-1]
    expected = clip_length * 16 * 16
    if feats.shape[2] != expected:
        raise RuntimeError(
            f"Unexpected VideoPrism token count per clip: got {feats.shape[2]}, "
            f"expected {expected} = clip_length({clip_length}) * 16 * 16."
        )
    feats = feats.reshape(A, n_clips, clip_length, 16, 16, D)
    feats = feats.mean(axis=(3, 4))
    feats = feats.reshape(A, T_padded, D)[:, :T]
    return feats.astype(np.float32, copy=False)


def _npy_save_f16(path: str, arr: np.ndarray) -> None:
    np.save(path, arr.astype(np.float16, copy=False))


# ----------------------------------------------------------------------
# Per-dataset per-video processing
# ----------------------------------------------------------------------

def _process_video_windowed(
    ds,
    video_id: str,
    indices: Sequence[int],
    out_dir: str,
    overwrite: bool,
    jit_apply,
    params,
    clip_length: int,
    image_size: int,
) -> int:
    """Used by MABe / CalMS21: all samples for one video → per-sample .pt + .npy.

    Matches resifuse/features/hf_mabe.py layout exactly: ds.samples is
    enumerated per-(mouse, window) pair in multi_mouse=True mode, so one
    video with K windows × A mice produces K*A samples. Each sample is
    saved with a SEQUENTIAL w index (0..N-1) — the filename's `w` is NOT
    the metadata window_idx; it is the sample's position in `indices`.
    This preserves per-mouse file multiplicity (3 files per window for
    MABe), so the downstream pretrain sees the right total sample count.

    Layout: {out_dir}/{video_id}_w{seq:03d}.pt  +  _vision.npy
    Sentinel = w000.pt (written LAST so a crash leaves no false 'done' marker).
    """
    sentinel = os.path.join(out_dir, f"{video_id}_w000.pt")
    if os.path.exists(sentinel) and not overwrite:
        return 0
    if not indices:
        return 0
    os.makedirs(out_dir, exist_ok=True)

    # Stage in sample-iteration order — keys are sequential positions, NOT
    # metadata window_idx (which collides across per-mouse samples of the
    # same window and would discard 2/3 of files in multi_mouse=True).
    staged_feats: List[np.ndarray] = []
    staged_payloads: List[dict] = []

    for idx in indices:
        sample = ds[idx]
        frames = sample.get("frames")
        if frames is None:
            frames = sample.get("cropped_video")
        if frames is None:
            raise RuntimeError(
                "Dataset sample has no 'frames'/'cropped_video' key — did you enable video loading?"
            )
        feats = _encode_window(frames, jit_apply, params, clip_length, image_size)
        payload = _extract_sample_payload(sample, use_vision=True)
        payload["video_id"] = video_id
        if hasattr(ds, "ds_stride"):
            payload["ds_stride"] = ds.ds_stride
        if hasattr(ds, "window_size"):
            payload["window_size"] = ds.window_size
        if hasattr(ds, "num_mice"):
            payload["num_mice"] = ds.num_mice
        staged_feats.append(feats)
        staged_payloads.append(payload)

    N = len(staged_feats)
    if N == 0:
        return 0

    # Inject the sequential window_idx + total count now that N is known.
    for w in range(N):
        staged_payloads[w]["window_idx"] = w
        staged_payloads[w]["num_windows"] = N

    # Write order: w>0 first, sentinel w=0 LAST (resume-safe).
    write_order = list(range(1, N)) + [0]
    for w in write_order:
        stem = f"{video_id}_w{w:03d}"
        _npy_save_f16(os.path.join(out_dir, f"{stem}_vision.npy"), staged_feats[w])
        torch.save(staged_payloads[w], os.path.join(out_dir, f"{stem}.pt"))
    return N


def _process_one_freely_window(
    ds,
    idx: int,
    save_root: str,
    overwrite: bool,
    jit_apply,
    params,
    clip_length: int,
    image_size: int,
) -> bool:
    """Used by Freely: mirror of resifuse/features/hf_freely.py output layout.

    For each dataset index, Freely exposes cache_entries[idx]["relpath"] or
    (video_id, mouse_id, window_idx) which we turn into a sidecar npy.
    """
    cache_entries = freely_cache_entries(ds)
    if cache_entries is not None:
        entry = cache_entries[idx]
        relpath = entry.get("relpath") or entry.get("path")
        if relpath is None:
            raise KeyError("Freely cache_entries[idx] missing relpath/path.")
    else:
        samples = getattr(ds, "samples", [])
        if idx >= len(samples):
            return False
        entry = samples[idx]
        video_id = entry.get("video_id", "unknown")
        mouse_id = entry.get("mouse_id", "mouse")
        window_idx = entry.get("window_idx", idx)
        relpath = os.path.join(video_id, str(mouse_id), f"win_{window_idx}.pt")

    out_path = os.path.join(save_root, relpath)
    out_dir = os.path.dirname(out_path)
    stem = os.path.splitext(os.path.basename(out_path))[0]
    npy_out = os.path.join(out_dir, f"{stem}_vision.npy")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(npy_out) and not overwrite:
        return False

    sample = ds[idx]
    frames = sample["frames"]
    feats = _encode_window(frames, jit_apply, params, clip_length, image_size)
    _npy_save_f16(npy_out, feats)
    return True


# ----------------------------------------------------------------------
# Sanity check
# ----------------------------------------------------------------------

def _sanity_check(out_dir: str, prefix: str, file_pattern: str) -> None:
    """Scan freshly written .npy under out_dir matching pattern for NaN/Inf."""
    import fnmatch
    vnpys = sorted(fn for fn in os.listdir(out_dir) if fnmatch.fnmatch(fn, file_pattern))
    if not vnpys:
        print(f"{prefix} sanity: no vision npys matching '{file_pattern}' in {out_dir}", flush=True)
        return
    bad, stats = [], None
    for fn in vnpys:
        arr = np.load(os.path.join(out_dir, fn), mmap_mode="r")
        if np.isnan(arr).any() or np.isinf(arr).any():
            bad.append(fn)
        elif stats is None:
            stats = (
                float(arr.mean(dtype=np.float32)),
                float(arr.std(dtype=np.float32)),
                float(np.abs(arr).max()),
            )
    if bad:
        raise RuntimeError(
            f"{prefix} sanity FAILED: {len(bad)}/{len(vnpys)} .npy files contain NaN/Inf. "
            f"Example: {bad[0]}. Aborting."
        )
    if stats is not None:
        print(
            f"{prefix} sanity OK ({len(vnpys)} files, mean={stats[0]:.4f} "
            f"std={stats[1]:.4f} absmax={stats[2]:.4f})",
            flush=True,
        )


# ----------------------------------------------------------------------
# Per-shard workers (one process per GPU)
# ----------------------------------------------------------------------

def _load_and_warm_jax(variant, clip_length, image_size, device_idx):
    """Common warmup: load model, JIT-compile on a dummy."""
    print(f"[VideoPrism][gpu{device_idx}] loading variant='{variant}' ...", flush=True)
    flax_model, params = _load_videoprism(variant)
    jit_apply = _make_jit_forward(flax_model)
    import jax.numpy as jnp
    t0 = time.time()
    dummy = jnp.zeros((1, clip_length, image_size, image_size, 3), dtype=jnp.float32)
    out = jit_apply(params, dummy)
    _ = np.asarray(out)
    print(
        f"[VideoPrism][gpu{device_idx}] first forward shape={tuple(out.shape)} "
        f"dtype={out.dtype} compile={time.time()-t0:.1f}s",
        flush=True,
    )
    if out.ndim != 3:
        raise RuntimeError(f"Unexpected VideoPrism output ndim={out.ndim}; expected 3.")
    return jit_apply, params


def _run_shard_mabe(device_idx, shard_video_ids, cfg_path, stage, split, cache_root,
                    variant, clip_length, image_size, overwrite):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)
    _configure_jax_env_before_import()
    cfg = load_cfg(cfg_path)
    ds = build_dataset_mabe(cfg, stage=stage, split=split)
    video_to_indices = build_video_index(ds)

    jit_apply, params = _load_and_warm_jax(variant, clip_length, image_size, device_idx)
    out_dir = os.path.join(cache_root, stage, split)
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"[Cache][mabe][{stage}/{split}][videoprism][gpu{device_idx}]"
    print(f"{prefix} assigned {len(shard_video_ids)} videos → {out_dir}", flush=True)

    first_checked = False
    for i, vid in enumerate(shard_video_ids, 1):
        n = _process_video_windowed(
            ds, vid, video_to_indices.get(vid, []),
            out_dir, overwrite, jit_apply, params, clip_length, image_size,
        )
        _render_progress(prefix, i, len(shard_video_ids), f"{vid} (+{n} windows)")
        if not first_checked and n > 0:
            first_checked = True
            _sanity_check(out_dir, prefix, f"{vid}_*_vision.npy")


def _run_shard_calms21(device_idx, shard_video_ids, cfg_path, split, cache_root,
                       variant, clip_length, image_size, overwrite):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)
    _configure_jax_env_before_import()
    cfg = load_cfg(cfg_path)
    ds = build_dataset_calms21(cfg, split=split)
    video_to_indices = build_video_index(ds)

    jit_apply, params = _load_and_warm_jax(variant, clip_length, image_size, device_idx)
    out_dir = os.path.join(cache_root, split)
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"[Cache][calms21][{split}][videoprism][gpu{device_idx}]"
    print(f"{prefix} assigned {len(shard_video_ids)} videos → {out_dir}", flush=True)

    first_checked = False
    for i, vid in enumerate(shard_video_ids, 1):
        n = _process_video_windowed(
            ds, vid, video_to_indices.get(vid, []),
            out_dir, overwrite, jit_apply, params, clip_length, image_size,
        )
        _render_progress(prefix, i, len(shard_video_ids), f"{vid} (+{n} windows)")
        if not first_checked and n > 0:
            first_checked = True
            _sanity_check(out_dir, prefix, f"{vid}_*_vision.npy")


def _run_shard_freely(device_idx, shard_indices, cfg_path, stage, split, save_root,
                      variant, clip_length, image_size, overwrite):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)
    _configure_jax_env_before_import()
    cfg = load_cfg(cfg_path)
    ds = build_dataset_freely(cfg, stage=stage, split=split)

    jit_apply, params = _load_and_warm_jax(variant, clip_length, image_size, device_idx)
    prefix = f"[Cache][freely][{stage}/{split}][videoprism][gpu{device_idx}]"
    print(f"{prefix} assigned {len(shard_indices)} windows → {save_root}", flush=True)
    first_checked = False
    first_out_dir = None
    for k, idx in enumerate(shard_indices, 1):
        wrote = _process_one_freely_window(
            ds, idx, save_root, overwrite,
            jit_apply, params, clip_length, image_size,
        )
        if wrote and not first_checked:
            # Record first real output dir for sanity scan later.
            entries = freely_cache_entries(ds)
            if entries is not None:
                relpath = entries[idx].get("relpath") or entries[idx].get("path")
            else:
                relpath = None
            if relpath is not None:
                first_out_dir = os.path.dirname(os.path.join(save_root, relpath))
                first_checked = True
        if k % max(1, len(shard_indices) // 20) == 0 or k == len(shard_indices):
            _render_progress(prefix, k, len(shard_indices),
                             f"(wrote ~{first_out_dir.split(os.sep)[-1] if first_out_dir else '?'})")
    if first_out_dir is not None:
        # Scan the first-seen output dir for NaN.
        try:
            _sanity_check(first_out_dir, prefix, "*_vision.npy")
        except RuntimeError as exc:
            raise


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def _split_round_robin(items, n):
    b = [[] for _ in range(n)]
    for i, v in enumerate(items):
        b[i % n].append(v)
    return b


def main():
    parser = argparse.ArgumentParser(description="JAX/Flax VideoPrism feature extractor.")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["mabe", "calms21", "freely"],
                        help="Which dataset to extract features for.")
    parser.add_argument("--config", type=str, required=True,
                        help="Config yaml matching the dataset.")
    parser.add_argument("--stage", type=str, choices=["pretrain", "finetune"], default="pretrain",
                        help="Only relevant for mabe / freely. Ignored for calms21.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--variant", type=str, default="videoprism_public_v1_large",
                        help="videoprism.models.CHECKPOINTS key "
                             "(base / large / lvt_base / lvt_large).")
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Override cache root. Defaults to the dataset-appropriate "
                             "config entry (mabe→mabe_data_cache_dir, "
                             "calms21→calms21_cache_dir, "
                             "freely→vision_cache_dir or cache_dir).")
    parser.add_argument("--devices", type=str, default="0",
                        help="Comma-separated CUDA indices for multi-GPU sharding.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    # Resolve cache root per dataset
    if args.cache_dir:
        cache_root = args.cache_dir
    elif args.dataset == "mabe":
        cache_root = cfg.get("mabe_data_cache_dir")
    elif args.dataset == "calms21":
        cache_root = cfg.get("calms21_cache_dir")
    else:  # freely
        cache_root = cfg.get("vision_cache_dir") or cfg.get("cache_dir")
    if not cache_root:
        raise RuntimeError(
            f"No cache dir resolved for dataset={args.dataset}. "
            "Pass --cache-dir or set the appropriate config field."
        )

    device_ids = [int(d.strip()) for d in args.devices.split(",") if d.strip()]
    if not device_ids:
        raise RuntimeError("No CUDA devices specified via --devices.")

    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    for split in args.splits:
        print(f"\n[VideoPrism] {args.dataset} {args.stage}/{split} ...", flush=True)

        if args.dataset == "mabe":
            ds = build_dataset_mabe(cfg, stage=args.stage, split=split)
            video_ids = sorted(build_video_index(ds).keys())
            del ds
            shards_or_ids = _split_round_robin(video_ids, len(device_ids))
            shard_worker = _run_shard_mabe
            shard_args_tpl = (args.config, args.stage, split, cache_root,
                              args.variant, args.clip_length, args.image_size, args.overwrite)
            print(f"[VideoPrism] {len(video_ids)} videos → sharded across {len(device_ids)} GPUs",
                  flush=True)

        elif args.dataset == "calms21":
            ds = build_dataset_calms21(cfg, split=split)
            video_ids = sorted(build_video_index(ds).keys())
            del ds
            shards_or_ids = _split_round_robin(video_ids, len(device_ids))
            shard_worker = _run_shard_calms21
            shard_args_tpl = (args.config, split, cache_root,
                              args.variant, args.clip_length, args.image_size, args.overwrite)
            print(f"[VideoPrism] {len(video_ids)} videos → sharded across {len(device_ids)} GPUs",
                  flush=True)

        else:  # freely
            ds = build_dataset_freely(cfg, stage=args.stage, split=split)
            # Freely shards by raw dataset index (each entry is a window).
            total = len(ds)
            all_indices = list(range(total))
            del ds
            shards_or_ids = _split_round_robin(all_indices, len(device_ids))
            shard_worker = _run_shard_freely
            shard_args_tpl = (args.config, args.stage, split, cache_root,
                              args.variant, args.clip_length, args.image_size, args.overwrite)
            print(f"[VideoPrism] {total} windows → sharded across {len(device_ids)} GPUs",
                  flush=True)

        # Single-GPU fast path (no mp overhead)
        if len(device_ids) == 1:
            shard_worker(device_ids[0], shards_or_ids[0], *shard_args_tpl)
            continue

        # Multi-GPU
        procs = []
        for dev_idx, shard in zip(device_ids, shards_or_ids):
            if not shard:
                continue
            p = ctx.Process(target=shard_worker, args=(dev_idx, shard) + shard_args_tpl)
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(
                    f"[VideoPrism] worker PID {p.pid} exited with code {p.exitcode}. "
                    "See logs above."
                )

    print("\n[VideoPrism] done.", flush=True)


if __name__ == "__main__":
    main()
