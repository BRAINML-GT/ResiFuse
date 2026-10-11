import argparse
import copy
import multiprocessing as mp
import os
import sys
from contextlib import nullcontext
from typing import Sequence

# Force line-buffered stdout/stderr so long-running runs stream progress in
# real time even when redirected to a file / tee. PYTHONUNBUFFERED is set so
# mp.spawn children inherit the same behavior (env var is re-read at Python
# startup, so setting it here only affects subprocess workers, not us).
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

import numpy as np
import torch

from resifuse.config import DEFAULT_CONFIG, load_config
from resifuse.data.datasets import OpenFieldDataset, MaBeMouseDataset
# Reuse encoder wrappers + helpers from precompute_mabe_cache to keep the
# two pipelines in lock-step.
from resifuse.features.hf_mabe import (
    ENCODER_CHOICES,
    DEFAULT_MODEL_IDS,
    HFImageEncoderForCache,
    VideoMAEEncoderForCache,
    build_qwen_encoder,
    _to_numpy_contiguous,
    _resolve_frame_resize,
)


def parse_dtype(dtype_str: str | None) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if not dtype_str:
        return torch.bfloat16
    return dtype_map.get(dtype_str.lower(), torch.bfloat16)


def load_cfg(path: str) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(load_config(path))
    if "seq_len_cache" not in cfg:
        cfg["seq_len_cache"] = cfg.get("seq_len", DEFAULT_CONFIG["seq_len"])
    if "seq_len_train" in cfg:
        cfg["seq_len"] = cfg["seq_len_train"]
    return cfg


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_dataset(cfg: dict, stage: str, split: str):
    dataset_name = str(cfg.get("dataset", "openfield") or "openfield").lower()
    if stage == "finetune":
        ft_train = cfg.get("finetune_train_ratio")
        ft_val = cfg.get("finetune_val_ratio")
        ft_test = cfg.get("finetune_test_ratio")
        train_ratio = ft_train if ft_train is not None else cfg.get("train_ratio", 0.8)
        val_ratio = ft_val if ft_val is not None else cfg.get("val_ratio", 0.1)
        test_ratio = ft_test if ft_test is not None else cfg.get("test_ratio", 0.1)
    else:
        train_ratio = cfg.get("train_ratio", 0.8)
        val_ratio = cfg.get("val_ratio", 0.1)
        test_ratio = cfg.get("test_ratio", 0.1)
    include_velocity = cfg.get("motion_include_velocity", True)
    include_confidence = cfg.get("motion_include_confidence", True)
    if dataset_name.startswith("mabe"):
        mabe_root = cfg.get("mabe_data_dir")
        if not mabe_root:
            raise RuntimeError("mabe_data_dir must be set in config for MABe vision caching.")
        print("Using Mabe dataset!", flush=True)
        stage_dir = os.path.join(mabe_root, stage)
        ds = MaBeMouseDataset(
            data_dir=stage_dir,
            index_file="training_index.json",
            split=split,
            stage=stage,
            train_ratio=train_ratio,
            ds_stride=cfg.get("ds_stride", 1),
            seq_len=cfg.get("seq_len", 32),
            multi_mouse=True,
            include_neighbor_feats=False,
            max_neighbors=cfg.get("max_neighbors", 2),
            motion_include_velocity=include_velocity,
            motion_include_confidence=include_confidence,
            frame_resize=cfg.get("frame_resize"),
            frame_resize_align=cfg.get("frame_resize_align", 32),
        )
    else:
        ds = OpenFieldDataset(
            root_dir=cfg["data_root"],
            seq_len=cfg.get("seq_len", 32),
            ds_stride=cfg.get("ds_stride", 1),
            frame_resize=cfg.get("frame_resize"),
            frame_resize_align=cfg.get("frame_resize_align", 32),
            pretrain_stride=cfg.get("pretrain_stride"),
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            stage=stage,
            dtype=parse_dtype(cfg.get("dtype") or cfg.get("torch_dtype")),
            max_neighbors=cfg.get("max_neighbors", 2),
            use_neighbor_feats=cfg.get("use_neighbor_feats", True),
            num_joints=cfg.get("num_joints"),
            conf_mask_thresh=cfg.get("conf_mask_thresh", 0.3),
            cache_dir=cfg.get("cache_dir"),
            use_cache=True,
            dynamic_cache_split=cfg.get("dynamic_cache_split"),
            oversample_grooming_factor=1,
            use_motion_scaler=cfg.get("use_motion_scaler", False),
            marker_subject_centric=cfg.get("marker_subject_centric", True),
            motion_include_velocity=include_velocity,
            motion_include_confidence=include_confidence,
            # This script iterates windows sequentially and forwards them
            # through a GPU encoder, so memory-mapped loads are much faster
            # than re-parsing every .pt from scratch.
            mmap_load=True,
        )

    print(f"[Debug], seq_len: {cfg.get('seq_len', 32)}, ds_stride:{cfg.get('ds_stride', 1)}", flush=True)

    return ds


def _encoder_for_path(out_path: str) -> tuple[str, str, str]:
    """Derive (out_dir, stem, vision_npy_path) for a given window cache path.

    The dataset's ``_maybe_load_vision_feats`` fast-path looks for
    ``<base>_vision.npy`` next to the .pt location encoded in cache_entries,
    matching the MaBe sidecar layout.
    """
    out_dir = os.path.dirname(out_path)
    stem = os.path.splitext(os.path.basename(out_path))[0]
    npy_out = os.path.join(out_dir, f"{stem}_vision.npy")
    return out_dir, stem, npy_out


def process_split(
    ds,
    encoder,
    device: torch.device,
    save_root: str,
    overwrite: bool = False,
    autocast_dtype: torch.dtype | None = None,
    max_samples: int | None = None,
    batch_size: int = 1,
    shard_rank: int = 0,
    shard_world_size: int = 1,
):
    total = len(ds)
    has_cache_entries = hasattr(ds, "cache_entries") and ds.cache_entries is not None
    has_samples = hasattr(ds, "samples")
    if not has_cache_entries and not has_samples:
        raise RuntimeError("Vision feature caching requires cache_entries or samples on the dataset.")
    # Round-robin shard: window idx `i` is owned by rank `i % world_size`.
    # .npy output paths are a deterministic function of the entry's relpath,
    # so disjoint shards produce disjoint files with no coordination needed.
    shard_world_size = max(1, int(shard_world_size))
    shard_rank = int(shard_rank) % shard_world_size
    shard_total = (total + shard_world_size - 1 - shard_rank) // shard_world_size
    prefix = f"[VisionCache][{device}][rank={shard_rank}/{shard_world_size}]"
    print(
        f"{prefix} Split={ds.split} | shard_windows={shard_total} (of {total}) | "
        f"save_root={save_root} | overwrite={overwrite} | encoder={type(encoder).__name__}",
        flush=True,
    )
    tick = max(1, min(100, shard_total // 20 if shard_total else 1))
    frames_batch: list[torch.Tensor] = []
    meta_batch: list[dict] = []
    processed = 0

    # Fail-fast sanity flags — we only check the VERY FIRST batch of the
    # run, at three points: input to the encoder, encoder output, and (just
    # once) scan the already-written .npy files on disk. This mirrors the
    # MABe / CalMS21 sanity checks: cheap, runs once per shard, aborts
    # early if the features are garbage so we don't waste NFS bandwidth.
    sanity_state = {"input_checked": False, "output_checked": False, "disk_checked": False}

    def _flush_batch():
        nonlocal processed
        if not frames_batch:
            return
        # Stack along batch dimension; encoder expects [B*A, T, C, H, W].
        frames_cat = torch.cat(frames_batch, dim=0)
        frames_dev = frames_cat.to(device=device)

        # Input sanity (first batch only).
        if not sanity_state["input_checked"]:
            sanity_state["input_checked"] = True
            finite = torch.isfinite(frames_dev)
            if not bool(finite.all()):
                n_bad = int((~finite).sum())
                raise RuntimeError(
                    f"{prefix} [INPUT SANITY] frames_dev has {n_bad} non-finite values "
                    f"BEFORE encoder forward. shape={tuple(frames_dev.shape)} "
                    f"dtype={frames_dev.dtype}. Fix the dataset / decode path, not the encoder."
                )
            print(
                f"{prefix} [INPUT SANITY] frames_dev shape={tuple(frames_dev.shape)} "
                f"dtype={frames_dev.dtype} "
                f"range=[{float(frames_dev.amin()):.2f}, {float(frames_dev.amax()):.2f}] "
                f"mean={float(frames_dev.float().mean()):.4f} (all finite)",
                flush=True,
            )

        ctx = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if device.type == "cuda" and autocast_dtype is not None
            else nullcontext()
        )
        with torch.inference_mode():
            with ctx:
                feats_flat = encoder._encode_real_frames(frames_dev)

        # Encoder output sanity (first batch only).
        if not sanity_state["output_checked"]:
            sanity_state["output_checked"] = True
            finite = torch.isfinite(feats_flat)
            if not bool(finite.all()):
                n_bad = int((~finite).sum())
                raise RuntimeError(
                    f"{prefix} [ENCODER SANITY] feats_flat has {n_bad} non-finite values "
                    f"immediately after encoder forward. shape={tuple(feats_flat.shape)} "
                    f"dtype={feats_flat.dtype}. Encoder is producing NaN — input was fine. "
                    f"Likely a VideoMAE-at-native-224 issue; try --encoder-image-size 128 or 208."
                )
            print(
                f"{prefix} [ENCODER SANITY] feats_flat shape={tuple(feats_flat.shape)} "
                f"dtype={feats_flat.dtype} "
                f"mean={float(feats_flat.float().mean()):.4f} "
                f"std={float(feats_flat.float().std()):.4f} (all finite)",
                flush=True,
            )

        B, T = frames_cat.shape[0], frames_cat.shape[1]
        feats_batch = feats_flat.view(B, T, -1)
        offset = 0
        for m in meta_batch:
            A_i = m["A"]
            feats_sample = feats_batch[offset : offset + A_i]
            offset += A_i
            # Save as .npy sidecar for mmap-friendly loading in the
            # dataset. Matches the MaBe cache layout so the shared
            # fast-path in OpenFieldDataset works unchanged.
            feats_np = _to_numpy_contiguous(feats_sample.to(dtype=autocast_dtype or feats_flat.dtype))
            np.save(m["npy_out"], feats_np)
            processed += 1
            if processed % tick == 0 or processed == shard_total:
                pct = processed / shard_total if shard_total else 1.0
                print(
                    f"{prefix} {ds.split}: cached {processed}/{shard_total} ({pct:.1%}) -> {m['npy_out']}",
                    flush=True,
                )

        # Disk sanity (first batch only): scan just-written .npy files for
        # NaN/Inf. Catches any downcast / to_numpy / float16-overflow bugs
        # between the tensor-level check above and the bytes actually on
        # disk — that path has historically hidden issues (e.g. bf16→f16
        # saturation at large values).
        if not sanity_state["disk_checked"] and meta_batch:
            sanity_state["disk_checked"] = True
            bad = []
            sample_stats = None
            for m in meta_batch:
                try:
                    arr = np.load(m["npy_out"], mmap_mode="r")
                except Exception:
                    continue
                if np.isnan(arr).any() or np.isinf(arr).any():
                    bad.append(m["npy_out"])
                elif sample_stats is None:
                    sample_stats = (
                        tuple(arr.shape), str(arr.dtype),
                        float(arr.mean()), float(arr.std()), float(np.abs(arr).max()),
                    )
            if bad:
                raise RuntimeError(
                    f"{prefix} [DISK SANITY] {len(bad)}/{len(meta_batch)} just-written "
                    f".npy files contain NaN/Inf. Example: {bad[0]}. "
                    f"Aborting before wasting time on the full run."
                )
            if sample_stats is not None:
                print(
                    f"{prefix} [DISK SANITY] OK ({len(meta_batch)} files, "
                    f"sample shape={sample_stats[0]} dtype={sample_stats[1]} "
                    f"mean={sample_stats[2]:.4f} std={sample_stats[3]:.4f} "
                    f"absmax={sample_stats[4]:.4f})",
                    flush=True,
                )

        frames_batch.clear()
        meta_batch.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for idx in range(total):
        if idx % shard_world_size != shard_rank:
            continue
        if max_samples is not None and idx >= max_samples:
            break
        entry = ds.cache_entries[idx] if has_cache_entries else ds.samples[idx]
        if has_cache_entries:
            relpath = entry.get("relpath") or entry.get("path")
            if relpath is None:
                raise KeyError("Cache entry missing relpath/path for vision cache export.")
            out_path = os.path.join(save_root, relpath)
        else:
            video_id = entry.get("video_id", "unknown")
            mouse_id = entry.get("mouse_id", "mouse")
            window_idx = entry.get("window_idx", idx)
            relpath = os.path.join(video_id, str(mouse_id), f"win_{window_idx}.pt")
            out_path = os.path.join(save_root, relpath)
        out_dir, stem, npy_out = _encoder_for_path(out_path)
        ensure_dir(out_dir)
        if os.path.exists(npy_out) and not overwrite:
            continue

        sample = ds[idx]
        frames = sample["frames"]
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
        if frames.dim() != 5:
            raise ValueError(f"Expected frames with 5 dims [A,T,C,H,W], got shape {tuple(frames.shape)}")
        frames_batch.append(frames)
        meta_batch.append(
            {
                "npy_out": npy_out,
                "stem": sample.get("stem") if isinstance(sample, dict) else None,
                "A": frames.shape[0],
                "T": frames.shape[1],
                "start": int(entry.get("start", 0)) if isinstance(entry, dict) else 0,
                "end": int(entry.get("end", frames.shape[1])) if isinstance(entry, dict) else int(frames.shape[1]),
            }
        )

        if len(frames_batch) >= batch_size:
            _flush_batch()

    # Unconditional final flush: each shard owns a subset of idxs under
    # round-robin, so the old `idx == total - 1` trigger only fires on one
    # shard; the other shards would silently drop any tail batch smaller
    # than batch_size, leaving the last windows of each split missing
    # (this caused the OFT_41/44/6 tail gaps we just hit).
    _flush_batch()


def build_encoder(
    cfg: dict,
    encoder_type: str,
    device: torch.device,
    max_len: int,
    encoder_model_id: str | None,
    encoder_image_size: int | None,
):
    """Build the vision encoder wrapper. Matches resifuse/features/hf_mabe.py."""
    if encoder_type == "qwen":
        return build_qwen_encoder(cfg, max_len, device)
    resolved_image_size = encoder_image_size
    if resolved_image_size is None:
        resolved_image_size = _resolve_frame_resize(cfg)
    if encoder_type == "videomae":
        return VideoMAEEncoderForCache(
            model_id=encoder_model_id,
            image_size=resolved_image_size,
            clip_length=int(cfg.get("videomae_clip_length", 16)),
        ).to(device=device)
    # clip / dinov3
    return HFImageEncoderForCache(
        encoder_type=encoder_type,
        model_id=encoder_model_id,
        image_size=resolved_image_size,
    ).to(device=device)


def _run_shard(
    cfg: dict,
    stage: str,
    split: str,
    save_root: str,
    overwrite: bool,
    max_samples: int | None,
    batch_size: int,
    encoder_type: str,
    encoder_model_id: str | None,
    encoder_image_size: int | None,
    device_str: str,
    shard_rank: int,
    shard_world_size: int,
):
    """Self-contained worker: build dataset + encoder, process assigned shard.

    Top-level function (not nested) so mp.spawn / mp.Process can pickle it.
    Each process pins a single CUDA device via the device string, builds its
    own encoder on that device, and writes .npy sidecars to the shared
    save_root. Round-robin sharding on window idx guarantees disjoint
    outputs across ranks with no coordination required.
    """
    device = torch.device(device_str)
    if device.type == "cuda" and device.index is not None:
        # Pin visible device so any HF / transformers code that reads
        # CUDA_VISIBLE_DEVICES / torch.cuda.current_device() DTRT in children.
        torch.cuda.set_device(device)
    ds = build_dataset(cfg, stage, split)
    max_len = cfg.get("seq_len", 512)
    encoder = build_encoder(
        cfg=cfg,
        encoder_type=encoder_type,
        device=device,
        max_len=max_len,
        encoder_model_id=encoder_model_id,
        encoder_image_size=encoder_image_size,
    )
    encoder.eval()
    dtype = parse_dtype(cfg.get("torch_dtype") or cfg.get("dtype"))
    autocast_dtype = dtype if device.type == "cuda" else None
    process_split(
        ds,
        encoder,
        device,
        save_root,
        overwrite=overwrite,
        autocast_dtype=autocast_dtype,
        max_samples=max_samples,
        batch_size=max(1, int(batch_size)),
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
    )


def main():
    parser = argparse.ArgumentParser(description="Precompute vision features (.npy sidecars) for cached Open Field windows.")
    parser.add_argument("--config", type=str, default="./config.yaml", help="Path to config.yaml.")
    parser.add_argument("--stage", type=str, choices=["pretrain", "finetune"], default=None, help="Stage to process.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to process (train/val/test).")
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help=(
            "Comma-separated CUDA devices for multi-GPU encoding (e.g. "
            "'cuda:0,cuda:1,cuda:2,cuda:3'). Windows are sharded round-robin. "
            "Falls back to --device / cfg['device'] when unset."
        ),
    )
    parser.add_argument("--device", type=str, default=None, help="Single device for inference (ignored if --devices given).")
    parser.add_argument("--overwrite", action="store_true", help="Recompute features even if .npy exists.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick tests.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of cached windows per vision forward.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset type: openfield or mabe.")
    parser.add_argument("--mabe-data-dir", type=str, default=None, help="MABe output root (contains pretrain/finetune).")
    parser.add_argument(
        "--encoder",
        type=str,
        choices=ENCODER_CHOICES,
        default="qwen",
        help="Vision encoder backend: qwen (default) | clip | dinov3 | videomae.",
    )
    parser.add_argument(
        "--encoder-model-id",
        type=str,
        default=None,
        help="Override HF model id for clip/dinov3/videomae backends.",
    )
    parser.add_argument(
        "--encoder-image-size",
        type=int,
        default=None,
        help=(
            "Input image size fed to clip/dinov3/videomae backbones. "
            "Defaults to cfg['frame_resize'] so the ablation compares encoders "
            "at matched resolution."
        ),
    )
    parser.add_argument("--frame-resize", type=int, default=None,
                        help="Override frame_resize of the config (frames are resized to this before the encoder).")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.frame_resize:
        cfg["frame_resize"] = args.frame_resize
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.mabe_data_dir:
        cfg["mabe_data_dir"] = args.mabe_data_dir
    stage = args.stage or cfg.get("stage", "pretrain")
    splits: Sequence[str] = args.splits or cfg.get("splits", ["train", "val", "test"])
    dataset_name = str(cfg.get("dataset", "openfield") or "openfield").lower()
    if dataset_name.startswith("mabe"):
        raise NotImplementedError(
            "Use resifuse/features/hf_mabe.py for MABe vision caching; this script targets openfield."
        )
    cache_root = cfg.get("cache_dir")
    if not cache_root:
        raise RuntimeError("cache_dir must be set in config to locate window cache.")
    save_root = cfg.get("vision_cache_dir") or cache_root

    devices_arg = args.devices or args.device or cfg.get("device", "cuda")
    device_list = [d.strip() for d in str(devices_arg).split(",") if d.strip()]
    if not device_list:
        device_list = ["cuda"]

    overwrite = args.overwrite
    max_samples = args.max_samples
    batch_size = max(1, int(args.batch_size))

    for split in splits:
        if len(device_list) == 1:
            _run_shard(
                cfg=cfg,
                stage=stage,
                split=split,
                save_root=save_root,
                overwrite=overwrite,
                max_samples=max_samples,
                batch_size=batch_size,
                encoder_type=args.encoder,
                encoder_model_id=args.encoder_model_id,
                encoder_image_size=args.encoder_image_size,
                device_str=device_list[0],
                shard_rank=0,
                shard_world_size=1,
            )
            continue

        # Multi-GPU: spawn one child per device. Each child builds its own
        # encoder on its pinned device; outputs are disjoint .npy files so
        # no locking is needed.
        ctx = mp.get_context("spawn")
        procs: list[mp.Process] = []
        world_size = len(device_list)
        for rank, dev in enumerate(device_list):
            p = ctx.Process(
                target=_run_shard,
                kwargs=dict(
                    cfg=cfg,
                    stage=stage,
                    split=split,
                    save_root=save_root,
                    overwrite=overwrite,
                    max_samples=max_samples,
                    batch_size=batch_size,
                    encoder_type=args.encoder,
                    encoder_model_id=args.encoder_model_id,
                    encoder_image_size=args.encoder_image_size,
                    device_str=dev,
                    shard_rank=rank,
                    shard_world_size=world_size,
                ),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                # Terminate siblings so we don't silently produce a half-sharded cache.
                for q in procs:
                    if q.is_alive():
                        q.terminate()
                raise RuntimeError(
                    f"Shard worker on device {device_list[procs.index(p)]} exited with code {p.exitcode}"
                )


if __name__ == "__main__":
    main()
