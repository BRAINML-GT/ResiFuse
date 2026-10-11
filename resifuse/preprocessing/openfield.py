"""Open Field window caches.

Writes the per-window caches that Open Field training and the VideoPrism
extractor read, under ``cache_dir`` of the config:

    <cache_dir>/<stage>/<split>/<idx>_<video>_s<start>_e<end>.pt   one window: keypoints,
        motion features, labels and video frames resized to cache_frame_resize
    <cache_dir>/<stage>_<split>_meta.json, <stage>_all_meta.json   window lists

Windows and the 50/10/40 video split come from OpenFieldDataset
(seq_len_cache frames per window, half-window stride for pretraining):

    python -m resifuse.preprocessing.openfield --config configs/openfield_pretrain.yaml
    python -m resifuse.preprocessing.openfield --config configs/openfield_finetune.yaml
"""

import argparse
import json
import os
from types import SimpleNamespace
from typing import List

import torch
from torch.utils.data import DataLoader, Subset

from resifuse.config import load_config_with_defaults
from resifuse.data.datasets import OpenFieldDataset

DTYPES = {"float32": torch.float32, "float": torch.float32, "fp32": torch.float32,
          "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
          "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}


def _atomic_torch_save(obj, path: str):
    """torch.save through a temp file and rename, so a killed run leaves no partial .pt."""
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _remove_stale_tmp_files(dir_path: str) -> int:
    removed = 0
    for name in os.listdir(dir_path):
        if ".pt.tmp" in name:
            os.remove(os.path.join(dir_path, name))
            removed += 1
    return removed


def _split_ratios(args, stage: str):
    if stage == "finetune":
        pick = lambda name: (getattr(args, f"finetune_{name}_ratio", None)
                             if getattr(args, f"finetune_{name}_ratio", None) is not None
                             else getattr(args, f"{name}_ratio"))
        return pick("train"), pick("val"), pick("test")
    return args.train_ratio, args.val_ratio, args.test_ratio


def cache_split(args, stage: str, split: str, limit: int | None = None):
    """Cache all windows of one stage/split; returns the meta dict (with its window entries)."""
    cache_root = args.cache_dir or os.path.join(args.data_root, "cache_windows")
    split_dir = os.path.join(cache_root, stage, split)
    os.makedirs(split_dir, exist_ok=True)
    stale = _remove_stale_tmp_files(split_dir)
    if stale:
        print(f"[Cache] removed {stale} stale *.pt.tmp files in {split_dir}", flush=True)

    train_ratio, val_ratio, test_ratio = _split_ratios(args, stage)
    frame_resize = getattr(args, "cache_frame_resize", None) or getattr(args, "frame_resize", None)
    seq_len = getattr(args, "seq_len_cache", args.seq_len)
    ds = OpenFieldDataset(
        root_dir=args.data_root,
        seq_len=seq_len,
        ds_stride=getattr(args, "cache_ds_stride", 1),
        frame_resize=frame_resize,
        frame_resize_align=getattr(args, "frame_resize_align", 32),
        pretrain_stride=args.pretrain_stride,
        split=split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stage=stage,
        dtype=DTYPES.get(str(args.dtype).lower(), torch.bfloat16),
        max_neighbors=args.max_neighbors,
        use_neighbor_feats=getattr(args, "use_neighbor_feats", True),
        num_joints=args.num_joints,
        conf_mask_thresh=getattr(args, "conf_mask_thresh", 0.3),
        cache_dir=None,
        use_cache=False,
        oversample_grooming_factor=1,
        use_motion_scaler=getattr(args, "use_motion_scaler", False),
        marker_subject_centric=getattr(args, "marker_subject_centric", True),
        motion_include_velocity=getattr(args, "motion_include_velocity", True),
        motion_include_confidence=getattr(args, "motion_include_confidence", True),
    )
    num_classes = getattr(args, "head_a_classes", 2)
    raw_counts = [0] * num_classes   # labeled frames per class in the full videos
    window_counts = [0] * num_classes  # labeled frames per class inside the cached windows
    if stage == "finetune":
        for entry in ds.samples:
            for v in OpenFieldDataset._read_labels(entry["labels"]):
                if v in (1, 2):    # 1 = Rearing, 2 = Grooming
                    raw_counts[int(v) - 1] += 1

    # The DataLoader's collation (ints -> tensors) is part of the cache format.
    n = len(ds) if limit is None else min(limit, len(ds))
    num_workers = int(getattr(args, "cache_num_workers", 0) or 0)
    loader = DataLoader(Subset(ds, range(n)), batch_size=1, shuffle=False, num_workers=num_workers,
                        pin_memory=False, persistent_workers=num_workers > 0)
    print(f"[Cache] stage={stage} split={split} | {n}/{len(ds)} windows | num_workers={num_workers}", flush=True)

    entries: List[dict] = []
    for idx, batch in enumerate(loader):
        sample = {k: (v[0] if (isinstance(v, torch.Tensor) and v.shape[0] == 1)
                      or (isinstance(v, list) and len(v) == 1) else v)
                  for k, v in batch.items()}
        win = ds.index[idx]
        stem, start, end = win["entry"]["stem"], win["start"], win["end"]
        cache_path = os.path.join(split_dir, f"{idx:08d}_{stem}_s{start}_e{end}.pt")
        if args.overwrite or not os.path.exists(cache_path):
            _atomic_torch_save(sample, cache_path)
        if stage == "finetune" and "frame_labels" in sample and "frame_mask" in sample:
            fl, fm = sample["frame_labels"].view(-1), sample["frame_mask"].view(-1)
            valid = (fm > 0) & (fl >= 0)
            for c in range(num_classes):
                window_counts[c] += int(((fl == c) & valid).sum().item())
        entries.append({
            "relpath": os.path.relpath(cache_path, cache_root),
            "stem": stem,
            "start": int(start),
            "end": int(end),
            "label": int(sample["label"]) if "label" in sample else None,
        })
        if (idx + 1) % max(1, n // 10) == 0 or idx == n - 1:
            print(f"  cached {idx + 1}/{n}", flush=True)

    meta = {
        "stage": stage,
        "split": split,
        "seq_len_cache": seq_len,
        "cache_ds_stride": getattr(args, "cache_ds_stride", 1),
        "pretrain_stride": args.pretrain_stride,
        "num_joints": int(ds.num_joints) if ds.num_joints is not None else None,
        "frame_resize": frame_resize,
        "frame_resize_align": getattr(args, "frame_resize_align", 32),
        "entries": entries,
    }
    with open(os.path.join(cache_root, f"{stage}_{split}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if stage == "finetune":
        print(f"[Cache][{split}] labeled frames per class: videos={raw_counts}, windows={window_counts}", flush=True)
    return meta


def main():
    p = argparse.ArgumentParser(description="Build the Open Field window caches.")
    p.add_argument("--config", required=True, help="configs/openfield_{pretrain,finetune}.yaml")
    p.add_argument("--stage", choices=["pretrain", "finetune", "both"], default=None,
                   help="default: `stage` of the config")
    p.add_argument("--splits", nargs="+", default=None, help="default: `splits` of the config")
    p.add_argument("--limit", type=int, default=None,
                   help="cache only the first N windows of each split (for testing)")
    p.add_argument("--overwrite", action="store_true", help="rewrite existing window files")
    cli = p.parse_args()

    args = SimpleNamespace(**load_config_with_defaults(cli.config))
    args.overwrite = cli.overwrite or bool(getattr(args, "overwrite", False))
    stage = cli.stage or args.stage
    splits = cli.splits or args.splits
    cache_root = args.cache_dir or os.path.join(args.data_root, "cache_windows")
    for st in (["pretrain", "finetune"] if stage == "both" else [stage]):
        metas = [cache_split(args, st, split, cli.limit) for split in splits]
        # One list over all splits, so training can re-split the windows.
        all_entries = [{**e, "orig_split": m["split"]} for m in metas for e in m["entries"]]
        if all_entries:
            meta_all = {k: v for k, v in metas[0].items() if k != "entries"}
            meta_all.update(split="all", entries=all_entries)
            path = os.path.join(cache_root, f"{st}_all_meta.json")
            with open(path, "w") as f:
                json.dump(meta_all, f, indent=2)
            print(f"[Cache] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
