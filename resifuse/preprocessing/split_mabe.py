#!/usr/bin/env python3
"""
Create train/val/test video splits for MABe pretrain/finetune datasets.

- Finetune split is created from finetune video_ids.
- Pretrain split is created from pretrain video_ids after excluding finetune val/test ids.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _load_video_ids(index_path: Path) -> List[str]:
    with index_path.open("r") as f:
        data = json.load(f)
    videos = data.get("videos", [])
    return [str(v["video_id"]) for v in videos if "video_id" in v]


def _load_video_label_counts(index_path: Path) -> tuple[Dict[str, List[int]], List[str]]:
    with index_path.open("r") as f:
        data = json.load(f)
    label_names = data.get("label_names") or []
    videos = data.get("videos", [])
    counts_by_video: Dict[str, List[int]] = {}
    if not label_names:
        return counts_by_video, []
    for vid in videos:
        video_id = str(vid.get("video_id", ""))
        if not video_id:
            continue
        counts = [0] * len(label_names)
        mice = vid.get("mice", {})
        for mouse_meta in mice.values():
            label_counts = mouse_meta.get("label_counts") or {}
            for idx, name in enumerate(label_names):
                counts[idx] += int(label_counts.get(name, 0) or 0)
        counts_by_video[video_id] = counts
    return counts_by_video, label_names


def _load_video_window_counts(index_path: Path) -> Dict[str, int]:
    with index_path.open("r") as f:
        data = json.load(f)
    videos = data.get("videos", [])
    counts_by_video: Dict[str, int] = {}
    for vid in videos:
        video_id = str(vid.get("video_id", ""))
        if not video_id:
            continue
        mice = vid.get("mice", {})
        total = 0
        for mouse_meta in mice.values():
            total += int(mouse_meta.get("num_windows", 0) or 0)
        counts_by_video[video_id] = total
    return counts_by_video


def _compute_split_counts(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    train_r, val_r, test_r = ratios
    raw = [train_r * n, val_r * n, test_r * n]
    counts = [int(x) for x in raw]
    remainder = n - sum(counts)
    if remainder > 0:
        frac = sorted(
            enumerate([r - int(r) for r in raw]),
            key=lambda x: x[1],
            reverse=True,
        )
        for i in range(remainder):
            counts[frac[i % 3][0]] += 1
    return counts[0], counts[1], counts[2]


def _split_ids(
    ids: Sequence[str],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Dict[str, List[str]]:
    ids = list(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_train, n_val, n_test = _compute_split_counts(len(ids), ratios)
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val : n_train + n_val + n_test]
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _split_ids_stratified(
    ids: Sequence[str],
    ratios: Tuple[float, float, float],
    seed: int,
    label_counts_by_video: Dict[str, List[int]],
    num_labels: int,
) -> Dict[str, List[str]]:
    ids = [vid for vid in ids if vid in label_counts_by_video]
    if not ids or num_labels <= 0:
        return _split_ids(ids, ratios, seed)

    rng = random.Random(seed)
    split_names = ["train", "val", "test"]
    target_sizes = dict(zip(split_names, _compute_split_counts(len(ids), ratios)))
    splits = {k: [] for k in split_names}
    split_sizes = {k: 0 for k in split_names}
    split_label_counts = {k: [0] * num_labels for k in split_names}

    label_present = {
        vid: [int(x > 0) for x in label_counts_by_video.get(vid, [0] * num_labels)]
        for vid in ids
    }
    total_label_counts = [0] * num_labels
    for counts in label_present.values():
        for i in range(num_labels):
            total_label_counts[i] += counts[i]
    target_label_counts = {
        split: [ratios[idx] * total_label_counts[i] for i in range(num_labels)]
        for idx, split in enumerate(split_names)
    }

    remaining = set(ids)

    def _best_split_for_vid(vid: str) -> str:
        counts = label_present[vid]
        candidates = [s for s in split_names if split_sizes[s] < target_sizes[s]]
        if not candidates:
            candidates = split_names
        scores = []
        for split in candidates:
            size_target = max(target_sizes[split], 1)
            size_score = (split_sizes[split] + 1) / size_target
            label_score = 0.0
            for i, has_label in enumerate(counts):
                if not has_label:
                    continue
                tgt = target_label_counts[split][i]
                tgt = tgt if tgt > 1e-6 else 1e-6
                label_score += (split_label_counts[split][i] + 1) / tgt
            scores.append((label_score + size_score, split))
        min_score = min(s for s, _ in scores)
        tied = [name for s, name in scores if s == min_score]
        return rng.choice(tied)

    while remaining:
        # Find rarest label among remaining
        label_remaining = [0] * num_labels
        for vid in remaining:
            counts = label_present[vid]
            for i in range(num_labels):
                label_remaining[i] += counts[i]
        if all(c == 0 for c in label_remaining):
            break
        rare_label = min(
            (i for i in range(num_labels) if label_remaining[i] > 0),
            key=lambda i: label_remaining[i],
        )
        vids_with_label = [vid for vid in remaining if label_present[vid][rare_label]]
        rng.shuffle(vids_with_label)
        for vid in vids_with_label:
            split = _best_split_for_vid(vid)
            splits[split].append(vid)
            split_sizes[split] += 1
            cur = split_label_counts[split]
            counts = label_present[vid]
            for i in range(num_labels):
                cur[i] += counts[i]
            remaining.discard(vid)

    if remaining:
        leftover = list(remaining)
        rng.shuffle(leftover)
        for vid in leftover:
            split = _best_split_for_vid(vid)
            splits[split].append(vid)
            split_sizes[split] += 1
            cur = split_label_counts[split]
            counts = label_present[vid]
            for i in range(num_labels):
                cur[i] += counts[i]
            remaining.discard(vid)

    return splits


def _write_split(path: Path, splits: Dict[str, List[str]], meta: Dict) -> None:
    payload = {"splits": splits, "meta": meta}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)

def _summarize_label_counts(
    splits: Dict[str, List[str]],
    label_counts_by_video: Dict[str, List[int]],
    label_names: List[str],
) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for split, vids in splits.items():
        counts = {name: 0 for name in label_names}
        for vid in vids:
            vals = label_counts_by_video.get(vid)
            if not vals:
                continue
            for idx, name in enumerate(label_names):
                counts[name] += int(vals[idx])
        summary[split] = counts
    return summary


def _summarize_window_counts(
    splits: Dict[str, List[str]],
    window_counts_by_video: Dict[str, int],
) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for split, vids in splits.items():
        total = 0
        for vid in vids:
            total += int(window_counts_by_video.get(vid, 0) or 0)
        summary[split] = total
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Split MABe videos for pretrain/finetune.")
    parser.add_argument("--finetune-index", type=Path, required=True)
    parser.add_argument("--pretrain-index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.4)
    parser.add_argument("--finetune-train-ratio", type=float, default=None)
    parser.add_argument("--finetune-val-ratio", type=float, default=None)
    parser.add_argument("--finetune-test-ratio", type=float, default=None)
    parser.add_argument("--pretrain-train-ratio", type=float, default=None)
    parser.add_argument("--pretrain-val-ratio", type=float, default=None)
    parser.add_argument("--pretrain-test-ratio", type=float, default=None)
    parser.add_argument("--pretrain-max-videos", type=int, default=1200)
    args = parser.parse_args()

    def _resolve_ratios(prefix: str, overrides: Tuple[float | None, float | None, float | None]) -> Tuple[float, float, float]:
        if all(v is None for v in overrides):
            ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
        else:
            if any(v is None for v in overrides):
                raise ValueError(f"{prefix} ratios must be all provided or all omitted")
            ratios = (overrides[0], overrides[1], overrides[2])
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"{prefix} train/val/test ratios must sum to 1.0")
        return ratios

    ft_ratios = _resolve_ratios(
        "finetune",
        (args.finetune_train_ratio, args.finetune_val_ratio, args.finetune_test_ratio),
    )
    pt_ratios = _resolve_ratios(
        "pretrain",
        (args.pretrain_train_ratio, args.pretrain_val_ratio, args.pretrain_test_ratio),
    )

    ft_ids = _load_video_ids(args.finetune_index)
    pt_ids = _load_video_ids(args.pretrain_index)

    ft_label_counts, ft_label_names = _load_video_label_counts(args.finetune_index)
    ft_window_counts = _load_video_window_counts(args.finetune_index)
    if ft_label_names:
        ft_splits = _split_ids_stratified(
            ft_ids,
            ft_ratios,
            args.seed,
            ft_label_counts,
            len(ft_label_names),
        )
    else:
        ft_splits = _split_ids(ft_ids, ft_ratios, args.seed)
    ft_meta = {
        "source": str(args.finetune_index),
        "seed": args.seed,
        "ratios": {"train": ft_ratios[0], "val": ft_ratios[1], "test": ft_ratios[2]},
        "total_videos": len(ft_ids),
        "label_names": ft_label_names,
        "stratified": bool(ft_label_names),
    }
    ft_out = args.out_dir / "finetune_split.json" if args.out_dir else args.finetune_index.parent / "finetune_split.json"
    _write_split(ft_out, ft_splits, ft_meta)

    exclude = set(ft_splits["val"]) | set(ft_splits["test"])
    pt_filtered = [vid for vid in pt_ids if vid not in exclude]
    rng = random.Random(args.seed)
    if args.pretrain_max_videos > 0 and len(pt_filtered) > args.pretrain_max_videos:
        pt_filtered = rng.sample(pt_filtered, args.pretrain_max_videos)
    pt_splits = _split_ids(pt_filtered, pt_ratios, args.seed)
    pt_window_counts = _load_video_window_counts(args.pretrain_index)
    pt_meta = {
        "source": str(args.pretrain_index),
        "seed": args.seed,
        "ratios": {"train": pt_ratios[0], "val": pt_ratios[1], "test": pt_ratios[2]},
        "total_videos": len(pt_ids),
        "filtered_videos": len(pt_filtered),
        "excluded_finetune_val_test": len(exclude),
        "pretrain_max_videos": int(args.pretrain_max_videos),
    }
    pt_out = args.out_dir / "pretrain_split.json" if args.out_dir else args.pretrain_index.parent / "pretrain_split.json"
    _write_split(pt_out, pt_splits, pt_meta)

    ft_counts = {k: len(v) for k, v in ft_splits.items()}
    pt_counts = {k: len(v) for k, v in pt_splits.items()}
    ft_total = max(sum(ft_counts.values()), 1)
    pt_total = max(sum(pt_counts.values()), 1)
    ft_ratios = {k: round(v / ft_total, 4) for k, v in ft_counts.items()}
    pt_ratios = {k: round(v / pt_total, 4) for k, v in pt_counts.items()}
    print("Finetune split:", ft_counts, "ratios:", ft_ratios)
    if ft_label_names:
        ft_label_summary = _summarize_label_counts(ft_splits, ft_label_counts, ft_label_names)
        print("Finetune label distribution:", ft_label_summary)
    ft_window_summary = _summarize_window_counts(ft_splits, ft_window_counts)
    print("Finetune windows:", ft_window_summary)
    print("Pretrain split:", pt_counts, "ratios:", pt_ratios)
    pt_window_summary = _summarize_window_counts(pt_splits, pt_window_counts)
    print("Pretrain windows:", pt_window_summary)
    print(f"Excluded from pretrain (finetune val/test): {len(exclude)}")


if __name__ == "__main__":
    main()
