#!/usr/bin/env python3
"""Split CalMS21 Task 1 videos into train/val/test (default 6:1:3).

Unlike resifuse/preprocessing/split_mabe.py, CalMS21 has a single unified video pool
(88 videos after preprocessing). There is no leakage concern between
pretrain and finetune because both stages train on the same 'train'
split and evaluate on the same 'val'/'test' splits — a single
split.json is shared by both stages.

The official CalMS21 train/test flag is ignored by default (the
--respect-official flag can keep the 19 official test videos as test).

Stratification: balance the 4 CalMS21 behavior classes across splits
by greedy assignment on rarest-class first (same idea as MABe's
_split_ids_stratified but simplified for dense single-label data).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _load_index(index_path: Path) -> dict:
    with index_path.open("r") as f:
        return json.load(f)


def _compute_split_counts(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    raw = [r * n for r in ratios]
    counts = [int(x) for x in raw]
    remainder = n - sum(counts)
    if remainder > 0:
        frac = sorted(
            enumerate([r - int(r) for r in raw]),
            key=lambda x: x[1], reverse=True,
        )
        for i in range(remainder):
            counts[frac[i % 3][0]] += 1
    return counts[0], counts[1], counts[2]


def _split_ids_stratified(
    videos: List[dict],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Dict[str, List[str]]:
    """Greedy stratified split by rarest-class-first.

    For each video we have per-class frame counts; the goal is to
    allocate videos to train/val/test so that (a) video counts match
    ratios and (b) every class has meaningful presence in val/test.
    """
    rng = random.Random(seed)
    split_names = ["train", "val", "test"]
    target_videos = dict(zip(split_names, _compute_split_counts(len(videos), ratios)))

    # Build per-video presence vector (which classes are non-trivial).
    label_names = list(videos[0]["label_counts"].keys())
    presence = {
        v["video_id"]: [int(v["label_counts"].get(n, 0) > 0) for n in label_names]
        for v in videos
    }
    totals = [sum(p[i] for p in presence.values()) for i in range(len(label_names))]
    target_label = {
        s: [ratios[idx] * totals[i] for i in range(len(label_names))]
        for idx, s in enumerate(split_names)
    }

    assigned: Dict[str, List[str]] = {s: [] for s in split_names}
    counts_videos = {s: 0 for s in split_names}
    counts_labels = {s: [0] * len(label_names) for s in split_names}

    remaining = {v["video_id"] for v in videos}

    def _best_split(vid: str) -> str:
        pres = presence[vid]
        under = [s for s in split_names if counts_videos[s] < target_videos[s]]
        if not under:
            under = split_names
        scores = []
        for s in under:
            size_tgt = max(target_videos[s], 1)
            score = (counts_videos[s] + 1) / size_tgt
            for i, has in enumerate(pres):
                if not has:
                    continue
                tgt = max(target_label[s][i], 1e-6)
                score += (counts_labels[s][i] + 1) / tgt
            scores.append((score, s))
        min_s = min(s for s, _ in scores)
        return rng.choice([n for s, n in scores if s == min_s])

    # Rarest-class first to ensure every split gets minority behaviors.
    while remaining:
        remaining_label = [0] * len(label_names)
        for vid in remaining:
            for i, has in enumerate(presence[vid]):
                remaining_label[i] += has
        if not any(c > 0 for c in remaining_label):
            break
        rare = min(
            (i for i in range(len(label_names)) if remaining_label[i] > 0),
            key=lambda i: remaining_label[i],
        )
        vids_with_rare = [v for v in remaining if presence[v][rare]]
        rng.shuffle(vids_with_rare)
        for vid in vids_with_rare:
            s = _best_split(vid)
            assigned[s].append(vid)
            counts_videos[s] += 1
            for i, has in enumerate(presence[vid]):
                counts_labels[s][i] += has
            remaining.discard(vid)

    if remaining:
        leftover = list(remaining)
        rng.shuffle(leftover)
        for vid in leftover:
            s = _best_split(vid)
            assigned[s].append(vid)
            counts_videos[s] += 1
            remaining.discard(vid)

    return assigned


def _split_ids_respect_official(
    videos: List[dict],
    train_ratios: Tuple[float, float],  # (train, val) over official train set
    seed: int,
) -> Dict[str, List[str]]:
    """Keep the official 19 test videos as test; split the rest into train/val."""
    rng = random.Random(seed)
    test_ids = [v["video_id"] for v in videos if v["split"] == "test"]
    train_pool = [v["video_id"] for v in videos if v["split"] == "train"]
    rng.shuffle(train_pool)
    n_train_final = int(len(train_pool) * train_ratios[0] / sum(train_ratios))
    return {
        "train": train_pool[:n_train_final],
        "val": train_pool[n_train_final:],
        "test": test_ids,
    }


def _summarize(splits: Dict[str, List[str]], videos: List[dict]) -> dict:
    by_id = {v["video_id"]: v for v in videos}
    label_names = list(videos[0]["label_counts"].keys())
    summary = {}
    for s, vids in splits.items():
        win = sum(by_id[v]["num_windows"] for v in vids)
        frames_by_class = {n: 0 for n in label_names}
        for v in vids:
            for n in label_names:
                frames_by_class[n] += by_id[v]["label_counts"].get(n, 0)
        summary[s] = {
            "num_videos": len(vids),
            "num_windows": win,
            "num_samples_A2": win * 2,
            "frames_by_class": frames_by_class,
        }
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, required=True,
                   help="Stage 1 training_index.json")
    p.add_argument("--out", type=Path, default=None,
                   help="Output split.json (default: next to index)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.3)
    p.add_argument("--respect-official", action="store_true",
                   help="Keep official 19-video test set as test; split train pool only")
    args = p.parse_args()

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit(f"ratios must sum to 1.0, got {sum(ratios)}")

    index = _load_index(args.index)
    videos = index["videos"]
    print(f"Loaded {len(videos)} videos from {args.index}")

    if args.respect_official:
        splits = _split_ids_respect_official(
            videos, (args.train_ratio, args.val_ratio), args.seed
        )
        print("Mode: respect-official (test = official 19 videos)")
    else:
        splits = _split_ids_stratified(videos, ratios, args.seed)
        print(f"Mode: stratified (ratios={ratios})")

    summary = _summarize(splits, videos)
    for s in ("train", "val", "test"):
        print(f"  {s}: {summary[s]}")

    out = args.out or args.index.parent / "calms21_split.json"
    payload = {
        "dataset": "calms21_task1",
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "respect_official": bool(args.respect_official),
        "splits": splits,
        "summary": summary,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
