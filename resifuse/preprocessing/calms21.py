"""CalMS21 Task 1 pose preprocessing.

Reads the official calms21_task1_{train,test}.json files and writes one
per-video cache ``<output-dir>/caches/<video_id>.pt`` (keypoints, centers,
labels and window metadata; no pixels) and ``<output-dir>/training_index.json``.
The VideoPrism extractor (resifuse.features.videoprism) later decodes the
mp4 files and crops each mouse.

* Windows are defined on the original 30 fps frame indices; temporal
  downsampling (``ds_stride`` of the config) happens when the windows are read.
* CalMS21 has 2 mice, 7 keypoints and no missing frames, so there is no
  valid-segment logic as for Mouse Triplet.
* The official train/test flag is kept in ``_meta["split"]``; the split used
  for training is ``calms21_split.json`` (splits/ in this repository, or
  resifuse.preprocessing.split_calms21).
* Sequences shorter than ``--min-valid-len`` are skipped (only mouse036, 96
  frames).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

LABEL_NAMES = ["attack", "investigation", "mount", "other"]
NUM_JOINTS = 7
NUM_MICE = 2
JOINT_NAMES = [
    "nose", "left_ear", "right_ear", "neck",
    "left_hip", "right_hip", "tail_base",
]


@dataclass
class Config:
    window_size: int          # in original frames (e.g. 256)
    stride: int               # in original frames (e.g. 128)
    min_valid_len: int        # sequences shorter than this are dropped
    smooth_window: int        # smoothing window for center trajectory


def setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("calms21.preprocess")


def probe_video(video_path: Path, logger: logging.Logger) -> Optional[Tuple[int, int, float, int]]:
    """Return (width, height, fps, nb_frames) via ffprobe, or None on failure."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning(f"ffprobe failed for {video_path}: {exc}")
        return None
    lines = out.strip().splitlines()
    if len(lines) != 4:
        logger.warning(f"Unexpected ffprobe output for {video_path}: {lines}")
        return None
    w, h, rate, nb = lines
    num, den = rate.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 0.0
    return int(w), int(h), fps, int(nb)


def smooth_centers(centers: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smooth along time. centers: (T, 2)."""
    if window <= 1 or len(centers) <= window:
        return centers
    pad = window // 2
    padded = np.pad(centers, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.stack(
        [np.convolve(padded[:, d], kernel, mode="valid") for d in range(2)],
        axis=-1,
    )
    return smoothed[: len(centers)]


def generate_sliding_windows(
    total_frames: int, config: Config
) -> List[Dict]:
    """Sliding windows over [0, total_frames).

    Returns list of dicts with start/end (inclusive)/real_len/padding.
    If total_frames < window_size but >= min_valid_len, return a single
    padded window. If total_frames < min_valid_len, return [].
    """
    W, S = config.window_size, config.stride
    if total_frames < config.min_valid_len:
        return []
    if total_frames < W:
        return [{
            "start": 0,
            "end": total_frames - 1,
            "real_len": total_frames,
            "padding": W - total_frames,
        }]
    windows = []
    t = 0
    while t + W <= total_frames:
        windows.append({
            "start": t,
            "end": t + W - 1,
            "real_len": W,
            "padding": 0,
        })
        t += S
    # Tail handling: if there's a meaningful remainder, add one last window
    # anchored at (total_frames - W) to cover the tail without padding.
    last = windows[-1]
    tail_remain = total_frames - (last["end"] + 1)
    if tail_remain >= config.min_valid_len and tail_remain > 0:
        final_start = total_frames - W
        if final_start != last["start"]:
            windows.append({
                "start": final_start,
                "end": total_frames - 1,
                "real_len": W,
                "padding": 0,
            })
    return windows


def windowize(
    arr: np.ndarray, windows: List[Dict], window_size: int, pad_value: float = 0.0
) -> np.ndarray:
    """Slice arr along axis 0 into N windows. arr: (T, ...) → (N, T=W, ...)."""
    if not windows:
        return np.zeros((0, window_size) + arr.shape[1:], dtype=arr.dtype)
    out = np.full(
        (len(windows), window_size) + arr.shape[1:], pad_value, dtype=arr.dtype
    )
    for i, w in enumerate(windows):
        s, rl = w["start"], w["real_len"]
        out[i, :rl] = arr[s : s + rl]
    return out


def process_sequence(
    seq_id: str,
    entry: Dict,
    split: str,
    video_path: Path,
    config: Config,
    logger: logging.Logger,
) -> Optional[Dict]:
    """Process one sequence -> dict ready for torch.save."""
    video_id = seq_id.split("/")[-1]  # e.g. mouse003_task1_annotator1

    kp_raw = np.asarray(entry["keypoints"], dtype=np.float32)    # (T, 2, 2, 7)
    sc_raw = np.asarray(entry["scores"], dtype=np.float32)       # (T, 2, 7)
    ann = np.asarray(entry["annotations"], dtype=np.int64)       # (T,)
    T = kp_raw.shape[0]

    if not (kp_raw.shape == (T, NUM_MICE, 2, NUM_JOINTS)
            and sc_raw.shape == (T, NUM_MICE, NUM_JOINTS)
            and ann.shape == (T,)):
        logger.error(
            f"{video_id}: unexpected shapes kp={kp_raw.shape} sc={sc_raw.shape} ann={ann.shape}"
        )
        return None

    # Probe video to verify alignment
    if not video_path.exists():
        logger.warning(f"{video_id}: video not found at {video_path}; skipping")
        return None
    probed = probe_video(video_path, logger)
    if probed is None:
        return None
    video_w, video_h, fps, nb_frames = probed
    if nb_frames != T:
        logger.warning(
            f"{video_id}: frame-count mismatch (video={nb_frames}, json={T}); skipping"
        )
        return None

    # Layout: (T, 2, 2, 7) -> per-mouse (A=2, T, K=7, 2)
    # Original is (mouse, xy, joint); we want (mouse, time, joint, xy).
    kp = np.transpose(kp_raw, (1, 0, 3, 2)).copy()   # (A, T, K, 2)
    sc = np.transpose(sc_raw, (1, 0, 2)).copy()      # (A, T, K)

    # Centers: mean of the 7 joints per frame -> (A, T, 2)
    centers = kp.mean(axis=2)
    for a in range(NUM_MICE):
        centers[a] = smooth_centers(centers[a], config.smooth_window)

    # Mask: True for all valid frames (CalMS21 has no missing data).
    mask = np.ones((NUM_MICE, T), dtype=bool)

    # Windows
    windows = generate_sliding_windows(T, config)
    if not windows:
        logger.info(f"{video_id}: T={T} < min_valid_len={config.min_valid_len}; skipping")
        return None

    # Windowize all tensors -> (N, W, ...) per mouse, stack into (N, A, W, ...)
    N, W = len(windows), config.window_size

    kp_win = np.stack([
        windowize(kp[a], windows, W, pad_value=0.0) for a in range(NUM_MICE)
    ], axis=1)                                                      # (N, A, W, K, 2)
    sc_win = np.stack([
        windowize(sc[a], windows, W, pad_value=0.0) for a in range(NUM_MICE)
    ], axis=1)                                                      # (N, A, W, K)
    ce_win = np.stack([
        windowize(centers[a], windows, W, pad_value=0.0) for a in range(NUM_MICE)
    ], axis=1)                                                      # (N, A, W, 2)

    ann_win = windowize(ann, windows, W, pad_value=-1)              # (N, W)

    # mask window: True where frame is real (< real_len), False where padded.
    mask_win = np.zeros((N, NUM_MICE, W), dtype=bool)
    for i, w in enumerate(windows):
        mask_win[i, :, : w["real_len"]] = True

    cached = {
        "_meta": {
            "video_id": video_id,
            "video_path": str(video_path.resolve()),
            "total_frames": int(T),
            "fps": float(fps),
            "video_width": int(video_w),
            "video_height": int(video_h),
            "split": split,                  # official: 'train' or 'test'
            "num_mice": NUM_MICE,
            "num_joints": NUM_JOINTS,
            "num_windows": int(N),
            "joint_names": JOINT_NAMES,
            "label_names": LABEL_NAMES,
        },
        "windows": windows,
        "keypoints":    torch.from_numpy(kp_win).float(),
        "confidence":   torch.from_numpy(sc_win).float(),
        "centers":      torch.from_numpy(ce_win).float(),
        "mask":         torch.from_numpy(mask_win),
        "frame_labels": torch.from_numpy(ann_win).long(),
    }

    # Per-class frame counts (useful for stratified split + class weights).
    label_counts: Dict[str, int] = {name: 0 for name in LABEL_NAMES}
    for c, name in enumerate(LABEL_NAMES):
        label_counts[name] = int(((ann_win == c) & (ann_win >= 0)).sum())
    cached["_meta"]["label_counts"] = label_counts

    return cached


def save_and_index_entry(cached: Dict, out_dir: Path) -> Dict:
    """Save one per-video .pt file and return its index entry."""
    meta = cached["_meta"]
    video_id = meta["video_id"]
    pt_path = out_dir / f"{video_id}.pt"
    torch.save(cached, pt_path)
    return {
        "video_id": video_id,
        "video_path": meta["video_path"],
        "pt_file": str(pt_path.resolve()),
        "total_frames": meta["total_frames"],
        "fps": meta["fps"],
        "video_width": meta["video_width"],
        "video_height": meta["video_height"],
        "split": meta["split"],
        "num_windows": meta["num_windows"],
        "num_mice": meta["num_mice"],
        "num_joints": meta["num_joints"],
        "label_counts": meta["label_counts"],
        "windows": cached["windows"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess CalMS21 Task 1 pose.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--json-dir", type=Path, default=Path("data/calms21/task1_classic_classification"),
                   help="Dir containing calms21_task1_{train,test}.json")
    p.add_argument("--video-dir", type=Path, default=Path("data/calms21/task1_videos_mp4"),
                   help="Root with {train,test}/*.mp4")
    p.add_argument("--output-dir", type=Path, default=Path("data/calms21/resifuse_calms21_data_w256_s128"),
                   help="Output directory for per-video .pt + training_index.json")
    p.add_argument("--window-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--min-valid-len", type=int, default=128,
                   help="Drop sequences shorter than this (original frames)")
    p.add_argument("--smooth-window", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logger = setup_logging(args.verbose)

    config = Config(
        window_size=args.window_size,
        stride=args.stride,
        min_valid_len=args.min_valid_len,
        smooth_window=args.smooth_window,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "caches"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Config: window_size={config.window_size}, stride={config.stride}, "
                f"min_valid_len={config.min_valid_len}, smooth={config.smooth_window}")

    all_entries: List[Tuple[str, str, Dict]] = []
    for split in ("train", "test"):
        json_path = args.json_dir / f"calms21_task1_{split}.json"
        if not json_path.exists():
            logger.error(f"Missing json: {json_path}")
            sys.exit(1)
        logger.info(f"Loading {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)
        seqs = data.get("annotator-id_0")
        if seqs is None:
            logger.error(f"{json_path}: missing 'annotator-id_0' top key")
            sys.exit(1)
        for seq_id, entry in seqs.items():
            all_entries.append((seq_id, split, entry))

    if args.limit is not None:
        all_entries = all_entries[: args.limit]
    logger.info(f"Loaded {len(all_entries)} sequences total")

    index_videos: List[Dict] = []
    n_skipped = 0
    n_total_windows = 0
    global_label_counts = {name: 0 for name in LABEL_NAMES}

    for i, (seq_id, split, entry) in enumerate(all_entries, 1):
        video_id = seq_id.split("/")[-1]
        video_path = args.video_dir / split / f"{video_id}.mp4"
        logger.info(f"[{i}/{len(all_entries)}] {video_id}  (split={split})")
        cached = process_sequence(seq_id, entry, split, video_path, config, logger)
        if cached is None:
            n_skipped += 1
            continue
        idx_entry = save_and_index_entry(cached, cache_dir)
        index_videos.append(idx_entry)
        n_total_windows += idx_entry["num_windows"]
        for name, cnt in idx_entry["label_counts"].items():
            global_label_counts[name] += cnt

    index = {
        "dataset": "calms21_task1",
        "num_mice": NUM_MICE,
        "num_joints": NUM_JOINTS,
        "label_names": LABEL_NAMES,
        "joint_names": JOINT_NAMES,
        "config": {
            "window_size": config.window_size,
            "stride": config.stride,
            "min_valid_len": config.min_valid_len,
            "smooth_window": config.smooth_window,
            "video_width": 1024,
            "video_height": 570,
            "fps": 30.0,
        },
        "precomputed_data_dir": str(cache_dir.resolve()),
        "videos": index_videos,
        "stats": {
            "num_videos": len(index_videos),
            "num_skipped": n_skipped,
            "num_windows_total": n_total_windows,
            "num_samples_with_a": n_total_windows * NUM_MICE,
            "label_counts_frames": global_label_counts,
        },
    }
    index_path = args.output_dir / "training_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    logger.info(f"Wrote {index_path}")
    logger.info(
        f"Summary: {len(index_videos)} videos kept, {n_skipped} skipped, "
        f"{n_total_windows} windows ({n_total_windows * NUM_MICE} samples at A=2)"
    )
    logger.info(f"Label frame counts: {global_label_counts}")


if __name__ == "__main__":
    main()
