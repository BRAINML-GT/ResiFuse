#!/usr/bin/env python3
"""Mouse Triplet (MABe22) pose preprocessing.

For every video with DeepLabCut pose (``*_el.h5``) and labels, this writes one
per-video cache ``<output-dir>/<stage>/caches/<video_id>.pt`` and an index
``<output-dir>/<stage>/training_index.json``:

1. Valid segments: frames whose mean keypoint confidence is at least
   ``--conf-threshold``; gaps of up to ``--max-gap`` low-confidence frames are
   bridged.
2. Windows: sliding windows over the valid segments (pretraining), or windows
   centered on labeled behavior segments (fine-tuning). Windows are kept only
   if all three mice have them.
3. Per mouse and window: keypoints, confidence, egocentric keypoints (relative
   to the mouse's center), the keypoints of another mouse in the same frame of
   reference, a validity mask, smoothed crop centers, a ``--crop-size`` video
   crop around the mouse, and (with labels) per-frame multi-label targets.

The defaults are the settings used in the paper; only ``--stage`` is required:

    python -m resifuse.preprocessing.mabe --stage pretrain    # 300-frame windows, stride 150
    python -m resifuse.preprocessing.mabe --stage finetune    # 200-frame windows, no overlap
"""

import argparse
import json
import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

DATA_ROOT = Path("data/mabe2022")
DEFAULT_OUTPUT_DIR = DATA_ROOT / "windows"
STAGE_WINDOWS = {"pretrain": (300, 150), "finetune": (200, 200)}  # (window size, stride)

KEYPOINTS = [
    'nose', 'left ear', 'right ear', 'neck', 'center back',
    'left forepaw', 'right forepaw', 'left hindpaw', 'right hindpaw',
    'tail base', 'tail middle', 'tail tip',
]

# Fine-tuning classes; a class is present if any of its member labels is.
FINETUNE_LABEL_GROUPS = [
    ("chases", ["chases"]),
    ("huddles", ["huddles"]),
    ("oral_contact", ["oral_ear_contact", "oral_oral_contact"]),
    ("oral_genital_contact", ["oral_genital_contact"]),
]

log = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Window and crop settings; stored in training_index.json."""
    conf_threshold: float = 0.2
    max_gap: int = 60
    window_size: int = 300
    stride: int = 150
    min_valid_len: int = 30
    downsample: int = 1          # recorded in the index only; no downsampling happens here
    crop_size: int = 160
    video_size: int = 448
    smooth_window: int = 10


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

_LABEL_DATA_CACHE: Dict[str, Dict[str, Any]] = {}


def load_label_data(label_path: Path) -> Dict[str, Any]:
    """Load the MABe22 label file and resolve FINETUNE_LABEL_GROUPS (cached per process)."""
    key = str(label_path)
    if key in _LABEL_DATA_CACHE:
        return _LABEL_DATA_CACHE[key]
    data = np.load(label_path, allow_pickle=True).item()
    vocab = data["vocabulary"]
    label_groups, label_names = [], []
    for name, members in FINETUNE_LABEL_GROUPS:
        missing = [m for m in members if m not in vocab]
        if missing:
            raise ValueError(f"Labels {missing} not found in vocabulary.")
        label_groups.append([vocab.index(m) for m in members])
        label_names.append(name)
    _LABEL_DATA_CACHE[key] = {
        "frame_number_map": data["frame_number_map"],
        "label_array": data["label_array"],
        "label_groups": label_groups,
        "label_names": label_names,
    }
    return _LABEL_DATA_CACHE[key]


def get_video_label_matrix(video_id: str, label_data: Dict[str, Any]) -> Optional[np.ndarray]:
    """Return the (C, T) bool label matrix of a video, or None if it has no labels."""
    frame_map = label_data["frame_number_map"]
    if video_id not in frame_map:
        log.warning(f"{video_id}: label mapping not found, skipping.")
        return None
    start, end = frame_map[video_id]
    label_array = label_data["label_array"]
    if end > label_array.shape[1]:
        log.warning(f"{video_id}: label range {start}:{end} exceeds label array length "
                    f"{label_array.shape[1]}, truncating.")
        end = label_array.shape[1]
    if end <= start:
        log.warning(f"{video_id}: empty label range {start}:{end}, skipping.")
        return None
    rows = []
    for indices in label_data["label_groups"]:
        vals = label_array[indices, start:end]
        rows.append(vals > 0.5 if vals.ndim == 1 else np.any(vals > 0.5, axis=0))
    return np.stack(rows, axis=0)


def build_frame_labels(label_matrix: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """(T, C) int64 multi-label targets; -1 on frames without any label or outside valid_mask."""
    num_classes, num_frames = label_matrix.shape
    frame_labels = np.full((num_frames, num_classes), -1, dtype=np.int64)
    keep = label_matrix.any(axis=0) & valid_mask
    if keep.any():
        frame_labels[keep] = label_matrix[:, keep].T.astype(np.int64)
    return frame_labels


# ---------------------------------------------------------------------------
# Segments and windows
# ---------------------------------------------------------------------------

def _segments_to_mask(segments: List[Tuple[int, int]], length: int) -> np.ndarray:
    """Convert inclusive (start, end) segments to a boolean mask."""
    mask = np.zeros(length, dtype=bool)
    for start, end in segments:
        if start < 0 or end < 0 or start >= length:
            continue
        end = min(end, length - 1)
        if end >= start:
            mask[start:end + 1] = True
    return mask


def _find_true_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Inclusive (start, end) runs of True in a 1D boolean mask."""
    segments, start = [], None
    for idx, val in enumerate(mask):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def find_valid_segments(mean_conf: np.ndarray, config: ProcessingConfig) -> List[Tuple[int, int]]:
    """Inclusive (start, end) runs of confident frames; gaps up to max_gap are bridged."""
    is_valid = mean_conf >= config.conf_threshold
    segments, start, invalid_count = [], None, 0
    for t in range(len(mean_conf)):
        if is_valid[t]:
            if start is None:
                start = t
            invalid_count = 0
        elif start is not None:
            invalid_count += 1
            if invalid_count > config.max_gap:
                end = t - invalid_count
                if end >= start:
                    segments.append((start, end))
                start, invalid_count = None, 0
    if start is not None:
        end = len(mean_conf) - 1 - invalid_count
        if end >= start:
            segments.append((start, end))
    return segments


def _window(seg_idx: int, start: int, end: int, real_len: int, padding: int) -> Dict:
    return {'segment_idx': seg_idx, 'start': start, 'end': end, 'real_len': real_len, 'padding': padding}


def generate_sliding_windows(segments: List[Tuple[int, int]], config: ProcessingConfig) -> List[Dict]:
    """Pretraining windows: slide over each valid segment.

    Segments shorter than min_valid_len are dropped; shorter than a window,
    they become one padded window. A tail of at least min_valid_len becomes a
    padded window; a shorter tail is covered by a full window ending at the
    segment end.
    """
    W, S = config.window_size, config.stride
    windows = []
    for seg_idx, (start, end) in enumerate(segments):
        seg_len = end - start + 1
        if seg_len < config.min_valid_len:
            continue
        if seg_len < W:
            windows.append(_window(seg_idx, start, end, seg_len, W - seg_len))
            continue
        t = start
        while t + W - 1 <= end:
            windows.append(_window(seg_idx, t, t + W - 1, W, 0))
            t += S
        if t <= end:
            remain = end - t + 1
            if remain >= config.min_valid_len:
                windows.append(_window(seg_idx, t, end, remain, W - remain))
            else:
                final_start = end - W + 1
                if final_start >= start and final_start != windows[-1]['start']:
                    windows.append(_window(seg_idx, final_start, end, W, 0))
    return windows


def _center_window_start(seg_start: int, seg_end: int, num_frames: int, window_size: int) -> int:
    """Start of a window that centers [seg_start, seg_end], clipped to the video."""
    if num_frames <= window_size:
        return 0
    start = (seg_start + seg_end) // 2 - window_size // 2
    return max(0, min(start, num_frames - window_size))


def _append_window(windows: List[Dict], seen: Set[Tuple[int, int]], start: int,
                   num_frames: int, window_size: int, segment_idx: int) -> None:
    """Append a (possibly padded) window, de-duplicated by (start, end)."""
    if start < 0 or start >= num_frames:
        return
    end = min(start + window_size - 1, num_frames - 1)
    if (start, end) in seen:
        return
    real_len = end - start + 1
    windows.append({'start': start, 'end': end, 'real_len': real_len,
                    'padding': window_size - real_len, 'segment_idx': segment_idx})
    seen.add((start, end))


def generate_labeled_windows(label_matrix: np.ndarray, segments: List[Tuple[int, int]],
                             config: ProcessingConfig) -> List[Dict]:
    """Fine-tuning windows centered on (or striding over) labeled segments of each class.

    Windows may extend beyond the labeled segment; those frames are masked in
    the frame labels.
    """
    num_classes, num_frames = label_matrix.shape
    windows: List[Dict] = []
    if num_classes == 0 or num_frames == 0:
        return windows
    valid_mask = _segments_to_mask(segments, num_frames)
    seen: Set[Tuple[int, int]] = set()
    W = config.window_size
    for label_idx in range(num_classes):
        cls_segments = _find_true_segments(label_matrix[label_idx] & valid_mask)
        for seg_idx, (seg_start, seg_end) in enumerate(cls_segments):
            seg_len = seg_end - seg_start + 1
            if seg_len < config.min_valid_len:
                continue
            if seg_len <= W:
                win_start = _center_window_start(seg_start, seg_end, num_frames, W)
                _append_window(windows, seen, win_start, num_frames, W, seg_idx)
            else:
                for win_start in range(seg_start, seg_end + 1, config.stride):
                    _append_window(windows, seen, win_start, num_frames, W, seg_idx)
    return windows


def _windowize_array(data: np.ndarray, windows: List[Dict], window_size: int,
                     pad_value: float = 0) -> np.ndarray:
    """(T, ...) -> (N, window_size, ...): each window's real frames, padded with pad_value."""
    if not windows:
        return np.empty((0, window_size) + data.shape[1:], dtype=data.dtype)
    out = []
    for win in windows:
        seg = data[win["start"]:win["start"] + win["real_len"]]
        if win["padding"] > 0:
            pad = np.full((win["padding"],) + seg.shape[1:], pad_value, dtype=data.dtype)
            seg = np.concatenate([seg, pad], axis=0)
        out.append(seg)
    return np.stack(out, axis=0)


def _windowize_video(frames: np.ndarray, windows: List[Dict], window_size: int) -> np.ndarray:
    """(T, H, W, 3) -> (N, window_size, H, W, 3) of raw frames; padded windows keep real video."""
    if not windows:
        return np.empty((0, window_size) + frames.shape[1:], dtype=frames.dtype)
    for win in windows:
        if win["start"] + window_size > len(frames):
            raise ValueError(f"Window {win['start']}:{win['start'] + window_size} "
                             f"exceeds video length {len(frames)}.")
    return np.stack([frames[w["start"]:w["start"] + window_size] for w in windows], axis=0)


# ---------------------------------------------------------------------------
# Pose, centers and crops
# ---------------------------------------------------------------------------

def extract_mouse_data(df: pd.DataFrame, mouse_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keypoints (T, K, 2), confidence (T, K) with NaN -> 0, and (T,) frames with any NaN confidence."""
    scorer = df.columns.get_level_values('scorer')[0]
    xy = [np.stack([df[(scorer, mouse_id, bp, 'x')].values,
                    df[(scorer, mouse_id, bp, 'y')].values], axis=1) for bp in KEYPOINTS]
    conf = [df[(scorer, mouse_id, bp, 'likelihood')].values for bp in KEYPOINTS]
    keypoints = np.stack(xy, axis=1)
    confidence = np.stack(conf, axis=1)
    conf_nonfinite = ~np.isfinite(confidence).all(axis=1)
    return keypoints, np.nan_to_num(confidence, nan=0.0), conf_nonfinite


def compute_center_trajectory(keypoints: np.ndarray, is_valid: np.ndarray,
                              video_size: int = 448) -> np.ndarray:
    """(T, 2) mean keypoint position; invalid frames hold the last valid center.

    Frames before the first valid one use the mean valid center (or the image
    center if no frame is valid).
    """
    centers = np.nanmean(keypoints, axis=1)
    is_valid = is_valid & ~np.isnan(centers).any(axis=1)
    if is_valid.any():
        fallback = centers[is_valid].mean(axis=0)
    else:
        fallback = np.array([video_size / 2, video_size / 2])
    last_valid = None
    for t in range(len(centers)):
        if is_valid[t]:
            last_valid = centers[t].copy()
        else:
            centers[t] = last_valid if last_valid is not None else fallback
    return centers


def smooth_centers(centers: np.ndarray, window: int) -> np.ndarray:
    """Moving average of a (T, 2) trajectory."""
    smoothed = np.copy(centers)
    for dim in range(2):
        smoothed[:, dim] = uniform_filter1d(centers[:, dim], size=window, mode='nearest')
    return smoothed


def crop_frame(frame: np.ndarray, center: np.ndarray, crop_size: int, video_size: int) -> np.ndarray:
    """(crop_size, crop_size, 3) crop around center (x, y); outside the frame is black."""
    half = crop_size // 2
    x1, y1 = int(center[0]) - half, int(center[1]) - half
    x2, y2 = x1 + crop_size, y1 + crop_size
    cropped = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(video_size, x2), min(video_size, y2)
    if sx2 > sx1 and sy2 > sy1:
        cropped[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = frame[sy1:sy2, sx1:sx2]
    return cropped


# ---------------------------------------------------------------------------
# Video I/O (ffmpeg)
# ---------------------------------------------------------------------------

def _parse_fraction(value: Optional[str]) -> Optional[float]:
    """Parse an ffprobe rate such as '30000/1001'."""
    if not value or value == "0/0":
        return None
    try:
        if "/" in value:
            num, den = (float(x) for x in value.split("/", 1))
            return num / den if den else None
        return float(value)
    except ValueError:
        return None


def probe_video(video_path: Path) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[int]]:
    """(width, height, fps, nb_frames) via ffprobe; Nones on failure."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames",
           "-of", "json", str(video_path)]
    try:
        stream = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)["streams"][0]
        nb = stream.get("nb_frames")
        return (int(stream.get("width")), int(stream.get("height")),
                _parse_fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
                int(nb) if nb and str(nb).isdigit() else None)
    except (subprocess.CalledProcessError, KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
        log.error(f"ffprobe failed for {video_path}: {exc}")
        return None, None, None, None


def read_video_frames(video_path: Path, width: int, height: int) -> Optional[np.ndarray]:
    """All frames as a (T, H, W, 3) uint8 BGR array, decoded by ffmpeg."""
    if width <= 0 or height <= 0:
        log.error(f"Invalid video dimensions: {width}x{height}")
        return None
    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video_path),
                           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
    if proc.returncode != 0:
        log.error(f"ffmpeg failed for {video_path}: {proc.stderr.decode(errors='ignore')}")
        return None
    frame_bytes = width * height * 3
    num_frames = len(proc.stdout) // frame_bytes
    if len(proc.stdout) % frame_bytes:
        log.warning(f"{video_path.name}: byte count not divisible by frame size; "
                    f"truncating to {num_frames} frames")
    return np.frombuffer(proc.stdout[:num_frames * frame_bytes], dtype=np.uint8).reshape(
        (num_frames, height, width, 3))


# ---------------------------------------------------------------------------
# One video
# ---------------------------------------------------------------------------

def _mouse_cache(keypoints: np.ndarray, confidence: np.ndarray, invalid: np.ndarray,
                 other_keypoints: np.ndarray, frames: np.ndarray, windows: List[Dict],
                 segments: List[Tuple[int, int]], frame_labels: Optional[np.ndarray],
                 config: ProcessingConfig) -> Dict[str, Any]:
    """Windowed tensors of one mouse (see the module docstring)."""
    W = config.window_size
    is_valid = (confidence.mean(axis=1) >= config.conf_threshold) & ~invalid
    centers = smooth_centers(compute_center_trajectory(keypoints, is_valid, config.video_size),
                             config.smooth_window)
    crops = np.stack([crop_frame(frame, centers[t], config.crop_size, config.video_size)
                      for t, frame in enumerate(frames[:len(keypoints)])], axis=0)
    # Egocentric coordinates use the unsmoothed center; the second mouse is the
    # first other individual in the file, in this mouse's frame of reference.
    origin = compute_center_trajectory(keypoints, is_valid, config.video_size)[:, None, :]
    cache = {
        'keypoints': torch.from_numpy(_windowize_array(keypoints, windows, W)).float(),            # (N, T, K, 2)
        'confidence': torch.from_numpy(_windowize_array(confidence, windows, W)).float(),          # (N, T, K)
        'ego_keypoints': torch.from_numpy(_windowize_array(keypoints - origin, windows, W)).float(),
        'neighbor_keypoints': torch.from_numpy(_windowize_array(other_keypoints - origin, windows, W)).float(),
        'cropped_video': torch.from_numpy(_windowize_video(crops, windows, W)).permute(0, 1, 4, 2, 3),  # (N, T, 3, H, W)
        'mask': torch.from_numpy(_windowize_array(is_valid, windows, W)).bool(),                   # (N, T)
        'centers': torch.from_numpy(_windowize_array(centers, windows, W)).float(),                # (N, T, 2)
        'segments': segments,
        'windows': windows,
    }
    if frame_labels is not None:
        labels = _windowize_array(frame_labels, windows, W, pad_value=-1)
        cache['frame_labels'] = torch.from_numpy(labels).long()                                    # (N, T, C)
        cache['frame_mask'] = torch.from_numpy((labels != -1).any(axis=-1)).bool()                 # (N, T)
    return cache


def process_single_video(video_path: Path, h5_path: Path, cache_dir: Path,
                         config: ProcessingConfig, stage: str, label_path: Path) -> Dict[str, Any]:
    """Write <cache_dir>/<video_id>.pt.

    Returns {"status": "ok" | "no_windows" | "frame_count_mismatch" | "failed",
    "metadata": index entry (status ok only), "warnings": window counts}.
    """
    video_id = video_path.stem
    warnings = {"filtered_windows": 0, "aligned_dropped_windows": 0}
    result = lambda status, metadata=None: {"video_id": video_id, "status": status,
                                            "metadata": metadata, "warnings": warnings}
    try:
        df = pd.read_hdf(h5_path)
        mouse_ids = df.columns.get_level_values('individuals').unique().tolist()
        if len(mouse_ids) != 3:
            log.warning(f"Skipping {video_id}: expected 3 mice, found {len(mouse_ids)}")
            return result("failed")
        log.info(f"Processing {video_id} with mice: {mouse_ids}")

        label_data = load_label_data(label_path)
        label_matrix = get_video_label_matrix(video_id, label_data)
        if label_matrix is None:
            return result("failed")
        label_names = label_data["label_names"]

        pose = {m: extract_mouse_data(df, m) for m in mouse_ids}
        num_frames = min(len(pose[mouse_ids[0]][0]), label_matrix.shape[1])
        if num_frames < len(pose[mouse_ids[0]][0]):
            log.warning(f"{video_id}: truncating keypoints to {num_frames} frames to match labels.")
        if num_frames < label_matrix.shape[1]:
            log.warning(f"{video_id}: truncating labels to {num_frames} frames to match keypoints.")
        label_matrix = label_matrix[:, :num_frames]

        # Frames with any non-finite keypoint or confidence are zeroed and invalid.
        keypoints, confidence, invalid, segments, windows, frame_labels = {}, {}, {}, {}, {}, {}
        for m in mouse_ids:
            kp, conf, conf_nonfinite = (x[:num_frames] for x in pose[m])
            bad = ~np.isfinite(kp).all(axis=(1, 2)) | conf_nonfinite
            if bad.any():
                kp, conf = kp.copy(), conf.copy()
                kp[bad] = 0
                conf[bad] = 0
            keypoints[m], confidence[m], invalid[m] = kp, conf, bad
            segments[m] = find_valid_segments(conf.mean(axis=1), config)
            valid_mask = _segments_to_mask(segments[m], num_frames) & ~bad
            frame_labels[m] = build_frame_labels(label_matrix, valid_mask)
            if stage == "finetune":
                windows[m] = generate_labeled_windows(label_matrix, segments[m], config)
            else:
                windows[m] = generate_sliding_windows(segments[m], config)
            log.debug(f"  Mouse {m}: {len(segments[m])} segments, {len(windows[m])} windows")
        if not any(windows.values()):
            log.info(f"{video_id}: no valid windows for any mouse; skipping video.")
            return result("no_windows")

        width, height, fps, probed_frames = probe_video(video_path)
        if width is None or height is None:
            log.warning(f"{video_id}: ffprobe failed; assuming {config.video_size}x{config.video_size}")
            width = height = config.video_size
        frames = read_video_frames(video_path, width, height)
        if frames is None:
            return result("failed")
        if probed_frames is not None and probed_frames != len(frames):
            log.warning(f"{video_id}: ffprobe frames={probed_frames} but decoded frames={len(frames)}")
        if len(frames) == num_frames + 1:
            log.warning(f"{video_id}: video has one frame more than the pose; dropping it.")
            frames = frames[:num_frames]
        elif len(frames) != num_frames:
            log.warning(f"{video_id}: frame count mismatch (video {len(frames)}, pose {num_frames}); "
                        "skipping video.")
            return result("frame_count_mismatch")

        # Video crops always span a full window, so drop windows that would run
        # past the video; then keep only windows that all mice share.
        for m in mouse_ids:
            kept = [w for w in windows[m] if w["start"] + config.window_size <= len(frames)]
            if len(kept) != len(windows[m]):
                log.warning(f"{video_id}:{m} dropped {len(windows[m]) - len(kept)} windows "
                            "that run past the video.")
                warnings["filtered_windows"] += len(windows[m]) - len(kept)
            windows[m] = kept
        common = sorted(set.intersection(*({(w["start"], w["end"]) for w in windows[m]} for m in mouse_ids)))
        if not common:
            log.info(f"{video_id}: no aligned windows across mice; skipping video.")
            return result("no_windows")
        for m in mouse_ids:
            by_key = {(w["start"], w["end"]): w for w in windows[m]}
            dropped = len(windows[m]) - len(common)
            if dropped:
                warnings["aligned_dropped_windows"] += dropped
                log.warning(f"{video_id}:{m} dropped {dropped} windows for cross-mouse alignment.")
            windows[m] = [by_key[k] for k in common]

        cached = {}
        for m in mouse_ids:
            other = next(o for o in mouse_ids if o != m)
            cached[m] = _mouse_cache(keypoints[m], confidence[m], invalid[m], keypoints[other],
                                     frames, windows[m], segments[m], frame_labels[m], config)
        cached["_meta"] = {
            "mouse_ids": mouse_ids,
            "video_size": config.video_size,
            # "truncated_windows" is always 0; kept so the cache format is unchanged.
            "warning_counts": {"filtered_windows": warnings["filtered_windows"], "truncated_windows": 0,
                               "aligned_dropped_windows": warnings["aligned_dropped_windows"]},
        }
        torch.save(cached, cache_dir / f"{video_id}.pt")

        metadata = {'video_id': video_id, 'video_path': str(video_path), 'h5_path': str(h5_path),
                    'total_frames': len(frames), 'fps': fps,
                    'mice': {m: _mouse_index_entry(cached[m], label_names) for m in mouse_ids}}
        log.info(f"Processed {video_id}")
        return result("ok", metadata)
    except Exception as exc:
        log.error(f"Error processing {video_id}: {exc}", exc_info=True)
        return result("failed")


def _mouse_index_entry(mouse_cache: Dict[str, Any], label_names: List[str]) -> Dict[str, Any]:
    entry = {'num_segments': len(mouse_cache['segments']), 'num_windows': len(mouse_cache['windows']),
             'segments': mouse_cache['segments'], 'windows': mouse_cache['windows']}
    if label_names:
        counts = {name: 0 for name in label_names}
        frame_labels = mouse_cache.get('frame_labels')
        if frame_labels is not None:
            fl = np.asarray(frame_labels.cpu().numpy() if isinstance(frame_labels, torch.Tensor) else frame_labels)
            for idx, name in enumerate(label_names):
                counts[name] = int((fl[..., idx] == 1).sum())
        entry['label_counts'] = counts
    return entry


def index_entry_from_cache(cache_path: Path, video_path: Path, h5_path: Path,
                           label_names: List[str]) -> Optional[Dict[str, Any]]:
    """Rebuild an index entry from an existing cache (for runs interrupted before the index was written)."""
    try:
        cached = torch.load(cache_path, map_location="cpu")
    except Exception as exc:
        log.warning(f"Failed to load cache {cache_path}: {exc}")
        return None
    mouse_ids = cached.get("_meta", {}).get("mouse_ids") or [k for k in cached if k != "_meta"]
    if not mouse_ids:
        return None
    _, _, fps, probed_frames = probe_video(video_path)
    windows = cached[mouse_ids[0]].get("windows") or []
    inferred = max((w.get("end", -1) for w in windows), default=-1) + 1 or None
    mice = {m: _mouse_index_entry(cached[m], label_names) for m in mouse_ids if cached.get(m)}
    if not mice:
        return None
    return {"video_id": video_path.stem, "video_path": str(video_path), "h5_path": str(h5_path),
            "total_frames": probed_frames or inferred, "fps": fps, "mice": mice}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Preprocess Mouse Triplet (MABe22) pose and video crops.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--stage', required=True, choices=list(STAGE_WINDOWS))
    p.add_argument('--h5-dir', type=Path, default=DATA_ROOT / 'deeplabcut_filtered',
                   help='DeepLabCut pose files (<video_id>DLC..._el.h5)')
    p.add_argument('--video-dir', type=Path, default=DATA_ROOT / 'mouse_submission_mp4_448')
    p.add_argument('--label-file', type=Path, default=DATA_ROOT / 'mouse_submission_labels.npy')
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                   help='writes <output-dir>/<stage>/{caches/,training_index.json}')
    p.add_argument('--conf-threshold', type=float, default=ProcessingConfig.conf_threshold)
    p.add_argument('--max-gap', type=int, default=ProcessingConfig.max_gap,
                   help='longest run of low-confidence frames bridged inside a segment')
    p.add_argument('--window-size', type=int, default=None, help='default: 300 (pretrain) / 200 (finetune)')
    p.add_argument('--stride', type=int, default=None, help='default: 150 (pretrain) / 200 (finetune)')
    p.add_argument('--min-valid-len', type=int, default=ProcessingConfig.min_valid_len)
    p.add_argument('--crop-size', type=int, default=ProcessingConfig.crop_size)
    p.add_argument('--video-ids', nargs='+', default=None, help='process only these videos')
    p.add_argument('--limit', type=int, default=None, help='process only the first N pose files')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--overwrite', action='store_true', help='rebuild existing caches and the index')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    default_w, default_s = STAGE_WINDOWS[args.stage]
    config = ProcessingConfig(conf_threshold=args.conf_threshold, max_gap=args.max_gap,
                              window_size=args.window_size or default_w, stride=args.stride or default_s,
                              min_valid_len=args.min_valid_len, crop_size=args.crop_size)
    stage_dir = args.output_dir / args.stage
    cache_dir = stage_dir / "caches"
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = stage_dir / 'training_index.json'
    log.info(f"Stage {args.stage}, config {asdict(config)}")

    if not args.label_file.exists():
        raise SystemExit(f"Label file not found: {args.label_file}")
    label_names = load_label_data(args.label_file)["label_names"]

    # Keep entries of an existing index (same settings) unless --overwrite.
    entries: Dict[str, Dict] = {}
    order: List[str] = []
    if index_path.exists() and not args.overwrite:
        existing = json.load(open(index_path))
        if existing.get('config') and existing['config'] != asdict(config):
            raise SystemExit(f"{index_path} was built with different settings; use --overwrite to rebuild.")
        if existing.get('stage', 'pretrain') != args.stage:
            raise SystemExit(f"{index_path} is for stage '{existing.get('stage')}'; use --overwrite to rebuild.")
        if existing.get('label_names') and existing['label_names'] != label_names:
            raise SystemExit(f"{index_path} has different labels; use --overwrite to rebuild.")
        order = [v['video_id'] for v in existing.get('videos', [])]
        entries = {v['video_id']: v for v in existing.get('videos', [])}

    h5_files = sorted(args.h5_dir.glob('*_el.h5'))
    if args.video_ids:
        wanted = set(args.video_ids)
        h5_files = [h for h in h5_files if h.name.split('DLC')[0] in wanted]
    if args.limit:
        h5_files = h5_files[:args.limit]
    log.info(f"Found {len(h5_files)} pose files")

    todo, n_existing = [], 0
    for h5_path in h5_files:
        video_id = h5_path.name.split('DLC')[0]
        video_path = args.video_dir / f"{video_id}.mp4"
        if not video_path.exists():
            log.warning(f"Video not found for {h5_path.name}")
            continue
        if video_id not in order:
            order.append(video_id)
        cache_path = cache_dir / f"{video_id}.pt"
        if cache_path.exists() and not args.overwrite:
            n_existing += 1
            if video_id not in entries:
                recovered = index_entry_from_cache(cache_path, video_path, h5_path, label_names)
                if recovered:
                    entries[video_id] = recovered
                else:
                    log.warning(f"Skipping {video_id}: cache exists but has no usable metadata "
                                "(use --overwrite to rebuild it)")
            continue
        todo.append((video_path, h5_path))
    log.info(f"{len(todo)} videos to process, {n_existing} with existing caches")

    status_counts = {"ok": 0, "no_windows": 0, "frame_count_mismatch": 0, "failed": 0}
    window_warnings = {"filtered_windows": 0, "aligned_dropped_windows": 0}
    jobs = [(v, h, cache_dir, config, args.stage, args.label_file) for v, h in todo]
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            results = ex.map(process_single_video, *zip(*jobs)) if jobs else []
            results = list(tqdm(results, total=len(jobs), desc="Processing videos"))
    else:
        results = [process_single_video(*job) for job in tqdm(jobs, desc="Processing videos")]
    for res in results:
        status_counts[res["status"]] += 1
        for k in window_warnings:
            window_warnings[k] += res["warnings"][k]
        if res["status"] == "ok":
            entries[res["video_id"]] = res["metadata"]

    index = {'config': asdict(config), 'stage': args.stage, 'label_names': label_names,
             'precomputed_data_dir': str(cache_dir),
             'videos': [entries[v] for v in order if v in entries]}
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    log.info(f"Wrote {index_path}: {len(index['videos'])} videos; this run: {status_counts}, "
             f"existing caches: {n_existing}, window warnings: {window_warnings}")


if __name__ == '__main__':
    main()
