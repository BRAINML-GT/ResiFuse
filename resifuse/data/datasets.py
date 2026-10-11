import json
import numpy as np
import subprocess
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import csv
from typing import List, Tuple, Dict, Any, Optional
from collections import OrderedDict
from pathlib import Path
import decord  # type: ignore
from decord import cpu as decord_cpu
from decord._ffi.base import DECORDError  # type: ignore
from sklearn.preprocessing import StandardScaler


def _is_rank0() -> bool:
    rank = os.environ.get("RANK")
    if rank is None:
        return True
    try:
        return int(rank) == 0
    except ValueError:
        return True


def _read_video(path: str):
    """
    Read video into torch.Tensor [T,H,W,C] uint8 without using torchvision.io to avoid deprecation warning.
    """

    os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")
    vr = decord.VideoReader(path, ctx=decord_cpu())
    total = len(vr)
    for n in range(total, 0, -1):
        try:
            frames = vr.get_batch(range(n))  # (n,H,W,3), uint8
            return torch.from_numpy(frames.asnumpy())
        except DECORDError:
            continue
    raise RuntimeError(f"Failed to read video {path} via decord (even after trimming EOF).")


# Per-process VideoReader cache: same video used by many consecutive windows,
# so keep readers open to save open+seek cost. Max 4 readers keeps memory
# bounded even when workers iterate all 20 Open Field videos.
_VIDEO_READER_CACHE: "OrderedDict[str, decord.VideoReader]" = OrderedDict()
_VIDEO_READER_CACHE_MAX = 4


def _get_video_reader(path: str) -> "decord.VideoReader":
    vr = _VIDEO_READER_CACHE.get(path)
    if vr is not None:
        _VIDEO_READER_CACHE.move_to_end(path)
        return vr
    os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")
    vr = decord.VideoReader(path, ctx=decord_cpu())
    _VIDEO_READER_CACHE[path] = vr
    _VIDEO_READER_CACHE.move_to_end(path)
    while len(_VIDEO_READER_CACHE) > _VIDEO_READER_CACHE_MAX:
        _VIDEO_READER_CACHE.popitem(last=False)
    return vr


def _read_video_window_fast(path: str, start: int, end: int, ds_stride: int = 1) -> torch.Tensor:
    """Decode only the frames in [start, end) with stride, using a cached VideoReader.

    Replaces the full-video decode path. For typical Open Field windows (T=128 per
    window in a 10K-frame video), this is 50-100x faster than _read_video + slice.
    """
    vr = _get_video_reader(path)
    total = len(vr)
    end = min(end, total)
    if start >= end:
        return torch.zeros(0, dtype=torch.uint8)
    indices = list(range(start, end, max(1, ds_stride)))
    try:
        frames = vr.get_batch(indices)
        return torch.from_numpy(frames.asnumpy())  # [T, H, W, 3]
    except DECORDError:
        kept = []
        for i in indices:
            try:
                kept.append(vr[i].asnumpy())
            except DECORDError:
                break
        if not kept:
            raise
        import numpy as _np
        return torch.from_numpy(_np.stack(kept, axis=0))


def _majority_vote_frame_labels(labels_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    if not labels_list:
        return torch.empty(0), torch.empty(0, dtype=torch.bool)
    normed = []
    for labels in labels_list:
        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        normed.append(labels)
    labels_t = torch.stack(normed, dim=0)
    valid = labels_t != -1
    valid_count = valid.sum(dim=0)
    pos_count = ((labels_t == 1) & valid).sum(dim=0)
    majority = pos_count > (valid_count / 2.0)
    maj_labels = torch.full_like(labels_t[0], -1)
    has_valid = valid_count > 0
    if has_valid.any():
        maj_labels[has_valid] = majority[has_valid].to(dtype=labels_t.dtype)
    maj_mask = has_valid.any(dim=-1)
    return maj_labels, maj_mask


class OpenFieldDataset(Dataset):
    """
    Open Field dataset (data/openfield).
    - markers: DeepLabCut CSV (keypoints)
    - videos: mp4
    - labels-hand: per-frame one-hot CSV (Background, Supported, Unsupported, Grooming)
    Two modes: pretrain (unlabeled sliding windows) and finetune (windows anchored on labels).
    """
    LABEL_ORDER = ["Background", "Supported", "Unsupported", "Grooming"]

    def __init__(
        self,
        root_dir: str = "data/openfield",
        seq_len: int = 64,
        pretrain_stride: int | None = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split: str = "train",
        stage: str = "pretrain",
        dtype: torch.dtype = torch.bfloat16,
        ds_stride: int = 1,
        max_neighbors: int = 2,
        use_neighbor_feats: bool = True,
        body_length: float | None = None,
        num_joints: int | None = None,
        min_body_length: float = 10.0,
        frame_resize: int | Tuple[int, int] | None = None,
        frame_resize_align: int = 32,
        cache_dir: str | None = None,
        use_cache: bool = False,
        conf_mask_thresh: float = 0.3,
        dynamic_cache_split: bool | None = None,
        oversample_grooming_factor: int = 1,
        vision_feat_dir: str | None = None,
        use_vision_cache: bool = False,
        use_motion_scaler: bool = False,
        motion_scaler: StandardScaler | None = None,
        marker_subject_centric: bool = True,
        motion_include_velocity: bool = True,
        motion_include_confidence: bool = True,
        drop_empty_frame_mask: bool = False,
        preload: bool = False,
        cache_size: int = 256,
        prefer_vision_npy: bool = True,
        mmap_load: bool = True,
        load_frames: bool = True,
        openfield_num_classes: int = 2,
    ):
        super().__init__()
        self.root = root_dir
        self.seq_len = seq_len                     # length after optional downsampling
        self.ds_stride = max(1, int(ds_stride))    # temporal stride applied at loading time
        self.seq_len_raw = self.seq_len * self.ds_stride
        self.pretrain_stride = pretrain_stride or max(1, seq_len // 2)  # in downsampled steps
        self.pretrain_stride_raw = self.pretrain_stride * self.ds_stride
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split = split
        self.stage = stage
        self.dtype = dtype
        self.max_neighbors = max_neighbors
        if use_neighbor_feats:
            print("[DataLoader] OpenFieldDataset forces use_neighbor_feats=False for compatibility.")
        self.max_neighbors = 0
        self.use_neighbor_feats = False
        self.body_length = body_length
        self.num_joints: int | None = num_joints
        self.min_body_length = min_body_length
        self.frame_resize_align = max(1, int(frame_resize_align))
        self._warned_resize_align = False
        self.frame_resize = self._normalize_resize_size(frame_resize)
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        # None -> auto: use combined meta if exists; True/False to force behavior
        self.dynamic_cache_split = dynamic_cache_split
        self.cache_meta: Dict[str, Any] | None = None
        self.cache_entries: List[Dict[str, Any]] | None = None
        self.vision_feat_dir = vision_feat_dir
        self.use_vision_cache = use_vision_cache and vision_feat_dir is not None
        self._warned_missing_vision_feat = False
        self.conf_mask_thresh = conf_mask_thresh
        self._logged_conf_mask = False  # avoid spamming mask stats
        self.oversample_grooming_factor = max(1, int(oversample_grooming_factor))
        # Label granularity. 2-class (default, main table): the two rearing classes
        # are fused into Rearing (see _read_labels), then mapped to head-A indices
        # {0=Rearing, 1=Grooming, -1=ignore}. 3-class (rebuttal diagnostic): keep
        # Supported/Unsupported/Grooming separate (_read_labels_3class) mapped to
        # {0=Supported, 1=Unsupported, 2=Grooming, -1=ignore}. Nothing else changes.
        self.openfield_num_classes = int(openfield_num_classes)
        if self.openfield_num_classes == 3:
            # _read_labels_3class already emits final class ids 0/1/2 (-1 ignore),
            # so label_map is identity over {0,1,2}.
            self.label_map = {0: 0, 1: 1, 2: 2}
        else:
            # Fused labels: Background=0 (ignored), Rearing(Supported/Unsupported)=1,
            # Grooming=2. Head A: 0=Rearing, 1=Grooming.
            self.label_map = {0: -1, 1: 0, 2: 1, 3: -1}
        self._warned_no_label_for_oversample = False

        self.video_dir = os.path.join(self.root, "videos")
        self.kp_dir = os.path.join(self.root, "markers")
        self.label_dir = os.path.join(self.root, "labels-hand")
        self.use_motion_scaler = use_motion_scaler
        self.motion_scaler: StandardScaler | None = motion_scaler
        self.marker_subject_centric = marker_subject_centric
        self.motion_include_velocity = motion_include_velocity
        self.motion_include_confidence = motion_include_confidence
        self.drop_empty_frame_mask = drop_empty_frame_mask
        self.load_frames = bool(load_frames)
        self.preload = bool(preload)
        self._cache_size = max(1, int(cache_size))
        self.prefer_vision_npy = bool(prefer_vision_npy)
        self.mmap_load = bool(mmap_load)
        self._pt_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._preloaded_data: Dict[str, Dict[str, Any]] = {}

        self.samples = self._gather_files()
        if self.use_cache and self.cache_dir:
            self._try_load_cache_meta()
        self.index = self.cache_entries if self.cache_entries is not None else self._build_index()
        if self.preload and self.cache_entries:
            self._preload_all_cache_files()
        if self.use_motion_scaler and self.motion_scaler is None:
            if self.cache_entries is not None:
                print("[DataLoader] use_motion_scaler=True but cache_entries loaded; please pass in a fitted motion_scaler.")
            elif self.split == "train":
                print("[DataLoader] Fitting motion StandardScaler on training windows...")
                self.motion_scaler = self._fit_motion_scaler()

    def _gather_files(self) -> List[Dict[str, str]]:
        files = []
        for fname in sorted(os.listdir(self.video_dir)):
            if not fname.endswith(".mp4"):
                continue
            stem = os.path.splitext(fname)[0]
            kp_path = os.path.join(self.kp_dir, f"{stem}_labeled.csv")
            label_path = os.path.join(self.label_dir, f"{stem}_labels.csv")
            if not (os.path.exists(kp_path) and os.path.exists(label_path)):
                continue
            files.append({
                "stem": stem,
                "video": os.path.join(self.video_dir, fname),
                "kps": kp_path,
                "labels": label_path,
            })
        # split train/val/test by ratios
        n = len(files)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        if self.split == "train":
            return files[:train_end]
        elif self.split == "val":
            return files[train_end:val_end]
        else:
            return files[val_end:]

    def _read_dlc_csv(self, path: str, num_joints: int | None = None) -> np.ndarray:
        """Parse DeepLabCut CSV -> (T, K, 3) float32, optionally slicing to first K joints."""
        with open(path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        data_rows = rows[3:]
        # drop potential trailing empty cells, coerce blanks to 0.0
        data_clean = []
        for r in data_rows:
            row_vals = r
            if row_vals and row_vals[-1] == "":
                row_vals = row_vals[:-1]
            parsed = []
            for x in row_vals:
                if x == "" or x is None:
                    parsed.append(0.0)
                else:
                    parsed.append(float(x))
            data_clean.append(parsed)
        data = np.array(data_clean, dtype=np.float32)
        # DLC CSV has a leading frame index column; remove it so each joint keeps (x, y, likelihood)
        if data.shape[1] % 3 == 1:
            data = data[:, 1:]
        if data.shape[1] % 3 != 0:
            raise ValueError(f"Malformed DLC CSV: columns ({data.shape[1]}) not divisible by 3 after trimming frame index: {path}")
        num_cols = data.shape[1]
        total_joints = num_cols // 3
        use_joints = num_joints if num_joints is not None else total_joints
        use_joints = min(use_joints, total_joints)
        data = data[:, : use_joints * 3]
        data = data.reshape(data.shape[0], use_joints, 3)
        return data

    def _resolve_cache_path(self, entry: Dict[str, Any], root_override: str | None = None) -> str:
        base = root_override or self.cache_dir
        if not base:
            raise RuntimeError("cache_dir is not set but cache path was requested.")
        if "relpath" in entry:
            return os.path.join(base, entry["relpath"])
        if "path" in entry:
            if os.path.isabs(entry["path"]):
                return entry["path"]
            return os.path.join(base, entry["path"])
        raise KeyError("Cache entry missing 'relpath' or 'path'.")

    def _torch_load_safe(self, path: str) -> Any:
        """torch.load with mmap when supported (PyTorch >= 2.1); graceful fallback."""
        if self.mmap_load:
            try:
                return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            except TypeError:
                pass  # older PyTorch without the mmap kwarg
            except RuntimeError:
                pass  # non-mmap-able payload (e.g. custom picklables)
        return torch.load(path, map_location="cpu")

    def _load_pt_file(self, path: str) -> Dict[str, Any]:
        """LRU-cached / preloaded torch.load. Caller MUST NOT mutate the returned dict in place."""
        if self.preload:
            cached = self._preloaded_data.get(path)
            if cached is not None:
                return cached
        if path in self._pt_cache:
            self._pt_cache.move_to_end(path)
            return self._pt_cache[path]
        data = self._torch_load_safe(path)
        self._pt_cache[path] = data
        self._pt_cache.move_to_end(path)
        while len(self._pt_cache) > self._cache_size:
            self._pt_cache.popitem(last=False)
        return data

    def _preload_all_cache_files(self) -> None:
        if not self.cache_entries:
            return
        if _is_rank0():
            print("[DataLoader] Preloading Open Field cache windows into memory...")
        seen_pt: set[str] = set()
        seen_vis: set[str] = set()
        for entry in self.cache_entries:
            cache_path = self._resolve_cache_path(entry)
            if cache_path not in seen_pt and os.path.exists(cache_path):
                self._preloaded_data[cache_path] = self._torch_load_safe(cache_path)
                seen_pt.add(cache_path)
            if self.use_vision_cache and self.vision_feat_dir:
                vpath = self._resolve_cache_path(entry, root_override=self.vision_feat_dir)
                if vpath in seen_vis:
                    continue
                seen_vis.add(vpath)
                if os.path.exists(vpath):
                    self._preloaded_data[vpath] = self._torch_load_safe(vpath)
        if _is_rank0():
            print(f"  Preloaded {len(seen_pt)} window files, {len(seen_vis)} vision files.")

    def _maybe_load_vision_feats(self, entry: Dict[str, Any]) -> tuple[torch.Tensor | None, int]:
        if not (self.use_vision_cache and self.vision_feat_dir):
            return None, 1
        feat_path = self._resolve_cache_path(entry, root_override=self.vision_feat_dir)
        # Fast path: .npy sidecar loaded via mmap. Expected as "<stem>_vision.npy"
        # next to the .pt, matching the MaBe cache layout (see resifuse/features/hf_mabe.py).
        # The .npy has no ds_stride metadata, so we trust the configured ds_stride.
        if self.prefer_vision_npy:
            base, _ext = os.path.splitext(feat_path)
            for npy_candidate in (base + "_vision.npy", base + ".npy"):
                if os.path.exists(npy_candidate):
                    # .copy() materializes the mmap window into writable RAM,
                    # matching MaBeMouseDataset and silencing PyTorch's
                    # "non-writable numpy array" warning on from_numpy.
                    arr = np.load(npy_candidate, mmap_mode="r").copy()
                    feats = torch.from_numpy(arr)
                    if feats.dim() == 2:
                        feats = feats.unsqueeze(0)
                    return feats, max(1, int(self.ds_stride))
        if not os.path.exists(feat_path):
            # Fail fast: silently returning None produces a sample dict without
            # the "vision_feats" key, which breaks default_collate in the next
            # batch (heterogeneous keys). Raising here tells the user exactly
            # which window's cache is missing so they can re-run the precompute.
            base_for_msg, _ = os.path.splitext(feat_path)
            raise RuntimeError(
                f"[DataLoader] use_vision_cache=True but vision feature file is missing.\n"
                f"  expected one of: {base_for_msg}_vision.npy | {base_for_msg}.npy | {feat_path}\n"
                f"  Re-run resifuse/features/hf_openfield.py (without --overwrite to "
                f"only fill missing files), or disable use_vision_cache."
            )
        data = self._load_pt_file(feat_path)
        feat_stride = 1
        feats = data
        if isinstance(data, dict):
            feats = data.get("vision_feats")
            feat_stride = int(data.get("ds_stride", 1) or 1)
        if feat_stride != self.ds_stride:
            raise RuntimeError(
                f"[DataLoader] vision feats ds_stride ({feat_stride}) != current ds_stride ({self.ds_stride}); "
                "regenerate vision cache or align ds_stride in config."
            )
        if feats is None:
            return None, max(1, feat_stride)
        if not isinstance(feats, torch.Tensor):
            feats = torch.tensor(feats)
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        return feats, max(1, feat_stride)

    @staticmethod
    def _read_labels(path: str) -> np.ndarray:
        """
        labels CSV: header [,Background,Supported,Unsupported,Grooming]
        returns fused labels int array shape (T,), values in:
        0=Background, 1=Rearing(Supported/Unsupported), 2=Grooming, -1 if no label.
        Priority when multi-hot:
        - Grooming beats Supported/Unsupported
        - Supported/Unsupported combine into Rearing
        """
        with open(path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        data_rows = rows[1:]  # skip header
        labels = []
        for r in data_rows:
            # skip frame index col; coerce empty cells to 0
            vals = []
            for x in r[1:]:
                if x == "" or x is None:
                    vals.append(0.0)
                else:
                    vals.append(float(x))
            if len(vals) == 0:
                labels.append(-1)
                continue
            bg, sup, unsup, groom = vals
            if groom > 0:  # grooming overrides all
                labels.append(2)
                continue
            if sup > 0 or unsup > 0:  # fuse supported/unsupported -> rearing
                labels.append(1)
                continue
            if bg > 0:
                labels.append(0)
            else:
                labels.append(-1)
        return np.array(labels, dtype=np.int64)

    @staticmethod
    def _read_labels_3class(path: str) -> np.ndarray:
        """3-class variant: DO NOT fuse the rearing classes.

        Returns per-frame ids in {-1, 0, 1, 2}:
          0=Supported rearing, 1=Unsupported rearing, 2=Grooming, -1=Background/none.
        Priority on multi-hot frames matches the 2-class reader (Grooming wins);
        Unsupported is preferred over Supported on the rare (0.33%) overlap. This
        is the only difference from ``_read_labels`` — everything downstream
        (windowing, ds_stride, masking) is identical.
        """
        with open(path, "r") as f:
            rows = list(csv.reader(f))
        labels = []
        for r in rows[1:]:  # skip header
            vals = [(0.0 if (x == "" or x is None) else float(x)) for x in r[1:]]
            if len(vals) == 0:
                labels.append(-1)
                continue
            bg, sup, unsup, groom = vals
            if groom > 0:
                labels.append(2)
            elif unsup > 0:
                labels.append(1)
            elif sup > 0:
                labels.append(0)
            else:
                labels.append(-1)   # background/none -> ignore
        return np.array(labels, dtype=np.int64)

    def _load_video_window(self, video_path: str, start: int, end: int, target_len: int, ds_stride: int = 1) -> torch.Tensor:
        """Load video frames [start,end) with optional temporal stride; fallback to zeros on failure.

        Uses the cached VideoReader + index-based decode to avoid decoding the
        whole video for every single window (O(N_windows × N_frames) → O(windows_T)).
        """
        vframes = _read_video_window_fast(video_path, start, end, ds_stride=ds_stride)  # [T, H, W, 3]
        if vframes.numel() == 0:
            # No frames decoded; return zeros of the expected target length (C=3, default HxW).
            zeros = torch.zeros((target_len, 3, 1, 1), dtype=self.dtype)
            return zeros
        frames = vframes.permute(0, 3, 1, 2)  # T,C,H,W
        if frames.shape[0] < target_len:
            pad = target_len - frames.shape[0]
            pad_tensor = torch.zeros((pad, *frames.shape[1:]), device=frames.device, dtype=frames.dtype)
            frames = torch.cat([frames, pad_tensor], dim=0)
        else:
            frames = frames[:target_len]
        return frames.to(dtype=self.dtype)

    def _compute_body_length(self, kps_xy, conf):
        nose = kps_xy[..., 0, :]  # (T, A, 2)
        tail = kps_xy[..., -1, :]  # (T, A, 2)
        # conf_pair = (conf[..., 0, 0] * conf[..., -1, 0])  # (T, A)
        dist = np.linalg.norm(nose - tail, axis=-1)  # (T, A)
        # if conf_pair.sum() > 1e-6:
        #     length = (dist * conf_pair).sum() / (conf_pair.sum() + 1e-6)
        # else:
        if dist.size == 0:
            return 1.0
        length = dist.mean()
        return length if np.isfinite(length) and length > 1e-6 else 1.0

    def _subject_centric(self, kps, normalize_motion: bool = True, use_subject_centric: bool = True):
        """
        kps: (T, A, K, 3) -> returns (A, T, K, 3) normalized and motion feats (A, T, D)
        """
        T, A, K, _ = kps.shape
        xy = kps[..., :2]
        conf = kps[..., 2:]
        # conf_sum = conf.sum(axis=2, keepdims=True) + 1e-6
        # center = (xy * conf).sum(axis=2, keepdims=True) / conf_sum  # (T, A, 1, 2)
        center = xy.mean(axis=2, keepdims=True)
        if use_subject_centric:
            scale = self._compute_body_length(xy, conf)
            xy_centered = xy - center
            xy_norm = xy_centered / scale
        else:
            xy_norm = xy
            scale = 1.0
        kps_norm = np.concatenate([xy_norm, conf], axis=-1).astype(np.float32)  # (T, A, K, 3)

        delta_xy = np.diff(xy_norm, axis=0, prepend=xy_norm[0:1])
        delta_xy = delta_xy[..., :2]  # (T, A, K, 2)

        neighbor_feats = None
        if self.use_neighbor_feats and self.max_neighbors > 0:
            centers_2d = center[..., 0, :]  # (T, A, 2) in original pixel space
            neighbor_feats_list = []
            for a in range(A):
                feats_t = []
                for t in range(T):
                    dists = np.linalg.norm(centers_2d[t] - centers_2d[t, a], axis=-1)  # (A,)
                    dists[a] = np.inf
                    idx_sorted = np.argsort(dists)[: self.max_neighbors]
                    cur = []
                    for j in idx_sorted:
                        if np.isinf(dists[j]):
                            cur.append([0.0, 0.0, 0.0, 0.0])
                        else:
                            dx, dy = centers_2d[t, j] - centers_2d[t, a]
                            theta = np.arctan2(dy, dx)
                            # normalize neighbor distance by body length for scale invariance
                            dist_norm = dists[j] / scale
                            c_nb = conf[t, j].mean()
                            cur.append([dist_norm, np.sin(theta), np.cos(theta), c_nb])
                    while len(cur) < self.max_neighbors:
                        cur.append([0.0, 0.0, 0.0, 0.0])
                    feats_t.append(cur)
                neighbor_feats_list.append(feats_t)
            # neighbor_feats shape: (A, T, M, 4) where M=max_neighbors and the last dim is [distance, sin(theta), cos(theta), neighbor_conf]
            neighbor_feats = np.array(neighbor_feats_list, dtype=np.float32)

        motion_feats = []
        mask_confidence = (self.stage == "pretrain")
        if mask_confidence:
            mask_all = (conf >= self.conf_mask_thresh).astype(np.float32)
            if not self._logged_conf_mask:
                masked_ratio = 1.0 - float(mask_all.mean())
                total_points = mask_all.size
                print(
                    f"[DataLoader] conf_mask_thresh={self.conf_mask_thresh} masked {masked_ratio*100:.2f}% "
                    f"({int(masked_ratio*total_points)}/{total_points}) of keypoints in first window (pretrain)."
                )
                self._logged_conf_mask = True
        else:
            mask_all = np.ones_like(conf, dtype=np.float32)
        for a in range(A):
            mask = mask_all[:, a]  # (T, K, 1)
            kps_masked = kps_norm[:, a].copy()
            kps_masked[..., :2] = kps_masked[..., :2] * mask  # mask xy only, keep conf
            vel = delta_xy[:, a] * mask  # broadcast (T, K, 1) to (T, K, 2)
            if self.motion_include_confidence:
                parts = [kps_masked.reshape(T, -1)]  # K*3
            else:
                parts = [kps_masked[..., :2].reshape(T, -1)]  # K*2
            if self.motion_include_velocity:
                parts.append(vel.reshape(T, -1))  # K*2
            if neighbor_feats is not None:
                neigh = neighbor_feats[a]
                parts.append(neigh.reshape(T, -1))               # M*4
            feat = np.concatenate(parts, axis=-1)
            motion_feats.append(feat.astype(np.float32))
        motion_feats = np.stack(motion_feats, axis=0)  # (A, T, D_in)
        ### ============================================== ###
        # if normalize_motion:
        #     mu = motion_feats.mean(axis=1, keepdims=True)
        #     sigma = motion_feats.std(axis=1, keepdims=True) + 1e-3
        #     motion_feats = (motion_feats - mu) / sigma
        return kps_norm.transpose(1, 0, 2, 3), motion_feats

    def build_motion_feats_from_keypoints(
        self,
        kps: torch.Tensor,
        motion_include_velocity: bool | None = None,
        motion_include_confidence: bool | None = None,
        use_neighbor_feats: bool | None = None,
        max_neighbors: int | None = None,
    ) -> torch.Tensor:
        """
        Build motion features from normalized keypoints.
        kps: [B, A, T, K, 3] (normalized xy + conf)
        """
        if motion_include_velocity is None:
            motion_include_velocity = self.motion_include_velocity
        if motion_include_confidence is None:
            motion_include_confidence = self.motion_include_confidence
        if use_neighbor_feats is None:
            use_neighbor_feats = self.use_neighbor_feats
        if max_neighbors is None:
            max_neighbors = self.max_neighbors

        B, A, T, K, _ = kps.shape
        xy = kps[..., :2]
        conf = kps[..., 2:3]
        vel = xy - xy.roll(1, dims=2)
        vel[:, :, 0] = 0.0

        if self.stage == "pretrain":
            mask = (conf > self.conf_mask_thresh).to(kps.dtype)
        else:
            mask = torch.ones_like(conf, dtype=kps.dtype)
        kps_masked = kps.clone()
        kps_masked[..., :2] = kps_masked[..., :2] * mask
        vel = vel * mask

        parts = []
        if motion_include_confidence:
            parts.append(kps_masked.reshape(B, A, T, -1))
        else:
            parts.append(kps_masked[..., :2].reshape(B, A, T, -1))
        if motion_include_velocity:
            parts.append(vel.reshape(B, A, T, -1))

        if use_neighbor_feats and max_neighbors > 0:
            neighbor_feats = torch.zeros(
                (B, A, T, max_neighbors, 4),
                device=kps.device,
                dtype=kps.dtype,
            )
            centers = xy.mean(dim=3)  # [B, A, T, 2]
            conf_mean = conf.mean(dim=3).squeeze(-1)  # [B, A, T]
            for b in range(B):
                for t in range(T):
                    c = centers[b, :, t]  # [A, 2]
                    if A <= 1:
                        continue
                    dists = torch.cdist(c, c, p=2)
                    dists.fill_diagonal_(float("inf"))
                    vals, idx = torch.topk(dists, k=max_neighbors, dim=1, largest=False)
                    for a in range(A):
                        for m in range(max_neighbors):
                            j = idx[a, m].item()
                            dist = vals[a, m]
                            if torch.isinf(dist):
                                continue
                            dx, dy = c[j] - c[a]
                            theta = torch.atan2(dy, dx)
                            sin_theta = torch.sin(theta)
                            cos_theta = torch.cos(theta)
                            c_nb = conf_mean[b, j, t]
                            neighbor_feats[b, a, t, m, 0] = dist
                            neighbor_feats[b, a, t, m, 1] = sin_theta
                            neighbor_feats[b, a, t, m, 2] = cos_theta
                            neighbor_feats[b, a, t, m, 3] = c_nb
            parts.append(neighbor_feats.reshape(B, A, T, -1))

        return torch.cat(parts, dim=-1)

    def _normalize_resize_size(self, size: int | Tuple[int, int] | None) -> Tuple[int, int] | None:
        """
        Ensure resize is a (H, W) tuple and snap to frame_resize_align multiples.
        """
        if size is None:
            return None
        if isinstance(size, int):
            target = (size, size)
        elif isinstance(size, (list, tuple)) and len(size) == 2:
            target = (int(size[0]), int(size[1]))
        else:
            raise ValueError(f"frame_resize must be int or (H, W) tuple, got {size}")
        align = max(1, int(self.frame_resize_align))
        if align > 1:
            def _snap(x):
                return max(align, int(round(x / align)) * align)
            snapped = (_snap(target[0]), _snap(target[1]))
            if snapped != target and not self._warned_resize_align:
                print(f"[DataLoader] frame_resize snapped from {target} to {snapped} to align with {align}.")
                self._warned_resize_align = True
            target = snapped
        return target

    def _maybe_resize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Optional spatial downscale to reduce visual tokens. Keeps dtype and shape [A, T, C, H, W].
        """
        if self.frame_resize is None or frames.numel() == 0:
            return frames
        orig_dtype = frames.dtype
        A, T, C, H, W = frames.shape
        flat = frames.reshape(A * T, C, H, W).float()
        resized = F.interpolate(flat, size=self.frame_resize, mode="bilinear", align_corners=False)
        resized = resized.reshape(A, T, C, *self.frame_resize)
        if resized.dtype != orig_dtype:
            resized = resized.to(orig_dtype)
        return resized

    def _try_load_cache_meta(self):
        meta_path_split = os.path.join(self.cache_dir, f"{self.stage}_{self.split}_meta.json")
        meta_path_all = os.path.join(self.cache_dir, f"{self.stage}_all_meta.json")
        explicit_dynamic = self.dynamic_cache_split is True
        has_all_meta = os.path.exists(meta_path_all)
        if explicit_dynamic and not has_all_meta:
            raise RuntimeError(
                f"[DataLoader] dynamic_cache_split=True but missing all_meta file: {meta_path_all}"
            )
        use_dynamic = explicit_dynamic or (self.dynamic_cache_split is None and has_all_meta)
        path_to_use = meta_path_all if (use_dynamic and has_all_meta) else meta_path_split
        if not os.path.exists(path_to_use):
            return
        if use_dynamic and has_all_meta:
            print(f"[DataLoader] Using all_meta for dynamic split: {meta_path_all}")
        with open(path_to_use, "r") as f:
            meta = json.load(f)
        entries = []
        if isinstance(meta, dict) and "entries" in meta:
            entries = meta.get("entries", [])
            if self.num_joints is None and isinstance(meta.get("num_joints"), int):
                self.num_joints = meta["num_joints"]
        elif isinstance(meta, list):
            entries = meta
        if not entries:
            return
        if use_dynamic and has_all_meta:
            split_map = self._split_cache_entries_by_ratio(entries)
            entries = split_map.get(self.split, [])
            if not entries:
                raise RuntimeError(
                    f"[DataLoader] Dynamic cache split produced no entries for split '{self.split}'. "
                    f"meta={meta_path_all}"
                )
        self.cache_meta = meta
        self.cache_entries = entries
        if self.drop_empty_frame_mask and self.stage == "finetune":
            self.cache_entries = self._filter_cache_entries_by_frame_mask(self.cache_entries)
        
        if self.num_joints is None and self.cache_entries:
            sample0 = self._torch_load_safe(self._resolve_cache_path(self.cache_entries[0]))
            self.num_joints = sample0["keypoints"].shape[2]

        if self.stage == "finetune" and self.split == "train" and self.oversample_grooming_factor > 1:
            self.cache_entries = self._apply_grooming_oversample(self.cache_entries)

    def _filter_cache_entries_by_frame_mask(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        kept = []
        dropped = 0
        for entry in entries:
            rel = entry.get("relpath") or entry.get("path")
            if not rel:
                kept.append(entry)
                continue
            path = self._resolve_cache_path(entry)
            try:
                sample = self._torch_load_safe(path)
            except Exception:
                kept.append(entry)
                continue
            frame_mask = sample.get("frame_mask")
            if frame_mask is None:
                kept.append(entry)
                continue
            if not isinstance(frame_mask, torch.Tensor):
                frame_mask = torch.tensor(frame_mask)
            if (frame_mask == 0).all().item():
                dropped += 1
                continue
            kept.append(entry)
        if dropped:
            print(f"[DataLoader] drop_empty_frame_mask=True removed {dropped} windows.")
        return kept

    def _split_cache_entries_by_ratio(self, entries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Split cached entries by video stem according to current ratios, preserving the
        original order from meta (no shuffling).
        """
        stems: List[str] = []
        seen = set()
        for e in entries:
            stem = e.get("stem")
            if stem is None or stem in seen:
                continue
            stems.append(stem)
            seen.add(stem)
        n = len(stems)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        train_end = min(train_end, n)
        val_end = min(val_end, n)
        train_stems = set(stems[:train_end])
        val_stems = set(stems[train_end:val_end])
        test_stems = set(stems[val_end:])

        def _filter(target_stems):
            return [e for e in entries if e.get("stem") in target_stems]

        return {
            "train": _filter(train_stems),
            "val": _filter(val_stems),
            "test": _filter(test_stems),
        }

    def _fit_motion_scaler(self) -> StandardScaler:
        """
        Fit StandardScaler on motion features from all windows in the current (train) split.
        Uses subject-centric keypoints with ds_stride applied and skips per-window normalization.
        """
        scaler = StandardScaler()
        use_partial = hasattr(scaler, "partial_fit")
        buffered: list[np.ndarray] = []
        for item in self.index:
            start, end, kps = item["start"], item["end"], item["kps"]
            kps_slice = kps[start:end]  # (T_win, K, 3)
            if self.ds_stride > 1:
                kps_slice = kps_slice[:: self.ds_stride]
            kps_slice = kps_slice[None, ...]  # (1, T_win, K, 3)
            _, motion_feats = self._subject_centric(
                kps_slice.transpose(1, 0, 2, 3),
                normalize_motion=False,
                use_subject_centric=self.marker_subject_centric,
            )
            flat = motion_feats.reshape(-1, motion_feats.shape[-1])
            if use_partial:
                scaler.partial_fit(flat)
            else:
                buffered.append(flat)
        if not use_partial and buffered:
            scaler.fit(np.concatenate(buffered, axis=0))
        return scaler

    def _apply_grooming_oversample(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Duplicate Grooming windows to mitigate class imbalance.
        Works for both cached meta entries (requires 'label' present) and raw index entries.
        """
        if self.oversample_grooming_factor <= 1:
            return entries
        groom = [e for e in entries if e.get("label") == 1]
        if not groom:
            if not self._warned_no_label_for_oversample:
                print("[DataLoader] Grooming oversample requested but no label info found; skip oversample.")
                self._warned_no_label_for_oversample = True
            return entries
        dup = []
        for e in groom:
            for _ in range(self.oversample_grooming_factor - 1):
                dup.append(dict(e))
        return entries + dup

    def _build_index(self) -> List[Dict[str, Any]]:
        index = []
        for entry in self.samples:
            kps = self._read_dlc_csv(entry["kps"], self.num_joints)
            if self.num_joints is None:
                self.num_joints = kps.shape[1]
            labels = self._read_labels(entry["labels"])
            T = kps.shape[0]
            if self.stage == "pretrain":
                stride = self.pretrain_stride_raw
                win = self.seq_len_raw
                for start in range(0, max(1, T - win + 1), stride):
                    end = min(start + win, T)
                    index.append({
                        "entry": entry,
                        "start": start,
                        "end": end,
                        "label": None,
                        "scenario": None,
                        "kps": kps,
                        "labels": labels,
                    })
            else:
                # finetune: slice windows strictly within single-class segments (ignore background)
                target_classes = [1, 2]  # 1=Rearing (Supported/Unsupported fused), 2=Grooming
                stride = max(1, self.seq_len_raw // 2)
                win = self.seq_len_raw
                for cls in target_classes:
                    allowed = (labels == cls) | (labels == 0)  # allow background inside, block other foreground
                    segments = []
                    start_seg = None
                    has_cls = False
                    for i in range(T):
                        if allowed[i]:
                            if start_seg is None:
                                start_seg = i
                                has_cls = False
                            if labels[i] == cls:
                                has_cls = True
                        else:
                            if start_seg is not None and has_cls:
                                segments.append((start_seg, i - 1))
                            start_seg = None
                            has_cls = False
                    if start_seg is not None and has_cls:
                        segments.append((start_seg, T - 1))
                    if not segments:
                        continue
                    for s, e in segments:
                        seg_len = e - s + 1
                        if seg_len >= win:
                            starts = list(range(s, e - win + 2, stride))
                            if (e - win + 1) not in starts:
                                starts.append(e - win + 1)
                            for st in starts:
                                ed = st + win
                                index.append({
                                    "entry": entry,
                                    "start": st,
                                    "end": ed,
                                    "label": cls - 1,
                                    "scenario": 0,
                                    "kps": kps,
                                    "labels": labels,
                                })
                        else:
                            # short segment: use single window covering the segment, will pad later
                            index.append({
                                "entry": entry,
                                "start": s,
                                "end": e + 1,
                                "label": cls - 1,
                                "scenario": 0,
                                "kps": kps,
                                "labels": labels,
                            })
        if self.stage == "finetune" and self.split == "train" and self.oversample_grooming_factor > 1:
            index = self._apply_grooming_oversample(index)
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if self.cache_entries is not None:
            entry = self.cache_entries[idx]
            cache_path = self._resolve_cache_path(entry)
            # Shallow copy so key replacements below don't leak into LRU / preload cache.
            # Returned tensors themselves are never mutated in-place downstream,
            # so sharing the underlying (possibly mmap'd) storage across calls is safe.
            sample = dict(self._load_pt_file(cache_path))
            sample["stem"] = entry.get("stem")
            # Window coordinates are needed downstream to build unique per-window
            # keys for global_masks lookup in finetune (mirrors mabe's
            # (video_id, window_idx, mouse_id) key).
            sample["start"] = int(entry.get("start", 0))
            sample["end"] = int(entry.get("end", 0))
            if not self.load_frames:
                # Drop the large ~7MB frames tensor entirely so DataLoader
                # workers don't pay the mmap-read / collate-stack / GPU-transfer
                # cost for data the model never consumes (finetune with vision cache).
                # All downstream frames-related processing is guarded by `in sample`.
                sample.pop("frames", None)
            vision_feats, vision_feat_stride = self._maybe_load_vision_feats(entry)
            vision_stride_factor = 1
            if vision_feats is not None:
                if self.ds_stride > vision_feat_stride:
                    if self.ds_stride % vision_feat_stride != 0 and not self._warned_missing_vision_feat:
                        print(
                            f"[DataLoader] vision_feats stride {vision_feat_stride} does not divide ds_stride {self.ds_stride}; "
                            "using floor division for downsample."
                        )
                        self._warned_missing_vision_feat = True
                    vision_stride_factor = max(1, self.ds_stride // vision_feat_stride)
                elif self.ds_stride < vision_feat_stride and not self._warned_missing_vision_feat:
                    print(
                        f"[DataLoader] vision_feats stride ({vision_feat_stride}) coarser than requested ds_stride ({self.ds_stride}); "
                        "features will be padded to match length."
                    )
                    self._warned_missing_vision_feat = True
            if self.ds_stride > 1:
                def _ds(x, axis=1):
                    return x.index_select(axis, torch.arange(0, x.shape[axis], self.ds_stride))
                def _ds_vis(x, axis=1):
                    return x.index_select(axis, torch.arange(0, x.shape[axis], vision_stride_factor))
                sample["keypoints"] = _ds(sample["keypoints"])
                sample["motion_feats"] = _ds(sample["motion_feats"])
                if "frames" in sample:
                    sample["frames"] = _ds(sample["frames"])
                if vision_feats is not None:
                    vision_feats = _ds_vis(vision_feats)
                if "frame_labels" in sample:
                    sample["frame_labels"] = _ds(sample["frame_labels"], axis=0)
                if "frame_mask" in sample:
                    sample["frame_mask"] = _ds(sample["frame_mask"], axis=0)
                cur_T = sample["keypoints"].shape[1]
                if cur_T < self.seq_len:
                    pad_t = self.seq_len - cur_T
                    A, K = sample["keypoints"].shape[0], sample["keypoints"].shape[2]
                    D = sample["motion_feats"].shape[-1]
                    pad_kps = torch.zeros((A, pad_t, K, 3), dtype=sample["keypoints"].dtype)
                    pad_motion = torch.zeros((A, pad_t, D), dtype=sample["motion_feats"].dtype)
                    sample["keypoints"] = torch.cat([sample["keypoints"], pad_kps], dim=1)
                    sample["motion_feats"] = torch.cat([sample["motion_feats"], pad_motion], dim=1)
                    if "frames" in sample:
                        pad_frames = torch.zeros((A, pad_t, *sample["frames"].shape[2:]), dtype=sample["frames"].dtype)
                        sample["frames"] = torch.cat([sample["frames"], pad_frames], dim=1)
                    if vision_feats is not None:
                        D_vis = vision_feats.shape[-1]
                        pad_vis = torch.zeros((vision_feats.shape[0], pad_t, D_vis), dtype=vision_feats.dtype)
                        vision_feats = torch.cat([vision_feats, pad_vis], dim=1)
                    if "frame_labels" in sample:
                        if sample["frame_labels"].dim() == 1:
                            pad_labels = torch.full((pad_t,), -1, dtype=sample["frame_labels"].dtype)
                        else:
                            pad_labels = torch.full(
                                (pad_t, sample["frame_labels"].shape[1]),
                                -1,
                                dtype=sample["frame_labels"].dtype,
                            )
                        sample["frame_labels"] = torch.cat([sample["frame_labels"], pad_labels], dim=0)
                    if "frame_mask" in sample:
                        pad_mask = torch.zeros((pad_t,), dtype=sample["frame_mask"].dtype)
                        sample["frame_mask"] = torch.cat([sample["frame_mask"], pad_mask], dim=0)
            if "frames" in sample:
                sample["frames"] = self._maybe_resize_frames(sample["frames"])
                target_len = sample["frames"].shape[1]
            else:
                target_len = sample["keypoints"].shape[1]
            if vision_feats is not None:
                if vision_feats.shape[1] < target_len:
                    pad_t = target_len - vision_feats.shape[1]
                    pad_vis = torch.zeros((vision_feats.shape[0], pad_t, vision_feats.shape[2]), dtype=vision_feats.dtype)
                    vision_feats = torch.cat([vision_feats, pad_vis], dim=1)
                elif vision_feats.shape[1] > target_len:
                    vision_feats = vision_feats[:, :target_len]
            K = sample["keypoints"].shape[2]
            total_dim = sample["motion_feats"].shape[-1]
            neighbor_dim_guess = self.max_neighbors * 4
            if not self.motion_include_confidence:
                base_dim_conf = K * 3
                remainder = total_dim - base_dim_conf
                conf_present = remainder in {0, K * 2, neighbor_dim_guess, neighbor_dim_guess + K * 2}
                if conf_present:
                    head = sample["motion_feats"][..., :base_dim_conf].reshape(*sample["motion_feats"].shape[:-1], K, 3)
                    head = head[..., :2].reshape(*sample["motion_feats"].shape[:-1], K * 2)
                    tail = sample["motion_feats"][..., base_dim_conf:]
                    sample["motion_feats"] = torch.cat([head, tail], dim=-1) if tail.numel() > 0 else head
                    total_dim = sample["motion_feats"].shape[-1]
            if not self.motion_include_velocity:
                base_dim = K * (2 if not self.motion_include_confidence else 3)
                remainder = total_dim - (base_dim + K * 2)
                vel_present = remainder in {0, neighbor_dim_guess}
                if vel_present:
                    head = sample["motion_feats"][..., :base_dim]
                    tail = sample["motion_feats"][..., base_dim + K * 2:]
                    sample["motion_feats"] = torch.cat([head, tail], dim=-1) if tail.numel() > 0 else head
                    total_dim = sample["motion_feats"].shape[-1]
            if not self.use_neighbor_feats:
                base_dim = K * (2 if not self.motion_include_confidence else 3)
                vel_dim = K * 2 if self.motion_include_velocity else 0
                expected_dim = base_dim + vel_dim
                if total_dim > expected_dim:
                    sample["motion_feats"] = sample["motion_feats"][..., :expected_dim]
            if self.use_motion_scaler and self.motion_scaler is not None:
                mf = sample["motion_feats"]
                orig_shape = mf.shape
                mf_np = mf.reshape(-1, orig_shape[-1]).cpu().numpy()
                mf_np = self.motion_scaler.transform(mf_np)
                sample["motion_feats"] = torch.tensor(mf_np.reshape(orig_shape), dtype=torch.float32)
            sample["keypoints"] = sample["keypoints"].to(self.dtype)
            sample["motion_feats"] = sample["motion_feats"].to(self.dtype)
            if "frames" in sample:
                sample["frames"] = sample["frames"].to(self.dtype)
            sample["count"] = sample["count"].to(self.dtype)
            if vision_feats is not None:
                sample["vision_feats"] = vision_feats.to(self.dtype)
            if "scenario" in sample:
                sample["scenario"] = sample["scenario"].long()
            if "label" in sample:
                sample["label"] = sample["label"].long()
            # In 3-class mode the cached frame_labels are the 2-class FUSED labels
            # (Supported/Unsupported already merged at cache-build time), so we MUST
            # re-derive from the CSV to recover the distinction. Force it here.
            if self.stage != "pretrain" and (
                self.openfield_num_classes == 3
                or "frame_labels" not in sample
                or "frame_mask" not in sample
            ):
                label_path = os.path.join(self.label_dir, f"{entry['stem']}_labels.csv")
                labels_full = (
                    self._read_labels_3class(label_path)
                    if self.openfield_num_classes == 3
                    else self._read_labels(label_path)
                )
                start, end = entry.get("start", 0), entry.get("end", self.seq_len)
                labels_slice = labels_full[start:end]
                if self.ds_stride > 1:
                    labels_slice = labels_slice[:: self.ds_stride]
                labels_slice = np.array([self.label_map.get(int(x), -1) for x in labels_slice], dtype=np.int64)
                frame_mask = (labels_slice != -1).astype(np.int64)
                if labels_slice.shape[0] < self.seq_len:
                    pad_len = self.seq_len - labels_slice.shape[0]
                    labels_slice = np.concatenate([labels_slice, -1 * np.ones(pad_len, dtype=np.int64)], axis=0)
                    frame_mask = np.concatenate([frame_mask, np.zeros(pad_len, dtype=np.int64)], axis=0)
                sample["frame_labels"] = torch.tensor(labels_slice, dtype=torch.long)
                sample["frame_mask"] = torch.tensor(frame_mask, dtype=torch.bool)
            return sample

        item = self.index[idx]
        start, end = item["start"], item["end"]
        entry = item["entry"]
        kps_slice = item["kps"][start:end]
        labels = item["labels"][start:end]
        kps_slice = kps_slice[:: self.ds_stride]
        labels = labels[:: self.ds_stride]
        kps_slice = kps_slice[None, ...]  # (1, T_win, K, 3)
        kps_norm, motion_feats = self._subject_centric(
            kps_slice.transpose(1,0,2,3),
            normalize_motion=(not self.use_motion_scaler or self.motion_scaler is None),
            use_subject_centric=self.marker_subject_centric,
        )  # expects (T_win,A,K,3); returns (A,T_win,K,3)
        cur_T = kps_norm.shape[1]
        if cur_T < self.seq_len:
            print(
                f"[DataLoader][Pad] keypoints_len={cur_T} < seq_len={self.seq_len}; "
                f"stem={entry['stem']} start={start} end={end} ds_stride={self.ds_stride}"
            )
            pad_t = self.seq_len - cur_T
            A, _, K, _ = kps_norm.shape
            D = motion_feats.shape[-1]
            pad_kps = np.zeros((A, pad_t, K, 3), dtype=kps_norm.dtype)
            pad_motion = np.zeros((A, pad_t, D), dtype=motion_feats.dtype)
            kps_norm = np.concatenate([kps_norm, pad_kps], axis=1)
            motion_feats = np.concatenate([motion_feats, pad_motion], axis=1)
            labels = np.concatenate([labels, -1 * np.ones(pad_t, dtype=labels.dtype)])
        frames = self._load_video_window(entry["video"], start, end, self.seq_len, ds_stride=self.ds_stride)
        frames = frames.unsqueeze(0)  # A=1
        frames = self._maybe_resize_frames(frames)

        # per-frame labels (remapped; padding as -1)
        frame_labels = np.array(labels, copy=True)
        frame_labels = np.array([self.label_map.get(int(x), -1) for x in frame_labels], dtype=np.int64)
        frame_mask = (frame_labels != -1).astype(np.int64)
        # Pad to seq_len
        if frame_labels.shape[0] < self.seq_len:
            print(
                f"[DataLoader][PadLabels] frame_labels_len={frame_labels.shape[0]} < seq_len={self.seq_len}; "
                f"stem={entry['stem']} start={start} end={end} ds_stride={self.ds_stride}"
            )
            pad_len = self.seq_len - frame_labels.shape[0]
            frame_labels = np.concatenate([frame_labels, -1 * np.ones(pad_len, dtype=np.int64)], axis=0)
            frame_mask = np.concatenate([frame_mask, np.zeros(pad_len, dtype=np.int64)], axis=0)

        if self.use_motion_scaler and self.motion_scaler is not None:
            orig_shape = motion_feats.shape
            motion_feats = self.motion_scaler.transform(motion_feats.reshape(-1, orig_shape[-1])).reshape(orig_shape)
        sample = {
            "keypoints": torch.tensor(kps_norm, dtype=self.dtype),       # (1, T, K, 3)
            "motion_feats": torch.tensor(motion_feats, dtype=self.dtype),# (1, T, D)
            "frames": frames.to(dtype=self.dtype),                       # (1, T, C, H, W)
            "count": torch.tensor(1.0, dtype=self.dtype),
            "frame_labels": torch.tensor(frame_labels, dtype=torch.long),
            "frame_mask": torch.tensor(frame_mask, dtype=torch.bool),
            "stem": entry["stem"],
        }
        if self.stage != "pretrain":
            sample["scenario"] = torch.tensor(item["scenario"], dtype=torch.long)
            sample["label"] = torch.tensor(item["label"], dtype=torch.long)
        return sample

class MaBeMouseDataset(Dataset):
    """
    PyTorch Dataset for MABe 2022 Mouse Behavior Data

    Loads preprocessed data from .pt cache files and provides windowed samples
    for training based on the training_index.json file.

    Each sample contains:
        - cropped_video: (A, T, 3, H, W) - cropped video frames
        - keypoints: (A, T, K, 3) - egocentric keypoints with confidence
        - mask: (A, T) - validity mask (True for valid frames)
        - confidence: (A, T, K) - keypoint confidence scores
        - metadata: dict with video_id, mouse_id, frame indices, etc.
    """

    def __init__(
        self,
        data_dir: str,
        index_file: str = 'training_index.json',
        split: str = 'train',
        stage: str = 'pretrain',
        train_ratio: float = 0.8,
        split_json: str | Path | None = None,
        transform=None,
        return_raw_keypoints: bool = True,
        preload: bool = False,
        cache_size: int = 50,
        ds_stride: int = 1,
        seq_len: int | None = None,
        multi_mouse: bool = False,
        include_neighbor_feats: bool = True,
        max_neighbors: int = 2,
        motion_include_velocity: bool = True,
        motion_include_confidence: bool = True,
        frame_resize: int | Tuple[int, int] | None = None,
        frame_resize_align: int = 32,
        dtype: torch.dtype = torch.float32,
        cache_dir: str | Path | None = None,
        use_cache: bool = False,
        use_vision_cache: bool = False,
        huddle_only_keep_ratio: float | None = None,
    ):
        """
        Initialize the dataset

        Args:
            data_dir: directory containing processed .pt files and training_index.json
            index_file: name of the training index file
            split: 'train', 'val', or 'test'
            stage: 'pretrain' or 'finetune'
            train_ratio: ratio of training data (rest is split equally for val/test)
            split_json: optional JSON file with explicit train/val/test video_id lists
            transform: optional transform to apply to video frames
            return_raw_keypoints: if True, also return raw (non-egocentric) keypoints
            preload: if True, load all .pt files into memory at initialization (faster but uses more RAM)
            cache_size: number of .pt files to keep in LRU cache (only used if preload=False)
            ds_stride: temporal downsampling stride (1=no downsampling, 2=half framerate, etc.)
            seq_len: expected window length after downsampling (must equal window_size / ds_stride)
            multi_mouse: if True, return all mice in A dimension (A=3) per window
            include_neighbor_feats: if True, return neighbor_feats as context input
            max_neighbors: max number of neighbors for neighbor_feats (A-1 recommended)
            motion_include_velocity: include velocity in motion_feats
            motion_include_confidence: include confidence in motion_feats
            frame_resize: optional resize for frames (int or (H, W))
            frame_resize_align: snap resize to multiples of this value
            dtype: dtype for returned tensors
            cache_dir: optional cache directory (from resifuse/features/hf_mabe.py)
            use_cache: if True, read precomputed cache files instead of raw window caches
            use_vision_cache: if True, prefer vision_feats from cache when available
            huddle_only_keep_ratio: optional keep ratio for huddles-only windows in finetune train split
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.stage = stage
        self.transform = transform
        self.return_raw_keypoints = return_raw_keypoints
        self.preload = preload
        self.cache_size = cache_size
        self.ds_stride = ds_stride
        self.multi_mouse = multi_mouse
        self.include_neighbor_feats = include_neighbor_feats
        self.max_neighbors = max(0, int(max_neighbors))
        self.motion_include_velocity = motion_include_velocity
        self.motion_include_confidence = motion_include_confidence
        self.frame_resize_align = max(1, int(frame_resize_align))
        self.frame_resize = self._normalize_resize_size(frame_resize)
        self._warned_resize_align = False
        self.dtype = dtype
        self.use_cache = bool(use_cache or cache_dir is not None)
        self.use_vision_cache = bool(use_vision_cache)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split_json = Path(split_json) if split_json else None
        self.huddle_only_keep_ratio = huddle_only_keep_ratio
        self._split_map: Dict[str, List[str]] | None = None
        self._blank_frame_cache: Dict[Tuple[int, int, int, torch.dtype], torch.Tensor] = {}
        self.num_joints: int | None = None
        self.num_mice: int | None = None
        self._transform_supports_batch: bool | None = None
        self._warned_neighbor_pad = False

        # Load training index
        index_path = self.data_dir / index_file
        with open(index_path, 'r') as f:
            self.index_data = json.load(f)
        index_stage = self.index_data.get('stage', 'pretrain')
        if index_stage != self.stage:
            raise ValueError(
                f"Index stage '{index_stage}' does not match requested stage '{self.stage}'."
            )
        self.label_names = self.index_data.get('label_names') or []
        precomputed_data_dir = self.index_data.get('precomputed_data_dir')
        self.pt_root = Path(precomputed_data_dir) if precomputed_data_dir else self.data_dir
        if self.use_cache:
            if self.cache_dir is None:
                raise ValueError("use_cache=True requires cache_dir.")
            cache_root = self.cache_dir
            stage_dir = cache_root / self.stage / self.split
            if stage_dir.exists():
                cache_root = stage_dir
            self.cache_dir = cache_root

        self.config = self.index_data['config']
        self.window_size_original = self.config['window_size']
        # Effective window size after downsampling
        if self.window_size_original % ds_stride != 0:
            raise ValueError(
                f"window_size ({self.window_size_original}) must be divisible by ds_stride ({ds_stride})."
            )
        self.window_size = self.window_size_original // ds_stride
        if seq_len is None:
            seq_len = self.window_size
            if _is_rank0():
                print(
                    f"[MaBeMouseDataset] inferred seq_len={seq_len} from window_size={self.window_size_original} "
                    f"and ds_stride={ds_stride}",
                    flush=True,
                )
        if seq_len != self.window_size:
            raise ValueError(
                f"seq_len ({seq_len}) must equal window_size/ds_stride ({self.window_size})."
            )
        self.seq_len = seq_len
        self.video_size = int(self.config.get('video_size', 448))
        self.video_width = int(self.config.get('video_width', self.video_size))
        self.video_height = int(self.config.get('video_height', self.video_size))

        # Build sample index
        if self.use_cache:
            self.samples = self._build_cache_index()
        else:
            self.samples = self._build_sample_index()
        # Split data
        if self.split_json:
            self._load_split_json()
            self.samples = self._apply_split_json(self.samples, split)
        else:
            # self.samples = self._split_data(self.samples, split, train_ratio)
            raise NotImplementedError
        self._maybe_filter_huddles_only()
        self.num_mice = self._infer_num_mice()
        self.num_joints = self._infer_num_joints()

        # Initialize cache
        should_log = _is_rank0()
        if self.preload:
            if should_log:
                print("Preloading all data into memory...")
            self._preloaded_data = {}
            unique_files = set(s['pt_file'] for s in self.samples)
            for pt_file in unique_files:
                self._preloaded_data[pt_file] = torch.load(pt_file)
            if should_log:
                print(f"  Loaded {len(self._preloaded_data)} .pt files")
        else:
            # Use LRU cache with OrderedDict
            self._cache = OrderedDict()
            self._cache_size = cache_size

        if should_log:
            print("Initialized MaBeMouseDataset:")
            print(f"  Split: {split}")
            print(f"  Stage: {self.stage}")
            print(f"  Number of samples: {len(self.samples)}")
            print(f"  Window size (original): {self.window_size_original}")
            print(f"  Multi-mouse: {self.multi_mouse}")
            if self.multi_mouse:
                print(f"  Include neighbor feats: {self.include_neighbor_feats}")
            if self.motion_include_confidence:
                print(f"  Include confidence feats: {self.motion_include_confidence}")
            if self.motion_include_velocity:
                print(f"  Include velocity feats: {self.motion_include_velocity}")
            if ds_stride > 1:
                print(f"  Downsampling stride: {ds_stride}")
                print(f"  Window size (effective seq len): {self.window_size}")
            if preload:
                print("  Mode: Preloaded (all data in memory)")
            else:
                print(f"  Mode: Cached (LRU cache size: {cache_size})")

    def _infer_num_mice(self) -> Optional[int]:
        if self.use_cache and self.samples:
            sample = self.samples[0]
            cache_path = sample.get("cache_path")
            if cache_path:
                cached = torch.load(cache_path, map_location="cpu")
                num_mice = cached.get("num_mice")
                if isinstance(num_mice, int):
                    return num_mice
                keypoints = cached.get("keypoints")
                if isinstance(keypoints, torch.Tensor):
                    # Per-window: [A, T, J, C], per-video: [N, A, T, J, C]
                    is_per_window = sample.get("cache_idx") is None
                    if is_per_window and keypoints.ndim >= 1:
                        return int(keypoints.shape[0])
                    if not is_per_window and keypoints.ndim >= 2:
                        return int(keypoints.shape[1])
        videos = self.index_data.get("videos", [])
        if not videos:
            return None
        mice = videos[0].get("mice", {})
        return len(mice) if mice else None

    def _infer_num_joints(self) -> Optional[int]:
        if not self.samples:
            return None
        sample = self.samples[0]
        if self.use_cache:
            cache_path = sample.get("cache_path")
            if not cache_path:
                return None
            try:
                cached = torch.load(cache_path, map_location="cpu")
            except Exception as exc:
                print(f"[MaBeMouseDataset] Failed to load {cache_path} for num_joints: {exc}")
                return None
            keypoints = cached.get("keypoints")
            if keypoints is None:
                return None
            # Per-window: [A, T, J, C] ndim=4, joints at dim 2
            # Per-video:  [N, A, T, J, C] ndim=5, joints at dim 3
            is_per_window = sample.get("cache_idx") is None
            if is_per_window:
                if keypoints.ndim >= 3:
                    return int(keypoints.shape[2])
            else:
                if keypoints.ndim >= 4:
                    return int(keypoints.shape[3])
                if keypoints.ndim >= 3:
                    return int(keypoints.shape[2])
            return None
        pt_file = sample.get("pt_file")
        if not pt_file:
            return None
        try:
            cached = torch.load(pt_file, map_location="cpu")
        except Exception as exc:
            print(f"[MaBeMouseDataset] Failed to load {pt_file} for num_joints: {exc}")
            return None
        mouse_ids = self._get_mouse_ids(cached)
        if not mouse_ids:
            return None
        keypoints = cached[mouse_ids[0]].get("keypoints")
        if keypoints is None:
            return None
        if keypoints.ndim >= 4:
            return int(keypoints.shape[2])
        if keypoints.ndim >= 3:
            return int(keypoints.shape[1])
        return None

    def _build_cache_index(self) -> List[Dict]:
        samples = []
        if self.cache_dir is None or not self.cache_dir.exists():
            return samples
        cache_files = sorted(self.cache_dir.glob("*.pt"))
        for cache_path in cache_files:
            stem = cache_path.stem
            # Per-window files: VIDEO_ID_wNNN.pt  (from resifuse/preprocessing/split_cache_windows.py)
            if "_w" in stem:
                video_id = stem.rsplit("_w", 1)[0]
                samples.append({
                    "cache_path": str(cache_path),
                    "cache_idx": None,
                    "video_id": video_id,
                })
            else:
                # Legacy per-video files: VIDEO_ID.pt (all windows stacked)
                try:
                    cached = torch.load(cache_path, map_location="cpu")
                except Exception as exc:
                    print(f"[MaBeMouseDataset] Failed to load cache {cache_path}: {exc}")
                    continue
                metadata = cached.get("metadata") or []
                for idx, meta in enumerate(metadata):
                    sample = {
                        "cache_path": str(cache_path),
                        "cache_idx": idx,
                    }
                    if isinstance(meta, dict):
                        if "video_id" in meta:
                            sample["video_id"] = meta["video_id"]
                        if "mouse_id" in meta:
                            sample["mouse_id"] = meta["mouse_id"]
                    samples.append(sample)
        return samples

    def _build_sample_index(self) -> List[Dict]:
        """
        Build a flat list of all samples across all videos and mice

        Returns:
            samples: list of sample metadata dicts
        """
        samples = []

        for video_meta in self.index_data['videos']:
            video_id = video_meta['video_id']
            pt_file = self.pt_root / f"{video_id}.pt"

            if not pt_file.exists():
                print(f"Warning: {pt_file} not found, skipping")
                continue

            mouse_items = list(video_meta['mice'].items())

            # Add samples for each mouse
            for mouse_id, mouse_meta in mouse_items:
                windows = mouse_meta['windows']

                for win_idx, window in enumerate(windows):
                    label = window.get('label')
                    label_name = window.get('label_name')
                    if self.stage == "finetune":
                        if label_name is None and label is not None and self.label_names and label < len(self.label_names):
                            label_name = self.label_names[label]
                    sample = {
                        'video_id': video_id,
                        'mouse_id': mouse_id,
                        'pt_file': str(pt_file),
                        'window_idx': win_idx,
                        'start': window['start'],
                        'end': window['end'],
                        'real_len': window['real_len'],
                        'padding': window['padding']
                    }
                    if self.stage == "finetune" and label is not None:
                        sample['label'] = label
                        sample['label_name'] = label_name
                    samples.append(sample)

        return samples

    def _load_split_json(self) -> None:
        if self._split_map is not None:
            return
        if not self.split_json or not self.split_json.exists():
            raise ValueError(f"split_json not found: {self.split_json}")
        with open(self.split_json, "r") as f:
            payload = json.load(f)
        splits = payload.get("splits") if isinstance(payload, dict) else None
        if splits is None:
            splits = payload
        if not isinstance(splits, dict):
            raise ValueError("split_json must contain a dict with train/val/test lists.")
        split_map: Dict[str, List[str]] = {}
        for key in ("train", "val", "test"):
            ids = splits.get(key)
            if ids is None:
                raise ValueError(f"split_json missing '{key}' list.")
            if not isinstance(ids, list):
                raise ValueError(f"split_json '{key}' must be a list.")
            split_map[key] = [str(x) for x in ids]
        self._split_map = split_map

    def _apply_split_json(self, samples: List[Dict], split: str) -> List[Dict]:
        if self._split_map is None:
            raise ValueError("split_json not loaded.")
        if split not in self._split_map:
            raise ValueError(f"split_json missing split '{split}'.")
        allowed = set(self._split_map[split])
        filtered = []
        missing = 0
        for sample in samples:
            video_id = sample.get("video_id")
            if not video_id:
                missing += 1
                continue
            if video_id in allowed:
                filtered.append(sample)
        if missing > 0:
            raise ValueError(
                f"split_json filtering found {missing} samples without video_id; "
                "ensure cache metadata includes video_id."
            )
        return filtered

    def _maybe_filter_huddles_only(self) -> None:
        if self.stage != "finetune" or self.split != "train":
            return
        if self.huddle_only_keep_ratio is None:
            return
        keep_ratio = float(self.huddle_only_keep_ratio)
        if keep_ratio >= 1.0:
            return
        if self.use_cache:
            print("[MaBeMouseDataset] huddle_only_keep_ratio ignored when use_cache=True.")
            return
        rng = np.random.default_rng(42)
        kept = []
        dropped = 0
        for sample in self.samples:
            label_name = sample.get("label_name")
            if label_name == "huddles":
                if rng.random() > keep_ratio:
                    dropped += 1
                    continue
            kept.append(sample)
        if dropped > 0:
            print(f"[MaBeMouseDataset] Dropped {dropped} huddles-only windows (keep_ratio={keep_ratio}).")
        self.samples = kept

    def _split_data(self, samples: List[Dict], split: str, train_ratio: float) -> List[Dict]:
        """
        Split samples into train/val/test

        Args:
            samples: list of all samples
            split: 'train', 'val', or 'test'
            train_ratio: ratio of training data

        Returns:
            split_samples: samples for the requested split
        """
        # Group by video to ensure videos don't span splits
        video_samples = {}
        for sample in samples:
            video_id = sample['video_id']
            if video_id not in video_samples:
                video_samples[video_id] = []
            video_samples[video_id].append(sample)

        # Sort video IDs for reproducibility
        video_ids = sorted(video_samples.keys())
        num_videos = len(video_ids)

        # Compute split indices
        train_end = int(num_videos * train_ratio)
        val_end = train_end + (num_videos - train_end) // 2

        if split == 'train':
            selected_videos = video_ids[:train_end]
        elif split == 'val':
            selected_videos = video_ids[train_end:val_end]
        elif split == 'test':
            selected_videos = video_ids[val_end:]
        else:
            raise ValueError(f"Invalid split: {split}")

        # Collect samples from selected videos
        split_samples = []
        for video_id in selected_videos:
            split_samples.extend(video_samples[video_id])

        return split_samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pt_file(self, pt_file: str) -> Dict:
        """
        Load a .pt file with caching

        Args:
            pt_file: path to .pt file

        Returns:
            cached_data: loaded data dict
        """
        if self.preload:
            # Use preloaded data
            return self._preloaded_data[pt_file]
        else:
            # Use LRU cache
            if pt_file in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(pt_file)
                return self._cache[pt_file]
            else:
                # Load from disk
                data = torch.load(pt_file)

                # Add to cache
                self._cache[pt_file] = data
                self._cache.move_to_end(pt_file)

                # Remove oldest if cache is full
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)

                return data

    def _get_mouse_ids(self, cached_data: Dict) -> List[str]:
        meta = cached_data.get("_meta", {})
        mouse_ids = meta.get("mouse_ids")
        if mouse_ids:
            return list(mouse_ids)
        return sorted([k for k in cached_data.keys() if not str(k).startswith("_")])

    def _get_blank_frames(
        self,
        num_animals: int,
        num_frames: int,
        crop_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (num_animals, num_frames, crop_size, dtype)
        cached = self._blank_frame_cache.get(key)
        if cached is None:
            cached = torch.zeros(
                (num_animals, num_frames, 3, crop_size, crop_size),
                dtype=dtype,
            )
            self._blank_frame_cache[key] = cached
        return cached

    def _build_motion_feats_from_keypoints(self, keypoints: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Build motion features from keypoints.
        keypoints: [A, T, K, 3] (xy + conf)
        mask: [A, T] bool
        """
        xy = keypoints[..., :2]
        conf = keypoints[..., 2:3]
        mask_f = mask.to(dtype=xy.dtype).unsqueeze(-1).unsqueeze(-1)
        xy = xy * mask_f
        conf = conf * mask_f
        vel = xy - xy.roll(1, dims=1)
        vel[:, 0] = 0.0
        mask_prev = mask_f.roll(1, dims=1)
        mask_prev[:, 0] = 0.0
        # vel = vel * mask_f * mask_prev
        vel *= mask_f

        parts = []
        if self.motion_include_confidence:
            parts.append(torch.cat([xy, conf], dim=-1).reshape(xy.shape[0], xy.shape[1], -1))
        else:
            parts.append(xy.reshape(xy.shape[0], xy.shape[1], -1))
        if self.motion_include_velocity:
            parts.append(vel.reshape(vel.shape[0], vel.shape[1], -1))
        return torch.cat(parts, dim=-1)

    def _build_neighbor_feats(
        self,
        centers: torch.Tensor,
        conf_mean: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Build neighbor context features.
        centers: [A, T, 2] absolute centers
        conf_mean: [A, T] mean confidence per mouse
        mask: [A, T] bool
        Returns: [A, T, M*4] where M=max_neighbors
        """
        A, T, _ = centers.shape
        M = max(0, self.max_neighbors)
        if M == 0:
            return torch.zeros((A, T, 0), dtype=centers.dtype)

        centers_t = centers.permute(1, 0, 2)  # [T, A, 2]
        valid_t = mask.permute(1, 0)  # [T, A]
        dists = torch.cdist(centers_t, centers_t)  # [T, A, A]
        diag = torch.eye(A, device=centers.device, dtype=torch.bool).unsqueeze(0)
        dists = dists.masked_fill(diag, float("inf"))
        if (~valid_t).any():
            valid_pairs = valid_t.unsqueeze(2) & valid_t.unsqueeze(1)
            dists = dists.masked_fill(~valid_pairs, float("inf"))

        vals, idx = torch.topk(dists, k=M, dim=-1, largest=False)  # [T, A, M]
        centers_t_exp = centers_t.unsqueeze(2).expand(-1, -1, M, -1)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, 2)
        neighbor_centers = torch.gather(centers_t_exp, 1, idx_exp)  # [T, A, M, 2]
        delta = neighbor_centers - centers_t.unsqueeze(2)
        dx = delta[..., 0]
        dy = delta[..., 1]
        theta = torch.atan2(dy, dx)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        dist_norm = vals / float(self.video_size)
        conf_t = conf_mean.permute(1, 0)  # [T, A]
        conf_t_exp = conf_t.unsqueeze(-1).expand(-1, -1, M)  # [T, A, M]
        neighbor_conf = conf_t_exp.gather(1, idx)

        finite = torch.isfinite(vals)
        dist_norm = torch.where(finite, dist_norm, torch.zeros_like(dist_norm))
        sin_theta = torch.where(finite, sin_theta, torch.zeros_like(sin_theta))
        cos_theta = torch.where(finite, cos_theta, torch.zeros_like(cos_theta))
        neighbor_conf = torch.where(finite, neighbor_conf, torch.zeros_like(neighbor_conf))

        feats = torch.stack([dist_norm, sin_theta, cos_theta, neighbor_conf], dim=-1)  # [T, A, M, 4]
        feats = feats.permute(1, 0, 2, 3)  # [A, T, M, 4]
        return feats.reshape(A, T, -1)

    def _normalize_resize_size(self, size: int | Tuple[int, int] | None) -> Tuple[int, int] | None:
        """
        Ensure resize is a (H, W) tuple and snap to frame_resize_align multiples.
        """
        if size is None:
            return None
        if isinstance(size, int):
            target = (size, size)
        elif isinstance(size, (list, tuple)) and len(size) == 2:
            target = (int(size[0]), int(size[1]))
        else:
            raise ValueError(f"frame_resize must be int or (H, W) tuple, got {size}")
        align = max(1, int(self.frame_resize_align))
        if align > 1:
            def _snap(x):
                return max(align, int(round(x / align)) * align)
            snapped = (_snap(target[0]), _snap(target[1]))
            if snapped != target and not self._warned_resize_align:
                print(f"[MaBeMouseDataset] frame_resize snapped from {target} to {snapped} to align with {align}.")
                self._warned_resize_align = True
            target = snapped
        return target

    def _maybe_resize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Optional spatial downscale to reduce visual tokens. Keeps dtype and shape [T,C,H,W] or [A,T,C,H,W].
        """
        if self.frame_resize is None or frames.numel() == 0:
            return frames
        orig_dtype = frames.dtype
        if frames.dim() == 4:
            T, C, H, W = frames.shape
            flat = frames.reshape(T, C, H, W).float()
            resized = F.interpolate(flat, size=self.frame_resize, mode="bilinear", align_corners=False)
            resized = resized.reshape(T, C, *self.frame_resize)
        elif frames.dim() == 5:
            A, T, C, H, W = frames.shape
            flat = frames.reshape(A * T, C, H, W).float()
            resized = F.interpolate(flat, size=self.frame_resize, mode="bilinear", align_corners=False)
            resized = resized.reshape(A, T, C, *self.frame_resize)
        else:
            raise ValueError(f"Unexpected frame shape {tuple(frames.shape)}")
        if resized.dtype != orig_dtype:
            resized = resized.to(orig_dtype)
        return resized

    def _apply_transform(self, frames: torch.Tensor) -> torch.Tensor:
        if self.transform is None:
            return frames
        if frames.dim() != 5:
            return self.transform(frames)
        if self._transform_supports_batch is None:
            supports_batch = getattr(self.transform, "supports_batch", None)
            if supports_batch is not None:
                self._transform_supports_batch = bool(supports_batch)
            else:
                try:
                    transformed = self.transform(frames)
                except Exception:
                    self._transform_supports_batch = False
                else:
                    self._transform_supports_batch = True
                    return transformed
        if self._transform_supports_batch:
            return self.transform(frames)
        return torch.stack([self.transform(frames[a]) for a in range(frames.shape[0])])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample

        Args:
            idx: sample index

        Returns:
            sample: dict containing:
                - cropped_video: (A, T, 3, H, W)
                - keypoints: (A, T, K, 3) - egocentric
                - mask: (A, T) - validity mask
                - confidence: (A, T, K)
                - metadata: dict
        """
        sample_meta = self.samples[idx]

        # Load cached data (with LRU cache or preload)
        if self.use_cache:
            return self._getitem_cached(sample_meta)
        cached_data = self._load_pt_file(sample_meta['pt_file'])
        return self._getitem_multi_mouse(sample_meta, cached_data)

    def _cast_sample_dtype(self, output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        float_keys = (
            "cropped_video",
            "keypoints",
            "motion_feats",
            "confidence",
            "frames",
            "vision_feats",
            "vision_patch_tokens",
            "neighbor_feats",
            "raw_keypoints",
        )
        for key in float_keys:
            if key in output and isinstance(output[key], torch.Tensor):
                output[key] = output[key].to(self.dtype)
        return output

    def _maybe_append_neighbor_feats(
        self,
        motion_feats: torch.Tensor,
        keypoints: torch.Tensor,
        mask: torch.Tensor,
        neighbor_feats: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.include_neighbor_feats or self.max_neighbors <= 0:
            return motion_feats
        K = keypoints.shape[2]
        base_dim = K * (3 if self.motion_include_confidence else 2)
        vel_dim = K * 2 if self.motion_include_velocity else 0
        neighbor_dim = self.max_neighbors * 4
        expected_total = base_dim + vel_dim + neighbor_dim
        if motion_feats.shape[-1] == expected_total:
            return motion_feats
        if motion_feats.shape[-1] != base_dim + vel_dim:
            raise RuntimeError("[MaBeMouseDataset] motion_feats dim wrong.")
        if neighbor_feats is None:
            raise RuntimeError("[MaBeMouseDataset] motion_feats missing neighbor dims; padding zeros for compatibility.")
        return torch.cat([motion_feats, neighbor_feats], dim=-1)

    def _getitem_cached(self, sample_meta: Dict) -> Dict[str, torch.Tensor]:
        cache_path = sample_meta.get("cache_path")
        cache_idx = sample_meta.get("cache_idx")
        if cache_path is None:
            raise ValueError("cache sample is missing cache_path")
        cached = self._load_pt_file(cache_path)

        # Per-window file (from resifuse/preprocessing/split_cache_windows.py): tensors have no window dim
        is_per_window = cache_idx is None

        ds_stride = int(cached.get("ds_stride", self.ds_stride))
        if ds_stride != int(self.ds_stride):
            raise ValueError(
                f"Cached ds_stride={ds_stride} does not match requested ds_stride={self.ds_stride}."
            )

        def _get(key):
            val = cached[key]
            return val if is_per_window else val[cache_idx]

        keypoints = _get("keypoints")
        confidence = _get("confidence")
        mask = _get("mask").bool()
        motion_feats = _get("motion_feats")
        motion_feats = self._maybe_append_neighbor_feats(motion_feats, keypoints, mask, None)

        frames = None
        _cache_p = Path(cache_path)
        _vision_npy = _cache_p.with_name(_cache_p.stem + "_vision.npy")
        _has_vision = "vision_feats" in cached or _vision_npy.exists()
        if "cropped_video" in cached:
            frames = _get("cropped_video")
        elif _has_vision:
            A, T = keypoints.shape[0], keypoints.shape[1]
            crop_size = int(self.config.get("crop_size", self.video_size))
            frames = self._get_blank_frames(A, T, crop_size, keypoints.dtype)
            if frames.device != keypoints.device:
                frames = frames.to(device=keypoints.device)
            if self.transform is not None:
                frames = frames.clone()

        if frames is None:
            raise RuntimeError("Cache sample missing frames and vision_feats.")

        frames = self._apply_transform(frames)
        frames = self._maybe_resize_frames(frames)

        if is_per_window:
            metadata = cached.get("metadata", {})
        else:
            metadata = cached.get("metadata", [])[cache_idx] if cached.get("metadata") else {}

        output = {
            "cropped_video": frames,
            "keypoints": keypoints,
            "motion_feats": motion_feats,
            "mask": mask,
            "confidence": confidence,
            "frames": frames,
            "metadata": metadata,
        }

        if self.use_vision_cache:
            # Prefer .npy sidecar files (mmap-friendly, near-instant IO).
            # Fall back to the .pt file if sidecars don't exist.
            # _cache_p and _vision_npy already computed above for the frames check.
            _patches_npy = _cache_p.with_name(_cache_p.stem + "_patches.npy")
            if _vision_npy.exists():
                output["vision_feats"] = torch.from_numpy(
                    np.load(str(_vision_npy), mmap_mode="r").copy()
                )
            elif "vision_feats" in cached:
                output["vision_feats"] = _get("vision_feats")
            if _patches_npy.exists():
                output["vision_patch_tokens"] = torch.from_numpy(
                    np.load(str(_patches_npy), mmap_mode="r").copy()
                )
                if "vision_patch_meta" in cached:
                    output["vision_patch_meta"] = cached["vision_patch_meta"]
            elif "vision_patch_tokens" in cached:
                output["vision_patch_tokens"] = _get("vision_patch_tokens")
                if "vision_patch_meta" in cached:
                    output["vision_patch_meta"] = cached["vision_patch_meta"]

        if self.stage == "finetune":
            frame_labels = cached.get("frame_labels")
            frame_mask = cached.get("frame_mask_majority")
            if frame_mask is None:
                frame_mask = cached.get("frame_mask")
            if frame_labels is not None:
                output["frame_labels"] = (frame_labels if is_per_window else frame_labels[cache_idx]).long()
            if frame_mask is not None:
                output["frame_mask"] = (frame_mask if is_per_window else frame_mask[cache_idx]).bool()

        return self._cast_sample_dtype(output)

    def _getitem_multi_mouse(self, sample_meta: Dict, cached_data: Dict) -> Dict[str, torch.Tensor]:
        start = sample_meta['start']
        real_len = sample_meta['real_len']
        end = start + real_len
        window_idx = sample_meta['window_idx']
        stride = max(1, int(self.ds_stride))
        real_len_ds = int((real_len + stride - 1) // stride)
        padding_ds = self.window_size - real_len_ds
        if padding_ds < 0:
            raise ValueError(
                f"Downsampled window length {real_len_ds} exceeds configured window_size {self.window_size}."
            )

        def _clean_nonfinite(t: torch.Tensor | None) -> torch.Tensor | None:
            if t is None or not torch.is_tensor(t) or not t.is_floating_point():
                return t
            return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            # return t

        def _maybe_normalize_keypoints(t: torch.Tensor | None) -> torch.Tensor | None:
            # if t is None or not torch.is_tensor(t) or not t.is_floating_point():
            #     return t
            # Avoid double-normalizing already normalized keypoints.
            # max_abs = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).abs().max()
            # if max_abs <= 2.0:
            #     return t
            scale = torch.tensor(
                [self.video_width, self.video_height],
                dtype=t.dtype,
                device=t.device,
            )
            return t / scale

        mouse_ids_all = self._get_mouse_ids(cached_data)
        center_mouse = sample_meta['mouse_id']
        if center_mouse in mouse_ids_all:
            mouse_order = [center_mouse] + [m for m in mouse_ids_all if m != center_mouse]
        else:
            mouse_order = list(mouse_ids_all)

        cropped_list = []
        ego_list = []
        conf_list = []
        mask_list = []
        centers_list = []

        assert self.return_raw_keypoints is True

        raw_list = [] if self.return_raw_keypoints else None

        for mouse_id in mouse_order:
            mouse_data = cached_data[mouse_id]
            cropped = mouse_data['cropped_video'][window_idx]
            ego_kps = mouse_data['ego_keypoints'][window_idx]
            conf = mouse_data['confidence'][window_idx]
            mouse_mask = mouse_data['mask'][window_idx]
            mouse_centers = mouse_data['centers'][window_idx]
            raw_kps = mouse_data['keypoints'][window_idx] if self.return_raw_keypoints else None

            # ego_kps = _clean_nonfinite(ego_kps)
            # conf = _clean_nonfinite(conf)
            # mouse_centers = _clean_nonfinite(mouse_centers)
            # if raw_kps is not None:
            #     raw_kps = _clean_nonfinite(raw_kps)

            ego_kps = _maybe_normalize_keypoints(ego_kps)
            if raw_kps is not None:
                raw_kps = _maybe_normalize_keypoints(raw_kps)

            if stride > 1:
                cropped = cropped[::stride]
                ego_kps = ego_kps[::stride]
                conf = conf[::stride]
                mouse_mask = mouse_mask[::stride]
                mouse_centers = mouse_centers[::stride]
                if raw_kps is not None:
                    raw_kps = raw_kps[::stride]

            cropped_list.append(cropped)
            ego_list.append(ego_kps)
            conf_list.append(conf)
            mask_list.append(mouse_mask)
            centers_list.append(mouse_centers)
            if raw_list is not None and raw_kps is not None:
                raw_list.append(raw_kps)

        cropped_video = torch.stack(cropped_list, dim=0)
        ego_kps = torch.stack(ego_list, dim=0)
        confidence = torch.stack(conf_list, dim=0)
        mask = torch.stack(mask_list, dim=0).bool()
        centers = torch.stack(centers_list, dim=0)
        raw_keypoints = torch.stack(raw_list, dim=0) if raw_list is not None else None

        # A, T, K, _ = ego_kps.shape
        # keypoints = torch.zeros(A, T, K, 3, dtype=ego_kps.dtype, device=ego_kps.device,
        # )
        # keypoints[:, :, :, :2] = ego_kps

        # change from keypoints to raw_keypoints!!!!!!!!!!!!! 
        A, T, K, _ = raw_keypoints.shape
        keypoints = torch.zeros(A, T, K, 3, dtype=raw_keypoints.dtype, device=raw_keypoints.device,
        )
        keypoints[:, :, :, :2] = raw_keypoints

        keypoints[:, :, :, 2] = confidence

        cropped_video = self._apply_transform(cropped_video)
        cropped_video = self._maybe_resize_frames(cropped_video)

        motion_feats = self._build_motion_feats_from_keypoints(keypoints, mask)
        neighbor_feats = None
        if self.include_neighbor_feats and self.max_neighbors > 0:
            conf_mean = confidence.mean(dim=2)
            neighbor_feats = self._build_neighbor_feats(centers, conf_mean, mask)
        motion_feats = self._maybe_append_neighbor_feats(motion_feats, keypoints, mask, neighbor_feats)

        output = {
            'cropped_video': cropped_video,
            'keypoints': keypoints,
            'motion_feats': motion_feats,
            'mask': mask,
            'confidence': confidence,
            'frames': cropped_video,
            'metadata': {
                'video_id': sample_meta['video_id'],
                'mouse_id': sample_meta['mouse_id'],
                'mouse_order': mouse_order,
                'window_idx': sample_meta['window_idx'],
                'start_frame': start,
                'end_frame': end,
                'padding': padding_ds,
                'ds_stride': self.ds_stride,
                'original_length': real_len
            }
        }

        if neighbor_feats is not None:
            output['neighbor_feats'] = neighbor_feats
        if self.stage == "finetune":
            labels_list = []
            for mid in mouse_ids_all:
                frame_labels = cached_data.get(mid, {}).get('frame_labels')
                if frame_labels is None:
                    continue
                if int(window_idx) >= int(frame_labels.shape[0]):
                    continue
                frame_labels = frame_labels[window_idx]
                if stride > 1:
                    frame_labels = frame_labels[::stride]
                if not isinstance(frame_labels, torch.Tensor):
                    frame_labels = torch.tensor(frame_labels)
                labels_list.append(frame_labels.long())
            if labels_list:
                maj_labels, maj_mask = _majority_vote_frame_labels(labels_list)
                output['frame_labels'] = maj_labels.long()
                output['frame_mask'] = maj_mask.bool()

        if self.return_raw_keypoints and raw_keypoints is not None:
            output['raw_keypoints'] = raw_keypoints

        return self._cast_sample_dtype(output)

    def get_full_sequence(self, video_id: str, mouse_id: str) -> Dict[str, torch.Tensor]:
        """
        Get the full processed sequence for a video and mouse

        Args:
            video_id: video identifier
            mouse_id: mouse identifier

        Returns:
            full_data: dict with full sequences
        """
        pt_file = str(self.data_dir / f"{video_id}.pt")
        if hasattr(self, "pt_root"):
            pt_file = str(self.pt_root / f"{video_id}.pt")
        cached_data = self._load_pt_file(pt_file)
        return cached_data[mouse_id]

    def get_window_size(self) -> int:
        """Get the window size used for sampling"""
        return self.window_size

    def get_config(self) -> Dict:
        """Get the preprocessing configuration"""
        return self.config


# class MaBeMouseRawDataset(Dataset):
#     """
#     Alternative dataset that provides full sequences without windowing

#     Useful for inference or visualization.
#     """

#     def __init__(
#         self,
#         data_dir: str,
#         index_file: str = 'training_index.json',
#         transform=None,
#         preload: bool = False,
#         cache_size: int = 20,
#         ds_stride: int = 1
#     ):
#         """
#         Initialize raw dataset

#         Args:
#             data_dir: directory containing processed .pt files
#             index_file: name of the training index file
#             transform: optional transform
#             preload: if True, load all .pt files into memory at initialization
#             cache_size: number of .pt files to keep in LRU cache (only used if preload=False)
#             ds_stride: temporal downsampling stride (1=no downsampling, 2=half framerate, etc.)
#         """
#         self.data_dir = Path(data_dir)
#         self.transform = transform
#         self.preload = preload
#         self.cache_size = cache_size
#         self.ds_stride = ds_stride

#         # Load index
#         index_path = self.data_dir / index_file
#         with open(index_path, 'r') as f:
#             self.index_data = json.load(f)
#         cache_dir = self.index_data.get('cache_dir')
#         self.pt_root = Path(cache_dir) if cache_dir else self.data_dir

#         # Build sequence index (one entry per video-mouse pair)
#         self.sequences = []
#         for video_meta in self.index_data['videos']:
#             video_id = video_meta['video_id']
#             pt_file = self.pt_root / f"{video_id}.pt"

#             if not pt_file.exists():
#                 continue

#             for mouse_id in video_meta['mice'].keys():
#                 self.sequences.append({
#                     'video_id': video_id,
#                     'mouse_id': mouse_id,
#                     'pt_file': str(pt_file)
#                 })

#         # Initialize cache
#         if self.preload:
#             print(f"Preloading all data into memory...")
#             self._preloaded_data = {}
#             unique_files = set(s['pt_file'] for s in self.sequences)
#             for pt_file in unique_files:
#                 self._preloaded_data[pt_file] = torch.load(pt_file)
#             print(f"  Loaded {len(self._preloaded_data)} .pt files")
#         else:
#             # Use LRU cache with OrderedDict
#             self._cache = OrderedDict()
#             self._cache_size = cache_size

#         print(f"Initialized MaBeMouseRawDataset:")
#         print(f"  Number of sequences: {len(self.sequences)}")
#         if ds_stride > 1:
#             print(f"  Downsampling stride: {ds_stride}")
#         if preload:
#             print(f"  Mode: Preloaded (all data in memory)")
#         else:
#             print(f"  Mode: Cached (LRU cache size: {cache_size})")

#     def __len__(self) -> int:
#         return len(self.sequences)

#     def _load_pt_file(self, pt_file: str) -> Dict:
#         """Load a .pt file with caching"""
#         if self.preload:
#             return self._preloaded_data[pt_file]
#         else:
#             if pt_file in self._cache:
#                 self._cache.move_to_end(pt_file)
#                 return self._cache[pt_file]
#             else:
#                 data = torch.load(pt_file)
#                 self._cache[pt_file] = data
#                 self._cache.move_to_end(pt_file)
#                 if len(self._cache) > self._cache_size:
#                     self._cache.popitem(last=False)
#                 return data

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         """Get a full sequence"""
#         seq_meta = self.sequences[idx]

#         # Load cached data (with LRU cache or preload)
#         cached_data = self._load_pt_file(seq_meta['pt_file'])
#         mouse_data = cached_data[seq_meta['mouse_id']].copy()

#         # Apply temporal downsampling if ds_stride > 1
#         if self.ds_stride > 1:
#             mouse_data['cropped_video'] = mouse_data['cropped_video'][::self.ds_stride]
#             mouse_data['keypoints'] = mouse_data['keypoints'][::self.ds_stride]
#             mouse_data['ego_keypoints'] = mouse_data['ego_keypoints'][::self.ds_stride]
#             mouse_data['neighbor_keypoints'] = mouse_data['neighbor_keypoints'][::self.ds_stride]
#             mouse_data['mask'] = mouse_data['mask'][::self.ds_stride]
#             mouse_data['confidence'] = mouse_data['confidence'][::self.ds_stride]
#             mouse_data['centers'] = mouse_data['centers'][::self.ds_stride]

#         # Apply transform if provided
#         if self.transform is not None:
#             mouse_data['cropped_video'] = self.transform(mouse_data['cropped_video'])

#         mouse_data['metadata'] = {
#             'video_id': seq_meta['video_id'],
#             'mouse_id': seq_meta['mouse_id'],
#             'ds_stride': self.ds_stride
#         }

#         return mouse_data


# def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
#     """
#     Custom collate function for batching samples

#     Args:
#         batch: list of samples from __getitem__

#     Returns:
#         batched: dict with batched tensors
#     """
#     # Stack tensors
#     batched = {
#         'cropped_video': torch.stack([b['cropped_video'] for b in batch]),
#         'keypoints': torch.stack([b['keypoints'] for b in batch]),
#         'mask': torch.stack([b['mask'] for b in batch]),
#         'confidence': torch.stack([b['confidence'] for b in batch]),
#         'metadata': [b['metadata'] for b in batch]  # Keep as list
#     }
#     if 'frames' in batch[0]:
#         batched['frames'] = torch.stack([b['frames'] for b in batch])
#     if 'neighbor_keypoints' in batch[0]:
#         batched['neighbor_keypoints'] = torch.stack([b['neighbor_keypoints'] for b in batch])
#     if 'motion_feats' in batch[0]:
#         batched['motion_feats'] = torch.stack([b['motion_feats'] for b in batch])
#     if 'neighbor_feats' in batch[0]:
#         batched['neighbor_feats'] = torch.stack([b['neighbor_feats'] for b in batch])

#     # Include raw keypoints if present
#     if 'raw_keypoints' in batch[0]:
#         batched['raw_keypoints'] = torch.stack([b['raw_keypoints'] for b in batch])
#     if 'label' in batch[0]:
#         batched['label'] = torch.stack([b['label'] for b in batch])
#     if 'label_name' in batch[0]:
#         batched['label_name'] = [b['label_name'] for b in batch]
#     if 'frame_labels' in batch[0]:
#         batched['frame_labels'] = torch.stack([b['frame_labels'] for b in batch])
#     if 'frame_mask' in batch[0]:
#         batched['frame_mask'] = torch.stack([b['frame_mask'] for b in batch])

#     return batched


# ============================================================================
# CalMS21Dataset
# ============================================================================

def _calms21_crop_frame(
    frame: np.ndarray,
    cx: float,
    cy: float,
    crop_size: int,
    video_w: int,
    video_h: int,
) -> np.ndarray:
    """Clamp-crop a square region from a (H, W, 3) frame.

    The crop is centered on (cx, cy) when that fits inside the frame,
    otherwise it is shifted inward to stay within the video bounds. No
    zero padding is introduced as long as crop_size <= min(video_w,
    video_h) (the common CalMS21 case with 1024x570 video and
    crop <= 570). For oversized crops the uncovered axis falls back to
    symmetric zero padding as a safety net.
    """
    half = crop_size // 2
    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    if crop_size <= video_w:
        x1 = max(0, min(x1, video_w - crop_size))
    else:
        x1 = (video_w - crop_size) // 2  # falls back to padded branch below
    if crop_size <= video_h:
        y1 = max(0, min(y1, video_h - crop_size))
    else:
        y1 = (video_h - crop_size) // 2
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(video_w, x2), min(video_h, y2)
    if sx1 == x1 and sy1 == y1 and sx2 == x2 and sy2 == y2:
        # Fully inside — direct view, no padding work.
        return frame[sy1:sy2, sx1:sx2].copy()
    # Oversize-crop fallback: pad the uncovered axis with zeros.
    out = np.zeros((crop_size, crop_size, 3), dtype=frame.dtype)
    dx1, dy1 = sx1 - x1, sy1 - y1
    dx2 = dx1 + (sx2 - sx1)
    dy2 = dy1 + (sy2 - sy1)
    if sx2 > sx1 and sy2 > sy1:
        out[dy1:dy2, dx1:dx2] = frame[sy1:sy2, sx1:sx2]
    return out


class CalMS21Dataset(Dataset):
    """PyTorch Dataset for CalMS21 Task 1 (resident-intruder).

    Supports two read paths:

    * ``use_vision_cache=False`` (*"raw"*): open Stage 1 per-video .pt,
      decode the corresponding mp4 segment via a fresh ``ffmpeg``
      subprocess per window, crop per mouse on CPU, downsample to
      ``ds_stride``. Used by Stage 3 to iterate over all windows
      during vision-encoder caching. Each decode spawns a clean
      ffmpeg process so there is no shared C++ state across
      DataLoader workers (previously decord was fork-unsafe and
      deadlocked on multi-worker runs).

    * ``use_vision_cache=True`` (*"cache"*): read Stage 3 per-window .pt
      + mmap-backed ``{video_id}_wNNN_vision.npy`` sidecar. This is the
      production training path.

    Both paths return the same dict keys (minus ``vision_feats`` in raw
    mode). The sample schema is:

        cropped_video : (A=2, T_ds, 3, H, W)   uint8 in raw mode (will
                        be cast to float in a downstream transform)
        keypoints     : (A=2, T_ds, K=7, 2)    float32 (orig px coords)
        confidence    : (A=2, T_ds, K=7)       float32
        centers       : (A=2, T_ds, 2)         float32
        mask          : (A=2, T_ds)            bool
        frame_labels  : (T_ds,)                long (-1 for padded frames)
        metadata      : dict
        vision_feats  : (A=2, T_ds, feat_dim)  only if use_vision_cache
        vision_patch_tokens : (A=2, T_ds, P, D)  if available
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        split_json: str | Path | None = None,
        index_file: str = "training_index.json",
        ds_stride: int = 4,
        crop_size: int = 448,
        frame_resize: int | None = 224,
        use_vision_cache: bool = False,
        cache_dir: str | Path | None = None,
        return_frame_labels: bool = True,
        dtype: torch.dtype = torch.float32,
        ffmpeg_bin: str | None = None,
    ) -> None:
        """
        Args:
            data_dir: dir containing Stage 1 ``training_index.json`` and
                ``caches/{video_id}.pt``.
            split: 'train' | 'val' | 'test' (keys inside split_json).
            split_json: path to calms21_split.json (defaults to
                ``{data_dir}/calms21_split.json``).
            index_file: name of the Stage 1 index file.
            ds_stride: downsample stride applied to each window
                (effective seq_len = window_size // ds_stride).
            crop_size: per-mouse crop side length (CPU crop).
            frame_resize: optional resize (int, square) applied after
                crop. None = keep crop_size.
            use_vision_cache: if True, read Stage 3 per-window cache;
                otherwise decode mp4 on the fly (raw path).
            cache_dir: Stage 3 output dir; required when
                use_vision_cache=True.
            return_frame_labels: emit ``frame_labels`` in the output
                (disable for pure unsupervised pretrain).
            dtype: float dtype for returned float tensors (the raw
                uint8 video stays uint8 until a transform casts it).
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.ds_stride = int(ds_stride)
        self.crop_size = int(crop_size)
        self.frame_resize = int(frame_resize) if frame_resize else None
        self.use_vision_cache = bool(use_vision_cache)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.return_frame_labels = bool(return_frame_labels)
        self.dtype = dtype
        self._ffmpeg_bin = ffmpeg_bin or os.environ.get("FFMPEG_BIN", "ffmpeg")

        if self.use_vision_cache and self.cache_dir is None:
            raise ValueError("use_vision_cache=True requires cache_dir")

        index_path = self.data_dir / index_file
        with open(index_path, "r") as f:
            self.index_data = json.load(f)
        self.config = self.index_data["config"]
        self.label_names = self.index_data["label_names"]
        self.num_joints = self.index_data["num_joints"]
        self.num_mice = self.index_data["num_mice"]
        self.window_size_original = int(self.config["window_size"])
        if self.window_size_original % self.ds_stride != 0:
            raise ValueError(
                f"window_size={self.window_size_original} not divisible by ds_stride={self.ds_stride}"
            )
        self.window_size = self.window_size_original // self.ds_stride  # effective T_ds
        self.video_width = int(self.config.get("video_width", 1024))
        self.video_height = int(self.config.get("video_height", 570))

        # Split filtering
        if split_json is None:
            split_json = self.data_dir / "calms21_split.json"
        split_json = Path(split_json)
        if not split_json.exists():
            raise FileNotFoundError(f"calms21_split.json not found at {split_json}")
        with open(split_json, "r") as f:
            split_payload = json.load(f)
        splits_dict = split_payload.get("splits", split_payload)
        if split not in splits_dict:
            raise ValueError(f"Split '{split}' not found in {split_json}")
        allowed_videos = set(splits_dict[split])

        # Build per-video lookups from Stage 1 index.
        self.video_meta: Dict[str, Dict] = {}
        for v in self.index_data["videos"]:
            if v["video_id"] in allowed_videos:
                self.video_meta[v["video_id"]] = v

        # Flat sample list: one entry per (video, window_idx).
        self.samples: List[Dict] = []
        for vid, meta in self.video_meta.items():
            for w_idx in range(meta["num_windows"]):
                self.samples.append({"video_id": vid, "window_idx": w_idx})

        # Per-video .pt cache (tiny: ~3 MB × 88 videos).
        self._pt_cache: Dict[str, Dict] = {}

        if _is_rank0():
            print(
                f"[CalMS21Dataset] split={split}: {len(self.video_meta)} videos, "
                f"{len(self.samples)} windows "
                f"(W_orig={self.window_size_original}, T_ds={self.window_size}, "
                f"crop={self.crop_size}, resize={self.frame_resize}, "
                f"vision_cache={'on' if self.use_vision_cache else 'off'})"
            )

    # ------------------------------------------------------------------
    # Stage 1 helpers (raw path)
    # ------------------------------------------------------------------
    def _load_video_pt(self, video_id: str) -> Dict:
        cached = self._pt_cache.get(video_id)
        if cached is not None:
            return cached
        meta = self.video_meta[video_id]
        cached = torch.load(meta["pt_file"], map_location="cpu", weights_only=False)
        self._pt_cache[video_id] = cached
        return cached

    def _ffmpeg_decode_window(
        self,
        video_path: str,
        start_frame: int,
        real_len: int,
        video_w: int,
        video_h: int,
        fps: float,
    ) -> np.ndarray:
        """Decode frames [start_frame, start_frame + real_len) via ffmpeg pipe.

        Returns ``(real_len, H, W, 3)`` uint8. Each invocation spawns
        a fresh ``ffmpeg`` subprocess, which is 100% fork-safe and
        immune to the C++-level deadlocks decord can hit under
        DataLoader multi-worker fork semantics.
        """
        if real_len <= 0:
            return np.zeros((0, video_h, video_w, 3), dtype=np.uint8)

        # Frame-accurate: ``-ss`` before ``-i`` triggers fast keyframe
        # seek, and ffmpeg >= 2.1 post-seek-discards to reach the
        # requested frame exactly.
        seek_ts = start_frame / float(fps) if fps > 0 else 0.0
        cmd = [
            self._ffmpeg_bin,
            "-v", "error",
            "-nostdin",
            "-hide_banner",
            "-ss", f"{seek_ts:.6f}",
            "-i", video_path,
            "-frames:v", str(real_len),
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ffmpeg decode failed ({video_path} @ frame {start_frame}, "
                f"n={real_len}): rc={e.returncode} "
                f"stderr={e.stderr.decode('utf-8', errors='replace')[:500]}"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"ffmpeg timed out ({video_path} @ frame {start_frame}, n={real_len})"
            )

        raw = np.frombuffer(proc.stdout, dtype=np.uint8)
        expected = real_len * video_h * video_w * 3
        if raw.size != expected:
            # Partial output: pad with zeros on the tail. Shouldn't
            # normally happen for a well-formed mp4, but better than
            # crashing a 3-hour job over a single weird frame.
            actual_frames = raw.size // (video_h * video_w * 3)
            decoded = raw[: actual_frames * video_h * video_w * 3].reshape(
                actual_frames, video_h, video_w, 3
            )
            out = np.zeros((real_len, video_h, video_w, 3), dtype=np.uint8)
            out[:actual_frames] = decoded
            return out
        return raw.reshape(real_len, video_h, video_w, 3)

    def _decode_cropped_window(
        self, cached: Dict, window_idx: int
    ) -> Tuple[np.ndarray, List[int]]:
        """Return (cropped_video: (A, T_ds, H, W, 3) uint8, ds_frame_indices).

        Decodes all ``real_len`` frames contiguously via ffmpeg, then
        subsamples every ``ds_stride``-th frame on the Python side. The
        extra pipe traffic (4x raw bytes) is cheaper than running a
        ``framestep`` filter because ffmpeg has to decode every frame
        to reach the ones we want anyway.
        """
        meta = cached["_meta"]
        video_path = meta["video_path"]
        video_w = int(meta["video_width"])
        video_h = int(meta["video_height"])
        fps = float(meta.get("fps", 30.0))

        window = cached["windows"][window_idx]
        start, real_len = int(window["start"]), int(window["real_len"])
        W_orig = self.window_size_original
        T_ds = self.window_size

        # Downsampled frame indices within the window: [0, ds, 2*ds, ...]
        ds_indices_rel = list(range(0, W_orig, self.ds_stride))  # len = T_ds
        abs_indices = [start + r for r in ds_indices_rel]

        # Decode contiguous frames covering real_len, then pick every ds_stride-th.
        decoded = self._ffmpeg_decode_window(
            video_path, start, real_len, video_w, video_h, fps
        )

        H = self.crop_size
        out = np.zeros((self.num_mice, T_ds, H, H, 3), dtype=np.uint8)
        centers = cached["centers"][window_idx].numpy()  # (A, W_orig, 2)

        for t_ds, rel in enumerate(ds_indices_rel):
            if rel >= real_len:
                continue  # padded frame stays zeros
            frame = decoded[rel]
            for a in range(self.num_mice):
                cx, cy = float(centers[a, rel, 0]), float(centers[a, rel, 1])
                out[a, t_ds] = _calms21_crop_frame(
                    frame, cx, cy, H, self.video_width, self.video_height
                )
        return out, abs_indices

    def _downsample_window_tensors(self, cached: Dict, window_idx: int) -> Dict[str, torch.Tensor]:
        """Slice and downsample the per-window tensors from Stage 1."""
        ds = self.ds_stride
        kp = cached["keypoints"][window_idx, :, ::ds]      # (A, T_ds, K, 2)
        conf = cached["confidence"][window_idx, :, ::ds]   # (A, T_ds, K)
        centers = cached["centers"][window_idx, :, ::ds]   # (A, T_ds, 2)
        mask = cached["mask"][window_idx, :, ::ds]         # (A, T_ds)
        labels = cached["frame_labels"][window_idx, ::ds]  # (T_ds,)
        return {
            "keypoints": kp.contiguous(),
            "confidence": conf.contiguous(),
            "centers": centers.contiguous(),
            "mask": mask.contiguous(),
            "frame_labels": labels.contiguous(),
        }

    def _maybe_resize(self, video: torch.Tensor) -> torch.Tensor:
        """Resize (A, T, 3, H, W) → (A, T, 3, H', W') if frame_resize set."""
        if self.frame_resize is None or self.frame_resize == video.shape[-1]:
            return video
        A, T = video.shape[:2]
        flat = video.reshape(A * T, 3, *video.shape[-2:]).float()
        resized = F.interpolate(
            flat,
            size=(self.frame_resize, self.frame_resize),
            mode="bilinear",
            align_corners=False,
        )
        return resized.reshape(A, T, 3, self.frame_resize, self.frame_resize)

    # ------------------------------------------------------------------
    # Stage 3 helpers (cache path)
    # ------------------------------------------------------------------
    def _cache_paths(self, video_id: str, window_idx: int) -> Dict[str, Path]:
        assert self.cache_dir is not None
        stem = f"{video_id}_w{window_idx:03d}"
        return {
            "pt": self.cache_dir / f"{stem}.pt",
            "vision_npy": self.cache_dir / f"{stem}_vision.npy",
            "patches_npy": self.cache_dir / f"{stem}_patches.npy",
            "motion_npy": self.cache_dir / f"{stem}_motion.npy",
        }

    def _load_cache_sample(self, video_id: str, window_idx: int) -> Dict[str, Any]:
        paths = self._cache_paths(video_id, window_idx)
        if not paths["pt"].exists():
            raise FileNotFoundError(f"Missing cache file: {paths['pt']}")
        cached = torch.load(paths["pt"], map_location="cpu", weights_only=False)

        # Concat confidence into keypoints → [A, T, K, 3] so the downstream
        # training loop (shared with MaBe) can index kps[..., 2] for conf.
        kps_xy = cached["keypoints"]
        conf = cached["confidence"]
        keypoints = torch.cat([kps_xy.to(torch.float32), conf.unsqueeze(-1).to(torch.float32)], dim=-1)

        out = {
            "keypoints": keypoints,
            "confidence": conf,
            "mask": cached["mask"].bool(),
            "frame_labels": cached["frame_labels"],
            "centers": cached.get("centers"),
            "metadata": cached.get("metadata", {"video_id": video_id, "window_idx": window_idx}),
        }

        # Mechanism B: mmap-backed vision features via .npy sidecar.
        # .copy() materializes a writable tensor while still benefiting
        # from the shared OS page cache on the underlying file.
        if paths["vision_npy"].exists():
            out["vision_feats"] = torch.from_numpy(
                np.load(str(paths["vision_npy"]), mmap_mode="r").copy()
            )
        elif "vision_feats" in cached:
            out["vision_feats"] = cached["vision_feats"]
        if paths["patches_npy"].exists():
            out["vision_patch_tokens"] = torch.from_numpy(
                np.load(str(paths["patches_npy"]), mmap_mode="r").copy()
            )
        elif "vision_patch_tokens" in cached:
            out["vision_patch_tokens"] = cached["vision_patch_tokens"]
        # motion_feats sidecar (offline-computed via resifuse/features/hf_calms21.py).
        if paths["motion_npy"].exists():
            out["motion_feats"] = torch.from_numpy(
                np.load(str(paths["motion_npy"]), mmap_mode="r").copy()
            )
        elif "motion_feats" in cached:
            out["motion_feats"] = cached["motion_feats"]

        return out

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        video_id = sample["video_id"]
        window_idx = sample["window_idx"]

        if self.use_vision_cache:
            out = self._load_cache_sample(video_id, window_idx)
            return self._finalize_sample_dtype(out)

        # Raw path: Stage 1 .pt + mp4 decode.
        cached = self._load_video_pt(video_id)
        meta = cached["_meta"]
        window = cached["windows"][window_idx]

        tensors = self._downsample_window_tensors(cached, window_idx)

        video_np, abs_indices = self._decode_cropped_window(cached, window_idx)
        # (A, T_ds, H, W, 3) uint8 → (A, T_ds, 3, H, W)
        video = torch.from_numpy(video_np).permute(0, 1, 4, 2, 3).contiguous()
        video = self._maybe_resize(video)

        out = {
            "cropped_video": video,
            "keypoints": tensors["keypoints"],
            "confidence": tensors["confidence"],
            "centers": tensors["centers"],
            "mask": tensors["mask"],
            "frame_labels": tensors["frame_labels"],
            "metadata": {
                "video_id": video_id,
                "window_idx": window_idx,
                "start": int(window["start"]),
                "end": int(window["end"]),
                "real_len": int(window["real_len"]),
                "padding": int(window["padding"]),
                "ds_stride": self.ds_stride,
                "abs_frame_indices": abs_indices,
                "split": meta.get("split"),
            },
        }
        if not self.return_frame_labels:
            out.pop("frame_labels", None)
        return self._finalize_sample_dtype(out)

    def _finalize_sample_dtype(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        float_keys = (
            "keypoints", "confidence", "centers",
            "vision_feats", "vision_patch_tokens",
        )
        for k in float_keys:
            if k in sample and torch.is_tensor(sample[k]):
                sample[k] = sample[k].to(self.dtype)
        # cropped_video stays uint8 here; a downstream transform (in the
        # training loop) will scale it to [0, 1] and cast to float.
        return sample
