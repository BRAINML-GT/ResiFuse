"""CalMS21 Task 1 Stage 3 cache precompute.

Iterates all windows of a given split via ``CalMS21Dataset`` (raw
path: Stage 1 per-video .pt + decord mp4 decode + per-mouse crop),
runs a frozen vision encoder, and writes:

    {out_dir}/{video_id}_wNNN.pt          small tensors + metadata
    {out_dir}/{video_id}_wNNN_vision.npy  (A, T_ds, D) float features
    {out_dir}/{video_id}_wNNN_patches.npy (A, T_ds, P, D) if requested

Design: mirrors ``resifuse/features/hf_mabe.py`` but uses the CalMS21
dataset class, has a single ``--split`` dimension (no stage), and
drops the ``refresh_nonvision`` mode because CalMS21 cache is rebuilt
from scratch each time.

Multi-GPU: pass ``--devices cuda:0,cuda:1,...`` to shard videos
round-robin across devices; each device runs in its own spawned
subprocess with an independent encoder.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

from resifuse.config import DEFAULT_CONFIG, load_config
from resifuse.data.datasets import CalMS21Dataset
# NOTE: the encoders in ``resifuse.features.hf_mabe`` (Qwen, VideoMAE)
# pull in flash_attn / transformers — heavy imports only needed when actually
# running the vision encoder. Defer them into build_encoder() so --motion-only
# works on hosts without flash_attn (e.g. login nodes).


ENCODER_CHOICES = ("qwen", "clip", "dinov3", "videomae")
DEFAULT_MODEL_IDS = {
    "clip": "openai/clip-vit-large-patch14",
    "dinov3": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "videomae": "OpenGVLab/VideoMAEv2-Large",
}


# ----------------------------------------------------------------------
# Config helpers (subset of resifuse/features/hf_mabe.py)
# ----------------------------------------------------------------------

def parse_dtype(dtype_str: str | None) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32, "float": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    if not dtype_str:
        return torch.bfloat16
    return dtype_map.get(dtype_str.lower(), torch.bfloat16)


def load_cfg(path: str) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(load_config(path))
    return cfg


def resolve_frame_resize(cfg: dict) -> int | None:
    """Return a square integer size derived from cfg['frame_resize'] (or None)."""
    fr = cfg.get("frame_resize")
    if fr is None:
        return None
    if isinstance(fr, (list, tuple)):
        if len(fr) == 0:
            return None
        vals = [int(v) for v in fr if v]
        return min(vals) if vals else None
    return int(fr)


def to_numpy_contiguous(t: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a contiguous numpy array (bf16 → f16)."""
    if t.dtype == torch.bfloat16:
        t = t.float().half()
    return t.detach().cpu().numpy().copy()


# ----------------------------------------------------------------------
# Motion feats builder (mirrors MaBeMouseDataset._build_motion_feats_from_keypoints +
# _build_neighbor_feats, but eats raw-pixel keypoints and normalizes by
# the original video dims — CalMS21 is 1024x570 non-square).
#
# Output: [A, T_ds, D] where D = K*{3 or 2} + (K*2 if velocity) + M*4
# For the CalMS21 default config (K=7, incl_conf=True, incl_vel=False, M=1) => D=25.
# ----------------------------------------------------------------------

def _build_motion_feats_for_window(
    keypoints: torch.Tensor,
    confidence: torch.Tensor,
    mask: torch.Tensor,
    centers: torch.Tensor,
    video_width: int,
    video_height: int,
    max_neighbors: int = 1,
    include_velocity: bool = False,
    include_confidence: bool = True,
) -> torch.Tensor:
    """Build motion_feats for one window.

    Args:
        keypoints:  [A, T, K, 2] raw-pixel xy
        confidence: [A, T, K]
        mask:       [A, T] bool (True = valid)
        centers:    [A, T, 2] raw-pixel per-mouse center
        video_width, video_height: original frame dims (for xy normalization)
        max_neighbors: number of nearest neighbors per mouse (0 = skip)
        include_velocity: append per-joint velocity (frame diff on normalized xy)
        include_confidence: include conf as a per-joint channel in base feats

    Returns:
        [A, T, D] float32 motion_feats. Invalid frames are zeroed.
    """
    A, T, K, _ = keypoints.shape
    dtype = torch.float32
    kp = keypoints.to(dtype)
    conf = confidence.to(dtype)
    msk = mask.to(dtype)

    scale = torch.tensor([float(video_width), float(video_height)], dtype=dtype)
    xy = (kp / scale) * msk.unsqueeze(-1).unsqueeze(-1)
    conf = conf * msk.unsqueeze(-1)

    parts: List[torch.Tensor] = []
    if include_confidence:
        parts.append(torch.cat([xy, conf.unsqueeze(-1)], dim=-1).reshape(A, T, K * 3))
    else:
        parts.append(xy.reshape(A, T, K * 2))

    if include_velocity:
        vel = xy - xy.roll(1, dims=1)
        vel[:, 0] = 0.0
        vel = vel * msk.unsqueeze(-1).unsqueeze(-1)
        parts.append(vel.reshape(A, T, K * 2))

    M = min(int(max_neighbors), A - 1) if A > 1 else 0
    if M > 0:
        scale_d = float(max(video_width, video_height))
        centers_f = centers.to(dtype)
        centers_t = centers_f.permute(1, 0, 2)            # [T, A, 2]
        valid_t = mask.permute(1, 0)                      # [T, A]
        dists = torch.cdist(centers_t, centers_t)         # [T, A, A]
        diag = torch.eye(A, dtype=torch.bool).unsqueeze(0)
        dists = dists.masked_fill(diag, float("inf"))
        valid_pairs = valid_t.unsqueeze(2) & valid_t.unsqueeze(1)
        dists = dists.masked_fill(~valid_pairs, float("inf"))

        vals, idx = torch.topk(dists, k=M, dim=-1, largest=False)  # [T, A, M]
        centers_exp = centers_t.unsqueeze(2).expand(-1, -1, M, -1)  # [T, A, M, 2]
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, 2)
        neighbor_centers = torch.gather(centers_exp, 1, idx_exp)    # [T, A, M, 2]
        delta = neighbor_centers - centers_t.unsqueeze(2)
        dx, dy = delta[..., 0], delta[..., 1]
        theta = torch.atan2(dy, dx)
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        dist_norm = vals / scale_d

        conf_mean = confidence.mean(dim=2).to(dtype)       # [A, T]
        conf_tp = conf_mean.permute(1, 0)                  # [T, A]
        conf_tp_exp = conf_tp.unsqueeze(-1).expand(-1, -1, M)
        neighbor_conf = conf_tp_exp.gather(1, idx)

        finite = torch.isfinite(vals)
        dist_norm = torch.where(finite, dist_norm, torch.zeros_like(dist_norm))
        sin_t = torch.where(finite, sin_t, torch.zeros_like(sin_t))
        cos_t = torch.where(finite, cos_t, torch.zeros_like(cos_t))
        neighbor_conf = torch.where(finite, neighbor_conf, torch.zeros_like(neighbor_conf))

        feats = torch.stack([dist_norm, sin_t, cos_t, neighbor_conf], dim=-1)  # [T, A, M, 4]
        feats = feats.permute(1, 0, 2, 3).reshape(A, T, M * 4)
        parts.append(feats)

    return torch.cat(parts, dim=-1)


def _resolve_motion_cfg(cfg: dict) -> dict:
    """Read motion-feat knobs from a loaded config dict with CalMS21 defaults."""
    return {
        "max_neighbors": int(cfg.get("max_neighbors", 1)),
        "include_velocity": bool(cfg.get("motion_include_velocity", False)),
        "include_confidence": bool(cfg.get("motion_include_confidence", True)),
    }


def render_progress(prefix: str, cur: int, total: int, detail: str = "") -> None:
    if total <= 0:
        return
    bar_len = 24
    filled = int(bar_len * cur / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    line = f"{prefix} [{bar}] {cur}/{total}"
    if detail:
        line += f" {detail}"
    end = "\n" if cur >= total else "\r"
    print(line, end=end, flush=True)


class HFImageEncoderForCache(nn.Module):
    """Minimal CLIP / DINOv3 image-encoder wrapper for cache export."""

    def __init__(
        self,
        encoder_type: str,
        model_id: str | None = None,
        image_size: int | None = None,
    ):
        super().__init__()
        import inspect
        from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel

        if encoder_type not in ("clip", "dinov3"):
            raise ValueError(f"HFImageEncoderForCache does not support encoder_type={encoder_type!r}")
        self.encoder_type = encoder_type
        self.model_id = model_id or DEFAULT_MODEL_IDS[encoder_type]

        processor = AutoImageProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        mean = list(processor.image_mean)
        std = list(processor.image_std)
        from resifuse.features.hf_mabe import processor_image_size
        processor_default = processor_image_size(processor)
        self._processor_default_size = int(processor_default)
        self.image_size = int(image_size) if image_size is not None else self._processor_default_size

        if encoder_type == "clip":
            self.backbone = CLIPVisionModel.from_pretrained(self.model_id)
            self._needs_interp_pos_flag = True
        else:
            self.backbone = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
            self._needs_interp_pos_flag = False
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        try:
            sig = inspect.signature(self.backbone.forward)
            self._supports_interp_pos = "interpolate_pos_encoding" in sig.parameters
        except (TypeError, ValueError):
            self._supports_interp_pos = False

        cfg = getattr(self.backbone, "config", None)
        vis_cfg = getattr(cfg, "vision_config", cfg)
        hidden = getattr(vis_cfg, "hidden_size", None)
        if hidden is None:
            raise RuntimeError(f"Could not resolve hidden_size from config for {self.model_id}")
        self.vision_dim = int(hidden)
        patch_size = getattr(vis_cfg, "patch_size", None)
        if patch_size is None:
            patch_size = 16 if encoder_type == "dinov3" else 14
        self._patch_size = int(patch_size)

        self.register_buffer("_pixel_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_pixel_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        non_native = self.image_size != self._processor_default_size
        if non_native and self._needs_interp_pos_flag:
            pos_mode = f"interpolate_pos_encoding={self._supports_interp_pos}"
        elif non_native:
            pos_mode = "native_2d_rope"
        else:
            pos_mode = "pretrained_resolution"
        print(
            f"[HFImageEncoderForCache] loaded encoder={encoder_type} model_id={self.model_id} "
            f"vision_dim={self.vision_dim} image_size={self.image_size} patch_size={self._patch_size} "
            f"pos_mode={pos_mode} (native={self._processor_default_size})",
            flush=True,
        )
        if non_native and self._needs_interp_pos_flag and not self._supports_interp_pos:
            raise RuntimeError(
                f"{self.model_id} does not expose interpolate_pos_encoding in forward() "
                "but uses learned absolute position embeddings; "
                f"cannot safely run at image_size={self.image_size} "
                f"(pretrained size={self._processor_default_size}). "
                "Upgrade transformers or pass --encoder-image-size "
                f"{self._processor_default_size}."
            )

    def _encode_real_frames(self, video_frames: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video_frames.shape
        x = video_frames.reshape(B * T, C, H, W)
        mean = self._pixel_mean.to(dtype=x.dtype, device=x.device)
        std = self._pixel_std.to(dtype=x.dtype, device=x.device)
        x = (x / 255.0 - mean) / std
        if H != self.image_size or W != self.image_size:
            x = F.interpolate(
                x.float(),
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).to(dtype=video_frames.dtype)
        forward_kwargs = {"pixel_values": x}
        if (
            self._needs_interp_pos_flag
            and self._supports_interp_pos
            and self.image_size != self._processor_default_size
        ):
            forward_kwargs["interpolate_pos_encoding"] = True
        outputs = self.backbone(**forward_kwargs)
        hidden = outputs.last_hidden_state
        n_patches = (self.image_size // self._patch_size) ** 2
        prefix = hidden.shape[1] - n_patches
        if prefix < 0:
            raise RuntimeError(
                f"Unexpected token count: hidden={hidden.shape[1]} n_patches={n_patches} "
                f"(image_size={self.image_size}, patch_size={self._patch_size})"
            )
        patch_tokens = hidden[:, prefix:, :]
        return patch_tokens.mean(dim=1)

    def _encode_real_frames_with_patch_tokens(self, video_frames: torch.Tensor):
        raise NotImplementedError(
            "Patch-token caching is only supported for the Qwen encoder; "
            "run without --save-vision-patch-tokens for CLIP / DINOv3."
        )




# ----------------------------------------------------------------------
# Dataset + Encoder builders
# ----------------------------------------------------------------------

def build_dataset(cfg: dict, split: str) -> CalMS21Dataset:
    data_dir = cfg.get("calms21_data_dir")
    if not data_dir:
        raise RuntimeError("calms21_data_dir must be set in config")
    return CalMS21Dataset(
        data_dir=data_dir,
        split=split,
        split_json=cfg.get("calms21_split_json"),
        ds_stride=int(cfg.get("ds_stride", 4)),
        crop_size=int(cfg.get("calms21_crop_size", 512)),
        frame_resize=cfg.get("frame_resize"),
        use_vision_cache=False,
        return_frame_labels=True,
    )


def build_video_index(ds: CalMS21Dataset) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for idx, sample in enumerate(ds.samples):
        out.setdefault(sample["video_id"], []).append(idx)
    return out


def build_encoder(
    cfg: dict,
    device: torch.device,
    max_len: int,
    encoder_type: str,
    encoder_model_id: str | None,
    encoder_image_size: int | None,
    save_patch_tokens: bool,
) -> nn.Module:
    # Deferred imports: only pay flash_attn / transformers cost when actually
    # building an encoder. --motion-only callers never reach this function.
    from resifuse.features.hf_mabe import VideoMAEEncoderForCache, build_qwen_encoder

    if encoder_type == "qwen":
        enc = build_qwen_encoder(cfg, max_len, device)
    else:
        if save_patch_tokens:
            raise ValueError(
                f"--save-vision-patch-tokens is not supported for encoder={encoder_type}; "
                "patch-token caching is Qwen-only."
            )
        resolved_image_size = encoder_image_size
        if resolved_image_size is None:
            resolved_image_size = resolve_frame_resize(cfg)
        if encoder_type == "videomae":
            enc = VideoMAEEncoderForCache(
                model_id=encoder_model_id,
                image_size=resolved_image_size,
                clip_length=int(cfg.get("videomae_clip_length", 16)),
            ).to(device=device)
        else:
            enc = HFImageEncoderForCache(
                encoder_type=encoder_type,
                model_id=encoder_model_id,
                image_size=resolved_image_size,
            ).to(device=device)
    enc.eval()
    return enc


def prepare_frames_for_encoder(frames_cat: torch.Tensor) -> torch.Tensor:
    """Canonicalize batched frames to float in [0, 255] for all encoder backends."""
    if not torch.is_floating_point(frames_cat):
        frames_cat = frames_cat.float()
    if frames_cat.numel() == 0:
        return frames_cat
    max_val = float(frames_cat.detach().amax().item())
    if max_val <= 1.5:
        frames_cat = frames_cat * 255.0
    return frames_cat


def encode_frames_batch(
    encoder: nn.Module,
    frames_list: List[torch.Tensor],
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    return_patch_tokens: bool = False,
):
    """Run encoder on a list of (A, T, 3, H, W) tensors, return list of (A, T, D)."""
    if not frames_list:
        return ([], [], {}) if return_patch_tokens else []
    frames_cat = prepare_frames_for_encoder(torch.cat(frames_list, dim=0))
    frames_dev = frames_cat.to(device=device)

    # Input-side diagnostic — on the very first batch of the run, print
    # stats and abort if the frames already contain NaN/Inf, so we can tell
    # whether NaN originates in the dataset or in the encoder forward.
    if not getattr(encode_frames_batch, "_input_checked", False):
        encode_frames_batch._input_checked = True
        finite_mask = torch.isfinite(frames_dev)
        all_finite = bool(finite_mask.all())
        if all_finite:
            print(
                f"[INPUT SANITY] frames_dev shape={tuple(frames_dev.shape)} "
                f"dtype={frames_dev.dtype} "
                f"range=[{float(frames_dev.amin()):.2f}, {float(frames_dev.amax()):.2f}] "
                f"mean={float(frames_dev.float().mean()):.4f} (all finite)",
                flush=True,
            )
        else:
            n_nan = int((~finite_mask).sum())
            raise RuntimeError(
                f"[INPUT SANITY] frames_dev already contains {n_nan} non-finite "
                f"values BEFORE encoder forward. shape={tuple(frames_dev.shape)} "
                f"dtype={frames_dev.dtype}. "
                f"Fix the dataset / decode path, not the encoder."
            )

    ctx = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if device.type == "cuda" and autocast_dtype is not None
        else nullcontext()
    )
    with torch.inference_mode(), ctx:
        if return_patch_tokens:
            feats_flat, patch_flat, patch_meta = encoder._encode_real_frames_with_patch_tokens(frames_dev)
        else:
            feats_flat = encoder._encode_real_frames(frames_dev)
            patch_flat = None
            patch_meta = None

    # Output-side check: if features are already non-finite right after the
    # encoder forward, print a concise diagnostic and abort. This localises
    # NaN to the encoder (versus the downstream save path).
    if not getattr(encode_frames_batch, "_output_checked", False) and feats_flat is not None:
        encode_frames_batch._output_checked = True
        finite_mask = torch.isfinite(feats_flat)
        if not bool(finite_mask.all()):
            n_nan = int((~finite_mask).sum())
            raise RuntimeError(
                f"[ENCODER SANITY] feats_flat has {n_nan} non-finite values "
                f"immediately after encoder forward. shape={tuple(feats_flat.shape)} "
                f"dtype={feats_flat.dtype}. Encoder is producing NaN — input was fine."
            )
        else:
            print(
                f"[ENCODER SANITY] feats_flat shape={tuple(feats_flat.shape)} "
                f"dtype={feats_flat.dtype} "
                f"mean={float(feats_flat.float().mean()):.4f} "
                f"std={float(feats_flat.float().std()):.4f} (all finite)",
                flush=True,
            )

    B, T = frames_cat.shape[0], frames_cat.shape[1]
    feats_batch = feats_flat.view(B, T, -1)
    patch_batch = None
    if patch_flat is not None:
        patch_batch = patch_flat.view(B, T, patch_flat.shape[1], patch_flat.shape[2])

    feats_out, patches_out = [], []
    off = 0
    for frames in frames_list:
        A = frames.shape[0]
        feats_out.append(feats_batch[off : off + A].cpu())
        if patch_batch is not None:
            patches_out.append(patch_batch[off : off + A].cpu())
        off += A
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if return_patch_tokens:
        meta_cpu = {}
        for k, v in (patch_meta or {}).items():
            meta_cpu[k] = v.detach().cpu() if torch.is_tensor(v) else v
        return feats_out, patches_out, meta_cpu
    return feats_out


# ----------------------------------------------------------------------
# Per-video processing
# ----------------------------------------------------------------------

def process_video(
    ds: CalMS21Dataset,
    video_id: str,
    indices: Sequence[int],
    out_dir: Path,
    overwrite: bool,
    encoder: nn.Module,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    batch_size: int,
    save_patch_tokens: bool,
    num_workers: int = 4,
    motion_cfg: dict | None = None,
) -> None:
    """Cache all windows of one video to per-window .pt + .npy.

    The sample-loading side (ffmpeg mp4 decode + per-mouse crop +
    resize) runs in ``num_workers`` threads via ThreadPoolExecutor so
    the GPU encoder forward can run on one batch while subsequent
    batches are prepped on CPU in the background. Threads (not
    processes) avoid the fork-after-CUDA-init hang that would
    otherwise deadlock the DataLoader-based approach.
    """
    sentinel = out_dir / f"{video_id}_w000.pt"
    if sentinel.exists() and not overwrite:
        return
    if not indices:
        return

    # Buffers indexed by window order (not dataset index)
    kp_buf: List[torch.Tensor] = []
    conf_buf: List[torch.Tensor] = []
    mask_buf: List[torch.Tensor] = []
    ctr_buf: List[torch.Tensor] = []
    lbl_buf: List[torch.Tensor] = []
    meta_buf: List[dict] = []
    feat_buf: List[torch.Tensor] = []
    patch_buf: List[torch.Tensor] = []
    shared_patch_meta: dict | None = None

    frames_batch: List[torch.Tensor] = []
    non_vision_batch: List[dict] = []

    def _flush():
        nonlocal shared_patch_meta
        if not non_vision_batch:
            return
        if save_patch_tokens:
            feats_list, patches_list, patch_meta = encode_frames_batch(
                encoder, frames_batch, device, autocast_dtype, return_patch_tokens=True
            )
            if shared_patch_meta is None:
                shared_patch_meta = patch_meta or {}
        else:
            feats_list = encode_frames_batch(
                encoder, frames_batch, device, autocast_dtype, return_patch_tokens=False
            )
            patches_list = []
        for j, payload in enumerate(non_vision_batch):
            kp_buf.append(payload["keypoints"])
            conf_buf.append(payload["confidence"])
            mask_buf.append(payload["mask"])
            ctr_buf.append(payload["centers"])
            lbl_buf.append(payload["frame_labels"])
            meta_buf.append(payload["metadata"])
            feat_buf.append(feats_list[j])
            if save_patch_tokens:
                patch_buf.append(patches_list[j])
        frames_batch.clear()
        non_vision_batch.clear()

    # Parallel sample loader via threads (NOT processes).
    #
    # IMPORTANT: PyTorch DataLoader(num_workers>0) forks the main
    # process, and because we already placed the vision tower on
    # cuda:<N>, CUDA runtime has initialized. Fork-after-CUDA-init
    # leaves the child with a broken CUDA context and the Python
    # interpreter hangs indefinitely on the first child import that
    # touches torch CUDA symbols. We hit this in Report_2880499 /
    # Report_2880556: both GPU shards froze at their first non-sentinel
    # video, regardless of whether the underlying video decode was
    # decord or ffmpeg subprocess.
    #
    # Threads avoid fork entirely. ffmpeg subprocess decoding and
    # numpy ops both release the GIL (ffmpeg blocks on stdout pipe,
    # numpy on C math), so N threads really do give N-way parallelism
    # on the CPU-bound sample loading path. The encoder forward still
    # runs on the main thread under autocast, uncontested.
    idx_list = list(indices)
    prefetch_depth = max(1, num_workers * 2) if num_workers > 0 else 1

    def _load_one(idx: int) -> dict:
        return ds[idx]

    if num_workers <= 0:
        # Sequential fallback — useful for debugging.
        sample_iter = (_load_one(i) for i in idx_list)
    else:
        pool = ThreadPoolExecutor(max_workers=num_workers)

        def _ordered_prefetch():
            in_flight: list = []
            cursor = 0
            total = len(idx_list)
            # Prime the pump.
            while cursor < total and len(in_flight) < prefetch_depth:
                in_flight.append(pool.submit(_load_one, idx_list[cursor]))
                cursor += 1
            while in_flight:
                fut = in_flight.pop(0)
                result = fut.result()
                if cursor < total:
                    in_flight.append(pool.submit(_load_one, idx_list[cursor]))
                    cursor += 1
                yield result

        sample_iter = _ordered_prefetch()

    try:
        for sample in sample_iter:
            frames_batch.append(sample["cropped_video"])
            non_vision_batch.append({
                "keypoints":    sample["keypoints"],
                "confidence":   sample["confidence"],
                "mask":         sample["mask"],
                "centers":      sample["centers"],
                "frame_labels": sample["frame_labels"],
                "metadata":     sample["metadata"],
            })
            if len(non_vision_batch) >= batch_size:
                _flush()
        _flush()
    finally:
        if num_workers > 0:
            pool.shutdown(wait=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    N = len(meta_buf)
    # Write order: all windows w > 0 first, sentinel w = 0 LAST.
    # Resume semantics depend on `{video_id}_w000.pt` being the
    # "done marker". Writing it last means a mid-write crash
    # (SIGKILL, node failure, OOM, etc.) leaves the sentinel absent
    # and the whole video gets re-processed cleanly on resume. If
    # we wrote w=0 first (the naive order), a crash mid-way through
    # the 80+ window writes would leave a sentinel on disk with only
    # a single window materialized, producing a permanently corrupt
    # cache that resume would silently skip.
    write_order = list(range(1, N)) + ([0] if N > 0 else [])
    for w in write_order:
        stem = f"{video_id}_w{w:03d}"
        window_data = {
            "video_id":    video_id,
            "window_idx":  w,
            "num_windows": N,
            "ds_stride":   ds.ds_stride,
            "keypoints":   kp_buf[w].clone(),
            "confidence":  conf_buf[w].clone(),
            "centers":     ctr_buf[w].clone(),
            "mask":        mask_buf[w].clone(),
            "frame_labels": lbl_buf[w].clone(),
            "metadata":    meta_buf[w],
        }
        # Vision features via .npy sidecar for mmap-friendly IO.
        np.save(
            out_dir / f"{stem}_vision.npy",
            to_numpy_contiguous(feat_buf[w]),
        )
        if save_patch_tokens:
            np.save(
                out_dir / f"{stem}_patches.npy",
                to_numpy_contiguous(patch_buf[w]),
            )
            if shared_patch_meta:
                window_data["vision_patch_meta"] = shared_patch_meta
        # Motion features via .npy sidecar. Same mmap-friendly read path
        # as vision, consumed by CalMS21Dataset._load_cache_sample.
        if motion_cfg is not None:
            motion_feats = _build_motion_feats_for_window(
                kp_buf[w], conf_buf[w], mask_buf[w], ctr_buf[w],
                video_width=ds.video_width,
                video_height=ds.video_height,
                max_neighbors=motion_cfg["max_neighbors"],
                include_velocity=motion_cfg["include_velocity"],
                include_confidence=motion_cfg["include_confidence"],
            )
            np.save(
                out_dir / f"{stem}_motion.npy",
                motion_feats.to(torch.float16).numpy(),
            )
        torch.save(window_data, out_dir / f"{stem}.pt")


def process_video_motion_only(
    ds: CalMS21Dataset,
    video_id: str,
    indices: Sequence[int],
    out_dir: Path,
    overwrite: bool,
    motion_cfg: dict,
) -> int:
    """Write {video_id}_wNNN_motion.npy sidecars without running the encoder.

    Requires the existing per-window .pt files from a prior full precompute
    pass (so motion sidecars align 1:1 with the vision cache). If a .pt
    file is missing, that window is skipped and counted toward the return
    value; caller can surface a summary.
    """
    if not indices:
        return 0
    # Load Stage 1 video .pt once (pt_file lives in ds.video_meta from
    # CalMS21Dataset.__init__).
    cached = ds._load_video_pt(video_id)
    meta = cached.get("_meta", {})
    video_w = int(meta.get("video_width", ds.video_width))
    video_h = int(meta.get("video_height", ds.video_height))

    out_dir.mkdir(parents=True, exist_ok=True)
    skipped_missing = 0
    for idx in indices:
        sample = ds.samples[idx]
        w = int(sample["window_idx"])
        stem = f"{video_id}_w{w:03d}"
        pt_path = out_dir / f"{stem}.pt"
        if not pt_path.exists():
            skipped_missing += 1
            continue
        sidecar = out_dir / f"{stem}_motion.npy"
        if sidecar.exists() and not overwrite:
            continue
        tensors = ds._downsample_window_tensors(cached, w)
        motion_feats = _build_motion_feats_for_window(
            tensors["keypoints"], tensors["confidence"], tensors["mask"], tensors["centers"],
            video_width=video_w, video_height=video_h,
            max_neighbors=motion_cfg["max_neighbors"],
            include_velocity=motion_cfg["include_velocity"],
            include_confidence=motion_cfg["include_confidence"],
        )
        np.save(str(sidecar), motion_feats.to(torch.float16).numpy())
    return skipped_missing


def process_shard_on_device(
    cfg: dict,
    split: str,
    video_ids: Sequence[str],
    out_dir: str,
    overwrite: bool,
    device_str: str,
    batch_size: int,
    encoder_type: str,
    encoder_model_id: str | None,
    encoder_image_size: int | None,
    save_patch_tokens: bool,
    num_workers: int = 4,
) -> None:
    device = torch.device(device_str)
    ds = build_dataset(cfg, split)
    video_to_indices = build_video_index(ds)
    max_len = int(ds.window_size)
    dtype = parse_dtype(cfg.get("torch_dtype") or cfg.get("dtype"))
    autocast_dtype = dtype if device.type == "cuda" else None
    motion_cfg = _resolve_motion_cfg(cfg)

    encoder = build_encoder(
        cfg,
        device,
        max_len=max_len,
        encoder_type=encoder_type,
        encoder_model_id=encoder_model_id,
        encoder_image_size=encoder_image_size,
        save_patch_tokens=save_patch_tokens,
    )

    total = len(video_ids)
    prefix = f"[Cache][{split}][{device_str}][{encoder_type}]"
    first_video_checked = False
    for i, vid in enumerate(video_ids, 1):
        indices = video_to_indices.get(vid, [])
        process_video(
            ds, vid, indices,
            out_dir=Path(out_dir),
            overwrite=overwrite,
            encoder=encoder,
            device=device,
            autocast_dtype=autocast_dtype,
            batch_size=batch_size,
            save_patch_tokens=save_patch_tokens,
            num_workers=num_workers,
            motion_cfg=motion_cfg,
        )
        render_progress(prefix, i, total, vid)

        # Fail-fast sanity check for videomae after the first video on each
        # shard. Loads the freshly-written vision .npy files and raises if
        # any contain NaN/Inf. Mirrors resifuse/features/hf_mabe.py's behaviour.
        if encoder_type == "videomae" and not first_video_checked:
            first_video_checked = True
            out_dir_path = Path(out_dir)
            vnpys = sorted(
                p for p in os.listdir(out_dir_path)
                if p.startswith(f"{vid}_") and p.endswith("_vision.npy")
            )
            if not vnpys:
                print(f"{prefix} videomae sanity: no vision npys found for {vid}", flush=True)
            else:
                bad = []
                stats = None
                for fn in vnpys:
                    arr = np.load(out_dir_path / fn, mmap_mode="r")
                    if np.isnan(arr).any() or np.isinf(arr).any():
                        bad.append(fn)
                    elif stats is None:
                        stats = (float(arr.mean()), float(arr.std()), float(np.abs(arr).max()))
                if bad:
                    raise RuntimeError(
                        f"{prefix} videomae sanity FAILED on first video {vid}: "
                        f"{len(bad)}/{len(vnpys)} vision npys contain NaN/Inf. "
                        f"Example: {bad[0]}. Aborting before wasting time on the full run."
                    )
                print(
                    f"{prefix} videomae sanity OK on {vid} "
                    f"({len(vnpys)} windows, mean={stats[0]:.4f} std={stats[1]:.4f} absmax={stats[2]:.4f})",
                    flush=True,
                )


def process_shard_motion_only(
    cfg: dict,
    split: str,
    video_ids: Sequence[str],
    out_dir: str,
    overwrite: bool,
) -> None:
    """Shard runner for --motion-only: no GPU, no encoder, no mp4 decode.

    Reuses CalMS21Dataset only as a Stage-1 .pt lookup + downsample helper
    (its __getitem__ would trigger ffmpeg decode, which we bypass here).
    """
    ds = build_dataset(cfg, split)
    video_to_indices = build_video_index(ds)
    motion_cfg = _resolve_motion_cfg(cfg)
    total = len(video_ids)
    prefix = f"[Motion][{split}]"
    total_missing = 0
    for i, vid in enumerate(video_ids, 1):
        indices = video_to_indices.get(vid, [])
        missing = process_video_motion_only(
            ds, vid, indices,
            out_dir=Path(out_dir),
            overwrite=overwrite,
            motion_cfg=motion_cfg,
        )
        total_missing += missing
        render_progress(prefix, i, total, vid)
    if total_missing:
        print(
            f"{prefix} skipped {total_missing} windows missing a .pt "
            f"(run a full precompute pass first to create them).",
            flush=True,
        )


def split_round_robin(items: Sequence[str], groups: int) -> List[List[str]]:
    buckets = [[] for _ in range(groups)]
    for i, v in enumerate(items):
        buckets[i % groups].append(v)
    return buckets


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CalMS21 Stage 3 vision cache precompute")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--encoder",
        type=str,
        choices=ENCODER_CHOICES,
        default="qwen",
        help="Vision encoder backend: qwen (default) | clip | dinov3.",
    )
    parser.add_argument(
        "--encoder-model-id",
        type=str,
        default=None,
        help="Override HF model id for clip/dinov3 backends.",
    )
    parser.add_argument(
        "--encoder-image-size",
        type=int,
        default=None,
        help=(
            "Input image size fed to clip/dinov3 backbones. "
            "Defaults to cfg['frame_resize'] so encoder ablations use the "
            "same resized crops unless explicitly overridden."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Windows per vision encoder forward")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers per GPU shard for parallel "
                             "mp4 decode + crop (0 = sequential in main process)")
    parser.add_argument("--devices", type=str, default="cuda",
                        help="Comma-separated CUDA devices (e.g. cuda:0,cuda:1)")
    parser.add_argument(
        "--save-vision-patch-tokens", action="store_true",
        help="Also cache patch tokens (WARNING: large on disk)",
    )
    parser.add_argument("--calms21-data-dir", type=str, default=None)
    parser.add_argument("--calms21-cache-dir", type=str, default=None)
    parser.add_argument("--calms21-split-json", type=str, default=None)
    parser.add_argument(
        "--motion-only", action="store_true",
        help=(
            "Only write {video_id}_wNNN_motion.npy sidecars into the existing "
            "cache dir; skip vision encoder build + mp4 decode. Requires a "
            "prior full precompute pass (existing .pt / _vision.npy files)."
        ),
    )
    parser.add_argument("--frame-resize", type=int, default=None,
                        help="Override frame_resize of the config (frames are resized to this before the encoder).")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.frame_resize:
        cfg["frame_resize"] = args.frame_resize
    if args.calms21_data_dir:
        cfg["calms21_data_dir"] = args.calms21_data_dir
    if args.calms21_split_json:
        cfg["calms21_split_json"] = args.calms21_split_json
    cache_root = args.calms21_cache_dir or cfg.get("calms21_cache_dir")
    if not cache_root:
        raise RuntimeError("calms21_cache_dir must be provided via CLI or config")
    if args.save_vision_patch_tokens and args.encoder != "qwen":
        raise ValueError(
            f"--save-vision-patch-tokens is only supported for --encoder qwen, "
            f"got --encoder {args.encoder}."
        )

    device_list = [d.strip() for d in args.devices.split(",") if d.strip()]
    if not device_list:
        device_list = ["cuda"]

    for split in args.splits:
        # Probe video list on CPU once (no encoder build) to shard evenly.
        ds = build_dataset(cfg, split)
        video_to_indices = build_video_index(ds)
        video_ids = sorted(video_to_indices.keys())
        out_dir = Path(cache_root) / split
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Cache] {split}: {len(video_ids)} videos, {len(ds)} windows → {out_dir}", flush=True)
        del ds

        if args.motion_only:
            # CPU-only path. No encoder, no mp4 decode — just iterate
            # Stage 1 .pt + downsample + compute motion sidecars.
            process_shard_motion_only(
                cfg, split, video_ids, str(out_dir),
                overwrite=args.overwrite,
            )
            continue

        if len(device_list) == 1:
            process_shard_on_device(
                cfg, split, video_ids, str(out_dir),
                overwrite=args.overwrite,
                device_str=device_list[0],
                batch_size=max(1, int(args.batch_size)),
                encoder_type=args.encoder,
                encoder_model_id=args.encoder_model_id,
                encoder_image_size=args.encoder_image_size,
                save_patch_tokens=args.save_vision_patch_tokens,
                num_workers=max(0, int(args.num_workers)),
            )
        else:
            ctx = mp.get_context("spawn")
            shards = split_round_robin(video_ids, len(device_list))
            procs = []
            for dev, shard in zip(device_list, shards):
                p = ctx.Process(
                    target=process_shard_on_device,
                    args=(
                        cfg, split, shard, str(out_dir),
                        args.overwrite, dev,
                        max(1, int(args.batch_size)),
                        args.encoder,
                        args.encoder_model_id,
                        args.encoder_image_size,
                        args.save_vision_patch_tokens,
                        max(0, int(args.num_workers)),
                    ),
                )
                p.start()
                procs.append(p)
            for p in procs:
                p.join()

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
