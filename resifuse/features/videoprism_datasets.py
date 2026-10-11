"""Dataset helpers for VideoPrism cache extraction (MABe / Freely / CalMS21).

Provides a small, JAX-env-safe surface for the three datasets used in this
project. Intentionally does NOT import `modules` (which requires
flash_attn / transformers / Qwen) so the script can run in a lean JAX env.

Required packages:
  torch (CPU-only is fine, just for .pt IO + Dataset)
  decord, numpy, sklearn, pyyaml (pulled in transitively by data_loader)
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from resifuse.config import DEFAULT_CONFIG, load_config
from resifuse.data.datasets import CalMS21Dataset, FreelyMovingMouseDataset, MaBeMouseDataset


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

def load_cfg(path: str) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(load_config(path))
    # Match resifuse/features/hf_freely.py behaviour: if the config
    # defines split-specific seq lengths, normalize them.
    if "seq_len_cache" not in cfg:
        cfg["seq_len_cache"] = cfg.get("seq_len", DEFAULT_CONFIG.get("seq_len", 32))
    if "seq_len_train" in cfg:
        cfg["seq_len"] = cfg["seq_len_train"]
    return cfg


def _resolve_train_ratio(cfg: dict, stage: str) -> float:
    if stage == "finetune":
        ft_train = cfg.get("finetune_train_ratio")
        return ft_train if ft_train is not None else cfg.get("train_ratio", 0.8)
    return cfg.get("train_ratio", 0.8)


def _validate_mabe_split_json(path: str) -> None:
    if not os.path.exists(path):
        raise RuntimeError(f"Split JSON not found: {path}")
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to read split JSON {path}: {exc}") from exc
    splits = payload.get("splits") if isinstance(payload, dict) else None
    if splits is None:
        splits = payload
    if not isinstance(splits, dict):
        raise RuntimeError(f"Invalid split JSON {path}: expected dict with splits.")
    for key in ("train", "val", "test"):
        ids = splits.get(key)
        if not isinstance(ids, list):
            raise RuntimeError(f"Invalid split JSON {path}: '{key}' must be a list.")


# ----------------------------------------------------------------------
# Dataset builders (one per dataset family)
# ----------------------------------------------------------------------

def build_dataset_mabe(cfg: dict, stage: str, split: str) -> MaBeMouseDataset:
    mabe_root = cfg.get("mabe_data_dir")
    if not mabe_root:
        raise RuntimeError("mabe_data_dir must be set in config for MABe cache export.")
    stage_dir = os.path.join(mabe_root, stage)
    split_json = os.path.join(stage_dir, f"{stage}_split.json")
    _validate_mabe_split_json(split_json)
    return MaBeMouseDataset(
        data_dir=stage_dir,
        index_file="training_index.json",
        split=split,
        stage=stage,
        train_ratio=_resolve_train_ratio(cfg, stage),
        split_json=split_json,
        ds_stride=cfg.get("ds_stride", 1),
        multi_mouse=True,
        include_neighbor_feats=cfg.get("use_neighbor_feats", True),
        max_neighbors=cfg.get("max_neighbors", 2),
        motion_include_velocity=cfg.get("motion_include_velocity", True),
        motion_include_confidence=cfg.get("motion_include_confidence", True),
        frame_resize=cfg.get("frame_resize"),
        frame_resize_align=cfg.get("frame_resize_align", 32),
        preload=False,
        cache_size=cfg.get("cache_size", 50),
        huddle_only_keep_ratio=cfg.get("finetune_huddle_keep_ratio", 1.0),
    )


def build_dataset_calms21(cfg: dict, split: str) -> CalMS21Dataset:
    data_dir = cfg.get("calms21_data_dir")
    if not data_dir:
        raise RuntimeError("calms21_data_dir must be set in config for CalMS21.")
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


def _parse_dtype(s: str | None) -> torch.dtype:
    m = {
        "float32": torch.float32, "float": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    return m.get((s or "").lower(), torch.float32)


def build_dataset_freely(cfg: dict, stage: str, split: str) -> FreelyMovingMouseDataset:
    data_root = cfg.get("data_root")
    if not data_root:
        raise RuntimeError("data_root must be set in config for Freely Moving Mouse dataset.")
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
    return FreelyMovingMouseDataset(
        root_dir=data_root,
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
        dtype=_parse_dtype(cfg.get("dtype") or cfg.get("torch_dtype")),
        max_neighbors=cfg.get("max_neighbors", 2),
        use_neighbor_feats=cfg.get("use_neighbor_feats", True),
        num_joints=cfg.get("num_joints"),
        conf_mask_thresh=cfg.get("conf_mask_thresh", 0.3),
        cache_dir=cfg.get("cache_dir"),
        use_cache=True,
    )


# ----------------------------------------------------------------------
# Video / window indexing
# ----------------------------------------------------------------------

def build_video_index(ds) -> Dict[str, List[int]]:
    """video_id -> sorted sample indices. Works for MaBe and CalMS21 where
    ds.samples is a list of dicts with a 'video_id' key."""
    out: Dict[str, List[int]] = {}
    samples = getattr(ds, "samples", None)
    if samples is None:
        raise RuntimeError("Dataset has no .samples attribute — cannot build video index.")
    for idx, sample in enumerate(samples):
        vid = sample.get("video_id", "unknown")
        out.setdefault(vid, []).append(idx)
    return out


def freely_cache_entries(ds):
    """Freely uses ds.cache_entries for its per-window layout. Returns the
    list or None if the dataset lacks it (pre-cache mode)."""
    return getattr(ds, "cache_entries", None)


# ----------------------------------------------------------------------
# Payload / progress
# ----------------------------------------------------------------------

def _extract_sample_payload(sample: dict, use_vision: bool) -> dict:
    payload = {}
    # Standardized tensor keys present on all three datasets (fall back gracefully).
    for key in ("keypoints", "confidence", "mask", "motion_feats",
                "frame_labels", "frame_mask", "label"):
        v = sample.get(key)
        if v is not None:
            payload[key] = v.cpu() if torch.is_tensor(v) else v
    if "label_name" in sample:
        payload["label_name"] = sample["label_name"]
    meta = sample.get("metadata", {})
    if meta:
        payload["metadata"] = meta
    if not use_vision and "frames" in sample:
        payload["cropped_video"] = sample["frames"].cpu()
    return payload


def _render_progress(prefix: str, current: int, total: int, detail: str = "") -> None:
    if total <= 0:
        return
    bar_len = 24
    filled = int(bar_len * current / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    line = f"{prefix} [{bar}] {current}/{total}"
    if detail:
        line += f" {detail}"
    end = "\n" if current >= total else "\r"
    print(line, end=end, flush=True)
