"""Freely-moving-mouse unsupervised DEC behavior segmentation.

Mirrors resifuse/training/unsup_mabe.py but uses :class:`FreelyMovingMouseDataset`.
Freely batches carry ``(stem, start, end)`` rather than the MaBe-style
``metadata`` dict, so :class:`FreelyDECWrapper` injects a synthetic
``metadata`` per sample (video_id=stem, start_frame=start, ds_stride=...)
so the shared unsup helpers (global-mask lookup, single-label F1 dedup) work
unchanged.

Usage:
    python -m resifuse.training.unsup_freely --config ./config_free_finetune.yaml
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import Dataset

from resifuse.config import load_config_with_defaults
from resifuse.data.datasets import FreelyMovingMouseDataset

from resifuse.training.unsup_mabe import (
    parse_args as _mabe_parse_args,
    run_unsup,
    _add_shared_cli_args,
    apply_dec_fix_overrides,
)


FREELY_DEFAULT_LABELS = ["Rearing", "Grooming"]


def parse_args(config_path: str | None = None):
    args = _mabe_parse_args(config_path)
    args.dataset = "freely_mouse"
    return args


class FreelyDECWrapper(Dataset):
    """Expose MaBe-style ``metadata`` dict on each freely sample so the shared
    unsup helpers can look up sample keys and dedup frames by video-id.
    """

    def __init__(self, base: FreelyMovingMouseDataset, ds_stride: int):
        self.base = base
        self.ds_stride = int(ds_stride)
        self.seq_len = int(base.seq_len)
        self.num_joints = int(base.num_joints)
        # Freely doesn't set self.label_names — leave empty and let run_unsup
        # fall back to FREELY_DEFAULT_LABELS.
        self.label_names = list(getattr(base, "label_names", []) or [])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s = dict(self.base[idx])
        stem = s.get("stem")
        stem_str = str(stem) if stem is not None else f"unknown_{idx}"
        start = int(s.get("start", 0) or 0)
        end = int(s.get("end", 0) or 0)
        s["metadata"] = {
            "video_id": stem_str,
            "window_idx": start,
            "mouse_id": "",
            "start_frame": start,
            "ds_stride": self.ds_stride,
        }
        return s


def _build_freely_dataset(args, split: str) -> FreelyDECWrapper:
    train_ratio = getattr(args, "finetune_train_ratio", None) or getattr(args, "train_ratio", 0.7)
    val_ratio = getattr(args, "finetune_val_ratio", None) or getattr(args, "val_ratio", 0.15)
    test_ratio = getattr(args, "finetune_test_ratio", None) or getattr(args, "test_ratio", 0.15)
    include_velocity = bool(getattr(args, "motion_include_velocity", False))
    include_confidence = bool(getattr(args, "motion_include_confidence", True))
    base = FreelyMovingMouseDataset(
        root_dir=args.data_root,
        seq_len=getattr(args, "seq_len", 128) or 128,
        ds_stride=int(getattr(args, "ds_stride", 1) or 1),
        frame_resize=getattr(args, "frame_resize", None),
        frame_resize_align=int(getattr(args, "frame_resize_align", 32) or 32),
        pretrain_stride=getattr(args, "pretrain_stride", None),
        split=split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stage="finetune",
        dtype=args.torch_dtype,
        max_neighbors=int(getattr(args, "max_neighbors", 2) or 0),
        use_neighbor_feats=bool(getattr(args, "use_neighbor_feats", False)),
        num_joints=int(getattr(args, "num_joints", 13) or 13),
        conf_mask_thresh=float(getattr(args, "conf_mask_thresh", 0.0) or 0.0),
        cache_dir=getattr(args, "cache_dir", None),
        use_cache=bool(getattr(args, "use_cache", False)),
        oversample_grooming_factor=1,
        vision_feat_dir=getattr(args, "vision_cache_dir", None),
        use_vision_cache=bool(getattr(args, "use_vision_cache", False)),
        use_motion_scaler=bool(getattr(args, "use_motion_scaler", False)),
        drop_empty_frame_mask=False,
        marker_subject_centric=bool(getattr(args, "marker_subject_centric", True)),
        motion_include_velocity=include_velocity,
        motion_include_confidence=include_confidence,
        load_frames=not bool(getattr(args, "use_vision_cache", False)),
    )
    return FreelyDECWrapper(base, ds_stride=int(getattr(args, "ds_stride", 1) or 1))


def _freely_build_datasets(args):
    train_ds = _build_freely_dataset(args, "train")
    args.seq_len = train_ds.seq_len
    val_ds = _build_freely_dataset(args, "val")
    test_ds = _build_freely_dataset(args, "test")
    label_names = list(train_ds.label_names) if train_ds.label_names else []
    num_classes = len(label_names) if label_names else int(getattr(args, "head_a_classes", 2) or 2)
    return train_ds, val_ds, test_ds, label_names, num_classes


def main(config_path: str | None = None,
         majority_map_mode: str | None = None,
         dec_train_agg: str | None = None,
         dec_eval_agg: str | None = None,
         animal_vote_tiebreak: str | None = None,
         dec_use_bigru: bool | None = None,
         vicreg_inv: float | None = None,
         vicreg_var: float | None = None,
         vicreg_cov: float | None = None,
         vicreg_variance_target: float | None = None,
         vicreg_shift: int | None = None,
         seed: int | None = None,
         dec_fixes=None):
    args = parse_args(config_path)
    dtype_map = {
        "float32": torch.float32, "float": torch.float32,
        "float16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    args.torch_dtype = dtype_map.get(str(args.torch_dtype).lower(), torch.bfloat16)
    args.dataset = "freely_mouse"
    if majority_map_mode is not None:
        args.majority_map_mode = majority_map_mode
    if dec_train_agg is not None:
        args.dec_train_agg = dec_train_agg
    if dec_eval_agg is not None:
        args.dec_eval_agg = dec_eval_agg
    if animal_vote_tiebreak is not None:
        args.animal_vote_tiebreak = animal_vote_tiebreak
    if dec_use_bigru is not None:
        args.dec_use_bigru = dec_use_bigru
    if vicreg_inv is not None:
        args.vicreg_inv = vicreg_inv
    if vicreg_var is not None:
        args.vicreg_var = vicreg_var
    if vicreg_cov is not None:
        args.vicreg_cov = vicreg_cov
    if vicreg_variance_target is not None:
        args.vicreg_variance_target = vicreg_variance_target
    if vicreg_shift is not None:
        args.vicreg_shift = vicreg_shift
    if seed is not None:
        args.seed = seed
    if dec_fixes is not None:
        apply_dec_fix_overrides(args, dec_fixes)
    run_unsup(args, _freely_build_datasets, label_names_fallback=list(FREELY_DEFAULT_LABELS))


def _parse_cli():
    parser = argparse.ArgumentParser(description="Freely unsupervised DEC segmentation entrypoint")
    _add_shared_cli_args(parser, default_cfg="./config_free_finetune.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    ns = _parse_cli()
    main(ns.config,
         majority_map_mode=ns.majority_map_mode,
         dec_train_agg=ns.dec_train_agg,
         dec_eval_agg=ns.dec_eval_agg,
         animal_vote_tiebreak=ns.animal_vote_tiebreak,
         dec_use_bigru=ns.dec_use_bigru,
         vicreg_inv=ns.vicreg_inv,
         vicreg_var=ns.vicreg_var,
         vicreg_cov=ns.vicreg_cov,
         vicreg_variance_target=ns.vicreg_variance_target,
         vicreg_shift=ns.vicreg_shift,
         seed=ns.seed,
         dec_fixes=ns)
