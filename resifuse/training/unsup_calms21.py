"""CalMS21 unsupervised DEC behavior segmentation.

Mirrors resifuse/training/unsup_mabe.py but uses :class:`CalMS21Dataset` wrapped by
:class:`CalMS21TrainWrapper` from train_entry_calms21, which turns the source
single-class frame label [T] into beast/bams-style multi-hot [T, 3] over
{attack, investigation, mount} (drops 'other' via frame_mask).

All DEC machinery — k-means init, DEC loss, global frame-select mask,
majority-map-based F1, best-val epoch summary — is imported from
resifuse.training.unsup_mabe.

Usage:
    python -m resifuse.training.unsup_calms21 --config configs/calms21_finetune.yaml
"""
from __future__ import annotations

import argparse
import os

import torch

from resifuse.config import load_config_with_defaults
from resifuse.data.datasets import CalMS21Dataset

from resifuse.training.calms21 import CalMS21TrainWrapper, CALMS21_TARGET_CLASSES
from resifuse.training.unsup_mabe import (
    parse_args as _mabe_parse_args,
    run_unsup,
    _add_shared_cli_args,
    apply_dec_fix_overrides,
)


def parse_args(config_path: str | None = None):
    # Reuse the mabe parser (same stage-override logic) and pin dataset to calms21.
    args = _mabe_parse_args(config_path)
    args.dataset = "calms21"
    return args


def _calms21_build_dataset(args, split: str) -> CalMS21TrainWrapper:
    data_dir = getattr(args, "calms21_data_dir", None)
    cache_root = getattr(args, "calms21_cache_dir", None)
    split_json = getattr(args, "calms21_split_json", None)
    if not data_dir or not cache_root:
        raise RuntimeError("calms21_data_dir and calms21_cache_dir must be set in the config.")
    cache_dir = os.path.join(cache_root, split)
    base = CalMS21Dataset(
        data_dir=data_dir,
        split=split,
        split_json=split_json,
        ds_stride=int(getattr(args, "ds_stride", 4)),
        crop_size=int(getattr(args, "calms21_crop_size", 448)),
        frame_resize=getattr(args, "frame_resize", None),
        use_vision_cache=bool(getattr(args, "use_vision_cache", True)),
        cache_dir=cache_dir,
        return_frame_labels=True,
        dtype=args.torch_dtype,
    )
    return CalMS21TrainWrapper(base, stage="unsup", dtype=args.torch_dtype)


def _calms21_build_datasets(args):
    train_ds = _calms21_build_dataset(args, "train")
    args.seq_len = train_ds.seq_len
    val_ds = _calms21_build_dataset(args, "val")
    test_ds = _calms21_build_dataset(args, "test")
    label_names = list(CALMS21_TARGET_CLASSES)
    num_classes = len(label_names)
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
    args.dataset = "calms21"
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
    run_unsup(args, _calms21_build_datasets, label_names_fallback=list(CALMS21_TARGET_CLASSES))


def _parse_cli():
    parser = argparse.ArgumentParser(description="CalMS21 unsupervised DEC segmentation entrypoint")
    _add_shared_cli_args(parser, default_cfg="configs/calms21_finetune.yaml")
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
