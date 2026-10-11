import copy
import json
import os
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    # Data paths
    "data_root": "data/freely_moving_mouse",
    "cache_dir": None,
    "mabe_data_dir": None,
    "mabe_label_file": None,

    # Dataset/window settings
    "seq_len": 64,             # length after optional ds_stride at training time
    "seq_len_cache": 64,       # length used when building cache (cache_ds_stride applied)
    "ds_stride": 1,
    "cache_ds_stride": 1,
    "pretrain_stride": None,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "finetune_train_ratio": None,
    "finetune_val_ratio": None,
    "finetune_test_ratio": None,
    "max_neighbors": 2,
    "use_neighbor_feats": True,
    "num_joints": None,
    "conf_mask_thresh": 0.3,
    "num_workers": 2,
    "dataset": "freely_mouse",
    "min_body_length": 10.0,
    "frame_resize": None,        # optional (H, W) or int; applied at load time even when using cache
    "frame_resize_align": 32,
    "vision_cache_dir": None,
    "mabe_vision_cache_dir": None,
    "use_vision_cache": False,
    "use_motion_scaler": False,
    "marker_subject_centric": True,
    "motion_include_velocity": True,
    "motion_include_confidence": True,
    "finetune_huddle_keep_ratio": 1.0,

    # Cache preprocessing
    "stage": "pretrain",  # pretrain|finetune|both (for prepare_cache)
    "splits": ["train", "val", "test"],
    "overwrite": False,

    # Training hyperparameters
    "batch_size": 32,
    "pretrain_lr": 1e-4,
    "finetune_lr": 1e-4,
    "pretrain_epochs": 100,
    "finetune_epochs": 100,
    "finetune_freeze_epochs": 0,
    "mask_ratio": 0.5,
    "profile_timing": False,
    "detect_anomaly": False,
    "torch_dtype": "bfloat16",
    "dtype": "bfloat16",
    "use_class_weights": True,
    "vq_loss_weight": 1.0,
    "vq_recon_weight": 1.0,
    "count_loss_weight": 1.0,
    "mfr_weight": 0.0,
    "mfr_activation_radius": 2,
    "vision_pred_weight": 0.0,
    "vision_pred_horizon": 8,
    "vision_pred_horizons": [4, 8, 16],
    "vision_pred_weights": [1.0, 0.7, 0.4],
    "vision_pred_warmup_epochs": 0,
    "vision_pred_loss": "cosine",
    "social_num_layers": 1,
    "social_num_heads": 0,
    "social_dropout": 0.1,
    "vision_pred_hidden_dim": 256,
    "vision_pred_dropout": 0.0,
    "finetune_pred_weight": 0.0,
    "vision_target_dim": None,
    "vision_target_use_adapter": False,
    "vision_target_adapter_hidden": None,
    "vision_target_adapter_dropout": 0.0,
    "vision_cf_weight": 0.0,
    "vision_cf_warmup_epochs": 0,
    "vision_cf_slow_alpha": 0.5,
    "vision_cf_prob": 1.0,
    "vision_cf_use_slow": True,
    "vision_cf_use_reverse": True,
    "vision_cf_horizon": 4,
    # Frame selection and fusion
    "warmup_epochs": 10,
    "budget_ratio": 0.25,
    "global_ratio": 0.3,
    "lambda_col": 0.1,
    "collision_temperature": 0.1,
    "vis_agg_layers": 2,
    "vis_agg_heads": 4,
    "lambda_sparse": 1e-3,
    "sparse_warmup_epochs": 0,
    "entropy_reg_lambda": 0.0,
    "save_dir": "./checkpoints",
    "log_interval": 10,
    "distributed": True,
    "master_port": 29500,
    "device": "cuda",
    "pretrain_ckpt": None,
    "lr_scheduler": "CosineAnnealingLR",  # StepLR | MultiStepLR | ExponentialLR | LinearLR | CosineAnnealingLR | none
    "lr_scheduler_params": {},
    "use_cache": True,
    "loss_type": "ce",  # ce | focal
    "focal_gamma": 2.0,
    "focal_alpha": None,
    "label_smoothing": 0.0,
    "oversample_grooming_factor": 1,

    # Model config
    "motion_dim": 256,
    "vision_dim": 2048,
    "num_layers": 4,
    "num_heads": 8,
    # Head A handles fused labels: Rearing (supported/unsupported) vs Grooming
    "head_a_classes": 2,
    "head_b_classes": 4,
    "head_c_classes": 3,
    "finetune_active_heads": ["A", "B", "C"],
    "vq_hidden_dim": 256,
    "vq_latent_dim": 256,
    "vq_codebook_size": 512,
    "vq_codebook_size2": 512,
    "vq_num_stride_blocks": 3,
    "vq_attn_layers": 1,
    "vq_attn_heads": 4,
    "vq_attn_layers_dec": None,
    "vq_beta": 0.25,
    "vq_dead_code_threshold": 1.0,  # revive entries with smoothed cluster_size below this
    "vq_revival_warmup_steps": 100,  # skip revival for the first N ema_update calls
    "vq_l2_normalize": True,  # pin codebook to unit sphere; set False to preserve encoder magnitude
    "reliability_lambda": 1.0,
    "quant_lambda": 1.0,
    "motion_lambda": 1.0,
    "entropy_lambda": 1.0,
    "dynamic_cache_split": None,
    # Neighbor context branch (finetune)
    "use_neighbor_branch": False,
    "neighbor_input_dim": 0,
    "neighbor_fuse_mode": "add",
    "neighbor_encoder_layers": 0,
    "neighbor_encoder_heads": 4,
    "neighbor_dropout": 0.0,
    # Checkpointing / logging
    "pretrain_save_dir": "./checkpoints_pretrain",
    "finetune_save_dir": "./checkpoints_finetune",
    "pretrain_save_interval": 5,
    "finetune_save_interval": 1,
    "run_name": "",
    "pretrain_resume_from": None,
    "finetune_resume_from": None,
}


def load_config(path: str | None) -> Dict[str, Any]:
    """Load YAML/JSON config into a dict. Returns {} if file is absent."""
    if not path or not os.path.exists(path) or os.path.isdir(path):
        return {}
    try:
        import yaml  # type: ignore
    except ImportError as e:
        if path.endswith(".json"):
            with open(path, "r") as f:
                return json.load(f)
        raise RuntimeError("PyYAML is required to read config.yaml. Install with `pip install pyyaml`.") from e
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a YAML mapping/object at the top level.")
    return cfg


def load_config_with_defaults(path: str | None = None) -> Dict[str, Any]:
    """Merge DEFAULT_CONFIG with file config (file overrides defaults)."""
    base = copy.deepcopy(DEFAULT_CONFIG)
    user_cfg = load_config(path or "./config.yaml")
    base.update(user_cfg)
    # Backward compatibility
    if "seq_len_cache" not in base:
        base["seq_len_cache"] = base.get("seq_len", DEFAULT_CONFIG["seq_len"])
    # If legacy seq_len_train provided, use it to override seq_len
    if "seq_len_train" in base:
        base["seq_len"] = base["seq_len_train"]
    if base.get("use_neighbor_branch") and int(base.get("neighbor_input_dim") or 0) <= 0:
        base["neighbor_input_dim"] = int(base.get("max_neighbors", 0) or 0) * 3
    return base
