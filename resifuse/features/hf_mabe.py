import argparse
import copy
import json
import os
import multiprocessing as mp
from contextlib import nullcontext
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from resifuse.config import DEFAULT_CONFIG, load_config
from resifuse.data.datasets import MaBeMouseDataset
from resifuse.models.modules import VisionMagnifier


ENCODER_CHOICES = ("qwen", "clip", "dinov3", "videomae")
DEFAULT_MODEL_IDS = {
    "clip": "openai/clip-vit-large-patch14",
    "dinov3": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "videomae": "OpenGVLab/VideoMAEv2-Large",
}
# VideoPrism is handled by the standalone resifuse/features/videoprism.py
# script (JAX/Flax runtime) — it is NOT a transformers-loadable encoder.


def _resolve_frame_resize(cfg: dict) -> int | None:
    """Return a square integer size derived from cfg['frame_resize'] (or None).

    Supports int, [H, W] list, or None. For non-square lists we take the
    smaller dim so the feature extractor never sees a crop larger than what
    the dataset actually produced.
    """
    fr = cfg.get("frame_resize")
    if fr is None:
        return None
    if isinstance(fr, (list, tuple)):
        if len(fr) == 0:
            return None
        vals = [int(v) for v in fr if v]
        return min(vals) if vals else None
    return int(fr)


def processor_image_size(processor, default: int = 224) -> int:
    """Pretraining resolution of an HF image processor.

    ``processor.size`` is a dict in transformers 4 and a SizeDict object in 5.
    """
    size = getattr(processor, "size", None)
    if size is None:
        return default
    if isinstance(size, (int, float)):
        return int(size)
    for key in ("shortest_edge", "height", "width"):
        val = size.get(key) if isinstance(size, dict) else getattr(size, key, None)
        if val:
            return int(val)
    return default


class HFImageEncoderForCache(nn.Module):
    """Minimal image-encoder wrapper used by the ablation cache pipeline.

    Exposes the subset of VisionMagnifier that resifuse/features/hf_mabe.py
    depends on (``vision_dim`` attribute and ``_encode_real_frames``), so the
    existing batching/saving code in ``_encode_frames`` and ``_process_video``
    can be reused unchanged.

    Features are mean-pooled over spatial patch tokens to mirror the pooling
    semantics used for the Qwen cache (resifuse/models/modules.py: ``patch_tokens.mean(dim=2)``).
    CLS / register tokens are skipped by deriving the prefix length from the
    actual ViT grid (``hidden.shape[1] - n_patches``), so the wrapper works for
    both CLIP (1 CLS) and DINOv3 (1 CLS + 4 register) without special-casing.

    ``image_size`` can be overridden to keep the encoder on the same pixel
    budget as the Qwen cache (typically 128 via cfg.frame_resize). When that
    size does not match the backbone's pretraining resolution (224 for both
    CLIP-L/14 and DINOv3-L/16), we pass ``interpolate_pos_encoding=True`` so
    the HF implementation bilinearly resamples the positional embeddings.
    """

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
        processor_default = processor_image_size(processor)
        self._processor_default_size = int(processor_default)
        self.image_size = int(image_size) if image_size is not None else self._processor_default_size

        if encoder_type == "clip":
            self.backbone = CLIPVisionModel.from_pretrained(self.model_id)
            # CLIP uses learned absolute pos embeddings tied to its pretraining
            # resolution, so running at a different image_size requires the
            # interpolate_pos_encoding kwarg on forward().
            self._needs_interp_pos_flag = True
        else:
            self.backbone = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
            # DINOv3 uses 2D-RoPE computed from the actual input shape every
            # forward pass (see transformers/models/dinov3_vit/modeling_*.py::
            # DINOv3ViTRopePositionEmbedding.forward), so arbitrary input sizes
            # are natively supported and there is no interpolate_pos_encoding
            # kwarg to pass.
            self._needs_interp_pos_flag = False
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        # Detect whether the backbone's forward takes interpolate_pos_encoding.
        # Only CLIP-style learned pos embeddings need this; 2D-RoPE models
        # (DINOv3) handle resolution changes internally.
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
        hidden = outputs.last_hidden_state  # [B*T, 1(+R)+P, D]
        n_patches = (self.image_size // self._patch_size) ** 2
        prefix = hidden.shape[1] - n_patches
        if prefix < 0:
            raise RuntimeError(
                f"Unexpected token count: hidden={hidden.shape[1]} n_patches={n_patches} "
                f"(image_size={self.image_size}, patch_size={self._patch_size})"
            )
        patch_tokens = hidden[:, prefix:, :]
        feats = patch_tokens.mean(dim=1)  # [B*T, D]
        return feats

    def _encode_real_frames_with_patch_tokens(self, video_frames: torch.Tensor):
        raise NotImplementedError(
            "Patch-token caching is only supported for the Qwen encoder; "
            "run without --save-vision-patch-tokens for CLIP / DINOv3."
        )


class VideoMAEEncoderForCache(nn.Module):
    """VideoMAE wrapper with per-frame output via non-overlapping clip chunking.

    VideoMAE processes video clips (default 16 frames) and produces patch tokens
    over spacetime. We:
      1. Split input [B*A, T, C, H, W] into clips of ``clip_length`` frames
         (pad last clip if needed).
      2. For each clip, run the backbone → mean-pool over all patch tokens →
         1 feature vector per clip.
      3. Broadcast back: each frame in a clip gets its clip's feature.

    Output is [B*A, T, D] matching the HFImageEncoderForCache interface.
    """

    def __init__(
        self,
        model_id: str | None = None,
        image_size: int | None = None,
        clip_length: int = 16,
    ):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        self.model_id = model_id or DEFAULT_MODEL_IDS["videomae"]
        processor = AutoImageProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        mean = list(processor.image_mean)
        std = list(processor.image_std)
        processor_default = processor_image_size(processor)
        self._processor_default_size = int(processor_default)
        self.image_size = int(image_size) if image_size is not None else self._processor_default_size

        # OpenGVLab VideoMAEv2 custom modeling calls `x.item()` on scalar
        # tensors inside __init__ (for drop_path_rate scheduling), which
        # breaks under transformers' default meta-device init. Force a
        # CPU device context to override the meta device that transformers
        # pushes internally, so `.item()` works.
        with torch.device("cpu"):
            self.backbone = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
            )
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        # OpenGVLab VideoMAEv2 stores vision-tower hyperparams inside a
        # nested `model_config` dict (key `embed_dim`, not `hidden_size`).
        # Other HF-native VideoMAE variants expose them directly on config.
        cfg = getattr(self.backbone, "config", None)
        model_cfg = getattr(cfg, "model_config", None) or {}
        if not isinstance(model_cfg, dict):
            model_cfg = {}

        def _cfg_get(key: str, default=None):
            if key in model_cfg:
                return model_cfg[key]
            return getattr(cfg, key, default)

        hidden = _cfg_get("embed_dim") or _cfg_get("hidden_size")
        if hidden is None:
            raise RuntimeError(f"Could not resolve hidden_size from config for {self.model_id}")
        self.vision_dim = int(hidden)

        self._patch_size = int(_cfg_get("patch_size", 16))
        self._tubelet_size = int(_cfg_get("tubelet_size", 2))
        num_frames_cfg = int(_cfg_get("num_frames", 16))
        self.clip_length = int(clip_length or num_frames_cfg)

        # Honor non-native image_size (e.g. cfg.frame_resize=128) by
        # regenerating the sinusoid pos_embed at the target resolution and
        # bypassing PatchEmbed's hard-coded 224 assertion. This works
        # because:
        #   (a) VideoMAEv2-Large uses `use_learnable_pos_emb=False`, so
        #       pos_embed is a *generated* buffer (not a checkpoint weight).
        #   (b) PatchEmbed is a Conv3d whose weights are resolution-
        #       agnostic (the same kernel convolves a smaller feature map).
        #   (c) Transformer + MLP weights are set-equivariant over tokens,
        #       so fewer tokens just means smaller attention matrices.
        # Features WILL differ slightly from native-224 inference because
        # token count and positional grid change, but the model still
        # produces reasonable features at non-native resolutions.
        if self.image_size != self._processor_default_size:
            self._retarget_pos_embed(num_frames_cfg)
        else:
            self._restore_native_pos_embed()

        self.register_buffer("_pixel_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_pixel_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        print(
            f"[VideoMAEEncoderForCache] loaded model_id={self.model_id} "
            f"vision_dim={self.vision_dim} image_size={self.image_size} "
            f"patch_size={self._patch_size} tubelet={self._tubelet_size} "
            f"clip_length={self.clip_length} (native={self._processor_default_size})",
            flush=True,
        )

    def _restore_native_pos_embed(self) -> None:
        """Recompute the fixed sinusoid pos_embed at the native resolution.

        pos_embed is a non-persistent buffer, so from_pretrained does not load
        it from the checkpoint, and with transformers 5 it can stay
        uninitialized (NaN / garbage features at 224). Rebuild it with the
        model's own get_sinusoid_encoding_table.
        """
        import sys

        vt = self.backbone.model  # VisionTransformer
        make_table = getattr(sys.modules[type(vt).__module__], "get_sinusoid_encoding_table", None)
        if make_table is None or "pos_embed" not in vt._buffers:
            return
        table = make_table(vt.patch_embed.num_patches, self.vision_dim).to(device="cpu", dtype=torch.float)
        del vt._buffers["pos_embed"]
        vt.register_buffer("pos_embed", table, persistent=False)

    def _retarget_pos_embed(self, num_frames_cfg: int) -> None:
        """Regenerate sinusoid pos_embed + relax img_size assert for non-native size."""
        import numpy as np

        vt = self.backbone.model  # VisionTransformer
        new_spatial = (self.image_size // self._patch_size) ** 2
        new_temporal = max(1, num_frames_cfg // self._tubelet_size)
        new_num_patches = new_spatial * new_temporal
        d_hid = self.vision_dim

        # Inline sinusoid encoding (same formula as the modeling module's
        # get_sinusoid_encoding_table, duplicated here to avoid importing
        # the dynamically-loaded module).
        def _pos_vec(pos: int):
            return [pos / np.power(10000, 2 * (j // 2) / d_hid) for j in range(d_hid)]

        table = np.array([_pos_vec(p) for p in range(new_num_patches)], dtype=np.float32)
        table[:, 0::2] = np.sin(table[:, 0::2])
        table[:, 1::2] = np.cos(table[:, 1::2])
        new_pos = torch.tensor(table, dtype=torch.float, device="cpu").unsqueeze(0)

        # Relax the hard-coded 224 assertion inside PatchEmbed.forward.
        vt.patch_embed.img_size = (self.image_size, self.image_size)
        vt.patch_embed.num_patches = new_num_patches

        # Overwrite the (non-persistent) pos_embed buffer with the new size.
        # Use register_buffer so the entry in _buffers is replaced cleanly.
        if "pos_embed" in vt._buffers:
            del vt._buffers["pos_embed"]
        vt.register_buffer("pos_embed", new_pos, persistent=False)
        print(
            f"[VideoMAEEncoderForCache] retargeted pos_embed: "
            f"image_size={self.image_size} (native={self._processor_default_size}), "
            f"num_patches {new_num_patches} (spatial={new_spatial}, temporal={new_temporal}).",
            flush=True,
        )

    def _forward_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """Run VisionTransformer manually and return per-tubelet features.

        Input:  x [B, C, T=clip_length, H, W]
        Output: temporal_feats [B, T_prime, D]
                where T_prime = clip_length // tubelet_size
                (i.e. one feature vector per `tubelet_size` input frames —
                the *finest* temporal granularity VideoMAE can give without
                sliding-window inference, because the Conv3d patch embedding
                already fuses tubelet_size frames into a single token).

        Compared to the default `backbone(pixel_values=x)`, which calls
        `forward_features` and pools across ALL spatiotemporal tokens
        (producing one clip-level feature), this method stops BEFORE the
        temporal mean-pool: it only mean-pools over spatial tokens, keeping
        the T_prime temporal tokens distinct. fc_norm is still applied per
        temporal position for distribution consistency with the pretrain.
        """
        vt = self.backbone.model  # VisionTransformer
        B = x.shape[0]
        # Manual replay of VisionTransformer.forward_features up to (but not
        # including) the `x.mean(1)` that collapses T'*H'*W' tokens.
        x = vt.patch_embed(x)  # [B, N=T'*H'*W', D]
        if vt.pos_embed is not None:
            pe = vt.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
            x = x + pe
        x = vt.pos_drop(x)
        for blk in vt.blocks:
            x = blk(x)  # [B, N, D]

        # Reshape tokens back to spacetime grid.
        HW = self.image_size // self._patch_size
        T_prime = self.clip_length // self._tubelet_size
        expected = T_prime * HW * HW
        if x.shape[1] != expected:
            raise RuntimeError(
                f"VideoMAE token count {x.shape[1]} does not match expected "
                f"T'={T_prime} x H'={HW} x W'={HW} = {expected}. "
                f"Check that clip_length ({self.clip_length}) and image_size "
                f"({self.image_size}) are consistent with the loaded model."
            )
        x = x.view(B, T_prime, HW, HW, self.vision_dim)
        # Spatial-only mean-pool → [B, T', D]
        temporal_feats = x.mean(dim=(2, 3))
        # Apply fc_norm per-temporal-position (LayerNorm normalizes last dim,
        # so this is well-defined and mirrors the pretrain's per-sample norm).
        if vt.fc_norm is not None:
            temporal_feats = vt.fc_norm(temporal_feats)
        return temporal_feats

    def _encode_real_frames(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_frames: [B*A, T, C, H, W]
        Returns:
            feats_flat: [B*A*T, D]
                Per-frame features where every `tubelet_size` adjacent frames
                SHARE one feature vector (i.e. temporal granularity = 2 frames
                with default tubelet_size=2). Across tubelet boundaries the
                feature vectors DO differ — unlike the previous clip-level
                pooling where all 16 frames in a clip shared one feature.
        """
        BA, T, C, H, W = video_frames.shape
        mean = self._pixel_mean.to(dtype=video_frames.dtype, device=video_frames.device)
        std = self._pixel_std.to(dtype=video_frames.dtype, device=video_frames.device)
        # Normalize: (pixels / 255 - mean) / std; [1,1,C,1,1] for broadcast
        x = (video_frames / 255.0 - mean.view(1, 1, C, 1, 1)) / std.view(1, 1, C, 1, 1)

        # Resize if needed
        if H != self.image_size or W != self.image_size:
            x_flat = x.reshape(BA * T, C, H, W).float()
            x_flat = F.interpolate(
                x_flat,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).to(dtype=video_frames.dtype)
            x = x_flat.view(BA, T, C, self.image_size, self.image_size)
            H = W = self.image_size

        # Pad T to multiple of clip_length
        clip_len = self.clip_length
        pad_t = (clip_len - T % clip_len) % clip_len
        if pad_t > 0:
            # Repeat last frame for padding
            last = x[:, -1:].expand(BA, pad_t, C, H, W)
            x = torch.cat([x, last], dim=1)
        T_padded = x.shape[1]
        n_clips = T_padded // clip_len

        # Reshape into clips and permute to the layout VideoMAEv2 PatchEmbed
        # expects: [B, C, T, H, W]. The asserting line is
        # `B, C, T, H, W = x.shape` in VideoMAEv2 PatchEmbed.forward.
        x = x.view(BA, n_clips, clip_len, C, H, W)
        x = x.reshape(BA * n_clips, clip_len, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B*n_clips, C, T, H, W]

        # Forward through VisionTransformer WITHOUT collapsing the temporal
        # dimension → per-tubelet features.
        temporal_feats = self._forward_temporal(x)  # [B*n_clips, T', D]
        T_prime = self.clip_length // self._tubelet_size
        # [BA, n_clips, T', D]
        temporal_feats = temporal_feats.view(BA, n_clips, T_prime, self.vision_dim)

        # Broadcast each temporal feature to its `tubelet_size` source frames:
        # [BA, n_clips, T', 1, D] → expand → [BA, n_clips, T', tubelet, D]
        tub = self._tubelet_size
        frame_feats = temporal_feats.unsqueeze(3).expand(
            BA, n_clips, T_prime, tub, self.vision_dim
        )
        # Flatten temporal dims: [BA, n_clips * T' * tubelet, D] = [BA, T_padded, D]
        frame_feats = frame_feats.reshape(BA, T_padded, self.vision_dim)
        # Trim padding
        frame_feats = frame_feats[:, :T]  # [BA, T, D]

        return frame_feats.reshape(BA * T, self.vision_dim)

    def _encode_real_frames_with_patch_tokens(self, video_frames: torch.Tensor):
        raise NotImplementedError("Patch-token caching is not supported for VideoMAE.")



def _to_numpy_contiguous(t: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a contiguous numpy array for np.save.

    bfloat16 has no numpy equivalent, so it's downcast to float16. The
    .copy() ensures the returned array owns its memory and doesn't share
    storage with any larger source tensor.
    """
    if t.dtype == torch.bfloat16:
        t = t.float().half()
    return t.detach().cpu().numpy().copy()


def parse_dtype(dtype_str: str | None) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if not dtype_str:
        return torch.bfloat16
    return dtype_map.get(dtype_str.lower(), torch.bfloat16)


def load_cfg(path: str) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(load_config(path))
    if "seq_len_cache" not in cfg:
        cfg["seq_len_cache"] = cfg.get("seq_len", DEFAULT_CONFIG["seq_len"])
    if "seq_len_train" in cfg:
        cfg["seq_len"] = cfg["seq_len_train"]
    return cfg


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _render_progress(prefix: str, current: int, total: int, detail: str = "") -> None:
    if total <= 0:
        return
    bar_len = 24
    filled = int(bar_len * current / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    line = f"{prefix} [{bar}] {current}/{total}"
    if detail:
        line = f"{line} {detail}"
    end = "\n" if current >= total else "\r"
    print(line, end=end, flush=True)


def _validate_split_json(path: str) -> None:
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


def build_dataset(cfg: dict, stage: str, split: str, preload: bool):
    mabe_root = cfg.get("mabe_data_dir")
    if not mabe_root:
        raise RuntimeError("mabe_data_dir must be set in config for MABe cache export.")
    stage_dir = os.path.join(mabe_root, stage)
    split_json = os.path.join(stage_dir, f"{stage}_split.json")
    # print(f"[Cache] split_json={split_json}", flush=True)
    _validate_split_json(split_json)
    include_velocity = cfg.get("motion_include_velocity", True)
    include_confidence = cfg.get("motion_include_confidence", True)
    huddle_keep_ratio = cfg.get("finetune_huddle_keep_ratio", 1.0)
    ds = MaBeMouseDataset(
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
        motion_include_velocity=include_velocity,
        motion_include_confidence=include_confidence,
        frame_resize=cfg.get("frame_resize"),
        frame_resize_align=cfg.get("frame_resize_align", 32),
        preload=preload,
        cache_size=cfg.get("cache_size", 50),
        huddle_only_keep_ratio=huddle_keep_ratio,
    )
    return ds


def _resolve_train_ratio(cfg: dict, stage: str) -> float:
    if stage == "finetune":
        ft_train = cfg.get("finetune_train_ratio")
        return ft_train if ft_train is not None else cfg.get("train_ratio", 0.8)
    return cfg.get("train_ratio", 0.8)


def build_video_index(ds: MaBeMouseDataset) -> Dict[str, List[int]]:
    video_to_indices: Dict[str, List[int]] = {}
    for idx, sample in enumerate(ds.samples):
        video_id = sample.get("video_id", "unknown")
        video_to_indices.setdefault(video_id, []).append(idx)
    return video_to_indices


def _encode_frames(
    encoder: VisionMagnifier,
    frames_batch: List[torch.Tensor],
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    return_patch_tokens: bool = False,
) -> List[torch.Tensor] | tuple[List[torch.Tensor], List[torch.Tensor], dict]:
    if not frames_batch:
        return ([], [], {}) if return_patch_tokens else []
    frames_cat = torch.cat(frames_batch, dim=0)
    frames_dev = frames_cat.to(device=device)
    ctx = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if device.type == "cuda" and autocast_dtype is not None
        else nullcontext()
    )
    with torch.inference_mode():
        with ctx:
            if return_patch_tokens:
                feats_flat, patch_flat, patch_meta = encoder._encode_real_frames_with_patch_tokens(frames_dev)
            else:
                feats_flat = encoder._encode_real_frames(frames_dev)
                patch_flat = None
                patch_meta = None
    B, T = frames_cat.shape[0], frames_cat.shape[1]
    feats_batch = feats_flat.view(B, T, -1)
    patch_batch = None
    if patch_flat is not None:
        patch_batch = patch_flat.view(B, T, patch_flat.shape[1], patch_flat.shape[2])
    feats_out: List[torch.Tensor] = []
    patch_out: List[torch.Tensor] = []
    offset = 0
    for frames in frames_batch:
        A = frames.shape[0]
        feats_out.append(feats_batch[offset : offset + A].cpu())
        if patch_batch is not None:
            patch_out.append(patch_batch[offset : offset + A].cpu())
        offset += A
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if return_patch_tokens:
        patch_meta_out = {}
        for key, value in (patch_meta or {}).items():
            patch_meta_out[key] = value.detach().cpu() if torch.is_tensor(value) else value
        return feats_out, patch_out, patch_meta_out
    return feats_out


def _extract_sample_payload(sample: dict, use_vision: bool) -> dict:
    payload = {
        "keypoints": sample["keypoints"].cpu(),
        "confidence": sample["confidence"].cpu(),
        "mask": sample["mask"].cpu(),
        "motion_feats": sample["motion_feats"].cpu(),
        "metadata": sample.get("metadata", {}),
    }
    if "frame_labels" in sample:
        payload["frame_labels"] = sample["frame_labels"].cpu()
    if "frame_mask" in sample:
        payload["frame_mask"] = sample["frame_mask"].cpu()
    if "label" in sample:
        payload["label"] = sample["label"].cpu()
    if "label_name" in sample:
        payload["label_name"] = sample["label_name"]
    if not use_vision:
        payload["cropped_video"] = sample["frames"].cpu()
    return payload


def _process_video(
    ds: MaBeMouseDataset,
    video_id: str,
    indices: Sequence[int],
    out_dir: str,
    overwrite: bool,
    use_vision: bool,
    encoder: VisionMagnifier | None,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    batch_size: int,
    refresh_nonvision: bool = False,
    save_patch_tokens: bool = False,
) -> None:
    # New per-window format: each window is a separate .pt + sidecar .npy files.
    # Skip the video entirely if the first window file exists (cheap check).
    sentinel_path = os.path.join(out_dir, f"{video_id}_w000.pt")
    if os.path.exists(sentinel_path) and not overwrite:
        return
    if not indices:
        return
    # Pre-validate indices against cached window counts to avoid OOB access.
    sample_meta0 = ds.samples[indices[0]]
    pt_file = sample_meta0.get("pt_file")
    if not pt_file:
        raise RuntimeError(f"Missing pt_file for video {video_id}.")
    cached_data = ds._load_pt_file(pt_file)
    valid_indices = []
    skipped = 0
    for idx in indices:
        sample_meta = ds.samples[idx]
        mouse_id = sample_meta.get("mouse_id")
        window_idx = sample_meta.get("window_idx")
        if mouse_id is None or window_idx is None:
            skipped += 1
            continue
        # Ensure all mice have enough windows for this index.
        ok = True
        for mid, mdata in cached_data.items():
            if str(mid).startswith("_"):
                continue
            crops = mdata.get("cropped_video") if isinstance(mdata, dict) else None
            if crops is None or int(crops.shape[0]) <= int(window_idx):
                ok = False
                break
        if not ok:
            skipped += 1
            continue
        valid_indices.append(idx)
    if skipped:
        print(f"[Cache] {video_id}: skipped {skipped} windows with missing/short cached data.", flush=True)
    if not valid_indices:
        return
    existing_vision = None
    existing_patch_tokens = None
    existing_patch_meta = None
    if refresh_nonvision:
        # New format: load existing vision feats from per-window .npy sidecars.
        existing_vision_list = []
        existing_patch_list = []
        for w in range(len(valid_indices)):
            stem = f"{video_id}_w{w:03d}"
            v_npy = os.path.join(out_dir, f"{stem}_vision.npy")
            if not os.path.exists(v_npy):
                print(f"[Cache][Refresh] missing {v_npy}", flush=True)
                return
            existing_vision_list.append(np.load(v_npy))
            p_npy = os.path.join(out_dir, f"{stem}_patches.npy")
            if os.path.exists(p_npy):
                existing_patch_list.append(np.load(p_npy))
            else:
                existing_patch_list.append(None)
        existing_vision = existing_vision_list
        existing_patch_tokens = existing_patch_list if any(p is not None for p in existing_patch_list) else None
        # vision_patch_meta is shared across windows; load it from the first .pt
        first_pt = os.path.join(out_dir, f"{video_id}_w000.pt")
        if os.path.exists(first_pt):
            try:
                first_data = torch.load(first_pt, map_location="cpu")
                existing_patch_meta = first_data.get("vision_patch_meta")
            except Exception:
                existing_patch_meta = None

    keys = {
        "keypoints": [],
        "confidence": [],
        "mask": [],
        "motion_feats": [],
        "metadata": [],
    }
    if use_vision:
        keys["vision_feats"] = []
        if save_patch_tokens:
            keys["vision_patch_tokens"] = []
    else:
        keys["cropped_video"] = []
    vision_patch_meta = None

    frames_batch: List[torch.Tensor] = []
    payload_batch: List[dict] = []

    def _flush_batch():
        nonlocal vision_patch_meta
        if not payload_batch:
            return
        if use_vision:
            if save_patch_tokens:
                feats_list, patch_list, patch_meta = _encode_frames(
                    encoder,
                    frames_batch,
                    device,
                    autocast_dtype,
                    return_patch_tokens=True,
                )
                if vision_patch_meta is None:
                    vision_patch_meta = patch_meta or {}
            else:
                feats_list = _encode_frames(encoder, frames_batch, device, autocast_dtype)
                patch_list = []
            for idx, (p, feats) in enumerate(zip(payload_batch, feats_list)):
                keys["vision_feats"].append(feats)
                if save_patch_tokens:
                    keys["vision_patch_tokens"].append(patch_list[idx])
                keys["keypoints"].append(p["keypoints"])
                keys["confidence"].append(p["confidence"])
                keys["mask"].append(p["mask"])
                keys["motion_feats"].append(p["motion_feats"])
                keys["metadata"].append(p["metadata"])
                if "frame_labels" in p:
                    keys.setdefault("frame_labels", []).append(p["frame_labels"])
                if "frame_mask" in p:
                    keys.setdefault("frame_mask", []).append(p["frame_mask"])
                if "label" in p:
                    keys.setdefault("label", []).append(p["label"])
                if "label_name" in p:
                    keys.setdefault("label_name", []).append(p["label_name"])
        else:
            for p in payload_batch:
                keys["cropped_video"].append(p["cropped_video"])
                keys["keypoints"].append(p["keypoints"])
                keys["confidence"].append(p["confidence"])
                keys["mask"].append(p["mask"])
                keys["motion_feats"].append(p["motion_feats"])
                keys["metadata"].append(p["metadata"])
                if "frame_labels" in p:
                    keys.setdefault("frame_labels", []).append(p["frame_labels"])
                if "frame_mask" in p:
                    keys.setdefault("frame_mask", []).append(p["frame_mask"])
                if "label" in p:
                    keys.setdefault("label", []).append(p["label"])
                if "label_name" in p:
                    keys.setdefault("label_name", []).append(p["label_name"])
        frames_batch.clear()
        payload_batch.clear()

    for idx in valid_indices:
        sample = ds[idx]
        payload = _extract_sample_payload(sample, use_vision=use_vision)
        if use_vision:
            frames_batch.append(sample["frames"])
            payload_batch.append(payload)
            if len(payload_batch) >= batch_size:
                _flush_batch()
        else:
            payload_batch.append(payload)
    if payload_batch:
        _flush_batch()

    ensure_dir(out_dir)

    # ===== Per-window save =====
    # Each window becomes its own small .pt file plus optional .npy sidecars
    # for the large vision tensors. This avoids the previous "load 500 MB
    # video file just to access one 15 MB window" pattern, and lets the
    # DataLoader open vision tensors via np.load(mmap_mode='r') for
    # near-instant, page-cache-shared IO.
    N = len(valid_indices)
    ds_stride = getattr(ds, "ds_stride", 1)
    window_size = getattr(ds, "window_size", None)
    num_mice = getattr(ds, "num_mice", None)
    shared_vision_patch_meta = (
        existing_patch_meta if existing_vision is not None else (vision_patch_meta or {})
    )

    for w in range(N):
        stem = f"{video_id}_w{w:03d}"
        out_pt = os.path.join(out_dir, f"{stem}.pt")

        window_data = {
            "video_id": video_id,
            "window_idx": w,
            "ds_stride": ds_stride,
            "window_size": window_size,
            "num_windows": N,
            "num_mice": num_mice,
            # .clone() ensures each saved tensor owns its storage and the
            # .pt file only contains this window's bytes (~1 MB), not the
            # whole video stack.
            "keypoints": keys["keypoints"][w].clone() if keys["keypoints"] else torch.empty(0),
            "confidence": keys["confidence"][w].clone() if keys["confidence"] else torch.empty(0),
            "mask": keys["mask"][w].clone() if keys["mask"] else torch.empty(0),
            "motion_feats": keys["motion_feats"][w].clone() if keys["motion_feats"] else torch.empty(0),
            "metadata": keys["metadata"][w] if w < len(keys["metadata"]) else {},
        }

        if "frame_labels" in keys:
            fl = keys["frame_labels"][w].clone()
            window_data["frame_labels"] = fl
            window_data["frame_labels_majority"] = fl
        if "frame_mask" in keys:
            fm = keys["frame_mask"][w].clone()
            window_data["frame_mask"] = fm
            window_data["frame_mask_majority"] = fm
        if "label" in keys:
            lbl = keys["label"][w]
            window_data["label"] = lbl.clone() if isinstance(lbl, torch.Tensor) else lbl
        if "label_name" in keys:
            window_data["label_name"] = keys["label_name"][w]

        # Vision: save large tensors as .npy sidecars (mmap-friendly).
        # vision_patch_meta is small and shared across windows — keep in .pt.
        if existing_vision is not None:
            vf = existing_vision[w]
            if isinstance(vf, torch.Tensor):
                vf_np = _to_numpy_contiguous(vf)
            else:
                vf_np = np.ascontiguousarray(vf)
            np.save(os.path.join(out_dir, f"{stem}_vision.npy"), vf_np)
            if existing_patch_tokens is not None and existing_patch_tokens[w] is not None:
                pt_t = existing_patch_tokens[w]
                if isinstance(pt_t, torch.Tensor):
                    pt_np = _to_numpy_contiguous(pt_t)
                else:
                    pt_np = np.ascontiguousarray(pt_t)
                np.save(os.path.join(out_dir, f"{stem}_patches.npy"), pt_np)
            if shared_vision_patch_meta:
                window_data["vision_patch_meta"] = shared_vision_patch_meta
        elif use_vision:
            vf = keys["vision_feats"][w]
            np.save(os.path.join(out_dir, f"{stem}_vision.npy"), _to_numpy_contiguous(vf))
            if save_patch_tokens:
                pt_t = keys["vision_patch_tokens"][w]
                np.save(os.path.join(out_dir, f"{stem}_patches.npy"), _to_numpy_contiguous(pt_t))
                if shared_vision_patch_meta:
                    window_data["vision_patch_meta"] = shared_vision_patch_meta
        else:
            window_data["cropped_video"] = keys["cropped_video"][w].clone()

        torch.save(window_data, out_pt)


def _init_worker(cfg: dict, stage: str, split: str, preload: bool, refresh_nonvision: bool):
    global G_DS, G_VIDEO_TO_INDICES, G_REFRESH
    G_DS = build_dataset(cfg, stage, split, preload=preload)
    G_VIDEO_TO_INDICES = build_video_index(G_DS)
    G_REFRESH = bool(refresh_nonvision)


def _process_video_worker(args):
    video_id, out_dir, overwrite = args
    refresh_nonvision = globals().get("G_REFRESH", False)
    indices = G_VIDEO_TO_INDICES.get(video_id, [])
    _process_video(
        G_DS,
        video_id,
        indices,
        out_dir,
        overwrite=overwrite,
        use_vision=False,
        encoder=None,
        device=torch.device("cpu"),
        autocast_dtype=None,
        batch_size=1,
        refresh_nonvision=refresh_nonvision,
    )
    return video_id


def _process_videos_on_device(
    cfg: dict,
    stage: str,
    split: str,
    video_ids: Sequence[str],
    out_dir: str,
    overwrite: bool,
    device: torch.device,
    batch_size: int,
    refresh_nonvision: bool,
    save_patch_tokens: bool,
    encoder_type: str = "qwen",
    encoder_model_id: str | None = None,
    encoder_image_size: int | None = None,
):
    ds = build_dataset(cfg, stage, split, preload=False)
    video_to_indices = build_video_index(ds)
    total = len(video_ids)
    prefix = f"[Cache][{stage}/{split}][{device}][{encoder_type}]"
    if refresh_nonvision:
        for idx, video_id in enumerate(video_ids, 1):
            indices = video_to_indices.get(video_id, [])
            _process_video(
                ds,
                video_id,
                indices,
                out_dir,
                overwrite=overwrite,
                use_vision=False,
                encoder=None,
                device=device,
                autocast_dtype=None,
                batch_size=batch_size,
                refresh_nonvision=True,
                save_patch_tokens=False,
            )
            _render_progress(prefix, idx, total, video_id)
        return

    dtype = parse_dtype(cfg.get("torch_dtype") or cfg.get("dtype"))
    max_len = getattr(ds, "seq_len", None)
    assert max_len is not None
    print(f"max_len is: {max_len}",flush=True)
    if encoder_type == "qwen":
        encoder = VisionMagnifier(
            cfg.get("qwen_vision_dim", 2048),  # Qwen vision tower output; vision_dim is the training feature size
            cfg.get("motion_dim", 256),
            use_real_model=cfg.get("use_real_qwen", True),
            qwen_model_id=cfg.get("qwen_model_id", "Qwen/Qwen3-VL-2B-Instruct"),
            agg_layers=cfg.get("vision_agg_layers", 2),
            agg_heads=cfg.get("vision_agg_heads", 4),
            max_len=max_len,
            enable_lora=cfg.get("vision_lora_enable", True),
            lora_r=cfg.get("vision_lora_r", 8),
            lora_alpha=cfg.get("vision_lora_alpha", 16),
            lora_layers=cfg.get("vision_lora_layers", 4),
            gate_thresh=cfg.get("gate_thresh", 0.5),
            enable_flash_attn=cfg.get("qwen_flash_attn", False),
            use_deepstack=cfg.get("vision_use_deepstack", True),
            use_frame_diff=cfg.get("vision_frame_diff", False),
            frame_diff_abs=cfg.get("vision_frame_diff_abs", False),
            frame_diff_fuse=cfg.get("vision_frame_diff_fuse", "residual"),
            frame_diff_dt_mode=cfg.get("vision_frame_diff_dt_mode", "divide"),
            frame_diff_dt_min=cfg.get("vision_frame_diff_dt_min", 1.0),
            use_temporal_conv=cfg.get("vision_temporal_conv", True),
            temporal_conv_kernel=cfg.get("vision_temporal_conv_kernel", 3),
            enable_grad_checkpoint=cfg.get("qwen_grad_checkpoint", False),
            allow_random_fallback=cfg.get("allow_random_vision_fallback", False),
        ).to(device=device)
    else:
        if save_patch_tokens:
            raise ValueError(
                f"--save-vision-patch-tokens is not supported for encoder={encoder_type}; "
                "patch-token caching is Qwen-only."
            )
        resolved_image_size = encoder_image_size
        if resolved_image_size is None:
            resolved_image_size = _resolve_frame_resize(cfg)
        if encoder_type == "videomae":
            encoder = VideoMAEEncoderForCache(
                model_id=encoder_model_id,
                image_size=resolved_image_size,
                clip_length=int(cfg.get("videomae_clip_length", 16)),
            ).to(device=device)
        else:
            encoder = HFImageEncoderForCache(
                encoder_type=encoder_type,
                model_id=encoder_model_id,
                image_size=resolved_image_size,
            ).to(device=device)
    encoder.eval()
    autocast_dtype = dtype if device.type == "cuda" else None
    first_video_checked = False
    for idx, video_id in enumerate(video_ids, 1):
        indices = video_to_indices.get(video_id, [])
        _process_video(
            ds,
            video_id,
            indices,
            out_dir,
            overwrite=overwrite,
            use_vision=True,
            encoder=encoder,
            device=device,
            autocast_dtype=autocast_dtype,
            batch_size=batch_size,
            refresh_nonvision=False,
            save_patch_tokens=save_patch_tokens,
        )
        _render_progress(prefix, idx, total, video_id)

        # Fail-fast sanity check for videomae after the first video on each
        # shard. Loads the freshly-written vision .npy files and raises if
        # any contain NaN/Inf. Non-videomae encoders are assumed-healthy
        # (well tested), and we only run this once to avoid NFS overhead.
        if encoder_type == "videomae" and not first_video_checked:
            first_video_checked = True
            vnpys = sorted(
                p for p in os.listdir(out_dir)
                if p.startswith(f"{video_id}_") and p.endswith("_vision.npy")
            )
            if not vnpys:
                print(f"{prefix} video-encoder sanity: no vision npys found for {video_id}", flush=True)
            else:
                bad = []
                stats = None
                for fn in vnpys:
                    arr = np.load(os.path.join(out_dir, fn), mmap_mode="r")
                    if np.isnan(arr).any() or np.isinf(arr).any():
                        bad.append(fn)
                    elif stats is None:
                        stats = (float(arr.mean()), float(arr.std()), float(np.abs(arr).max()))
                if bad:
                    raise RuntimeError(
                        f"{prefix} video-encoder sanity FAILED on first video {video_id}: "
                        f"{len(bad)}/{len(vnpys)} vision npys contain NaN/Inf. "
                        f"Example: {bad[0]}. Aborting before wasting time on the full run."
                    )
                print(
                    f"{prefix} video-encoder sanity OK on {video_id} "
                    f"({len(vnpys)} windows, mean={stats[0]:.4f} std={stats[1]:.4f} absmax={stats[2]:.4f})",
                    flush=True,
                )


def _split_round_robin(items: Sequence[str], groups: int) -> List[List[str]]:
    buckets = [[] for _ in range(groups)]
    for idx, item in enumerate(items):
        buckets[idx % groups].append(item)
    return buckets


def main():
    parser = argparse.ArgumentParser(description="Precompute MABe window caches per video.")
    parser.add_argument("--config", type=str, default="./config.yaml", help="Path to config.yaml.")
    parser.add_argument("--stage", type=str, choices=["pretrain", "finetune"], default=None, help="Stage to process.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to process.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files.")
    parser.add_argument("--use-vision-encoder", action="store_true", help="Precompute vision features.")
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
            "Defaults to cfg['frame_resize'] (same crop size as the Qwen "
            "cache pipeline) so the ablation compares encoders at matched "
            "resolution. Pass e.g. 224 to force the backbone's native "
            "pretraining resolution instead."
        ),
    )
    parser.add_argument(
        "--refresh-nonvision",
        action="store_true",
        help="Reuse existing vision_feats while recomputing non-vision fields.",
    )
    parser.add_argument(
        "--save-vision-patch-tokens",
        action="store_true",
        help="Also cache per-frame Qwen patch tokens as vision_patch_tokens [N,A,T,P,D].",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Windows per vision batch.")
    parser.add_argument("--processes", type=int, default=4, help="CPU worker processes (no vision).")
    parser.add_argument("--devices", type=str, default=None, help="Comma-separated devices for vision encoding.")
    parser.add_argument("--mabe-data-dir", type=str, default=None, help="MABe root containing pretrain/finetune.")
    parser.add_argument("--mabe-data-cache-dir", type=str, default=None, help="Output cache root.")
    parser.add_argument("--preload", action="store_true", help="Preload .pt files in each worker.")
    parser.add_argument("--frame-resize", type=int, default=None,
                        help="Override frame_resize of the config (frames are resized to this before the encoder).")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.frame_resize:
        cfg["frame_resize"] = args.frame_resize
    if args.mabe_data_dir:
        cfg["mabe_data_dir"] = args.mabe_data_dir
    stage = args.stage or cfg.get("stage", "pretrain")
    splits: Sequence[str] = args.splits or cfg.get("splits", ["train", "val", "test"])
    cache_root = args.mabe_data_cache_dir or cfg.get("mabe_data_cache_dir")
    if not cache_root:
        raise RuntimeError("mabe_data_cache_dir must be provided via CLI or config.")

    for split in splits:
        ds = build_dataset(cfg, stage, split, preload=args.preload)
        video_to_indices = build_video_index(ds)
        video_ids = sorted(video_to_indices.keys())
        out_dir = os.path.join(cache_root, stage, split)
        ensure_dir(out_dir)
        if video_ids:
            print(f"[Cache] {stage}/{split}: {len(video_ids)} videos", flush=True)

        refresh_nonvision = bool(args.refresh_nonvision)
        save_patch_tokens = bool(args.save_vision_patch_tokens or cfg.get("save_vision_patch_tokens", False))
        if save_patch_tokens and args.encoder != "qwen":
            raise ValueError(
                f"--save-vision-patch-tokens is only supported for --encoder qwen, "
                f"got --encoder {args.encoder}."
            )
        if args.use_vision_encoder and not refresh_nonvision:
            devices_arg = args.devices or cfg.get("device", "cuda")
            device_list = [d.strip() for d in str(devices_arg).split(",") if d.strip()]
            if not device_list:
                device_list = ["cuda"]
            if len(device_list) == 1:
                _process_videos_on_device(
                    cfg,
                    stage,
                    split,
                    video_ids,
                    out_dir,
                    overwrite=args.overwrite,
                    device=torch.device(device_list[0]),
                    batch_size=max(1, int(args.batch_size)),
                    refresh_nonvision=False,
                    save_patch_tokens=save_patch_tokens,
                    encoder_type=args.encoder,
                    encoder_model_id=args.encoder_model_id,
                    encoder_image_size=args.encoder_image_size,
                )
            else:
                ctx = mp.get_context("spawn")
                shards = _split_round_robin(video_ids, len(device_list))
                procs = []
                for dev, shard in zip(device_list, shards):
                    p = ctx.Process(
                        target=_process_videos_on_device,
                        args=(
                            cfg,
                            stage,
                            split,
                            shard,
                            out_dir,
                            args.overwrite,
                            torch.device(dev),
                            max(1, int(args.batch_size)),
                            False,
                            save_patch_tokens,
                            args.encoder,
                            args.encoder_model_id,
                            args.encoder_image_size,
                        ),
                    )
                    p.start()
                    procs.append(p)
                for p in procs:
                    p.join()
        else:
            if refresh_nonvision:
                processes = args.processes or max(1, os.cpu_count() or 1)
                ctx = mp.get_context("spawn")
                with ctx.Pool(
                    processes=processes,
                    initializer=_init_worker,
                    initargs=(cfg, stage, split, args.preload, True),
                ) as pool:
                    tasks = [(vid, out_dir, args.overwrite) for vid in video_ids]
                    total = len(tasks)
                    prefix = f"[Cache][{stage}/{split}][cpu]"
                    for idx, vid in enumerate(pool.imap_unordered(_process_video_worker, tasks), 1):
                        _render_progress(prefix, idx, total, vid)
            else:
                processes = args.processes or max(1, os.cpu_count() or 1)
                ctx = mp.get_context("spawn")
                with ctx.Pool(
                    processes=processes,
                    initializer=_init_worker,
                    initargs=(cfg, stage, split, args.preload, False),
                ) as pool:
                    tasks = [(vid, out_dir, args.overwrite) for vid in video_ids]
                    total = len(tasks)
                    prefix = f"[Cache][{stage}/{split}][cpu]"
                    for idx, vid in enumerate(pool.imap_unordered(_process_video_worker, tasks), 1):
                        _render_progress(prefix, idx, total, vid)


if __name__ == "__main__":
    main()
