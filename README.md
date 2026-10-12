# ResiFuse

Code for **Learning When Visual Context Matters for Mouse Behavior Analysis** (NeurIPS 2026).

Weihan Li<sup>1</sup>, Jingyang Ke<sup>1</sup>, Qinheng Pu<sup>2</sup>, Yule Wang<sup>1</sup>, Chengrui Li<sup>1</sup>, Anqi Wu<sup>1</sup>

<sup>1</sup>School of Computational Science and Engineering, Georgia Tech<br>
<sup>2</sup>Wallace H. Coulter Department of Biomedical Engineering, Georgia Tech and Emory University

ResiFuse pretrains a pose-based motion encoder with residual vector quantization (RVQ) and an auxiliary cross-modal loss. At fine-tuning time, the quantization residual of each frame decides which frames receive visual features: only the frames with the largest residuals are decoded and encoded by the vision encoder. The same fused representation is used for supervised behavior classification and for unsupervised segmentation with deep embedded clustering (DEC).

## Repository layout

```
configs/                 one pretraining and one fine-tuning config per dataset
splits/                  train / val / test splits used in the paper
scripts/                 pretrain.sh, finetune.sh, unsup.sh
checkpoints/             released pretrained weights (downloaded, see "Pretrained checkpoints")
resifuse/
  train.py               pretraining and supervised fine-tuning (dispatches on `dataset` in the config)
  config.py              config loading and defaults
  models/                motion encoder (tokenizer, RVQ, context and social Transformers),
                         residual-guided fusion, classifier
  data/                  datasets for Mouse Triplet, CalMS21 and Open Field
  training/              per-dataset training loops; unsup_*.py run unsupervised segmentation (DEC)
  preprocessing/         pose preprocessing and windowing, split generation
  features/              VideoPrism feature extraction (videoprism.py); Hugging Face encoders
                         (hf_*.py: VideoMAE, DINOv3, CLIP, Qwen) for the encoder comparison
```

All Python entry points are run as modules from the repository root, e.g. `python -m resifuse.train --config ...`.

## Installation

We use [uv](https://docs.astral.sh/uv/). Training uses PyTorch; VideoPrism feature extraction uses JAX, so we keep two environments.

**Training environment** (Python 3.12, CUDA 12):

```bash
uv sync                      # creates .venv with Python 3.12 (.python-version)
# flash-attn must match your CUDA / PyTorch build (torch 2.9.1). Install a prebuilt
# wheel from https://github.com/Dao-AILab/flash-attention/releases, or build it:
uv pip install flash-attn --no-build-isolation
```

Activate the environment (`source .venv/bin/activate`) before running the commands below; the scripts in `scripts/` call `python` and `torchrun` directly. `uv sync` removes packages that are not in `pyproject.toml`, so use `uv sync --inexact` later to keep flash-attn. `requirements.txt` lists the same dependencies for pip.

**VideoPrism environment:**

```bash
uv venv .venv-videoprism --python 3.12
source .venv-videoprism/bin/activate
uv pip install "jax[cuda12]" flax "git+https://github.com/google-deepmind/videoprism.git" \
  numpy pyyaml decord scikit-learn                                   # windows are read with resifuse.data
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build; only used for data loading
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

All commands below are run from the repository root.

## Data preparation

Paths in `configs/` are relative to the repository root (`data/...`). Place the data there or edit the paths. Each dataset needs pose windows, a split, and VideoPrism features; the VideoPrism extractor writes the features to the cache directory named in the config.

### Mouse Triplet (MABe22)

We use the videos and labels of the MABe22 mouse triplet task with pose re-estimated by DeepLabCut (see the paper, Appendix C.1).

- Pose: [WeihanLikk/ResiFuse-MABe22](https://huggingface.co/datasets/WeihanLikk/ResiFuse-MABe22) (CC BY-NC-SA 4.0); `hf download WeihanLikk/ResiFuse-MABe22 --repo-type dataset --local-dir data/mabe2022` puts it in `data/mabe2022/deeplabcut_filtered/`.
- Videos and labels: download `mouse_submission_videos_resized_224.zip` and `mouse_submission_labels.npy` from the MABe22 release on CaltechDATA ([doi:10.22002/rdsa8-rde65](https://doi.org/10.22002/rdsa8-rde65)) into `data/mabe2022/`.

The pose is in pixels of the videos upscaled to 448 x 448, so upscale the videos first (our videos were also converted to mp4 at 224 x 224 before upscaling, so the frames can differ slightly from ours):

```bash
cd data/mabe2022 && unzip -q mouse_submission_videos_resized_224.zip && mkdir -p mouse_submission_mp4_448
for v in mouse_submission_videos_resized_224/*.avi; do
  ffmpeg -loglevel error -i "$v" -vf scale=448:448:flags=lanczos -c:v libx264 -preset medium -crf 18 \
    -pix_fmt yuv420p "mouse_submission_mp4_448/$(basename "${v%.avi}").mp4"
done
cd -
```

```bash
# 1. pose windows and 160 x 160 crops around each mouse (defaults = paper settings:
#    pretraining 300-frame windows with stride 150, fine-tuning 200-frame windows)
for stage in pretrain finetune; do
  python -m resifuse.preprocessing.mabe --stage $stage --num-workers 4
done

# 2. splits used in the paper (fine-tuning 50/10/40; pretraining 70/20/10, excluding
#    fine-tuning val/test videos). resifuse.preprocessing.split_mabe generates new ones.
D=data/mabe2022/windows
cp splits/mabe_pretrain_split.json $D/pretrain/pretrain_split.json
cp splits/mabe_finetune_split.json $D/finetune/finetune_split.json

# 3. VideoPrism features (VideoPrism environment)
for stage in pretrain finetune; do
  python -m resifuse.features.videoprism --dataset mabe --config configs/mabe_${stage}.yaml \
    --stage $stage --splits train val test --devices 0
done
```

### CalMS21

Task 1 keypoints and videos from the CalMS21 release, as `data/calms21/task1_classic_classification/calms21_task1_{train,test}.json` and `data/calms21/task1_videos_mp4/{train,test}/*.mp4`.

```bash
# 1. pose windows (defaults = paper settings: 256 frames, stride 128). The configs read the
#    split used in the paper from splits/calms21_split.json (the 19 official test videos; 10% of
#    the official training videos become val); resifuse.preprocessing.split_calms21 makes new ones.
python -m resifuse.preprocessing.calms21

# 2. VideoPrism features (VideoPrism environment); one cache serves both stages
python -m resifuse.features.videoprism --dataset calms21 --config configs/calms21_pretrain.yaml \
  --splits train val test --devices 0
```

### Open Field

Keypoints and videos of Sturman et al. (2020) under `data/openfield/annotated_segments/`.

```bash
# 1. 256-frame windows with pose, labels and 192 x 192 frames; the 50/10/40 video split is
#    deterministic and made by the dataset class
python -m resifuse.preprocessing.openfield --config configs/openfield_pretrain.yaml
python -m resifuse.preprocessing.openfield --config configs/openfield_finetune.yaml

# 2. VideoPrism features (VideoPrism environment)
for stage in pretrain finetune; do
  python -m resifuse.features.videoprism --dataset openfield --config configs/openfield_${stage}.yaml \
    --stage $stage --splits train val test --devices 0
done
```

### Other vision encoders (Mouse Triplet)

The encoder comparison uses VideoMAE(V2), DINOv3, CLIP and Qwen3.5 features on Mouse Triplet. They are extracted with the training environment; frames are resized to `frame_resize` (128) before every encoder, as in the paper, and `qwen` is the Qwen3.5-2B vision tower:

```bash
for enc in videomae dinov3 clip qwen; do
  for stage in pretrain finetune; do
    python -m resifuse.features.hf_mabe --config configs/mabe_${stage}.yaml --stage $stage \
      --use-vision-encoder --encoder $enc --mabe-data-cache-dir data/mabe2022/features_$enc --devices cuda:0
  done
done
```

To train on one of them, set `vision_dim` to its feature size (1024; 2048 for Qwen3.5) and point the feature directory in the configs to it: `mabe_data_cache_dir` for Mouse Triplet. The same four encoders can be extracted for CalMS21 with `resifuse.features.hf_calms21` (directory key `calms21_cache_dir`) and for Open Field with `resifuse.features.hf_openfield` (`vision_cache_dir`); the paper compares encoders on Mouse Triplet only.

## Training

### Pretrained checkpoints

We release the pretrained motion encoders used in the paper. Download them into `checkpoints/`:

| Dataset | File |
|---|---|
| Mouse Triplet | `checkpoints/resifuse_pretrain_mabe.pth` |
| CalMS21 | `checkpoints/resifuse_pretrain_calms21.pth` |
| Open Field | `checkpoints/resifuse_pretrain_openfield.pth` |

Download them from the [GitHub releases](https://github.com/BRAINML-GT/ResiFuse/releases) page.

`scripts/finetune.sh` and `scripts/unsup.sh` use these files by default. Each file holds the model weights only (`{"model": state_dict}`).

### Pretraining from scratch (optional)

```bash
bash scripts/pretrain.sh mabe 2      # dataset, number of GPUs
bash scripts/pretrain.sh calms21 2
bash scripts/pretrain.sh openfield 2
```

`batch_size` in the configs is per GPU. Checkpoints are written to `pretrain_save_dir` of the pretraining config (`checkpoints_pretrain_<dataset>/<run>/best_pretrain_ep*.pth`). `SEED=<int>` fixes the random seed. To make a run resumable, give it a fixed directory: `RUN_DIR=runs/pretrain_mabe SAVE_INTERVAL=5 bash scripts/pretrain.sh mabe 2`; running the same command again continues from `RUN_DIR/latest.pth`. To fine-tune from your own checkpoint, set `PRETRAIN_CKPT=<path/to/best_pretrain_ep*.pth>` for the commands below.

### Supervised classification

```bash
bash scripts/finetune.sh <dataset> <budget> [seed] [residual|uniform]
```

`budget` is the fraction of frames that receive visual features. The budgets in Table 1 are 0.75 (Open Field), 1.0 (CalMS21), and 0.75 (Mouse Triplet). `uniform` selects frames evenly spaced in time instead of by residual.

### Unsupervised segmentation

```bash
bash scripts/unsup.sh <dataset> <budget> [seed] [residual|uniform]
```

The script applies the DEC settings of Appendix B (30 epochs; K = 50 / 50 / 30 for Mouse Triplet / CalMS21 / Open Field; VICReg 0.5 / 1.0 / 0.1; per-epoch target distribution; empty-cluster reseeding; a fixed centering of the visual term). The budgets in Table 1 are 0.25 (Open Field), 0.75 (CalMS21), and 0.75 (Mouse Triplet).

### Reading results

Each run evaluates the validation and test sets after every epoch and ends with the test F1 (per class and macro) of the epoch with the highest validation macro F1.

## Citation

The citation will be added once the paper is published.

## License

The code is released under the [MIT License](LICENSE). The re-estimated Mouse Triplet pose is distributed separately under CC BY-NC-SA, the license of MABe22.
