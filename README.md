# Latent Denoising for Large Multimodal Models

Reference implementation for training large multimodal models with a
**latent denoising** objective: an auxiliary self-supervised loss
applied to a chosen mid-layer of the language model that asks the network
to reconstruct, contrast, and relate corrupted versions of its own visual
representations.

The same recipe is implemented for three backbones:

| Backbone              | Vision encoder           | Language model | Training mode      |
|-----------------------|--------------------------|----------------|--------------------|
| LLaVA-1.5 + CLIP      | OpenAI CLIP ViT-L/14-336 | Vicuna-7B-v1.5 | Instruction tuning |
| LLaVA + SigLIP        | SigLIP-SO400M-patch14-384| Vicuna-7B-v1.5 | Pretrain + tune    |
| Qwen2.5-VL-7B         | Qwen ViT (built-in)      | Qwen-7B        | Post-tuning        |

This repo contains code only — no checkpoints, no datasets, no evaluation
result tables. See "Datasets" below for download pointers.

---

## Repository layout

```
.
├── llava/                 # Model + training + corruption code
│   ├── model/             # LLaVA / Qwen architectures with denoising heads
│   ├── train/             # train.py (LLaVA) and train_qwen.py (Qwen)
│   ├── corruption.py      # 15 ImageNet-C corruption types in 4 categories
│   ├── conversation.py    # conv templates
│   └── ...
├── scripts/               # Training launchers + DeepSpeed configs
│   ├── finetune_llava_clip.sh
│   ├── pretrain_llava_siglip.sh
│   ├── finetune_llava_siglip.sh
│   ├── posttune_qwen.sh
│   └── zero{2,2_qwen,3,3_offload,3_resume}.json
├── lmms_eval_patches/     # Patched lmms-eval model wrappers (corruption support)
│   ├── llava.py
│   ├── qwen2_5_vl.py
│   └── README.md
├── robustness/            # Frost overlay assets used by weather corruption
└── pyproject.toml
```

---

## Setup

We use **two separate conda environments** because LLaVA-1.5 and
Qwen2.5-VL pin incompatible versions of `transformers` / `torch` /
`deepspeed`. Activate the right one before training or evaluating.

### Environment 1 — LLaVA (CLIP/Vicuna and SigLIP/Vicuna)

```bash
conda create -n llava python=3.10 -y
conda activate llava

# install this repo (pulls torch, transformers, deepspeed, peft, etc. via pyproject.toml)
pip install -e .

# extras for full training functionality
pip install ninja
pip install flash-attn --no-build-isolation

# eval framework + corruption patches (see Evaluation section)
pip install lmms-eval==0.5.0
```

### Environment 2 — Qwen

```bash
conda create -n qwen_sgld python=3.10 -y
conda activate qwen_sgld

pip install "torch>=2.5.0" torchvision
pip install "transformers>=4.49.0" accelerate
pip install deepspeed peft bitsandbytes
pip install qwen-vl-utils
pip install opencv-python-headless scipy pillow numpy einops
pip install wandb tensorboard sentencepiece protobuf
pip install lmms-eval==0.5.0

# install this repo as a package so `llava.corruption` is importable
pip install -e .
```

After both environments exist, apply the lmms-eval patches in **each** env
(see [lmms_eval_patches/README.md](lmms_eval_patches/README.md)).

---

## Datasets

We do not redistribute any data. Acquire the standard LLaVA-1.5 training
data from upstream:

| Stage                | Dataset                           | Source                                                    |
|----------------------|-----------------------------------|-----------------------------------------------------------|
| Projector pretrain   | `blip_laion_cc_sbu_558k.json`     | [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) |
| Instruction tuning   | `llava_v1_5_mix665k.json`         | [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) |
| Image folders        | COCO, GQA, OCR-VQA, TextVQA, VG   | per LLaVA-1.5 download instructions                       |

Place everything under `playground/data/` to match the default paths in the
training scripts, or override `DATA_PATH` and `IMAGE_FOLDER` env vars.

For evaluation datasets, `lmms-eval` downloads them automatically from
HuggingFace on first run.

---

## Training

All training scripts:

* derive `REPO_ROOT` from their own location, so they can be launched from
  anywhere
* require `CUDA_VISIBLE_DEVICES` to be set explicitly (no implicit default)
* respect `CHECKPOINT_ROOT`, `OUTPUT_DIR`, `DATA_PATH`, `IMAGE_FOLDER`,
  `LOG_DIR`, `SAVE_STEPS`, `SAVE_TOTAL_LIMIT` env vars
* support `DRY_RUN=1` to print configuration and exit without launching

Set the relevant API key env vars before launch if you want logging:

```bash
export WANDB_API_KEY=...        # optional, for W&B logging
```

### LLaVA-1.5 (CLIP + Vicuna-7B)

The CLIP variant skips a separate projector pretraining stage and starts
from the official LLaVA-1.5 Stage-1 projector. Download `mm_projector.bin`
from [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b)
and place it at `checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin`
(or override `PRETRAIN_PROJECTOR`).

```bash
conda activate llava
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    bash scripts/finetune_llava_clip.sh
```

### LLaVA + SigLIP (SigLIP-SO400M + Vicuna-7B)

Two stages: projector pretraining (LM-only objective) followed by
latent-denoising fine-tuning.

```bash
conda activate llava

# Stage 1 — projector pretrain (558k caption pairs)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    bash scripts/pretrain_llava_siglip.sh

# Stage 2 — full fine-tune with the latent-denoising objective (665k mix)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    bash scripts/finetune_llava_siglip.sh
```

### Qwen2.5-VL-7B post-tune

Loads the released `Qwen/Qwen2.5-VL-7B-Instruct` checkpoint and post-tunes
LLM + visual merger + denoising heads.

```bash
conda activate qwen_sgld
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    bash scripts/posttune_qwen.sh
```

---

## Evaluation

We use [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval) for all
benchmark, robustness, and corruption-perturbation evaluation. The two
patched files in [`lmms_eval_patches/`](lmms_eval_patches/) extend the
upstream `llava` and `qwen2_5_vl` model wrappers with corruption support
— apply them once per environment as shown in
[`lmms_eval_patches/README.md`](lmms_eval_patches/README.md).

### 1. Standard benchmarks (clean)

LLaVA model:

```bash
conda activate llava
CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
    --model llava \
    --model_args pretrained=/path/to/llava-clip-vicuna-7b-finetune \
    --tasks vqav2_val,gqa,mmbench_en_dev,mmstar,mme,pope,mmmu_val,scienceqa_img,textvqa_val,ocrbench,chartqa \
    --batch_size 1 \
    --output_path ./eval_results/llava_clean
```

Qwen model:

```bash
conda activate qwen_sgld
CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=/path/to/qwen25vl-7b-posttune \
    --tasks vqav2_val,gqa,mmbench_en_dev,mmstar,mme,pope,mmmu_val,scienceqa_img,textvqa_val,ocrbench,chartqa \
    --batch_size 1 \
    --output_path ./eval_results/qwen_clean
```

### 2. Robustness benchmarks (clean images, harder distributions)

```bash
python -m lmms_eval \
    --model <llava|qwen2_5_vl> \
    --model_args pretrained=/path/to/checkpoint \
    --tasks naturalbench,qbench_a_dev,realworldqa,vizwiz_vqa_val \
    --batch_size 1 \
    --output_path ./eval_results/robustness
```

### 3. Corruption-perturbation evaluation

Pass `corruption_category` and `corruption_severity` through `--model_args`:

```bash
python -m lmms_eval \
    --model <llava|qwen2_5_vl> \
    --model_args "pretrained=/path/to/checkpoint,corruption_category=noise,corruption_severity=3" \
    --tasks vqav2_val,gqa,mmbench_en_dev,mmstar,mme,pope,mmmu_val,scienceqa_img,textvqa_val,ocrbench,chartqa \
    --batch_size 1 \
    --output_path ./eval_results/noise_sev3
```

Valid values:

* `corruption_category ∈ {noise, blur, weather, digital}`
* `corruption_severity ∈ {1, 2, 3, 4, 5}`

For each input image the wrapper samples one corruption from the requested
category pool, deterministically seeded by image index. The 15 corruption
types and 5 severity levels follow Hendrycks & Dietterich (ICLR 2019). The
weather pool reuses six frost overlay images shipped under
[`robustness/`](robustness/).

To get the full corruption result reported in the paper, run all
`(category × severity)` combinations and average per benchmark.

---

## Implementation notes

* **DeepSpeed:** ZeRO-2 is the safe default when training with the
  latent-denoising objective (the monkey-patch asymmetry in our auxiliary
  heads is incompatible with ZeRO-3 partitioning). Use `scripts/zero3.json`
  only for plain LM-only baselines.
* **torchrun vs deepspeed launcher:** Qwen training requires the
  `torchrun` launcher; LLaVA training uses the `deepspeed` launcher.
  The provided scripts already pick the right one.
* **Mid-layer supervision:** the auxiliary loss is applied at a single
  LLM mid-layer (default: layer 15/32 for LLaVA, 13/28 for Qwen).
  Override via the `--sgld_student_layer` training flag.
* **Always-on differentiable corruption:** the noise/mask masks are
  applied as float weights, not boolean indices, so that gradients flow
  to the auxiliary parameters even on steps where K=0. This matches the
  LLaVA per-image dynamic K behavior and avoids DeepSpeed IPG desync.
* **Training-time CLI flags** for the auxiliary objective are namespaced
  as `--sgld_*` (an internal name kept for backward-compatibility with the
  config plumbing in `train.py` / `train_qwen.py`); they correspond to
  the latent-denoising hyperparameters described in the paper.

---

## License

This repository is released under the Apache License 2.0
(see [LICENSE](LICENSE)).

It builds directly on:

* [LLaVA](https://github.com/haotian-liu/LLaVA) — Apache 2.0
* [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) — Apache 2.0
* [ImageNet-C corruptions](https://github.com/hendrycks/robustness)
  (Hendrycks & Dietterich, ICLR 2019) — MIT
* [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) — MIT
