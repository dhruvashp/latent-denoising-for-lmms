#!/usr/bin/env bash
set -euo pipefail

# Latent-denoising instruction tuning for LLaVA-1.5 (Vicuna-7B + CLIP ViT-L/14-336).
#
# Loads raw Vicuna-7B + the official LLaVA-1.5 Stage-1 projector and trains
# LLM + projector + denoising decoder + tau embedding; vision tower frozen.
# Mid-layer supervision at LLM layer 15.
#
# Default: 8 GPU x batch 4 x grad_accum 4 = effective batch 128.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "llava" ]]; then
    if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        . "${HOME}/anaconda3/etc/profile.d/conda.sh"
    fi
    conda activate llava
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "llava" ]]; then
    echo "ERROR: failed to activate conda env 'llava'" >&2
    exit 1
fi

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints}"
DATA_PATH="${DATA_PATH:-./playground/data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-./playground/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/llava-clip-vicuna-7b-finetune}"
LOG_DIR="${LOG_DIR:-./logs}"
REQUIRED_GPUS="${REQUIRED_GPUS:-}"
SAVE_STEPS="${SAVE_STEPS:-1298}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/finetune_llava_clip_$(date +%Y%m%d_%H%M%S).log"

# Base LLM: raw Vicuna-7b (same as original LLaVA finetune recipe)
MODEL_NAME="${MODEL_NAME:-lmsys/vicuna-7b-v1.5}"
# Stage 1 pretrained projector: official LLaVA-1.5 release (LM-only pretrain)
PRETRAIN_PROJECTOR="${PRETRAIN_PROJECTOR:-${CHECKPOINT_ROOT}/llava-v1.5-7b-pretrain/mm_projector.bin}"

for required_file in "${PRETRAIN_PROJECTOR}" "${DATA_PATH}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: missing required file: ${required_file}" >&2
        exit 1
    fi
done

if [[ ! -d "${IMAGE_FOLDER}" ]]; then
    echo "ERROR: missing image folder: ${IMAGE_FOLDER}" >&2
    exit 1
fi

export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES must be set explicitly (e.g., CUDA_VISIBLE_DEVICES=0,1)" >&2
    exit 1
fi
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found; cannot validate GPU count" >&2
    exit 1
fi

if ! gpu_query_output="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1)"; then
    echo "ERROR: nvidia-smi failed while querying GPUs" >&2
    echo "${gpu_query_output}" >&2
    exit 1
fi

available_gpu_count="$(printf "%s\n" "${gpu_query_output}" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"

if [[ -z "${REQUIRED_GPUS}" ]]; then
    REQUIRED_GPUS="$(echo "${CUDA_VISIBLE_DEVICES}" | tr -d ' ' | awk -F',' '{print NF}')"
fi

if ! [[ "${REQUIRED_GPUS}" =~ ^[0-9]+$ ]] || [[ "${REQUIRED_GPUS}" -lt 1 ]]; then
    echo "ERROR: REQUIRED_GPUS must be a positive integer (got '${REQUIRED_GPUS}')" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES
visible_gpu_count="$(echo "${CUDA_VISIBLE_DEVICES}" | tr -d ' ' | awk -F',' '{print NF}')"
if [[ "${available_gpu_count}" -lt "${REQUIRED_GPUS}" ]]; then
    echo "ERROR: required ${REQUIRED_GPUS} GPUs, but only ${available_gpu_count} detected" >&2
    exit 1
fi
if [[ "${visible_gpu_count}" -lt "${REQUIRED_GPUS}" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES exposes ${visible_gpu_count} GPUs, need ${REQUIRED_GPUS}" >&2
    exit 1
fi

echo "Launching latent-denoising fine-tune (LLaVA-1.5 / CLIP / Vicuna-7B; mid-layer 15)"
echo "  Conda env: ${CONDA_DEFAULT_ENV}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  Available GPU count: ${available_gpu_count}"
echo "  Required GPU count: ${REQUIRED_GPUS}"
echo "  Base LLM: ${MODEL_NAME}"
echo "  Pretrain projector: ${PRETRAIN_PROJECTOR}"
echo "  Data path: ${DATA_PATH}"
echo "  Image folder: ${IMAGE_FOLDER}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Save steps: ${SAVE_STEPS}"
echo "  Save total limit: ${SAVE_TOTAL_LIMIT}"
echo "  Log file: ${RUN_LOG}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1 set; skipping deepspeed launch."
    exit 0
fi

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path "${MODEL_NAME}" \
    --version v1 \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter "${PRETRAIN_PROJECTOR}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --sgld_enable True \
    --sgld_stage2_enable True \
    --sgld_mode scale \
    --sgld_schedule_type whd \
    --sgld_warmup_ratio 0.05 \
    --sgld_decay_ratio 0.20 \
    --sgld_rho_noise 0.10 \
    --sgld_rho_mask 0.02 \
    --sgld_tau_s 0.07 \
    --sgld_sigma 1.0 \
    --sgld_tau_max 0.15 \
    --sgld_use_rec True \
    --sgld_use_rel True \
    --sgld_use_con True \
    --sgld_lambda_rec 0.10 \
    --sgld_lambda_rel 0.025 \
    --sgld_lambda_con 0.025 \
    --sgld_tau_r 0.10 \
    --sgld_tau_c 0.07 \
    --sgld_tau_bins 8 \
    --sgld_use_tau_embed True \
    --sgld_saliency_type cls_attn \
    --sgld_decoder_lr 2e-5 \
    --sgld_student_layer 15 \
    2>&1 | tee "${RUN_LOG}"

echo "Latent-denoising fine-tune complete. Logs: ${RUN_LOG}"
