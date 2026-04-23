#!/usr/bin/env bash
set -euo pipefail

# Latent-denoising post-tune for Qwen2.5-VL-7B-Instruct.
#
# Loads the released Qwen2.5-VL-7B-Instruct checkpoint and fine-tunes
# LLM + visual merger + denoising decoder + tau embedding with the latent-denoising objective;
# vision encoder frozen. Mid-layer latent supervision at LLM layer 13.
# Teacher = post-merger features (3584-dim); saliency = L2-norm.
#
# Default: 8 GPU x batch 2 x grad_accum 8 = effective batch 128 (ZeRO-2).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "qwen_sgld" ]]; then
    if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        . "${HOME}/anaconda3/etc/profile.d/conda.sh"
    fi
    conda activate qwen_sgld
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "qwen_sgld" ]]; then
    echo "ERROR: failed to activate conda env 'qwen_sgld'" >&2
    exit 1
fi

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints}"
DATA_PATH="${DATA_PATH:-./playground/data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-./playground/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/qwen25vl-7b-posttune}"
LOG_DIR="${LOG_DIR:-./logs}"
REQUIRED_GPUS="${REQUIRED_GPUS:-}"
SAVE_STEPS="${SAVE_STEPS:-1298}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/posttune_qwen_$(date +%Y%m%d_%H%M%S).log"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"

for required_file in "${DATA_PATH}"; do
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
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

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

echo "Launching Latent-denoising post-tune (Qwen2.5-VL-7B-Instruct; mid-layer 13)"
echo "  Hardware: 8x RTX 6000 Ada (49GB)"
echo "  Conda env: ${CONDA_DEFAULT_ENV}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  Available GPU count: ${available_gpu_count}"
echo "  Required GPU count: ${REQUIRED_GPUS}"
echo "  Base model: ${MODEL_NAME}"
echo "  Data path: ${DATA_PATH}"
echo "  Image folder: ${IMAGE_FOLDER}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Save steps: ${SAVE_STEPS}"
echo "  Save total limit: ${SAVE_TOTAL_LIMIT}"
echo "  Log file: ${RUN_LOG}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1 set; skipping torchrun launch."
    exit 0
fi

# Use torchrun launcher (matches Qwen3-VL official training scripts).
# DeepSpeed ZeRO-2 is still used via --deepspeed arg to HF Trainer.
torchrun --nproc_per_node="${REQUIRED_GPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    llava/train/train_qwen.py \
    --deepspeed ./scripts/zero2_qwen.json \
    --model_name_or_path "${MODEL_NAME}" \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --image_max_pixels 451584 \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
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
    --group_by_modality_length True \
    --dataloader_num_workers 4 \
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
    --sgld_saliency_type l2_norm \
    --sgld_decoder_lr 2e-5 \
    --sgld_student_layer 13 \
    2>&1 | tee "${RUN_LOG}"

echo "Latent-denoising post-tune complete. Logs: ${RUN_LOG}"
