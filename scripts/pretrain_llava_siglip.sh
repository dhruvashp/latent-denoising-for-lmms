#!/usr/bin/env bash
set -euo pipefail

# Latent-denoising pretrain (LLaVA-style, SigLIP → Vicuna projector pretraining)
#
# Trains ONLY the mm_projector (1152 → 4096 → 4096) to align SigLIP features
# to Vicuna's embedding space. LLM and vision tower frozen. No latent-denoising losses.
#
# Uses llava env (transformers 4.37, deepspeed + ZeRO-3).
# 8 GPU x batch 16 x grad_accum 1 = effective 128

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
DATA_PATH="${DATA_PATH:-./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-./playground/data/LLaVA-Pretrain}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/llava-siglip-vicuna-7b-pretrain}"
LOG_DIR="${LOG_DIR:-./logs}"
REQUIRED_GPUS="${REQUIRED_GPUS:-}"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/pretrain_llava_siglip_$(date +%Y%m%d_%H%M%S).log"

MODEL_NAME="${MODEL_NAME:-lmsys/vicuna-7b-v1.5}"

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
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES must be set explicitly" >&2
    exit 1
fi
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

if ! gpu_query_output="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1)"; then
    echo "ERROR: nvidia-smi failed" >&2; exit 1
fi
available_gpu_count="$(printf "%s\n" "${gpu_query_output}" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"

if [[ -z "${REQUIRED_GPUS}" ]]; then
    REQUIRED_GPUS="$(echo "${CUDA_VISIBLE_DEVICES}" | tr -d ' ' | awk -F',' '{print NF}')"
fi
export CUDA_VISIBLE_DEVICES
visible_gpu_count="$(echo "${CUDA_VISIBLE_DEVICES}" | tr -d ' ' | awk -F',' '{print NF}')"
if [[ "${visible_gpu_count}" -lt "${REQUIRED_GPUS}" ]]; then
    echo "ERROR: need ${REQUIRED_GPUS} GPUs, have ${visible_gpu_count}" >&2; exit 1
fi

echo "Launching Latent-denoising pretrain (SigLIP → Vicuna projector pretraining)"
echo "  Conda env: ${CONDA_DEFAULT_ENV}"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "  Base LLM: ${MODEL_NAME}"
echo "  Vision tower: google/siglip-so400m-patch14-384"
echo "  Data: ${DATA_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Log: ${RUN_LOG}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1; skipping launch."; exit 0
fi

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "${MODEL_NAME}" \
    --version plain \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
    --sgld_enable False \
    2>&1 | tee "${RUN_LOG}"

echo "Projector pretrain complete. Logs: ${RUN_LOG}"
