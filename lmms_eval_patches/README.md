# lmms-eval patches

This directory contains drop-in replacements for two `lmms-eval` model wrappers
that add support for the corruption-perturbation evaluation regime used in this
repository.

## What changed

Both files are based on `lmms_eval==0.5.0`. The patches add two extra
`__init__` arguments to each model wrapper:

| argument              | type    | values                                       |
|-----------------------|---------|----------------------------------------------|
| `corruption_category` | `str`   | `noise`, `blur`, `weather`, `digital`, `None`|
| `corruption_severity` | `int`   | `1..5`                                       |

When `corruption_category` is set, every input image is corrupted in-place
before being handed to the model. Corruption is performed by `llava.corruption.apply_corruption`
(see `llava/corruption.py` in this repository). For each image the wrapper
samples one corruption type from the requested category pool (deterministically
seeded by image index, so the same image always gets the same corruption).

The `llava.py` wrapper also accepts an optional `model_base` argument, used
when loading LoRA-style checkpoints.

## How to apply

```bash
pip install lmms-eval==0.5.0

LMMS_DIR=$(python -c "import lmms_eval, os; print(os.path.dirname(lmms_eval.__file__))")
cp lmms_eval_patches/llava.py       "${LMMS_DIR}/models/simple/llava.py"
cp lmms_eval_patches/qwen2_5_vl.py  "${LMMS_DIR}/models/simple/qwen2_5_vl.py"
```

## How to use

Pass the new arguments through `--model_args` like any other model arg:

```bash
python -m lmms_eval \
    --model llava \
    --model_args "pretrained=/path/to/checkpoint,corruption_category=blur,corruption_severity=5" \
    --tasks vqav2,gqa,mmbench_en_dev \
    --batch_size 1 \
    --output_path ./eval_results/blur_sev5
```

To run a clean (non-corrupted) evaluation, simply omit `corruption_category`.
