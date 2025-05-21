# How to Use `finetune_gpt.py`

This document explains how to use the `finetune_gpt.py` script for instruction fine-tuning a custom GPT model.

## 1. Purpose

The `finetune_gpt.py` script is designed to adapt a pre-trained GPT model to follow specific instructions. It takes a dataset of instruction-input-output triplets and fine-tunes the model to generate appropriate responses based on given instructions and optional inputs.

Key features include:
*   **Instruction Formatting:** Converts instruction data (e.g., from Dolly 15k format) into a structured prompt format: `[BOS] Instruction: {instruction} Input: {input} Response: {output} [EOS]`.
*   **Masked Loss:** Calculates the training loss only on the `Response` part of the sequence, ensuring the model learns to generate the desired output without being penalized for the prompt tokens.
*   **Resumption Capability:** Allows fine-tuning to be paused and resumed from saved checkpoints, including optimizer and mixed-precision scaler states.
*   **Mixed-Precision Training:** Supports `bfloat16` or `float16` for faster training and reduced memory usage on compatible GPUs.
*   **Flexible Configuration:** Uses command-line arguments for easy customization of paths, hyperparameters, and training settings.

## 2. Prerequisites

Before running the script, ensure you have the following:

1.  **A Trained Tokenizer:** A Hugging Face `tokenizers` compatible tokenizer file (e.g., `custom_gpt_tokenizer.json`). This tokenizer should include special tokens like `[BOS]`, `[EOS]`, and ideally `[PAD]`. The script will attempt to use `[EOS]` as `[PAD]` if `[PAD]` is not explicitly found.
2.  **A Pre-trained GPT Model Checkpoint:** A `.pt` file containing the state dictionary and model arguments (`model_args`) of your pre-trained GPT model (e.g., `out/pretrain/best_model.pt`). This is required if you are not resuming from a fine-tuning checkpoint.
3.  **Instruction Fine-tuning Data:** A JSON Lines (`.jsonl`) file where each line is a JSON object containing `instruction`, `output`, and optionally `input` keys (e.g., `data/finetuning_data/dolly_15k_instructions.jsonl`).
4.  **Python Environment:** A Python environment with PyTorch, Hugging Face `tokenizers`, and other necessary libraries installed (refer to `requirements.txt` if available).
5.  **GPU (Recommended):** An NVIDIA GPU is highly recommended for feasible training times, especially with mixed-precision.

## 3. Command-Line Arguments

The script accepts various command-line arguments to control its behavior:

### Path Arguments:
*   `--tokenizer_path`: (Required) Path to the trained tokenizer file.
    *   Example: `data/tokenizer/custom_gpt_tokenizer.json`
*   `--data_path`: (Required) Path to the fine-tuning data (JSONL format).
    *   Example: `data/finetuning_data/dolly_15k_instructions.jsonl`
*   `--pretrained_ckpt_path`: Path to the base pre-trained model checkpoint (`.pt` file). Required if not resuming fine-tuning with `--resume_from_ckpt_path`.
    *   Example: `out/pretrain/best_model.pt`
*   `--resume_from_ckpt_path`: Path to a fine-tuning checkpoint (`.pt` file saved by this script) to resume training from. If provided, `--pretrained_ckpt_path` is ignored for model loading.
    *   Example: `out/finetune/ckpt_iter_500.pt`
*   `--out_dir`: Output directory for saving fine-tuned model checkpoints and logs.
    *   Default: `out/finetune`

### Training Hyperparameters:
*   `--batch_size`: Batch size for training. Adjust based on GPU memory.
    *   Default: `2`
*   `--learning_rate`: Learning rate for the AdamW optimizer. Typically smaller for fine-tuning than for pre-training.
    *   Default: `3e-5`
*   `--num_epochs`: Number of training epochs.
    *   Default: `1`
*   `--max_iters`: Maximum number of training iterations. Overrides `num_epochs` if set to a value greater than 0. If resuming, this is the *total* target iterations.
    *   Default: `-1` (calculated from `num_epochs` and dataset size)
*   `--gradient_accumulation_steps`: Number of steps to accumulate gradients before performing an optimizer step. Increases effective batch size.
    *   Default: `8`

### Logging and Saving:
*   `--log_interval`: Log training status (loss, time per iteration) every N iterations.
    *   Default: `10`
*   `--save_interval`: Save a model checkpoint every N iterations. Checkpoints are also saved at the end of training.
    *   Default: `100`

### Device and Precision:
*   `--device`: Device to use for training.
    *   Choices: `cuda`, `cpu`
    *   Default: `cuda` (falls back to `cpu` if CUDA is not available)
*   `--dtype`: Data type for mixed-precision training.
    *   Choices: `float16`, `bfloat16`
    *   Default: `bfloat16` if supported by CUDA, otherwise `float16`. Ignored if `device` is `cpu`.

## 4. Usage Examples

### Example 1: Starting a New Fine-tuning Run

This command starts a new fine-tuning process using a pre-trained model.

```bash
python finetune_gpt.py \
    --tokenizer_path data/tokenizer/custom_gpt_tokenizer.json \
    --data_path data/finetuning_data/dolly_15k_instructions.jsonl \
    --pretrained_ckpt_path out/pretrain/best_model.pt \
    --out_dir out/finetune_run1 \
    --batch_size 2 \
    --learning_rate 3e-5 \
    --num_epochs 1 \
    --gradient_accumulation_steps 8 \
    --log_interval 10 \
    --save_interval 200 \
    --device cuda \
    --dtype bfloat16
```

### Example 2: Resuming a Fine-tuning Run

This command resumes fine-tuning from a checkpoint saved during a previous run (e.g., `ckpt_iter_200.pt`).

```bash
python finetune_gpt.py \
    --tokenizer_path data/tokenizer/custom_gpt_tokenizer.json \
    --data_path data/finetuning_data/dolly_15k_instructions.jsonl \
    --resume_from_ckpt_path out/finetune_run1/ckpt_iter_200.pt \
    --out_dir out/finetune_run1 \
    --batch_size 2 \
    --learning_rate 3e-5 \
    --num_epochs 1 \
    --max_iters 500 \ # Optional: set a new total target iteration count
    --gradient_accumulation_steps 8 \
    --log_interval 10 \
    --save_interval 100 \
    --device cuda \
    --dtype bfloat16
```
**Note:** When resuming:
*   The `tokenizer_path` and `data_path` are still required to set up the dataset.
*   The `out_dir` should typically be the same to continue saving in the same location.
*   The learning rate can be adjusted if needed.
*   `max_iters` or `num_epochs` will determine how many *additional* iterations are run, considering the `iter_num` loaded from the checkpoint. If `max_iters` is specified, training continues until `iter_num` reaches `max_iters`.

## 5. Output

The script will create the specified output directory (`--out_dir`). Inside this directory, it will save:
*   **Model Checkpoints:** Files named like `ckpt_iter_XXX.pt` or `ckpt_final_iter_XXX.pt`. These checkpoints contain:
    *   `model_state_dict`: The fine-tuned model's weights.
    *   `optimizer_state_dict`: The state of the AdamW optimizer.
    *   `scaler_state_dict`: The state of the `GradScaler` (for mixed precision).
    *   `model_args`: The arguments used to initialize the GPT model.
    *   `iter_num`: The iteration number at which the checkpoint was saved.
    *   `config`: The command-line arguments used for the training run.
*   **Logs:** Training progress (loss, iteration time) will be printed to the console.

## 6. Troubleshooting

*   **`AttributeError` related to tokens:** Ensure your tokenizer (`--tokenizer_path`) has `[BOS]`, `[EOS]`, and ideally `[PAD]` tokens defined. The script uses these exact strings.
*   **CUDA Out of Memory:** Reduce `batch_size` or increase `gradient_accumulation_steps`. Ensure `dtype` is `bfloat16` or `float16` if using a GPU.
*   **`RuntimeError: Expected all tensors to be on the same device...` when resuming:** This should be handled by the script, but if it occurs, ensure your PyTorch version is up-to-date.
*   **Slow Training on CPU:** Fine-tuning large models is computationally intensive. Using a GPU (`--device cuda`) is strongly recommended.

This guide should help you effectively use the `finetune_gpt.py` script for your instruction fine-tuning tasks.