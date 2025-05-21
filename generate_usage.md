# Using `generate.py` for Text Generation

This document explains how to use the `generate.py` script to generate text using your fine-tuned GPT model.

## 1. Purpose

The `generate.py` script loads a pre-trained or fine-tuned GPT model checkpoint and a tokenizer to generate text based on a user-provided prompt. It allows customization of generation parameters like temperature, top-k sampling, and repetition penalty.

## 2. How to Run

You can run the script from your terminal using Python:

```bash
python generate.py --prompt "Your instruction here" [OPTIONS]
```

**Example:**

```bash
python generate.py \
    --checkpoint_path "out/finetune/ckpt.pt" \
    --tokenizer_path "data/tokenizer/custom_gpt_tokenizer.json" \
    --prompt "Explain the concept of neural networks in simple terms." \
    --max_new_tokens 150 \
    --temperature 0.7 \
    --top_k 50 \
    --repetition_penalty 1.15
```

## 3. Command-Line Arguments

The script accepts the following command-line arguments:

*   `--checkpoint_path` (str, default: `"out/finetune/ckpt.pt"`)
    *   Path to the fine-tuned model checkpoint (`.pt`) file.
*   `--tokenizer_path` (str, default: `"data/tokenizer/custom_gpt_tokenizer.json"`)
    *   Path to the tokenizer JSON file (e.g., `custom_gpt_tokenizer.json` created by `train_tokenizer.py`).
*   `--prompt` (str, **required**)
    *   The main instruction or question for the model (e.g., "Translate this to French", "Write a poem about spring").
*   `--input_text` (str, default: `""`)
    *   Optional additional input text if your fine-tuning prompt structure used a separate input field (e.g., text to be summarized or translated if the prompt is "Summarize the following text:").
*   `--max_new_tokens` (int, default: `150`)
    *   The maximum number of new tokens the model should generate after the prompt.
*   `--temperature` (float, default: `0.8`)
    *   Controls the randomness of the output. Lower values (e.g., 0.2-0.5) make the output more deterministic and focused. Higher values (e.g., 0.8-1.0) make it more random and creative.
*   `--top_k` (int, default: `50`)
    *   Filters the vocabulary to the `k` most likely next tokens at each step. Set to `0` to disable top-k sampling. Helps prevent very unlikely tokens from being generated.
*   `--repetition_penalty` (float, default: `1.0`)
    *   Penalizes tokens that have recently appeared in the generated text or prompt. Values greater than 1.0 (e.g., 1.1, 1.2) discourage repetition. `1.0` means no penalty.
*   `--device` (str, default: `None`)
    *   The device to run the model on (e.g., `"cuda"`, `"cpu"`). If `None`, it auto-detects CUDA availability.
*   `--seed` (int, default: `1337`)
    *   A random seed for reproducibility of the generation process.

## 4. Expected Prompt Format

The script formats the user's `--prompt` and `--input_text` into the structure the model was likely fine-tuned on:

`[BOS] Instruction: {user_prompt} Input: {user_input_text} Response:`

*   `[BOS]` is the beginning-of-sequence token.
*   The `Instruction:` part comes from your `--prompt` argument.
*   The `Input:` part (if `--input_text` is provided) comes from that argument.
*   The script appends `Response:` to signal the model where to start generating its answer.

The tokenizer (loaded from `--tokenizer_path`) is expected to handle the `[BOS]` token automatically due to its post-processing configuration (as set up in `train_tokenizer.py`). The content passed to the tokenizer's `encode` method is `Instruction: ... Input: ... Response:`.

## 5. Interpreting the Output

The script will print:
1.  Setup information (device, paths, model parameters).
2.  The content string being passed to the tokenizer's `encode` method.
3.  The final generated output, including the initial prompt structure and the model's response. The `[BOS]` and `[EOS]` (end-of-sequence) tokens might be visible depending on the tokenizer's decoding settings.

## 6. Troubleshooting Common Issues

*   **Repetitive Output (e.g., "::::::", "the the the"):**
    *   This is a common issue with language models.
    *   **Solution:** Increase the `--repetition_penalty` (e.g., try values from 1.1 to 1.5 or higher). You might also try lowering `--temperature`.
*   **Nonsensical or Irrelevant Output:**
    *   Could be due to the model quality, the prompt, or generation parameters.
    *   **Solutions:**
        *   Try different `--temperature` values. Lower might make it more coherent.
        *   Adjust `--top_k`.
        *   Rephrase your `--prompt` to be more specific or clear.
        *   The fine-tuned model might need more training or better quality data. The reported losses (e.g., pretrain loss ~3.9, finetune loss ~3.3) give some indication of model learning, but lower loss doesn't always guarantee perfect generation.
*   **`[UNK]` Tokens in Output:**
    *   Indicates unknown tokens. This should be rare if using the same tokenizer the model was trained with.
    *   **Check:** Ensure `--tokenizer_path` points to the correct tokenizer file used during training/fine-tuning.
*   **Unicode Replacement Characters (�):**
    *   This can happen if the model generates token ID sequences that don't form valid UTF-8 characters when decoded. This often points to deeper issues with the model's training or stability.
    *   **Solutions:** Primarily, this would require investigating the model training process. From the inference side, experimenting with generation parameters might sometimes help, but it's less likely to be a complete fix.
*   **`KeyError: 'model'` or `KeyError: 'model_state_dict'` during checkpoint loading:**
    *   The script tries common keys (`'model'`, `'state_dict'`, `'model_state_dict'`) for the model's weights in the checkpoint file. If your checkpoint uses a different key, you might need to modify the script.
*   **Errors related to tokenizer loading or special tokens:**
    *   Ensure the `custom_gpt_tokenizer.json` file is valid and accessible.
    *   The script expects `[BOS]` and `[EOS]` tokens to be defined in the tokenizer.

## 7. Note on Model-Dependent Output Quality

The quality, coherence, and relevance of the generated text heavily depend on the quality of the pre-trained base model, the fine-tuning data, and the fine-tuning process itself. The `generate.py` script provides the tools to run inference, but it cannot inherently fix underlying issues with the model.