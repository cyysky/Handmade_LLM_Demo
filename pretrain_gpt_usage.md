# Usage Guide: `pretrain_gpt.py`

## 1. Purpose

The [`pretrain_gpt.py`](pretrain_gpt.py:1) script is designed for pre-training a custom Generative Pre-trained Transformer (GPT) model. It handles loading a custom tokenizer, processing a large text corpus for training, instantiating the GPT model architecture, and executing the training loop using PyTorch. Key features include data caching, mixed-precision training, gradient accumulation, periodic checkpoint saving, and the ability to resume training from a saved state.

## 2. Prerequisites

Before running the pre-training script, ensure the following are in place:

*   **Custom Tokenizer:** A tokenizer file, typically in JSON format, trained for your specific vocabulary.
    *   Example: [`data/tokenizer/custom_gpt_tokenizer.json`](data/tokenizer/custom_gpt_tokenizer.json)
*   **Pre-training Corpus:** A plain text file containing the raw data for pre-training the model.
    *   Example: [`data/pretraining_data/wikitext-103_train_corpus.txt`](data/pretraining_data/wikitext-103_train_corpus.txt)
*   **GPT Model Definition:** The Python script defining the GPT model architecture.
    *   Example: [`model.py`](model.py:1)
*   **Installed Python Packages:** All necessary Python libraries as listed in the [`requirements.txt`](requirements.txt:1) file. You can install them using:
    ```bash
    pip install -r requirements.txt
    ```

## 3. Configuration

Key configuration parameters are located at the top of the [`pretrain_gpt.py`](pretrain_gpt.py:1) script. You will need to modify these variables directly in the script to suit your setup.

*   `TOKENIZER_PATH`: Path to your custom tokenizer file.
    *   Example: `"data/tokenizer/custom_gpt_tokenizer.json"`
*   `CORPUS_PATH`: Path to your pre-training corpus text file.
    *   Example: `"data/pretraining_data/wikitext-103_train_corpus.txt"`
*   `CACHED_DATA_PATH`: Path where the tokenized and processed training data will be saved as a PyTorch tensor file (`.pt`). This speeds up subsequent runs by avoiding re-processing the entire corpus.
    *   Example: `"data/pretraining_data/wikitext-103_train_corpus.pt"`
*   `MODEL_OUTPUT_DIR`: Directory where model checkpoints and the best model will be saved.
    *   Example: `"out/pretrain"`
*   `RESUME_FROM_CHECKPOINT`: Controls whether to resume training from a previously saved checkpoint.
    *   Set to `None` (default) to start training from scratch.
    *   Set to the path of a specific checkpoint file (e.g., `"out/pretrain/ckpt_iter_1000.pt"`) to resume from that point.
    *   Set to `"best"` to resume from the `best_model.pt` saved in `MODEL_OUTPUT_DIR`.
*   **Training Hyperparameters:**
    *   `BATCH_SIZE`: Number of sequences processed in each training step.
    *   `LEARNING_RATE`: The learning rate for the optimizer.
    *   `NUM_EPOCHS`: The total number of times the training script will iterate over the entire dataset.
    *   `GRADIENT_ACCUMULATION_STEPS`: Number of steps to accumulate gradients before performing an optimizer step. This is useful for simulating larger batch sizes with limited GPU memory.

## 4. How to Run

To start the pre-training process, execute the script from your terminal:

```bash
python pretrain_gpt.py
```

All configuration is done by editing the variables within the [`pretrain_gpt.py`](pretrain_gpt.py:1) script itself before running.

## 5. Output

During and after training, the script produces several outputs:

*   **Console Output:** The script logs information to the console, including:
    *   Configuration parameters being used.
    *   Progress of data loading and caching.
    *   Training progress, including current epoch, iteration, and training loss.
    *   Information about saved checkpoints.
*   **Model Checkpoints (`MODEL_OUTPUT_DIR`):**
    *   `ckpt_iter_X.pt`: Checkpoints saved periodically during training, where `X` is the iteration number. These files contain the model state, optimizer state, and other training metadata, allowing you to resume training.
    *   `best_model.pt`: The model checkpoint that achieved the lowest validation loss (if validation is implemented) or lowest training loss observed so far.
    *   `final_model_epochs_done.pt`: The model state saved at the very end of the training process after all epochs are completed.
*   **Cached Tokenized Data (`CACHED_DATA_PATH`):**
    *   A `.pt` file (e.g., [`data/pretraining_data/wikitext-103_train_corpus.pt`](data/pretraining_data/wikitext-103_train_corpus.pt)) containing the tokenized training data. This file is created during the first run (if it doesn't exist) and loaded in subsequent runs to save time.

## 6. Resuming Training

To resume training from a previously saved checkpoint:

1.  Locate the checkpoint file you wish to resume from (e.g., `out/pretrain/ckpt_iter_5000.pt` or `out/pretrain/best_model.pt`).
2.  Open the [`pretrain_gpt.py`](pretrain_gpt.py:1) script.
3.  Set the `RESUME_FROM_CHECKPOINT` variable to the path of your chosen checkpoint file or to the string `"best"`.
    *   Example (specific checkpoint): `RESUME_FROM_CHECKPOINT = "out/pretrain/ckpt_iter_5000.pt"`
    *   Example (best model): `RESUME_FROM_CHECKPOINT = "best"`
4.  Ensure other configurations (like `MODEL_OUTPUT_DIR`, `TOKENIZER_PATH`, etc.) are consistent with the run you are resuming.
5.  Run the script as usual: `python pretrain_gpt.py`.

The script will load the model weights, optimizer state, and other necessary information from the checkpoint and continue training.

## 7. Understanding Training Loss

The training loss, printed to the console during training, is a key indicator of how well the model is learning. Generally, a lower loss value indicates that the model is making better predictions on the training data.

For a more detailed explanation of training loss, what different loss values might mean, and strategies for improving your model's performance based on loss, please refer to the [`training_loss_explanation.md`](training_loss_explanation.md:1) document.