# Custom Small Instruction-Following GPT Model

## 1. Project Overview

This project focuses on creating, pre-training, and instruction fine-tuning a custom small Generative Pre-trained Transformer (GPT) model. The goal is to build a model capable of understanding and responding to instructions.

（Instruction Fine-tuning still Work in progress)

## 2. Setup & Dependencies

### Python Version
*   Python 3.9+ is recommended.

### Libraries
The necessary libraries for this project are:
*   `torch`
*   `datasets`
*   `tokenizers`

You can install these dependencies using pip. It's recommended to use a virtual environment.

### Installation
1.  Clone the repository (if applicable).
2.  Navigate to the project directory.
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    # If torch is not included in requirements.txt or you need a specific version (e.g., with CUDA support):
    # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # (Adjust the CUDA version as per your system)
    ```
    Ensure your [`requirements.txt`](requirements.txt:1) file includes `torch`, `datasets`, and `tokenizers`.

## 3. Directory Structure

*   `data/`: Contains datasets for pre-training and fine-tuning, as well as the trained tokenizer.
    *   `data/pretraining_data/`: Stores the processed pre-training corpus (e.g., WikiText-103).
    *   `data/finetuning_data/`: Stores the processed instruction fine-tuning dataset (e.g., Dolly 15k).
    *   `data/tokenizer/`: Stores the trained custom tokenizer.
*   `out/`: Stores the output model checkpoints from pre-training and fine-tuning.
    *   `out/pretrain/`: Contains checkpoints from the pre-training phase.
    *   `out/finetune/`: Contains checkpoints from the fine-tuning phase.
*   `*.py`: Various Python scripts for different stages of the project (data processing, tokenizer training, model definition, training, inference).
*   `*.md`: Markdown files providing explanations and usage instructions for different components.

## 4. Step-by-Step Workflow

### Step 1: Dataset Preparation

*   **Pre-training Data (WikiText-103):**
    *   Script: [`process_wikitext.py`](process_wikitext.py:1)
    *   Purpose: Downloads and processes the WikiText-103 dataset into a single text file for pre-training.
    *   Expected Output: `data/pretraining_data/wikitext-103_train_corpus.txt`
*   **Instruction Fine-tuning Data (Dolly 15k):**
    *   Script: [`process_dolly.py`](process_dolly.py:1)
    *   Purpose: Processes the Dolly 15k dataset into a JSONL format suitable for instruction fine-tuning.
    *   Expected Output: `data/finetuning_data/dolly_15k_instructions.jsonl`

### Step 2: Tokenizer Training

*   Script: [`train_tokenizer.py`](train_tokenizer.py:1)
*   Purpose: Trains a custom Byte Pair Encoding (BPE) tokenizer on the pre-training corpus.
*   Input: `data/pretraining_data/wikitext-103_train_corpus.txt`
*   Expected Output: `data/tokenizer/custom_gpt_tokenizer.json`

### Step 3: Model Architecture

*   Script: [`model.py`](model.py:1)
*   Purpose: Defines the `GPT` class, which specifies the architecture of the custom GPT model.
*   Key Parameters (example values, can be configured):
    *   `vocab_size`: (Determined by the tokenizer, e.g., 30000)
    *   `n_layer`: Number of transformer blocks (e.g., 6)
    *   `n_head`: Number of attention heads (e.g., 6)
    *   `n_embd`: Embedding dimension (e.g., 384)
    *   `block_size`: Context window size (e.g., 256)

### Step 4: Pre-training

*   Script: [`pretrain_gpt.py`](pretrain_gpt.py:1)
*   Purpose: Pre-trains the custom GPT model on the prepared pre-training corpus.
*   Inputs:
    *   Trained Tokenizer (`data/tokenizer/custom_gpt_tokenizer.json`)
    *   Pre-training Corpus (`data/pretraining_data/wikitext-103_train_corpus.txt`)
    *   Model Definition (from [`model.py`](model.py:1))
*   Key Command-Line Arguments:
    *   `--tokenizer_path`: Path to the tokenizer file.
    *   `--data_path`: Path to the pre-training data file.
    *   `--checkpoint_dir`: Directory to save model checkpoints.
    *   Other arguments for batch size, learning rate, epochs, etc.
*   Expected Output: Pre-trained model checkpoints saved in the specified directory (e.g., `out/pretrain/ckpt_epoch_X.pt`).
*   For context on initial training and loss, refer to [`training_loss_explanation.md`](training_loss_explanation.md:1).

### Step 5: Instruction Fine-tuning

*   Script: [`finetune_gpt.py`](finetune_gpt.py:1)
*   Purpose: Fine-tunes the pre-trained GPT model on an instruction-following dataset.
*   Inputs:
    *   Pre-trained Model Checkpoint (e.g., `out/pretrain/ckpt_final.pt` or a specific epoch's checkpoint)
    *   Trained Tokenizer (`data/tokenizer/custom_gpt_tokenizer.json`)
    *   Instruction Dataset (`data/finetuning_data/dolly_15k_instructions.jsonl`)
*   Key Command-Line Arguments:
    *   `--pretrained_ckpt_path`: Path to the pre-trained model checkpoint.
    *   `--tokenizer_path`: Path to the tokenizer file.
    *   `--dataset_path`: Path to the instruction dataset.
    *   `--finetuned_model_dir`: Directory to save fine-tuned model checkpoints.
*   Expected Output: Fine-tuned model checkpoints saved in the specified directory (e.g., `out/finetune/ckpt_final_iter_XXX.pt`).
*   For usage details, refer to [`finetune_gpt_usage.md`](finetune_gpt_usage.md:1).

### Step 6: Text Generation (Inference)

*   Script: [`generate.py`](generate.py:1)
*   Purpose: Generates text using the fine-tuned (or pre-trained) model given a prompt.
*   Inputs:
    *   Fine-tuned (or Pre-trained) Model Checkpoint
    *   Trained Tokenizer
*   Key Command-Line Arguments:
    *   `--prompt`: The input text prompt for the model.
    *   `--checkpoint_path`: Path to the model checkpoint (e.g., `out/finetune/ckpt_final_iter_938.pt`).
    *   `--tokenizer_path`: Path to the tokenizer file (e.g., `data/tokenizer/custom_gpt_tokenizer.json`).
    *   `--max_new_tokens`: Maximum number of new tokens to generate.
    *   `--temperature`: Controls randomness in generation.
    *   `--repetition_penalty`: Penalizes token repetition.
*   Example Command:
    ```bash
    python generate.py --prompt "What is the capital of France?" --checkpoint_path "out/finetune/ckpt_final_iter_938.pt" --tokenizer_path "data/tokenizer/custom_gpt_tokenizer.json" --max_new_tokens 50 --repetition_penalty 1.2 --temperature 0.7
    ```
*   For usage details, refer to [`generate_usage.md`](generate_usage.md:1).

## 5. Current Status & Next Steps

The model has undergone initial pre-training and instruction fine-tuning. Based on an example prompt like "What is the capital of France?", the model provides a relevant answer, indicating successful learning.

**Suggestions for Improvement:**
*   **Extended Pre-training/Fine-tuning:** Train for more epochs/iterations to improve understanding and fluency.
*   **Hyperparameter Tuning:** Experiment with learning rates, batch sizes, model architecture (layers, heads, embedding size), and optimizer settings.
*   **Larger/More Diverse Datasets:** Incorporate more varied and extensive datasets for both pre-training and fine-tuning.
*   **Advanced Tokenization:** Explore more sophisticated tokenization strategies if needed.
*   **Regularization Techniques:** Implement or tune dropout, weight decay, etc., to prevent overfitting.
*   **Evaluation Metrics:** Implement robust evaluation metrics (e.g., perplexity, ROUGE, BLEU, human evaluation) to systematically track performance.
*   **Learning Rate Schedulers:** Experiment with different learning rate schedulers.

## 6. File Manifest

### Python Scripts:
*   [`model.py`](model.py:1): Defines the GPT model architecture.
*   [`process_wikitext.py`](process_wikitext.py:1): Processes the WikiText-103 dataset for pre-training.
*   [`process_dolly.py`](process_dolly.py:1): Processes the Dolly 15k dataset for fine-tuning.
*   [`train_tokenizer.py`](train_tokenizer.py:1): Trains the custom BPE tokenizer.
*   [`pretrain_gpt.py`](pretrain_gpt.py:1): Script for pre-training the GPT model.
*   [`finetune_gpt.py`](finetune_gpt.py:1): Script for instruction fine-tuning the GPT model.
*   [`generate.py`](generate.py:1): Script for generating text using a trained model.

### Key Markdown Files:
*   [`README.md`](README.md:1): This file - provides an overview of the project.
*   [`training_loss_explanation.md`](training_loss_explanation.md:1): Explains the initial training loss behavior.
*   [`finetune_gpt_usage.md`](finetune_gpt_usage.md:1): Usage instructions for the fine-tuning script.
*   [`generate_usage.md`](generate_usage.md:1): Usage instructions for the text generation script.
*   [`requirements.txt`](requirements.txt:1): Lists project dependencies.
*   [`suggested_datasets.md`](suggested_datasets.md:1): Lists suggested datasets for further exploration.
