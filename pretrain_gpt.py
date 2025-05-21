import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import os
import time
from model import GPT # Assuming model.py is in the same directory or accessible in PYTHONPATH

# --- Configuration ---
# Paths
TOKENIZER_PATH = "data/tokenizer/custom_gpt_tokenizer.json"
CORPUS_PATH = "data/pretraining_data/wikitext-103_train_corpus.txt" # Ensure this file exists
MODEL_OUTPUT_DIR = "out/pretrain/"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Training Hyperparameters
BATCH_SIZE = 4 # Adjusted for 12GB VRAM with block_size=1024
LEARNING_RATE = 3e-4
NUM_EPOCHS = 1 # Start with 1-3 epochs
# MAX_ITERS = 200000 # Alternative to epochs, e.g. for large datasets
EVAL_INTERVAL = 200 # How often to evaluate (if an eval set is used)
LOG_INTERVAL = 10 # How often to log training loss
GRADIENT_ACCUMULATION_STEPS = 8 # Accumulate gradients for 8 steps
SAVE_INTERVAL = 500 # Save model every N iterations
BLOCK_SIZE = 1024 # Context length

# Model Configuration (should match the trained tokenizer and model.py)
VOCAB_SIZE = 32000
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 # for mixed precision
PTCOMPILE = False # PyTorch 2.0 compile, set to True for potential speedup

# --- Tokenizer Loading ---
print(f"Loading tokenizer from {TOKENIZER_PATH}...")
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
assert tokenizer.get_vocab_size() == VOCAB_SIZE, f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model VOCAB_SIZE {VOCAB_SIZE}"
print("Tokenizer loaded.")

# --- Dataset and DataLoader ---
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size):
        self.block_size = block_size
        self.tokenizer = tokenizer
        print(f"Reading and tokenizing data from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Tokenize the entire text
        # For very large files, consider streaming and tokenizing in chunks
        tokenized_output = self.tokenizer.encode(text)
        self.tokens = torch.tensor(tokenized_output.ids, dtype=torch.long)
        print(f"Loaded {len(self.tokens)} tokens.")
        if len(self.tokens) < block_size + 1:
            raise ValueError(f"Corpus too small for block_size {block_size}. Need at least {block_size + 1} tokens, got {len(self.tokens)}")

    def __len__(self):
        # Number of possible sequences of block_size
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        # Grab a chunk of (block_size + 1) tokens
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

print("Initializing dataset and dataloader...")
train_dataset = TextDataset(CORPUS_PATH, tokenizer, BLOCK_SIZE)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
print(f"Dataset size: {len(train_dataset)} samples. Dataloader ready.")

# --- Model Instantiation ---
model_args = dict(n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, block_size=BLOCK_SIZE,
                  bias=False, vocab_size=VOCAB_SIZE, dropout=0.1) # Add dropout if desired
print(f"Initializing model with args: {model_args}")
model = GPT(**model_args)
model.to(DEVICE)
print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

if PTCOMPILE:
    print("Compiling the model... (PyTorch 2.0)")
    try:
        model = torch.compile(model) # requires PyTorch 2.0
        print("Model compiled.")
    except Exception as e:
        print(f"Model compilation failed: {e}. Running uncompiled.")
        PTCOMPILE = False


# --- Optimizer ---
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)
print("Optimizer AdamW initialized.")

# --- Mixed Precision Training ---
scaler = torch.cuda.amp.GradScaler(enabled=(DTYPE == torch.float16 and DEVICE == 'cuda'))
print(f"Mixed precision scaler enabled: {scaler.is_enabled()}")
ctx = torch.amp.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DTYPE != torch.float32 and DEVICE == 'cuda'))


# --- Training Loop ---
def train():
    model.train()
    iter_num = 0
    best_loss = float('inf') # For saving the best model based on training loss (simple)

    for epoch in range(NUM_EPOCHS):
        print(f"Starting Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_start_time = time.time()
        
        for step, (X, Y) in enumerate(train_dataloader):
            iter_start_time = time.time()
            X, Y = X.to(DEVICE), Y.to(DEVICE)

            # Forward pass with autocasting
            with ctx:
                logits, loss = model(X, Y)
            
            loss = loss / GRADIENT_ACCUMULATION_STEPS # Scale loss for accumulation

            # Backward pass
            scaler.scale(loss).backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # Clip gradients
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Optional
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                iter_num += 1

                if iter_num % LOG_INTERVAL == 0:
                    current_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS # Unscale for logging
                    iter_end_time = time.time()
                    print(f"Epoch {epoch+1}, Iter {iter_num}, Loss: {current_loss:.4f}, Time/iter: {(iter_end_time - iter_start_time)*GRADIENT_ACCUMULATION_STEPS:.2f}s")
                
                if iter_num % SAVE_INTERVAL == 0:
                    checkpoint_path = os.path.join(MODEL_OUTPUT_DIR, f"ckpt_iter_{iter_num}.pt")
                    print(f"Saving checkpoint to {checkpoint_path}")
                    torch.save({
                        'iter_num': iter_num,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': current_loss, # Log the unscaled loss
                        'model_args': model_args,
                    }, checkpoint_path)

                # Simple evaluation based on training loss (can be expanded with a validation set)
                if iter_num % EVAL_INTERVAL == 0:
                    # For now, just use current training loss as a proxy
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_model_path = os.path.join(MODEL_OUTPUT_DIR, "best_model.pt")
                        print(f"New best training loss: {best_loss:.4f}. Saving model to {best_model_path}")
                        torch.save({
                            'iter_num': iter_num,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': best_loss,
                            'model_args': model_args,
                        }, best_model_path)
            
            # if MAX_ITERS is not None and iter_num >= MAX_ITERS:
            #     print(f"Reached max_iters ({MAX_ITERS}). Stopping training.")
            #     return

        epoch_end_time = time.time()
        print(f"Epoch {epoch+1} completed in {(epoch_end_time - epoch_start_time)/60:.2f} minutes.")

    # Final save
    final_model_path = os.path.join(MODEL_OUTPUT_DIR, "final_model.pt")
    print(f"Training finished. Saving final model to {final_model_path}")
    torch.save({
        'iter_num': iter_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item() * GRADIENT_ACCUMULATION_STEPS, # Last unscaled loss
        'model_args': model_args,
    }, final_model_path)

# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    if DTYPE != torch.float32 and DEVICE == 'cuda':
        print(f"Using mixed precision with dtype: {DTYPE}")
    
    # Check if corpus file exists
    if not os.path.exists(CORPUS_PATH):
        print(f"ERROR: Pretraining corpus not found at {CORPUS_PATH}")
        print("Please ensure the wikitext-103_train_corpus.txt (or your chosen corpus) is correctly placed.")
        exit(1)
    
    # Check if tokenizer file exists
    if not os.path.exists(TOKENIZER_PATH):
        print(f"ERROR: Tokenizer file not found at {TOKENIZER_PATH}")
        print("Please ensure the custom_gpt_tokenizer.json is correctly placed.")
        exit(1)

    train()
    print("Pre-training script finished.")