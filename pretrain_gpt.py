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
CACHED_DATA_PATH = "data/pretraining_data/wikitext-103_train_corpus.pt" # Path for cached tokenized data
MODEL_OUTPUT_DIR = "out/pretrain/"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CACHED_DATA_PATH), exist_ok=True) # Ensure directory for cache exists
RESUME_FROM_CHECKPOINT = None # Path to checkpoint file (e.g., "out/pretrain/best_model.pt" or "out/pretrain/ckpt_iter_X.pt") or "best"
# RESUME_FROM_CHECKPOINT = "out/pretrain/best_model.pt" # Example: Resume from best model
# RESUME_FROM_CHECKPOINT = "best" # Example: Resume from best_model.pt in MODEL_OUTPUT_DIR

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
    def __init__(self, file_path, tokenizer, block_size, cache_path):
        self.block_size = block_size
        self.tokenizer = tokenizer # Keep tokenizer for potential future use, though not strictly needed if only using cached IDs

        if os.path.exists(cache_path):
            print(f"Loading tokenized data from cache: {cache_path}")
            self.tokens = torch.load(cache_path)
            print(f"Loaded {len(self.tokens)} tokens from cache.")
        else:
            print(f"Cache not found. Reading and tokenizing data from {file_path}...")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Corpus file not found: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Tokenize the entire text
            # For very large files, consider streaming and tokenizing in chunks
            print("Tokenizing text (this may take a while)...")
            tokenized_output = self.tokenizer.encode(text) # Use the passed tokenizer instance
            self.tokens = torch.tensor(tokenized_output.ids, dtype=torch.long)
            print(f"Tokenized and loaded {len(self.tokens)} tokens.")
            print(f"Saving tokenized data to cache: {cache_path}")
            torch.save(self.tokens, cache_path)
            print("Cached data saved.")

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
train_dataset = TextDataset(CORPUS_PATH, tokenizer, BLOCK_SIZE, CACHED_DATA_PATH)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
print(f"Dataset size: {len(train_dataset)} samples. Dataloader ready.")

# --- Checkpoint Loading and Model/Optimizer Initialization ---
# Define initial iter_num and best_loss, may be overridden by checkpoint
iter_num = 0
best_loss = float('inf')
model = None
optimizer = None
active_model_args = None # Will store the actual model args used

# Determine checkpoint path
checkpoint_path_to_load = None
if RESUME_FROM_CHECKPOINT:
    if RESUME_FROM_CHECKPOINT == "best":
        checkpoint_path_to_load = os.path.join(MODEL_OUTPUT_DIR, "best_model.pt")
    else:
        checkpoint_path_to_load = RESUME_FROM_CHECKPOINT

# Attempt to load checkpoint
if checkpoint_path_to_load and os.path.exists(checkpoint_path_to_load):
    print(f"Resuming training from checkpoint: {checkpoint_path_to_load}")
    checkpoint = torch.load(checkpoint_path_to_load, map_location=DEVICE)
    
    # Use model_args from checkpoint
    # Use model_args from checkpoint
    loaded_checkpoint_model_args = checkpoint['model_args']
    active_model_args = loaded_checkpoint_model_args # Set active_model_args
    print(f"Initializing model from checkpoint with args: {active_model_args}")
    model = GPT(**active_model_args)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    iter_num = checkpoint.get('iter_num', 0) # Use .get for backward compatibility if iter_num wasn't saved
    best_loss = checkpoint.get('loss', float('inf')) # Use 'loss' as best_loss from checkpoint
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1) # Re-init with potentially new LR
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded optimizer state from checkpoint. Moving optimizer state to device...")
        # Move optimizer state to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)
        print("Optimizer state moved to device.")
    else:
        print("Optimizer state not found in checkpoint, initializing new optimizer.")
    
    print(f"Resumed from iter {iter_num}, best loss {best_loss:.4f}")

else:
    if RESUME_FROM_CHECKPOINT:
        print(f"Checkpoint not found at {checkpoint_path_to_load}. Starting from scratch.")
    else:
        print("No checkpoint specified. Starting from scratch.")
    
    # --- Model Instantiation (if not resuming) ---
    script_default_model_args = dict(n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, block_size=BLOCK_SIZE,
                                     bias=False, vocab_size=VOCAB_SIZE, dropout=0.1) # Add dropout if desired
    active_model_args = script_default_model_args # Set active_model_args
    print(f"Initializing new model with args: {active_model_args}")
    model = GPT(**active_model_args)
    
    # --- Optimizer (if not resuming) ---
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)

model.to(DEVICE)
print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

if PTCOMPILE and model is not None: # Ensure model is initialized
    print("Compiling the model... (PyTorch 2.0)")
    try:
        model = torch.compile(model) # requires PyTorch 2.0
        print("Model compiled.")
    except Exception as e:
        print(f"Model compilation failed: {e}. Running uncompiled.")
        PTCOMPILE = False

print("Optimizer AdamW initialized (or loaded).")

# --- Mixed Precision Training ---
autocast_enabled = (DEVICE == 'cuda' and DTYPE != torch.float32)
scaler = torch.amp.GradScaler(enabled=autocast_enabled) # Use torch.amp.GradScaler
print(f"Mixed precision scaler enabled: {scaler.is_enabled()}")
ctx = torch.amp.autocast(device_type=DEVICE, dtype=DTYPE, enabled=autocast_enabled)


# --- Training Loop ---
# iter_num and best_loss are now defined in the global scope and potentially loaded from checkpoint
def train(start_iter_num, current_best_loss):
    model.train()
    # Use the passed-in values
    iter_num_local = start_iter_num
    best_loss_local = current_best_loss

    for epoch in range(NUM_EPOCHS): # NUM_EPOCHS might need adjustment if resuming mid-epoch effectively
        print(f"Starting Epoch {epoch+1}/{NUM_EPOCHS} (Effective start iter: {iter_num_local})")
        epoch_start_time = time.time()
        
        for step, (X, Y) in enumerate(train_dataloader):
            iter_start_time = time.time()
            X, Y = X.to(DEVICE), Y.to(DEVICE)

            # Forward pass with autocasting
            with ctx:
                logits, loss = model(X, Y)
            
            loss_for_log_and_eval = loss.item() # Get Python number for loss before scaling
            loss = loss / GRADIENT_ACCUMULATION_STEPS # Scale loss for accumulation

            # Backward pass
            scaler.scale(loss).backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # Clip gradients
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Optional
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                iter_num_local += 1

                if iter_num_local % LOG_INTERVAL == 0:
                    iter_end_time = time.time()
                    print(f"Epoch {epoch+1}, Iter {iter_num_local}, Loss: {loss_for_log_and_eval:.4f}, Time/iter: {(iter_end_time - iter_start_time)*GRADIENT_ACCUMULATION_STEPS:.2f}s")
                
                if iter_num_local % SAVE_INTERVAL == 0:
                    checkpoint_save_path = os.path.join(MODEL_OUTPUT_DIR, f"ckpt_iter_{iter_num_local}.pt")
                    print(f"Saving checkpoint to {checkpoint_save_path}")
                    torch.save({
                        'iter_num': iter_num_local,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_for_log_and_eval,
                        'model_args': active_model_args, # Save the actual model args used
                    }, checkpoint_save_path)

                # Simple evaluation based on training loss (can be expanded with a validation set)
                if iter_num_local % EVAL_INTERVAL == 0:
                    if loss_for_log_and_eval < best_loss_local:
                        best_loss_local = loss_for_log_and_eval
                        best_model_path = os.path.join(MODEL_OUTPUT_DIR, "best_model.pt")
                        print(f"New best training loss: {best_loss_local:.4f}. Saving model to {best_model_path}")
                        torch.save({
                            'iter_num': iter_num_local,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': best_loss_local,
                            'model_args': active_model_args, # Save the actual model args used
                        }, best_model_path)
            
            # if MAX_ITERS is not None and iter_num_local >= MAX_ITERS:
            #     print(f"Reached max_iters ({MAX_ITERS}). Stopping training.")
            #     # Consider saving final model here if MAX_ITERS is the primary condition
            #     return iter_num_local, best_loss_local

        epoch_end_time = time.time()
        print(f"Epoch {epoch+1} completed in {(epoch_end_time - epoch_start_time)/60:.2f} minutes.")

    # Final save after all epochs
    final_model_path = os.path.join(MODEL_OUTPUT_DIR, "final_model_epochs_done.pt")
    print(f"Training finished after {NUM_EPOCHS} epochs. Saving final model to {final_model_path}")
    torch.save({
        'iter_num': iter_num_local,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_for_log_and_eval, # Last logged loss
        'model_args': active_model_args, # Save the actual model args used
    }, final_model_path)
    return iter_num_local, best_loss_local

# --- Main Execution Block ---
if __name__ == "__main__":
    # Safeguard: Ensure model, optimizer, and active_model_args are initialized if RESUME_FROM_CHECKPOINT was set but failed.
    # The logic above should handle this, but this is an extra check.
    if model is None or optimizer is None or active_model_args is None:
        print("Warning: Model, optimizer, or active_model_args not set. Initializing with script defaults.")
        # This implies RESUME_FROM_CHECKPOINT might have been set, but failed, and the 'else' for fresh start wasn't hit as expected.
        # Or, RESUME_FROM_CHECKPOINT was None, and the fresh start logic had an issue.
        # Re-initialize active_model_args for a fresh start if it's None
        if active_model_args is None:
             active_model_args = dict(n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, block_size=BLOCK_SIZE,
                                     bias=False, vocab_size=VOCAB_SIZE, dropout=0.1)
        if model is None:
            model = GPT(**active_model_args)
            model.to(DEVICE)
        if optimizer is None:
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)


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

    final_iter_num, final_best_loss = train(iter_num, best_loss)
    print(f"Pre-training script finished. Final iter: {final_iter_num}, Final best loss: {final_best_loss:.4f}")