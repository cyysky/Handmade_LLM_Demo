import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import os
import time
import math
import glob
from model import GPT

# --- Configuration ---
# Paths
TOKENIZER_PATH = "data/tokenizer/custom_gpt_tokenizer.json"
CORPUS_PATH = "data/pretraining_data/wikitext-103_train_corpus.txt"
TEST_CORPUS_PATH = "data/pretraining_data/wikitext-103_test_corpus.txt"  # Add test corpus
CACHED_DATA_PATH = "data/pretraining_data/wikitext-103_train_corpus.pt"
CACHED_TEST_DATA_PATH = "data/pretraining_data/wikitext-103_test_corpus.pt"  # Add test cache
MODEL_OUTPUT_DIR = "out/pretrain/"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CACHED_DATA_PATH), exist_ok=True)
RESUME_FROM_CHECKPOINT = "auto"  # "auto" to find latest, "best" for best model, path for specific, None for fresh start

# Optimized configuration for 10GB VRAM
BATCH_SIZE = 8  # Start with 8, can try 12-16 if memory allows
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 32
BLOCK_SIZE = 512  # Reduce from 1024 to 512 for memory savings
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
NUM_EPOCHS = 1
EVAL_INTERVAL = 1000  # Reduce evaluation frequency
LOG_INTERVAL = 25
SAVE_INTERVAL = 1000
GRAD_CLIP = 1.0
WARMUP_ITERS = 100

# Model Configuration - Smaller model for 10GB
VOCAB_SIZE = 32000
N_LAYER = 8  # Reduce from 12 to 8 layers
N_HEAD = 8   # Reduce from 12 to 8 heads  
N_EMBD = 512 # Reduce from 768 to 512

# Device Configuration - Support specific CUDA devices
DEVICE_ID = "cuda:0"  # Change this to "cuda:4" or any specific device you want
DEVICE = DEVICE_ID if torch.cuda.is_available() and DEVICE_ID.startswith("cuda") else "cpu"
if DEVICE != "cpu":
    # Validate the specific CUDA device exists
    try:
        torch.cuda.set_device(DEVICE)
        print(f"Using device: {DEVICE}")
    except:
        print(f"Warning: {DEVICE} not available, falling back to cuda:0 or cpu")
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
PTCOMPILE = False

# --- Helper Functions ---
def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory"""
    checkpoint_pattern = os.path.join(output_dir, "ckpt_iter_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        return None
    
    # Extract iteration numbers and find the latest
    latest_checkpoint = None
    latest_iter = -1
    
    for checkpoint_file in checkpoint_files:
        try:
            # Extract iteration number from filename
            iter_num = int(os.path.basename(checkpoint_file).split('_')[-1].split('.')[0])
            if iter_num > latest_iter:
                latest_iter = iter_num
                latest_checkpoint = checkpoint_file
        except:
            continue
    
    return latest_checkpoint

# --- Learning Rate Schedule ---
def get_lr(iter_num, max_iters, learning_rate, warmup_iters=100):
    """Learning rate schedule with warmup and cosine decay"""
    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters
    return learning_rate * 0.5 * (1 + math.cos(math.pi * (iter_num - warmup_iters) / (max_iters - warmup_iters)))

# --- Tokenizer Loading ---
print(f"Loading tokenizer from {TOKENIZER_PATH}...")
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
assert tokenizer.get_vocab_size() == VOCAB_SIZE, f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model VOCAB_SIZE {VOCAB_SIZE}"
print("Tokenizer loaded.")

# --- Dataset and DataLoader ---
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size, cache_path):
        self.block_size = block_size
        self.tokenizer = tokenizer

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
            
            print("Tokenizing text (this may take a while)...")
            tokenized_output = self.tokenizer.encode(text, add_special_tokens=False)
            self.tokens = torch.tensor(tokenized_output.ids, dtype=torch.long)
            print(f"Tokenized and loaded {len(self.tokens)} tokens.")
            print(f"Saving tokenized data to cache: {cache_path}")
            torch.save(self.tokens, cache_path)
            print("Cached data saved.")

        if len(self.tokens) < block_size + 1:
            raise ValueError(f"Corpus too small for block_size {block_size}. Need at least {block_size + 1} tokens, got {len(self.tokens)}")

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

print("Initializing dataset and dataloader...")
train_dataset = TextDataset(CORPUS_PATH, tokenizer, BLOCK_SIZE, CACHED_DATA_PATH)
# Optimized DataLoader settings
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=2,  # Reduce workers to save CPU memory
    pin_memory=False,  # Disable pin_memory to save GPU memory
    persistent_workers=False,  # Don't keep workers persistent
    prefetch_factor=1  # Reduce prefetch
)
# Initialize test dataset if test corpus exists
test_dataset = None
test_dataloader = None
if os.path.exists(TEST_CORPUS_PATH):
    print("Initializing test dataset...")
    test_dataset = TextDataset(TEST_CORPUS_PATH, tokenizer, BLOCK_SIZE, CACHED_TEST_DATA_PATH)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Test dataset size: {len(test_dataset)} samples.")
else:
    print(f"Test corpus not found at {TEST_CORPUS_PATH}. Evaluation will use training loss.")

print(f"Training dataset size: {len(train_dataset)} samples. Dataloader ready.")

# --- Evaluation Function ---
@torch.no_grad()
def evaluate(model, dataloader, max_eval_batches=100):
    """Evaluate model on given dataloader"""
    if dataloader is None:
        return float('inf')
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    try:
        for batch_idx, (X, Y) in enumerate(dataloader):
            if batch_idx >= max_eval_batches:
                break
                
            X, Y = X.to(DEVICE, non_blocking=True), Y.to(DEVICE, non_blocking=True)
            
            with ctx:
                logits, loss = model(X, Y)
            
            # Check for NaN or Inf values
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss detected at batch {batch_idx}: {loss.item()}")
                continue
                
            total_loss += loss.item()
            num_batches += 1
            
            # Clear cache every few batches during evaluation
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        model.train()
        return float('inf')
    
    model.train()
    
    if num_batches == 0:
        print("Warning: No valid batches processed during evaluation")
        return float('inf')
        
    avg_loss = total_loss / num_batches
    print(f"Evaluation completed: {num_batches} batches processed, avg loss: {avg_loss:.4f}")
    return avg_loss

# --- Checkpoint Loading and Model/Optimizer Initialization ---
iter_num = 0
best_loss = float('inf')
model = None
optimizer = None
scaler = None
active_model_args = None

# Determine checkpoint path with improved logic
checkpoint_path_to_load = None
if RESUME_FROM_CHECKPOINT:
    if RESUME_FROM_CHECKPOINT == "auto":
        checkpoint_path_to_load = find_latest_checkpoint(MODEL_OUTPUT_DIR)
        if checkpoint_path_to_load:
            print(f"Auto-detected latest checkpoint: {checkpoint_path_to_load}")
        else:
            print("No checkpoints found for auto-resume.")
    elif RESUME_FROM_CHECKPOINT == "best":
        checkpoint_path_to_load = os.path.join(MODEL_OUTPUT_DIR, "best_model.pt")
    else:
        checkpoint_path_to_load = RESUME_FROM_CHECKPOINT

# Attempt to load checkpoint
if checkpoint_path_to_load and os.path.exists(checkpoint_path_to_load):
    print(f"Resuming training from checkpoint: {checkpoint_path_to_load}")
    checkpoint = torch.load(checkpoint_path_to_load, map_location=DEVICE)
    
    # Use model_args from checkpoint
    loaded_checkpoint_model_args = checkpoint['model_args']
    active_model_args = loaded_checkpoint_model_args
    print(f"Initializing model from checkpoint with args: {active_model_args}")
    model = GPT(**active_model_args)
    
    # Load model state
    state_dict = checkpoint['model_state_dict']
    # Remove DDP prefix if present
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    
    iter_num = checkpoint.get('iter_num', 0)
    best_loss = checkpoint.get('loss', float('inf'))
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded optimizer state from checkpoint. Moving optimizer state to device...")
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)
        print("Optimizer state moved to device.")
    else:
        print("Optimizer state not found in checkpoint, initializing new optimizer.")
    
    # Initialize scaler
    scaler = torch.amp.GradScaler(enabled=(DTYPE == torch.float16 and DEVICE == 'cuda'))
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    print(f"Resumed from iter {iter_num}, best loss {best_loss:.4f}")

else:
    if RESUME_FROM_CHECKPOINT:
        print(f"Checkpoint not found at {checkpoint_path_to_load}. Starting from scratch.")
    else:
        print("No checkpoint specified. Starting from scratch.")
    
    # Initialize new model
    script_default_model_args = dict(
        n_layer=N_LAYER, 
        n_head=N_HEAD, 
        n_embd=N_EMBD, 
        block_size=BLOCK_SIZE,
        bias=False, 
        vocab_size=VOCAB_SIZE, 
        dropout=0.1
    )
    active_model_args = script_default_model_args
    print(f"Initializing new model with args: {active_model_args}")
    model = GPT(**active_model_args)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    
    # Initialize scaler
    scaler = torch.amp.GradScaler(enabled=(DTYPE == torch.float16 and DEVICE == 'cuda'))

model.to(DEVICE)
print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

if PTCOMPILE and model is not None:
    print("Compiling the model... (PyTorch 2.0)")
    try:
        model = torch.compile(model)
        print("Model compiled.")
    except Exception as e:
        print(f"Model compilation failed: {e}. Running uncompiled.")
        PTCOMPILE = False

print("Optimizer AdamW initialized (or loaded).")

# --- Mixed Precision Training ---
autocast_enabled = (DEVICE == 'cuda' and DTYPE != torch.float32)
print(f"Mixed precision scaler enabled: {scaler.is_enabled()}")
ctx = torch.amp.autocast(device_type=DEVICE, dtype=DTYPE, enabled=autocast_enabled)

# Calculate max iterations for learning rate scheduling
max_iters = NUM_EPOCHS * (len(train_dataloader) // GRADIENT_ACCUMULATION_STEPS)

def log_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3     # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

# --- Training Loop ---
def train(start_iter_num, current_best_loss):
    model.train()
    iter_num_local = start_iter_num
    best_loss_local = current_best_loss
    total_loss = 0.0
    
    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    total_steps_per_epoch = len(train_dataloader) // GRADIENT_ACCUMULATION_STEPS
    total_dataset_size = len(train_dataset)

    for epoch in range(NUM_EPOCHS):
        print(f"Starting Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_start_time = time.time()
        
        for step, (X, Y) in enumerate(train_dataloader):
            X, Y = X.to(DEVICE, non_blocking=True), Y.to(DEVICE, non_blocking=True)

            # Calculate progress metrics
            current_step = (step + 1) // GRADIENT_ACCUMULATION_STEPS
            progress_percent = (current_step / total_steps_per_epoch) * 100

            # Dynamic learning rate
            lr = get_lr(iter_num_local, max_iters, LEARNING_RATE, WARMUP_ITERS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Forward pass with autocasting
            with ctx:
                logits, loss = model(X, Y)
            
            loss_for_log_and_eval = loss.item()
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            total_loss += loss.item()

            # Backward pass
            scaler.scale(loss).backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # Clip gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                iter_num_local += 1

                # Clear cache periodically to prevent memory fragmentation
                if iter_num_local % 50 == 0:
                    torch.cuda.empty_cache()

                if iter_num_local % LOG_INTERVAL == 0:
                    avg_loss = total_loss * GRADIENT_ACCUMULATION_STEPS / LOG_INTERVAL
                    total_loss = 0.0
                    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Step {current_step}/{total_steps_per_epoch} ({progress_percent:.2f}%), "
                          f"Iter {iter_num_local}, Loss: {avg_loss:.4f}, LR: {lr:.8f}")
                    log_memory_usage()  # Monitor memory usage
                
                                # Evaluation on test set every EVAL_INTERVAL
                if iter_num_local % EVAL_INTERVAL == 0:
                    print("Evaluating...")
                    
                    # Clear cache before evaluation
                    torch.cuda.empty_cache()
                    
                    # Evaluate on smaller batches to avoid memory issues
                    train_loss = evaluate(model, train_dataloader, max_eval_batches=20)
                    test_loss = evaluate(model, test_dataloader, max_eval_batches=50) if test_dataloader else float('inf')
                    
                    print(f"Iter {iter_num_local} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
                    
                    # Use test loss for best model selection if available and valid, otherwise train loss
                    if test_dataloader is not None and not (torch.isinf(torch.tensor(test_loss)) or torch.isnan(torch.tensor(test_loss))):
                        eval_loss = test_loss
                        loss_type = "test"
                    else:
                        eval_loss = train_loss
                        loss_type = "train"
                        if test_dataloader is not None:
                            print("Warning: Test loss is invalid, using train loss for model selection")
                    
                    if eval_loss < best_loss_local and not (torch.isinf(torch.tensor(eval_loss)) or torch.isnan(torch.tensor(eval_loss))):
                        best_loss_local = eval_loss
                        best_model_path = os.path.join(MODEL_OUTPUT_DIR, "best_model.pt")
                        print(f"New best {loss_type} loss: {best_loss_local:.4f}. Saving model to {best_model_path}")
                        torch.save({
                            'iter_num': iter_num_local,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'loss': best_loss_local,
                            'test_loss': test_loss if test_loss != float('inf') else None,
                            'train_loss': train_loss,
                            'model_args': active_model_args,
                        }, best_model_path)
                    
                    # Clear cache after evaluation
                    torch.cuda.empty_cache()
                
                if iter_num_local % SAVE_INTERVAL == 0:
                    checkpoint_save_path = os.path.join(MODEL_OUTPUT_DIR, f"ckpt_iter_{iter_num_local}.pt")
                    print(f"Saving checkpoint to {checkpoint_save_path}")
                    torch.save({
                        'iter_num': iter_num_local,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'loss': loss_for_log_and_eval,
                        'model_args': active_model_args,
                    }, checkpoint_save_path)

        epoch_end_time = time.time()
        epoch_progress = ((epoch + 1) / NUM_EPOCHS) * 100
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} completed ({epoch_progress:.1f}% of training) in {(epoch_end_time - epoch_start_time)/60:.2f} minutes.")

    # Final save after all epochs
    final_model_path = os.path.join(MODEL_OUTPUT_DIR, f"final_model_epochs_done.pt")
    print(f"Training finished after {NUM_EPOCHS} epochs. Saving final model to {final_model_path}")
    torch.save({
        'iter_num': iter_num_local,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss_for_log_and_eval,
        'model_args': active_model_args,
    }, final_model_path)
    return iter_num_local, best_loss_local

# --- Main Execution Block ---
if __name__ == "__main__":
    # Safeguard checks
    if model is None or optimizer is None or active_model_args is None:
        print("Warning: Model, optimizer, or active_model_args not set. Initializing with script defaults.")
        if active_model_args is None:
             active_model_args = dict(
                 n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, block_size=BLOCK_SIZE,
                 bias=False, vocab_size=VOCAB_SIZE, dropout=0.1
             )
        if model is None:
            model = GPT(**active_model_args)
            model.to(DEVICE)
        if optimizer is None:
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
        if scaler is None:
            scaler = torch.amp.GradScaler(enabled=(DTYPE == torch.float16 and DEVICE == 'cuda'))

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

    print(f"Starting pretraining with max_iters: {max_iters}")
    final_iter_num, final_best_loss = train(iter_num, best_loss)
    print(f"Pre-training script finished. Final iter: {final_iter_num}, Final best loss: {final_best_loss:.4f}")