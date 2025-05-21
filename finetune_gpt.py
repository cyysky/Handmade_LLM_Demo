import os
import time
import json
import math
import argparse
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from model import GPT # Assuming model.py is in the same directory or PYTHONPATH

# --- Constants for prompt formatting ---
PROMPT_INSTRUCTION_KEY = "Instruction: "
PROMPT_INPUT_KEY = "Input: "
PROMPT_RESPONSE_KEY = "Response: "
IGNORE_INDEX = -1 # Should match the ignore_index in model.py's CrossEntropyLoss

# --- Special Token Strings ---
# These should match the special tokens used in your tokenizer
BOS_TOKEN_STRING = "[BOS]"
EOS_TOKEN_STRING = "[EOS]"
PAD_TOKEN_STRING = "[PAD]" # Common pad token, or use EOS if not present

class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.bos_id = self.tokenizer.token_to_id(BOS_TOKEN_STRING)
        self.eos_id = self.tokenizer.token_to_id(EOS_TOKEN_STRING)

        if self.bos_id is None:
            raise ValueError(f"BOS token '{BOS_TOKEN_STRING}' not found in tokenizer vocabulary.")
        if self.eos_id is None:
            raise ValueError(f"EOS token '{EOS_TOKEN_STRING}' not found in tokenizer vocabulary.")

        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item['instruction']
        input_text = item.get('input', '') # Use .get for optional input
        output_text = item['output']

        if input_text:
            prompt_prefix_str = f"{BOS_TOKEN_STRING}{PROMPT_INSTRUCTION_KEY}{instruction} {PROMPT_INPUT_KEY}{input_text} {PROMPT_RESPONSE_KEY}"
        else:
            prompt_prefix_str = f"{BOS_TOKEN_STRING}{PROMPT_INSTRUCTION_KEY}{instruction} {PROMPT_RESPONSE_KEY}"
        
        full_str = f"{prompt_prefix_str}{output_text}{EOS_TOKEN_STRING}"

        # Tokenize the full string (without adding special tokens automatically, as we've manually added them)
        encoded_full = self.tokenizer.encode(full_str, add_special_tokens=False)
        input_ids = encoded_full.ids

        # Tokenize the prefix part to determine how much to mask
        # We don't add eos_token to prefix_str for masking calculation
        encoded_prefix = self.tokenizer.encode(prompt_prefix_str, add_special_tokens=False)
        len_prefix_tokens = len(encoded_prefix.ids)

        # Create labels
        labels = list(input_ids) # Make a copy
        labels[:len_prefix_tokens] = [IGNORE_INDEX] * len_prefix_tokens
        
        # Truncate if longer than block_size
        if len(input_ids) > self.block_size:
            input_ids = input_ids[:self.block_size]
            labels = labels[:self.block_size]
        
        # Ensure the last token is EOS if truncated within response, this is complex.
        # For simplicity, we just truncate. If block_size is too small, this can be an issue.
        # A more robust way would be to ensure EOS is preserved if possible.

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def collate_fn(batch, pad_token_id, ignore_index):
    input_ids, labels = zip(*batch)

    max_len = max(len(ids) for ids in input_ids)

    padded_input_ids = torch.full((len(input_ids), max_len), pad_token_id, dtype=torch.long)
    padded_labels = torch.full((len(labels), max_len), ignore_index, dtype=torch.long)

    for i, ids in enumerate(input_ids):
        padded_input_ids[i, :len(ids)] = ids
    for i, lbls in enumerate(labels):
        padded_labels[i, :len(lbls)] = lbls
        
    return padded_input_ids, padded_labels

def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune a GPT model on instruction data.")
    
    # Paths
    parser.add_argument('--tokenizer_path', type=str, default="data/tokenizer/custom_gpt_tokenizer.json", help="Path to the trained tokenizer file.")
    parser.add_argument('--data_path', type=str, default="data/finetuning_data/dolly_15k_instructions.jsonl", help="Path to the fine-tuning data (JSONL).")
    parser.add_argument('--pretrained_ckpt_path', type=str, required=True, help="Path to the pre-trained model checkpoint (.pt file).")
    parser.add_argument('--out_dir', type=str, default="out/finetune", help="Output directory for saving fine-tuned models and logs.")

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=3e-5, help="Learning rate for AdamW optimizer.")
    parser.add_argument('--num_epochs', type=int, default=1, help="Number of training epochs.")
    parser.add_argument('--max_iters', type=int, default=-1, help="Maximum training iterations. Overrides num_epochs if set > 0.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help="Number of steps to accumulate gradients before an optimizer step.")
    
    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=10, help="Log training status every N iterations.")
    parser.add_argument('--save_interval', type=int, default=100, help="Save checkpoint every N iterations. Also saves at the end.") # Renamed from eval_interval for clarity

    # Device
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help="Device to use for training (cuda or cpu).")
    
    # Mixed precision
    parser.add_argument('--dtype', type=str, default='bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16', help='float16 or bfloat16')

    return parser.parse_args()

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = 'cpu'
    
    torch.manual_seed(1337) # For reproducibility
    if device == 'cuda':
        torch.cuda.manual_seed(1337)

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

    # --- Tokenizer Loading ---
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    # Determine pad_token_id
    pad_token_id = tokenizer.token_to_id(PAD_TOKEN_STRING)
    eos_token_id_for_pad_fallback = tokenizer.token_to_id(EOS_TOKEN_STRING) # Get EOS ID for fallback

    if pad_token_id is None:
        print(f"Tokenizer does not have a '{PAD_TOKEN_STRING}' token. Attempting to use EOS token '{EOS_TOKEN_STRING}' as PAD token.")
        if eos_token_id_for_pad_fallback is not None:
            pad_token_id = eos_token_id_for_pad_fallback
            print(f"Using EOS token (ID: {pad_token_id}) as PAD token.")
        else:
            # This should have been caught by InstructionDataset init if EOS_TOKEN_STRING is invalid
            raise ValueError(f"Neither '{PAD_TOKEN_STRING}' nor '{EOS_TOKEN_STRING}' (for fallback) found in tokenizer for PAD token.")
    
    if pad_token_id is None: # Should be redundant
        raise ValueError("Could not determine a PAD token ID.")

    print(f"Tokenizer loaded. Vocab size: {tokenizer.get_vocab_size()}. Using PAD token ID: {pad_token_id}")
    
    # Get BOS/EOS IDs for logging (already validated in InstructionDataset if it's instantiated first)
    bos_id_for_log = tokenizer.token_to_id(BOS_TOKEN_STRING)
    eos_id_for_log = tokenizer.token_to_id(EOS_TOKEN_STRING)
    
    # It's possible InstructionDataset is not initialized before this logging if there's an early exit,
    # so we check IDs here too for robustness of logging.
    if bos_id_for_log is None: print(f"Warning: BOS token '{BOS_TOKEN_STRING}' not found for logging.")
    if eos_id_for_log is None: print(f"Warning: EOS token '{EOS_TOKEN_STRING}' not found for logging.")

    print(f"BOS token: '{BOS_TOKEN_STRING}' (ID: {bos_id_for_log}), EOS token: '{EOS_TOKEN_STRING}' (ID: {eos_id_for_log})")


    # --- Model Instantiation and Checkpoint Loading ---
    print(f"Loading pre-trained model from {args.pretrained_ckpt_path}...")
    checkpoint = torch.load(args.pretrained_ckpt_path, map_location=device)
    model_args = checkpoint.get('model_args', None)
    if model_args is None:
        raise ValueError("Checkpoint must contain 'model_args' dictionary for model instantiation.")
    
    # Ensure vocab_size from model_args matches tokenizer, or update if necessary
    # This assumes pretraining used a compatible tokenizer vocab size.
    if model_args['vocab_size'] != tokenizer.get_vocab_size():
        print(f"Warning: Model vocab_size ({model_args['vocab_size']}) differs from tokenizer vocab_size ({tokenizer.get_vocab_size()}). Using model_args' vocab_size.")
        # This could be problematic if the embedding matrix size is wrong.
        # For fine-tuning, it's crucial they match or are handled (e.g. resizing embeddings).
        # For now, we trust model_args from pretraining.
        pass # vocab_size in model_args will be used for GPT instantiation.

    model = GPT(**model_args)
    state_dict = checkpoint['model_state_dict']
    
    # Fix potential issues with state_dict keys (e.g. from DataParallel)
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Model loaded. Trainable parameters: {model.get_num_params()/1e6:.2f}M")

    # --- Dataset and DataLoader ---
    # block_size should come from the loaded model configuration
    block_size = model_args['block_size'] 
    train_dataset = InstructionDataset(args.data_path, tokenizer, block_size)
    # The collate_fn needs pad_token_id and ignore_index
    collate_partial = lambda batch: collate_fn(batch, pad_token_id=pad_token_id, ignore_index=IGNORE_INDEX)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_partial, num_workers=0) # num_workers=0 for simplicity

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # If optimizer state was saved in checkpoint and you want to resume:
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # --- Training Loop ---
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16' and device == 'cuda'))
    
    if args.max_iters <= 0:
        args.max_iters = args.num_epochs * len(train_loader) // args.gradient_accumulation_steps
    
    print(f"Starting fine-tuning for {args.max_iters} iterations (effective epochs: {args.num_epochs}).")
    print(f"Batch size: {args.batch_size}, Grad accum steps: {args.gradient_accumulation_steps}, Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")

    iter_num = 0
    total_loss = 0.0
    model.train() # Set model to training mode

    # Determine starting epoch and iteration if resuming (not fully implemented here)
    # iter_num = checkpoint.get('iter_num', 0) 
    # best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    start_time = time.time()
    
    # Training loop
    for epoch in range(args.num_epochs): # Outer loop for epochs, though max_iters is the primary stop condition
        for step, (X, Y) in enumerate(train_loader):
            if iter_num >= args.max_iters:
                break

            X, Y = X.to(device), Y.to(device)

            # Forward pass with autocasting
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / args.gradient_accumulation_steps # Scale loss for accumulation

            # Backward pass
            scaler.scale(loss).backward()
            total_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Clip gradients (optional, but good practice)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Max norm
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                iter_num += 1

                if iter_num % args.log_interval == 0:
                    avg_loss = total_loss * args.gradient_accumulation_steps / args.log_interval # Correct averaging
                    total_loss = 0.0
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    print(f"Epoch {epoch+1}, Iter {iter_num}/{args.max_iters}, Loss: {avg_loss:.4f}, Time/iter: {elapsed_time*1000/args.log_interval:.2f}ms")
                    start_time = current_time # Reset start_time for next log_interval

                if iter_num % args.save_interval == 0 or iter_num == args.max_iters:
                    checkpoint_path = os.path.join(args.out_dir, f"ckpt_iter_{iter_num}.pt")
                    save_checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'config': vars(args) # Save training configuration
                    }
                    print(f"Saving checkpoint to {checkpoint_path}")
                    torch.save(save_checkpoint, checkpoint_path)
            
        if iter_num >= args.max_iters:
            print("Reached max_iters. Training finished.")
            break
    
    # Final save if not already saved at max_iters
    if iter_num % args.save_interval != 0 and iter_num == args.max_iters :
        checkpoint_path = os.path.join(args.out_dir, f"ckpt_final_iter_{iter_num}.pt")
        final_checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'config': vars(args)
        }
        print(f"Saving final checkpoint to {checkpoint_path}")
        torch.save(final_checkpoint, checkpoint_path)

if __name__ == '__main__':
    args = get_args()
    main(args)