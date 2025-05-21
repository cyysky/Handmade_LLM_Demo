import torch
import torch.nn.functional as F
import argparse
import json
import os
from model import GPT # Assuming model.py is in the same directory or accessible

# Define special token strings (ensure these match your tokenizer's definitions)
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
PAD_TOKEN = "[PAD]" # Though not explicitly used in generation logic here, good to be aware of

def load_tokenizer(tokenizer_path):
    """Loads the tokenizer data from a JSON file."""
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    # Assuming the JSON structure contains 'token_to_id' and 'id_to_token'
    # Adjust if your tokenizer JSON has a different structure (e.g., Hugging Face tokenizers export)
    if 'token_to_id' in tokenizer_data and 'id_to_token' in tokenizer_data:
        token_to_id = tokenizer_data['token_to_id']
        id_to_token = {int(k): v for k,v in tokenizer_data['id_to_token'].items()} # Ensure keys are int for id_to_token
    elif 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']: # Common for HF tokenizers saved with .save_pretrained() then json loaded
        vocab = tokenizer_data['model']['vocab']
        token_to_id = vocab
        id_to_token = {v: k for k, v in vocab.items()}
    else:
        raise ValueError("Tokenizer JSON structure not recognized. Expected 'token_to_id' and 'id_to_token', or HF-like structure.")

    # Ensure special tokens are in the vocabulary
    for token_str in [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]:
        if token_str not in token_to_id:
            print(f"Warning: Special token '{token_str}' not found in tokenizer vocabulary.")
            # Decide how to handle: error out, or assign a placeholder if your model expects it
            # For now, we'll assume they must exist if used in prompt formatting or EOS detection.

    return token_to_id, id_to_token

def generate_text(
    model,
    token_to_id,
    id_to_token,
    prompt_text,
    device,
    max_new_tokens=150,
    temperature=0.8,
    top_k=50
):
    """Generates text using the fine-tuned model."""
    model.eval()

    bos_id = token_to_id.get(BOS_TOKEN)
    eos_id = token_to_id.get(EOS_TOKEN)

    if bos_id is None:
        raise ValueError(f"BOS token '{BOS_TOKEN}' not found in tokenizer.")
    if eos_id is None:
        print(f"Warning: EOS token '{EOS_TOKEN}' not found in tokenizer. Generation might not stop correctly.")


    # Tokenize the prompt
    prompt_tokens = [token_to_id.get(token, token_to_id.get("[UNK]")) for token in prompt_text.split()] # Basic whitespace split
    # A more robust tokenizer would handle subwords, etc. This assumes pre-tokenized string or simple words.
    # If your tokenizer is more complex (e.g. BPE), you'd use its encode method.
    # For this script, we'll assume the prompt is constructed with known tokens.
    # A better approach for prompt tokenization:
    # tokenizer_encode_fn = lambda s: [token_to_id.get(t) for t in s.split()] # placeholder
    # For now, let's assume the prompt is already somewhat tokenized or simple.
    # A simple way to tokenize the prompt string:
    
    # For the given prompt structure, we need to be careful.
    # Let's assume the prompt_text is the full string including [BOS], Instruction, etc.
    # And we need to convert this string into a list of token IDs.
    # This part is highly dependent on how `custom_gpt_tokenizer.json` was created and what it can do.
    # If it's just a vocab map, manual tokenization is needed.
    
    # Simplified tokenization for the example prompt structure:
    # The prompt is expected to be like: "[BOS] Instruction: ... Response:"
    # We need to convert this string to IDs.
    # A robust solution would use the actual tokenizer's encoding logic.
    # For now, we'll manually construct the token list for the known parts and user input.
    # This is a placeholder for actual tokenization of the prompt string.
    # A better way: if tokenizer has an encode method: `start_ids = tokenizer.encode(prompt_text)`
    # Assuming simple space-based tokenization for the user-provided parts of the prompt.
    
    # Let's assume `prompt_text` is the full string to be tokenized.
    # A very basic tokenizer:
    words = prompt_text.split(' ') # Very naive
    start_ids = [token_to_id.get(word, token_to_id.get("[UNK]", 0)) for word in words if word]


    input_ids = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    generated_ids = input_ids.tolist()[0]

    for _ in range(max_new_tokens):
        # Crop context if it exceeds block size
        current_ids_cond = input_ids if input_ids.size(1) <= model.block_size else input_ids[:, -model.block_size:]
        
        with torch.no_grad():
            logits, _ = model(current_ids_cond) # (B, T, vocab_size)
        
        # Get logits for the last token
        logits = logits[:, -1, :] / temperature # (B, vocab_size)

        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1) # (B, vocab_size)
        next_token_id = torch.multinomial(probs, num_samples=1) # (B, 1)

        generated_ids.append(next_token_id.item())
        input_ids = torch.cat((input_ids, next_token_id), dim=1)

        if next_token_id.item() == eos_id:
            break
            
    # Decode generated tokens
    decoded_text = " ".join([id_to_token.get(token_id, "[UNK]") for token_id in generated_ids])
    # A better decoding would use the tokenizer's decode method to handle subwords, special tokens, and cleanup.
    # E.g. `tokenizer.decode(generated_ids)`
    
    return decoded_text

def main():
    parser = argparse.ArgumentParser(description="Generate text using a fine-tuned GPT model.")
    parser.add_argument('--checkpoint_path', type=str, default="out/finetune/ckpt.pt",
                        help="Path to the fine-tuned model checkpoint (.pt file).")
    parser.add_argument('--tokenizer_path', type=str, default="data/tokenizer/custom_gpt_tokenizer.json",
                        help="Path to the tokenizer JSON file.")
    parser.add_argument('--prompt', type=str, required=True,
                        help="User instruction for the model.")
    parser.add_argument('--input_text', type=str, default="",
                        help="Optional input text for the prompt (if your prompt structure uses it).")
    parser.add_argument('--max_new_tokens', type=int, default=150,
                        help="Maximum number of new tokens to generate.")
    parser.add_argument('--temperature', type=float, default=0.8,
                        help="Temperature for sampling (e.g., 0.7-0.9).")
    parser.add_argument('--top_k', type=int, default=50,
                        help="Top-k filtering (e.g., 50). Set to 0 to disable.")
    parser.add_argument('--device', type=str, default=None,
                        help="Device to use ('cuda', 'cpu'). Auto-detects if None.")
    parser.add_argument('--seed', type=int, default=1337, help="Random seed for reproducibility.")


    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed) # For multi-GPU, though not explicitly handled here

    print(f"Using device: {args.device}")

    # 1. Load Tokenizer
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    try:
        # This part was correctly updated in the previous partial apply,
        # load_hf_tokenizer is the new function name.
        tokenizer = load_hf_tokenizer(args.tokenizer_path)
        print(f"Tokenizer loaded successfully. Vocab size: {tokenizer.get_vocab_size()}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Define BOS_TOKEN here again as it's used in generate_text's logic for prompt construction
    # This is a bit of a workaround because of the complex interaction of post-processing.
    # Ideally, the tokenizer object itself would be solely responsible.
    BOS_TOKEN = "[BOS]"


    # 2. Load Model
    print(f"Loading model checkpoint from: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Model checkpoint not found at {args.checkpoint_path}")
        return
        
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model_args_dict = checkpoint.get('model_args')
    if model_args_dict is None:
        print("Error: 'model_args' not found in checkpoint. Cannot initialize model.")
        return
    
    # Override vocab_size from tokenizer to ensure consistency
    model_args_dict['vocab_size'] = tokenizer.get_vocab_size()
    print(f"Setting model vocab_size from tokenizer: {model_args_dict['vocab_size']}")

    # Ensure all necessary args for GPT are present
    # Default GPT args: vocab_size, n_layer, n_head, n_embd, block_size, dropout, bias
    required_gpt_args = ['vocab_size', 'n_layer', 'n_head', 'n_embd', 'block_size']
    for req_arg in required_gpt_args:
        if req_arg not in model_args_dict:
            print(f"Error: Required model argument '{req_arg}' not found in checkpoint's model_args.")
            return
    # Set defaults for dropout and bias if not in checkpoint (though they should be)
    model_args_dict.setdefault('dropout', 0.1)
    model_args_dict.setdefault('bias', True)


    model = GPT(**model_args_dict)
    
    state_dict = None
    # Try common keys for the model's state_dict
    possible_state_dict_keys = ['model', 'state_dict', 'model_state_dict']
    for key_candidate in possible_state_dict_keys:
        if key_candidate in checkpoint:
            state_dict = checkpoint[key_candidate]
            print(f"Found model state dictionary under key: '{key_candidate}'")
            break
    
    if state_dict is None:
        print(f"Error: Could not find model state_dict in checkpoint. Tried keys: {possible_state_dict_keys}.")
        print(f"Available keys in checkpoint: {list(checkpoint.keys())}")
        return # Exit if state_dict not found
    # Clean up state_dict if it was saved with DataParallel or DistributedDataParallel
    unwanted_prefix = '_orig_mod.' 
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        # also handle 'module.' prefix from DataParallel
        if k.startswith('module.'):
            state_dict[k[len('module.'):]] = state_dict.pop(k)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("This might be due to a mismatch in model architecture or saved keys.")
        print("Ensure the GPT class definition matches the one used for training.")
        return
        
    model.to(args.device)
    model.eval()
    print("Model loaded successfully.")

    # 3. Format Prompt
    # The tokenizer's post-processor (from train_tokenizer.py) likely adds [BOS] and [EOS].
    # So, the input to tokenizer.encode() should be the content part.
    # Original desired format: "[BOS] Instruction: {user_instruction} Input: {user_input_if_any} Response:"
    # String to pass to tokenizer.encode(): "Instruction: {user_instruction} Input: {user_input_if_any} Response:"
    
    prompt_content = f"Instruction: {args.prompt}"
    if args.input_text:
        prompt_content += f" Input: {args.input_text}"
    prompt_content += " Response:"
    
    print(f"\nContent for tokenizer (before [BOS]/[EOS] by post-processor):\n{prompt_content}\n")

    # 4. Generate Text
    print("Generating text...")
    generated_text = generate_text(
        model,
        tokenizer, # Pass the loaded HF tokenizer object
        prompt_content,
        args.device,
        args.max_new_tokens,
        args.temperature,
        args.top_k
    )

    # 5. Output
    print("\nGenerated Output:\n")
    print(generated_text)

if __name__ == '__main__':
    main()