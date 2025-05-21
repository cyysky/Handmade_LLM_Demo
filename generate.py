import torch
import torch.nn.functional as F
import argparse
import os
from model import GPT # Assuming model.py is in the same directory or accessible
from tokenizers import Tokenizer # Import Hugging Face Tokenizer

# Define special token strings that might be needed for logic
EOS_TOKEN = "[EOS]"
BOS_TOKEN = "[BOS]" # Needed for checking if tokenizer adds it, or for manual construction if necessary

def load_hf_tokenizer(tokenizer_path):
    """Loads a Hugging Face Tokenizer from a file."""
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer

def generate_text(
    model,
    tokenizer, # Expect a HF Tokenizer object
    prompt_content_for_tokenizer, # This is the content like "Instruction: ... Response:"
    device,
    max_new_tokens=150,
    temperature=0.8,
    top_k=50
):
    """Generates text using the fine-tuned model and HF tokenizer."""
    model.eval()

    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id is None:
        print(f"Warning: EOS token '{EOS_TOKEN}' not found in tokenizer. Generation might not stop correctly.")

    # Encode the prompt content.
    # The tokenizer (as configured in train_tokenizer.py) has a post-processor:
    # TemplateProcessing(single="[BOS] $A [EOS]", ...)
    # So, tokenizer.encode("Instruction: ... Response:") will produce:
    # [BOS_ID, tokens("Instruction: ... Response:"), EOS_ID]
    encoded_prompt = tokenizer.encode(prompt_content_for_tokenizer)
    start_ids = encoded_prompt.ids

    # For generation, we want to feed the model the prompt *up to* where it should start generating.
    # If the template added an EOS at the end of our prompt, we should remove it
    # so the model doesn't think the sequence is already complete.
    if start_ids and start_ids[-1] == eos_id:
        # Check if the prompt was just BOS + EOS or something very short.
        # This logic assumes the prompt_content_for_tokenizer is substantial enough
        # that an EOS at its end is from the template, not part of the actual desired starting prompt.
        # A more robust check might involve comparing length against expected BOS + content.
        # For now, if EOS is last, assume it's template-added and remove for generation.
        if len(start_ids) > 1 : # Ensure there's more than just an EOS token (or BOS + EOS)
            start_ids = start_ids[:-1]

    if not start_ids:
        print("Error: Tokenized prompt is empty. Cannot generate.")
        return "Error: Tokenized prompt is empty."

    input_ids = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated_ids = list(start_ids) # Keep a Python list for appending

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
        next_token_id_tensor = torch.multinomial(probs, num_samples=1) # (B, 1)
        next_token_id = next_token_id_tensor.item()

        generated_ids.append(next_token_id)
        input_ids = torch.cat((input_ids, next_token_id_tensor), dim=1)

        if eos_id is not None and next_token_id == eos_id:
            break
            
    # Decode generated tokens using the HF tokenizer
    # skip_special_tokens=False to see the full structure including any BOS/EOS.
    # Set to True if you only want the "content" part of the generation.
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=False) 
    
    return decoded_text

def main():
    parser = argparse.ArgumentParser(description="Generate text using a fine-tuned GPT model.")
    parser.add_argument('--checkpoint_path', type=str, default="out/finetune/ckpt.pt",
                        help="Path to the fine-tuned model checkpoint (.pt file).")
    parser.add_argument('--tokenizer_path', type=str, default="data/tokenizer/custom_gpt_tokenizer.json",
                        help="Path to the tokenizer JSON file.")
    parser.add_argument('--prompt', type=str, required=True,
                        help="User instruction for the model (e.g., 'Translate this to French').")
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
        torch.cuda.manual_seed_all(args.seed)

    print(f"Using device: {args.device}")

    # 1. Load Tokenizer
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    try:
        tokenizer = load_hf_tokenizer(args.tokenizer_path)
        print(f"Tokenizer loaded successfully. Vocab size: {tokenizer.get_vocab_size()}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
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
    
    # Ensure vocab_size in model_args matches the loaded tokenizer
    model_args_dict['vocab_size'] = tokenizer.get_vocab_size()
    print(f"Setting model vocab_size from tokenizer: {model_args_dict['vocab_size']}")

    required_gpt_args = ['vocab_size', 'n_layer', 'n_head', 'n_embd', 'block_size']
    for req_arg in required_gpt_args:
        if req_arg not in model_args_dict:
            print(f"Error: Required model argument '{req_arg}' not found in checkpoint's model_args.")
            return
    model_args_dict.setdefault('dropout', 0.1) # Default if not in checkpoint
    model_args_dict.setdefault('bias', True)    # Default if not in checkpoint

    model = GPT(**model_args_dict)
    
    state_dict = None
    possible_state_dict_keys = ['model', 'state_dict', 'model_state_dict'] 
    for key_candidate in possible_state_dict_keys:
        if key_candidate in checkpoint:
            state_dict = checkpoint[key_candidate]
            print(f"Found model state dictionary under key: '{key_candidate}'")
            break
    
    if state_dict is None:
        print(f"Error: Could not find model state_dict in checkpoint. Tried keys: {possible_state_dict_keys}.")
        print(f"Available keys in checkpoint: {list(checkpoint.keys())}")
        return

    # Clean up state_dict prefixes if necessary
    unwanted_prefixes = ['_orig_mod.', 'module.']
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        original_k = k
        for prefix in unwanted_prefixes:
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned_state_dict[k] = v
    state_dict = cleaned_state_dict

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

    # 3. Format Prompt Content for Tokenizer
    # This is the string that the tokenizer's `encode` method will process.
    # The tokenizer's post-processor (from train_tokenizer.py) adds [BOS] at the start
    # and [EOS] at the end of this content.
    # Fine-tuning prompt structure: "[BOS] Instruction: {user_instruction} Input: {user_input_if_any} Response:"
    # So, the content we provide to the tokenizer should be:
    # "Instruction: {user_instruction} Input: {user_input_if_any} Response:"
    
    prompt_content_for_tokenizer = f"Instruction: {args.prompt}"
    if args.input_text:
        prompt_content_for_tokenizer += f" Input: {args.input_text}"
    prompt_content_for_tokenizer += " Response:"
    
    print(f"\nContent passed to tokenizer.encode():\n'{prompt_content_for_tokenizer}'\n")

    # 4. Generate Text
    print("Generating text...")
    generated_text = generate_text(
        model,
        tokenizer,
        prompt_content_for_tokenizer, 
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