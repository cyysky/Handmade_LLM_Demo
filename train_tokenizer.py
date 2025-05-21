import os
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tokenizers.processors import TemplateProcessing

def train_and_save_tokenizer():
    """
    Trains a Byte-Pair Encoding (BPE) tokenizer on a given corpus,
    saves it, and demonstrates its usage.
    """

    # 1. Define paths and parameters
    corpus_file = "data/pretraining_data/wikitext-103_train_corpus.txt"
    tokenizer_save_dir = "data/tokenizer/"
    tokenizer_filename = "custom_gpt_tokenizer.json"
    full_tokenizer_path = os.path.join(tokenizer_save_dir, tokenizer_filename)

    vocab_size = 32000
    # Special tokens for the GPT-style model
    special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
    min_frequency_for_token = 2 # Standard minimum frequency

    # 2. Ensure output directory exists
    os.makedirs(tokenizer_save_dir, exist_ok=True)
    print(f"Output directory '{tokenizer_save_dir}' ensured.")

    # 3. Initialize and train the tokenizer
    print(f"Initializing ByteLevelBPETokenizer...")
    # Initialize a ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer()

    print(f"Starting tokenizer training on '{corpus_file}'...")
    print(f"Parameters: vocab_size={vocab_size}, min_frequency={min_frequency_for_token}, special_tokens={special_tokens}")
    # Train the tokenizer
    tokenizer.train(
        files=[corpus_file],
        vocab_size=vocab_size,
        min_frequency=min_frequency_for_token,
        special_tokens=special_tokens,
        show_progress=True,
    )
    print("Tokenizer training complete.")

    # 4. Set post-processor for BOS/EOS tokens (common for GPT-style models)
    # This will automatically add [BOS] at the beginning and [EOS] at the end of sequences.
    bos_token_id = tokenizer.token_to_id("[BOS]")
    eos_token_id = tokenizer.token_to_id("[EOS]")

    if bos_token_id is not None and eos_token_id is not None:
        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            pair="[BOS] $A [EOS] [BOS] $B [EOS]", # For encoding pairs of sequences
            special_tokens=[
                ("[BOS]", bos_token_id),
                ("[EOS]", eos_token_id),
            ],
        )
        print("Set TemplateProcessing with [BOS] and [EOS] tokens.")
    else:
        print("Warning: [BOS] or [EOS] token ID not found after training. Post-processor for BOS/EOS not set.")

    # 5. Save the tokenizer
    tokenizer.save(full_tokenizer_path)
    print(f"Tokenizer saved to '{full_tokenizer_path}'")

    # 6. Example usage (demonstrates loading, encoding, and decoding)
    print("\n--- Tokenizer Usage Example ---")
    
    # Load the saved tokenizer
    # Note: Can be loaded with `Tokenizer.from_file` or specific type if known.
    loaded_tokenizer = Tokenizer.from_file(full_tokenizer_path)
    print(f"Tokenizer loaded from '{full_tokenizer_path}'.")

    # Configure padding on the loaded tokenizer if needed for batching
    # (This step is for demonstration; actual padding depends on batching strategy)
    pad_token_id_loaded = loaded_tokenizer.token_to_id("[PAD]")
    if pad_token_id_loaded is not None:
        loaded_tokenizer.enable_padding(pad_id=pad_token_id_loaded, pad_token="[PAD]")
        print(f"Padding enabled with PAD token ID: {pad_token_id_loaded}")
    else:
        print("Warning: [PAD] token not found in loaded tokenizer. Padding not configured.")

    example_text = "Hello, this is a test sentence for our new custom GPT tokenizer."
    print(f"\nOriginal text: '{example_text}'")

    # Encode the sample text
    encoded_output = loaded_tokenizer.encode(example_text)
    
    print(f"Encoded Tokens: {encoded_output.tokens}")
    print(f"Encoded IDs: {encoded_output.ids}")

    # Decode the token IDs back to text
    decoded_text = loaded_tokenizer.decode(encoded_output.ids)
    print(f"Decoded text: '{decoded_text}'")

    # Example with a pair of sentences (if post-processor for pair is set)
    example_text_pair_A = "First sentence."
    example_text_pair_B = "Second sentence."
    encoded_pair_output = loaded_tokenizer.encode(example_text_pair_A, example_text_pair_B)
    print(f"\nOriginal text pair: ('{example_text_pair_A}', '{example_text_pair_B}')")
    print(f"Encoded Pair Tokens: {encoded_pair_output.tokens}")
    print(f"Encoded Pair IDs: {encoded_pair_output.ids}")
    decoded_pair_text = loaded_tokenizer.decode(encoded_pair_output.ids)
    print(f"Decoded Pair text: '{decoded_pair_text}'")


if __name__ == "__main__":
    # Ensure you have the 'tokenizers' library installed:
    # pip install tokenizers
    
    # Also, ensure the pretraining corpus exists at the specified path:
    # 'data/pretraining_data/wikitext-103_train_corpus.txt'
    
    train_and_save_tokenizer()