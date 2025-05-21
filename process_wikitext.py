import os
from datasets import load_dataset

def download_and_process_wikitext():
    """
    Downloads the WikiText-103 dataset, concatenates train, validation, and test splits,
    removes empty lines, and saves it to a single text file.
    """
    print("Loading WikiText-103 dataset...")
    # Load the wikitext-103-raw-v1 dataset
    dataset_train = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    dataset_validation = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
    dataset_test = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')

    print("Concatenating splits...")
    all_text = []
    for split_name, data_split in [("train", dataset_train), ("validation", dataset_validation), ("test", dataset_test)]:
        print(f"Processing {split_name} split...")
        for example in data_split:
            text = example['text']
            if text.strip(): # Keep non-empty lines
                all_text.append(text)

    combined_text = "\n".join(all_text)

    # Define the output directory and filename
    output_dir = "data/pretraining_data/"
    output_filename = "wikitext-103_train_corpus.txt"
    output_path = os.path.join(output_dir, output_filename)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving combined text to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined_text)

    print(f"Successfully processed and saved WikiText-103 to {output_path}")

if __name__ == "__main__":
    download_and_process_wikitext()