import os
import json
from datasets import load_dataset

def download_and_process_dolly():
    """
    Downloads the Databricks Dolly 15k dataset, reformats each record,
    and saves it to a JSONL file.
    """
    print("Loading Databricks Dolly 15k dataset...")
    # Load the databricks-dolly-15k dataset
    dataset = load_dataset('databricks/databricks-dolly-15k', split='train')

    processed_records = []
    print("Processing records...")
    for record in dataset:
        instruction = record.get('instruction', '')
        context = record.get('context', '') # This will be our 'input'
        response = record.get('response', '') # This will be our 'output'

        # Ensure all required fields are present, even if some are empty strings
        # as per Hugging Face dataset viewer, all records have these fields.
        # If context is empty, input should be an empty string.
        formatted_record = {
            "instruction": instruction,
            "input": context, # 'context' from dataset becomes 'input'
            "output": response
        }
        processed_records.append(formatted_record)

    # Define the output directory and filename
    output_dir = "data/finetuning_data/"
    output_filename = "dolly_15k_instructions.jsonl"
    output_path = os.path.join(output_dir, output_filename)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving processed records to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for record in processed_records:
            f.write(json.dumps(record) + "\n")

    print(f"Successfully processed and saved Databricks Dolly 15k to {output_path}")

if __name__ == "__main__":
    download_and_process_dolly()