# Suggested Datasets for Small GPT Model (100-300M Parameters)

This document outlines suggested publicly available datasets for pre-training and instruction fine-tuning a custom small GPT model.

## I. Pre-training Datasets

These datasets are intended to help your model learn general language patterns and knowledge.

1.  **WikiText-103**
    *   **Description:** A high-quality dataset composed of over 100 million tokens extracted from verified "Good" and "Featured" articles on Wikipedia. It's known for its clean and well-formatted text.
    *   **Suitability:** Its size (~500MB compressed, ~1.8GB uncompressed text) is manageable. The quality and encyclopedic nature of the text make it a good choice for foundational language understanding.
    *   **Access Link:** [https://huggingface.co/datasets/wikitext](https://huggingface.co/datasets/wikitext) (specifically, look for configurations like `wikitext-103-raw-v1` or `wikitext-103-v1`).

2.  **Simple Wikipedia (20220301.simple dump)**
    *   **Description:** This dataset is derived from the Simple English Wikipedia, which uses a more basic English vocabulary and simpler grammatical structures compared to the standard Wikipedia.
    *   **Suitability:** It provides clean, encyclopedic text that is easier to process and can be beneficial for a smaller model. The dump size is a few hundred megabytes, making it very manageable.
    *   **Access Link:** [https://huggingface.co/datasets/wikipedia](https://huggingface.co/datasets/wikipedia) (you would use the `20220301.simple` configuration when loading via the Hugging Face `datasets` library).

## II. Instruction Fine-tuning Datasets

These datasets will help your pre-trained model learn to follow instructions and respond to prompts effectively.

1.  **Databricks Dolly 15k (`databricks-dolly-15k`)**
    *   **Description:** Contains around 15,000 high-quality, human-generated instruction-following records. It covers a diverse set of capabilities including brainstorming, classification, question answering, generation, information extraction, and summarization.
    *   **Suitability:** Being human-generated, the quality is generally high. The dataset size is appropriate for initial fine-tuning, and it's available in a convenient JSONL format.
    *   **Access Link:** [https://huggingface.co/datasets/databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

2.  **Alpaca (Cleaned Version - `yahma/alpaca-cleaned`)**
    *   **Description:** This dataset provides approximately 52,000 instruction-following examples. It was originally generated using OpenAI's `text-davinci-003` based on seed instructions. The `yahma/alpaca-cleaned` version is a community-cleaned version which removes some low-quality or problematic examples.
    *   **Suitability:** It's a widely used dataset for instruction tuning, offering a larger number of examples than Dolly. The JSON format is standard for instruction datasets.
    *   **Access Link:** [https://huggingface.co/datasets/yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned)