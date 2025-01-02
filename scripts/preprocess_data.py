import os
from datasets import load_from_disk, DatasetDict
from transformers import PreTrainedTokenizerFast

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_SPLIT_DIR = os.path.join(PROJECT_ROOT, "../data/split/train")
TEST_SPLIT_DIR = os.path.join(PROJECT_ROOT, "../data/split/test")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "../models/tokenizer")
TOKENIZED_DIR = os.path.join(PROJECT_ROOT, "../data/tokenized")

# Ensure output directory exists
os.makedirs(TOKENIZED_DIR, exist_ok=True)

def tokenize_and_save(data_dir, tokenizer_dir, output_dir):
    """
    Tokenizes a dataset split and saves it.

    Args:
        data_dir (str): Path to the dataset split directory.
        tokenizer_dir (str): Path to the directory containing the trained tokenizer.
        output_dir (str): Directory to save the tokenized dataset.
    """
    print(f"Loading dataset from {data_dir}...")
    dataset = load_from_disk(data_dir)
    
    print(f"Loading tokenizer from {tokenizer_dir}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=512
        )
    
    print(f"Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4)

    # Save the tokenized dataset
    split_name = os.path.basename(data_dir)
    output_path = os.path.join(output_dir, split_name)
    print(f"Saving tokenized dataset to {output_path}...")
    tokenized_dataset.save_to_disk(output_path)

if __name__ == "__main__":
    # Tokenize train and test splits
    tokenize_and_save(TRAIN_SPLIT_DIR, TOKENIZER_DIR, TOKENIZED_DIR)
    tokenize_and_save(TEST_SPLIT_DIR, TOKENIZER_DIR, TOKENIZED_DIR)
