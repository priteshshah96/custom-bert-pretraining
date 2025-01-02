from datasets import load_from_disk
from tokenizers import BertWordPieceTokenizer
import os

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_SPLIT_DIR = os.path.join(PROJECT_ROOT, "../data/split/train")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "../models/tokenizer")

# Ensure the tokenizer directory exists
os.makedirs(TOKENIZER_DIR, exist_ok=True)

def train_tokenizer(data_dir, save_dir, vocab_size=30522):
    """
    Trains a WordPiece tokenizer on the given dataset.

    Args:
        data_dir (str): Directory containing training data.
        save_dir (str): Directory to save the trained tokenizer.
        vocab_size (int): Vocabulary size for the tokenizer.
    """
    print(f"Loading dataset from {data_dir}...")
    dataset = load_from_disk(data_dir)
    
    # Prepare training data for the tokenizer
    texts = [example["text"] for example in dataset]

    # Train the tokenizer
    print(f"Training tokenizer with vocab size {vocab_size}...")
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train_from_iterator(
        texts, 
        vocab_size=vocab_size, 
        min_frequency=2, 
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    
    # Save the tokenizer
    print(f"Saving tokenizer to {save_dir}...")
    tokenizer.save_model(save_dir)

if __name__ == "__main__":
    train_tokenizer(TRAIN_SPLIT_DIR, TOKENIZER_DIR)
