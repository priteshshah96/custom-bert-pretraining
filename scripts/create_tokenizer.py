from datasets import load_from_disk
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast
import os

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_SPLIT_DIR = os.path.join(PROJECT_ROOT, "../data/split/train")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "../models/tokenizer")

# Ensure the tokenizer directory exists
os.makedirs(TOKENIZER_DIR, exist_ok=True)

def train_tokenizer(data_dir, save_dir, vocab_size=30522):
    """
    Trains a WordPiece tokenizer on the given dataset and saves it.

    Args:
        data_dir (str): Directory containing training data.
        save_dir (str): Directory to save the trained tokenizer.
        vocab_size (int): Vocabulary size for the tokenizer.
    """
    print(f"Loading dataset from {data_dir}...")
    dataset = load_from_disk(data_dir)
    
    # Prepare training data for the tokenizer
    texts = [example["text"] for example in dataset]

    # Train the WordPiece tokenizer
    print(f"Training WordPiece tokenizer with vocab size {vocab_size}...")
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train_from_iterator(
        texts, 
        vocab_size=vocab_size, 
        min_frequency=2, 
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    
    # Save the tokenizer as a WordPiece tokenizer
    print(f"Saving WordPiece tokenizer to {save_dir}...")
    tokenizer.save_model(save_dir)
    
    # Create a Hugging Face-compatible tokenizer
    print("Creating Hugging Face-compatible tokenizer...")
    hf_tokenizer = BertTokenizerFast.from_pretrained(save_dir)
    
    # Save Hugging Face-compatible tokenizer files
    hf_tokenizer.save_pretrained(save_dir)
    print(f"Hugging Face-compatible tokenizer saved to {save_dir}")

if __name__ == "__main__":
    train_tokenizer(TRAIN_SPLIT_DIR, TOKENIZER_DIR)
