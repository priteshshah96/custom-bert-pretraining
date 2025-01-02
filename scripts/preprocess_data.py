import os
from datasets import Dataset
from transformers import AutoTokenizer

# Directories
RAW_DATA_DIR = "data/raw"
TOKENIZED_DATA_DIR = "data/tokenized"
PREPROCESSED_DATA_DIR = "data/preprocessed"
TOKENIZER_PATH = "models/tokenizer"

# Checking if directories exist
TOKENIZED_DATA_DIR = "data/tokenized"
os.makedirs(TOKENIZED_DATA_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)

def read_files(directory):

    data =  []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswirh(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                data.append({"text": content})
    return data

def tokenize_data(raw_data, tokenizer):

    def tokenizer_function(examples):
        return tokenizer(examples["text"], truncation = True, padding ="max_length", max_length = 512)
    
    dataset = Dataset.from_list(raw_data)
    tokenized_dataset = dataset.map(tokenizer_function, batched=True)
    return tokenized_dataset


def preprocess():

    print(" Reading raw data file .......")
    raw_data = read_files(RAW_DATA_DIR)

    print(f"Loaded {len(raw_data)} files from {RAW_DATA_DIR}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    print("Tokenizing data...")
    tokenized_dataset = tokenize_data(raw_data, tokenizer)

    print("Saving tokenized data...")
    tokenized_output_path = os.path.join(TOKENIZED_DATA_DIR, "tokenized.arrow")
    tokenized_dataset.save_to_disk(tokenized_output_path)
    print(f"Tokenized data saved to {tokenized_output_path}")

    print("Saving preprocessed data as text...")
    preprocessed_output_path = os.path.join(PREPROCESSED_DATA_DIR, "preprocessed.jsonl")
    tokenized_dataset.to_json(preprocessed_output_path)
    print(f"Preprocessed data saved to {preprocessed_output_path}")
    print("Preprocessing complete!")

if __name__ == "__main__":
    preprocess()