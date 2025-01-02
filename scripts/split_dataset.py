from datasets import load_from_disk

RAW_DATA_DIR = "data/raw"
OUTPUT_DIR = "data/split"

def split_dataset(raw_data_dir, test_size=0.1):
    """
    Splits the dataset into training and testing sets.

    Args:
        raw_data_dir (str): Path to the raw dataset directory.
        test_size (float): Proportion of the dataset to include in the test split.
    """
    print(f"Loading dataset from {raw_data_dir}")
    dataset = load_from_disk(raw_data_dir)

    print(f"Splitting dataset with test size {test_size}")
    split = dataset.train_test_split(test_size=test_size)
    split["train"].save_to_disk(f"{OUTPUT_DIR}/train")
    split["test"].save_to_disk(f"{OUTPUT_DIR}/test")

    print(f"Training and testing datasets saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    split_dataset(RAW_DATA_DIR)
