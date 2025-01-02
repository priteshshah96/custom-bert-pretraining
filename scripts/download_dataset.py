from datasets import load_dataset

OUTPUT_DIR = "data/raw"

def download_dataset(dataset_name="cc_news", split="train"):
    """
    Downloads a dataset and saves it to disk.

    Args:
        dataset_name (str): Name of the dataset to download.
        split (str): Dataset split to download.
    """
    print(f"Downloading dataset: {dataset_name}, split: {split}")
    dataset = load_dataset(dataset_name, split=split)

    dataset.save_to_disk(OUTPUT_DIR)
    print(f"Dataset saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    download_dataset()
