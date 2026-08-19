"""Download and extract the BraTS 2021 (Task 1) dataset from Kaggle."""

import os
import tarfile

import kagglehub

from config import RAW_DATASET_PATH

DATASET_ID = "dschettler8845/brats-2021-task1"


def download_dataset() -> str:
    """Download the dataset via kagglehub and return the local cache path."""
    print("Downloading BraTS 2021 dataset, this may take a while...")
    path = kagglehub.dataset_download(DATASET_ID)
    print(f"Download complete. Files located at:\n{path}")
    return path


def extract_dataset(tar_path: str, extract_path: str = RAW_DATASET_PATH) -> None:
    """Extract the downloaded .tar archive into `extract_path`."""
    os.makedirs(extract_path, exist_ok=True)

    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_path)

    print("Extraction completed to:", extract_path)


def list_patients(dataset_path: str = RAW_DATASET_PATH):
    """Return the sorted list of patient folder names."""
    patients = sorted(
        p for p in os.listdir(dataset_path) if p.startswith("BraTS2021_")
    )
    print("Total patients found:", len(patients))
    return patients


if __name__ == "__main__":
    cache_path = download_dataset()

    # The kagglehub cache usually contains a single .tar archive.
    tar_file = os.path.join(cache_path, "BraTS2021_Training_Data.tar")
    if os.path.exists(tar_file):
        extract_dataset(tar_file)
    else:
        print(
            "Could not find the expected .tar file automatically. "
            f"Please check the downloaded contents at: {cache_path}"
        )

    list_patients()
