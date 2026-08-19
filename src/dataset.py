"""Build the BraTS file list, train/val/test split, and DataLoaders."""

import os

from sklearn.model_selection import train_test_split
from monai.data import Dataset, DataLoader

from config import (
    RAW_DATASET_PATH,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
    BATCH_SIZE,
    NUM_WORKERS,
)
from transforms import get_transforms


def build_file_list(dataset_path: str = RAW_DATASET_PATH):
    """List all patients and build the {image: [...], label: ...} dicts."""
    patients = sorted(
        p for p in os.listdir(dataset_path) if p.startswith("BraTS2021_")
    )

    data = []
    for patient in patients:
        patient_path = os.path.join(dataset_path, patient)
        data.append(
            {
                "image": [
                    os.path.join(patient_path, f"{patient}_t1.nii.gz"),
                    os.path.join(patient_path, f"{patient}_t1ce.nii.gz"),
                    os.path.join(patient_path, f"{patient}_t2.nii.gz"),
                    os.path.join(patient_path, f"{patient}_flair.nii.gz"),
                ],
                "label": os.path.join(patient_path, f"{patient}_seg.nii.gz"),
            }
        )

    print("Total cases:", len(data))
    return data


def split_data(data):
    """70/15/15-style split: test first, then val from the remaining train."""
    train_files, test_files = train_test_split(
        data, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_files, val_files = train_test_split(
        train_files, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    print("Training  :", len(train_files))
    print("Validation:", len(val_files))
    print("Testing   :", len(test_files))

    return train_files, val_files, test_files


def build_datasets(dataset_path: str = RAW_DATASET_PATH):
    data = build_file_list(dataset_path)
    train_files, val_files, test_files = split_data(data)

    transforms = get_transforms()

    train_dataset = Dataset(data=train_files, transform=transforms)
    val_dataset = Dataset(data=val_files, transform=transforms)
    test_dataset = Dataset(data=test_files, transform=transforms)

    return train_dataset, val_dataset, test_dataset


def build_dataloaders(dataset_path: str = RAW_DATASET_PATH):
    train_dataset, val_dataset, test_dataset = build_datasets(dataset_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader
