"""Central configuration: paths and hyperparameters."""

import os
import torch

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
# Folder containing the extracted BraTS2021 patient folders
# (e.g. RAW_DATASET_PATH/BraTS2021_00000/BraTS2021_00000_t1.nii.gz)
RAW_DATASET_PATH = os.environ.get("BRATS_DATASET_PATH", r"C:\BraTS2021")

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
SPATIAL_SIZE = (128, 128, 128)
NATIVE_SHAPE = (240, 240, 155)  # original BraTS resolution

TEST_SIZE = 0.15
VAL_SIZE = 0.1765  # of the remaining train set -> ~70/15/15 overall split
RANDOM_STATE = 42

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
IN_CHANNELS = 4   # T1, T1ce, T2, FLAIR
OUT_CHANNELS = 3  # WT, TC, ET
CHANNELS = (32, 64, 128, 256, 512)
STRIDES = (2, 2, 2, 2)
NUM_RES_UNITS = 2

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
BATCH_SIZE = 1
NUM_WORKERS = 0
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
LOG_EVERY_N_BATCHES = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
EMPTY_MASK_PENALTY = 373.13  # mm, applied to HD95 when one mask is empty
