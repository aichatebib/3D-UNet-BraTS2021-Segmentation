# BraTS 2021 — 3D U-Net Brain Tumor Segmentation

3D multi-class brain tumor segmentation on the **BraTS 2021** dataset (Task 1)
using a MONAI `UNet` (3D, 5 resolution levels) trained on the four MRI
modalities (T1, T1ce, T2, FLAIR) to predict three overlapping tumor
sub-regions: **Whole Tumor (WT)**, **Tumor Core (TC)**, and
**Enhancing Tumor (ET)**.

This repository was extracted and reorganized from an exploratory research
notebook into a clean, runnable pipeline.

## Pipeline overview

1. **Download & extract** the BraTS 2021 dataset (via `kagglehub`).
2. **Preprocess** each case: load 4 modalities + segmentation mask,
   reorient to RAS, per-channel intensity normalization, resize to
   `128×128×128`, and convert the label into 3 binary channels
   (WT / TC / ET).
3. **Train** a 3D U-Net with `DiceLoss`, Adam, and mixed precision (AMP),
   saving the best checkpoint on validation loss (with resume support).
4. **Evaluate** in *native space* (240×240×155): predictions are
   upsampled back to the original resolution before computing
   **Dice** and **95th-percentile Hausdorff Distance (HD95)** per region.
5. **Visualize** predictions vs. ground truth overlaid on the FLAIR slice
   with the largest predicted tumor area, and plot per-patient
   Dice/HD95 curves.

## Repository structure

```
.
├── requirements.txt
├── src/
│   ├── config.py          # paths & hyperparameters
│   ├── data_download.py   # download + extract BraTS 2021 (kagglehub)
│   ├── transforms.py      # MONAI transforms + label-to-multichannel conversion
│   ├── dataset.py         # file listing, train/val/test split, DataLoaders
│   ├── model.py            # 3D U-Net definition
│   ├── train.py            # training loop with checkpoint/resume
│   ├── evaluate.py         # native-space Dice / HD95 evaluation
│   └── visualize.py        # qualitative + per-patient result plots
├── checkpoints/             # saved model weights (created at runtime)
└── outputs/                  # CSV results & figures (created at runtime)
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Download the dataset (requires a Kaggle account/API token)
python src/data_download.py

# 2. Train the model
python src/train.py

# 3. Evaluate on the test set (native-space Dice / HD95)
python src/evaluate.py

# 4. Visualize a specific patient's result
python src/visualize.py --patient-id 27
```

## Notes

- Update `RAW_DATASET_PATH` in `src/config.py` to point to your local
  extracted BraTS 2021 folder.
- Training defaults to `batch_size=1`, `128³` volumes, and AMP — tune
  `src/config.py` for your GPU memory budget.
- The label convention follows the standard BraTS mapping:
  `1` = necrotic/non-enhancing tumor core, `2` = peritumoral edema,
  `4` = enhancing tumor. These are combined into WT / TC / ET as:
  - WT = `{1, 2, 4}`
  - TC = `{1, 4}`
  - ET = `{4}`

## Citation

If you use the BraTS 2021 dataset, please cite the official BraTS
challenge papers as instructed on the [official BraTS site](http://braintumorsegmentation.org/).
