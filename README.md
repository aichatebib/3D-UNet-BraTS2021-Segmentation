# Compact 3D U-Net for Multi-Region Brain Tumor Segmentation on BraTS 2021 under a 6 GB Memory Budget

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22011395.svg)](https://doi.org/10.5281/zenodo.22011395)

## Overview

This repository provides the code for a compact 3D U-Net trained on the
BraTS 2021 dataset (1251 multimodal cases; T1, contrast-enhanced T1, T2,
FLAIR) to jointly segment three tumor sub-regions (whole tumor, tumor core,
enhancing tumor). Evaluated in native acquisition resolution on 188
independent test cases, the model achieves a mean DSC of 0.8076 (95% CI
0.7890–0.8248) and a mean HD95 of 9.04 mm (95% CI 6.19–12.95 mm), while
training on a single 6 GB consumer-grade GPU for 20 epochs.

## Hardware

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM)
- Peak GPU memory (training step): 836.00 MB (reserved) / 719.81 MB (allocated)
- Epochs: 20
- Training time: ≈ 9.12 h
- Inference time: 164.77 ms/case

## Environment

- Python 3.10
- PyTorch 2.8.0 (CUDA 12.8 build, `2.8.0+cu128`)
- TorchVision 0.23.0
- TorchAudio 2.8.0
- MONAI 1.6.0
- Other dependencies: see `requirements.txt` (NumPy, nibabel, SciPy, Matplotlib, pandas)

## Data

This repository does **not** include the BraTS 2021 dataset. The data are
distributed by the organizers of the BraTS challenge; access requires
registration and acceptance of the official BraTS Data Usage Agreement. See the
official BraTS 2021 data access and registration procedure:
<https://www.med.upenn.edu/cbica/brats2021/>

`src/data_download.py` retrieves a copy of BraTS 2021 from **Kaggle** via
`kagglehub`. This is a third-party mirror, not the official distribution: the
exact dataset identifier is set in `src/data_download.py`, and its licence and
terms of use are those stated on its Kaggle page (a Kaggle account / API
credentials are required). Users remain responsible for complying with the
BraTS Data Usage Agreement regardless of where the files are obtained.

### Data split

The 1251 cases are split at the patient level into **875 training / 188
validation / 188 test** cases (≈ 70 / 15 / 15 %) using scikit-learn's
`train_test_split` with `random_state=42`. All reported results are on the 188
held-out test cases. The training procedure itself does not set a global PyTorch
seed, so re-training may give slightly different weights and metrics.

## Reproducing

### 1. Download and extract the BraTS 2021 dataset

```
python src/data_download.py
```

Downloads and extracts the BraTS 2021 dataset via `kagglehub` (see **Data**).

### 2. Train the 3D U-Net

```
python src/train.py
```

Batch size 1, 128³ input volumes, Dice loss, Adam optimizer, mixed-precision
training (AMP). Preprocessing (reorientation, resampling, normalization,
WT/TC/ET label derivation) is applied on the fly through `transforms.py` and
`dataset.py` — there is no separate preprocessing script.

### 3. Evaluate the trained model

```
python src/evaluate.py
```

Reprojects predictions to native BraTS resolution (240 × 240 × 155, 1 mm
isotropic, RAS) and computes DSC and HD95 against the original BraTS
segmentation for WT, TC, and ET (`--split` can be set to `test`, `val`,
`train`, or `all`; defaults to `test`). When a region is empty in the
prediction or in the ground truth, HD95 is undefined; the native-space image
diagonal (373.13 mm) is then used as a penalty value.

### 4. Generate visualizations

Qualitative figures for one or more test-set patients (`--patient-id` is
1-indexed):

```
python src/visualize.py --patient-id 75 53 173
```

For each patient this writes three figures (`Axial`, `Coronal`, `Sagittal`) with
the panels *FLAIR MRI | Ground Truth | 3D U-Net Prediction*, as
`Patient_<id>_<Plane>_600dpi.png` and `Patient_<id>_<Plane>.pdf` (12 × 4.7 in,
600 dpi, no patient/DSC/HD95 header). The displayed slice in each plane is the
one with the largest union of ground-truth and predicted tumor. These figures
are drawn on the 128³ model grid (not in native space).

Per-patient Dice and HD95 plots (Fig. 3 and Fig. 4) from the evaluation CSV:

```
python src/visualize.py --summary --results-csv outputs/native_space_test.csv
```

This writes `Fig3_Per_Patient_Dice_Native_Space.{png,pdf}` and
`Fig4_Per_Patient_HD95_Native_Space.{png,pdf}` (600 dpi).

Useful options:

| Option | Meaning |
| --- | --- |
| `--channel N` | Input channel shown as the MRI: `0`=T1, `1`=T1ce, `2`=T2, `3`=FLAIR (default `3`) |
| `--equal-aspect` | Disable the z-axis aspect correction in coronal/sagittal views |
| `--out-dir DIR` | Output directory (default: `OUTPUT_DIR` from `config.py`) |
| `--no-show` | Save figures without opening a window (headless servers) |

`--patient-id` and `--summary` can be combined in one call.

## Expected results

Test set, 188 cases, native resolution. SD = sample standard deviation
(ddof = 1).

| Region | DSC (mean ± SD) | DSC median | HD95 mm (mean ± SD)\* | HD95 mm median\* |
| --- | --- | --- | --- | --- |
| WT | 0.8813 ± 0.0741 | 0.9052 | 7.34 ± 13.02 | 3.08 |
| TC | 0.8014 ± 0.2083 | 0.8887 | 6.78 ± 10.15 | 3.00 |
| ET | 0.7401 ± 0.1764 | 0.7878 | 5.19 ± 9.15 | 2.45 |
| **Overall (per-patient)** | **0.8076** (95% CI 0.7890–0.8248) | **0.8504** | **9.04** (95% CI 6.19–12.95)† | **3.59** (IQR 2.54–7.72)† |

\* Per-region HD95 statistics are computed **excluding** cases where HD95 is
undefined (1 TC case and 3 ET cases; no WT case), i.e. without the penalty.

† Overall HD95 is the per-patient value including the penalty (373.13 mm,
native-space image diagonal) applied to those 1 TC and 3 ET cases. Because of
this penalty, the mean is strongly influenced by a few cases; the median and IQR
are more representative.

## Citation

If you use this code, please cite both the article and the software:

- Article: The full article citation will be added after acceptance and publication.
- Software: DOI 10.5281/zenodo.22011395

## License

This project is licensed under the MIT License. See `LICENSE` for details. The
BraTS 2021 data are **not** covered by this licence.
