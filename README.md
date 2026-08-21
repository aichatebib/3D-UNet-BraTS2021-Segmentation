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
- PyTorch 2.8.0
- TorchVision 0.23.0
- TorchAudio 2.8.0
- MONAI 1.6.0
- Other dependencies: see `requirements.txt` (NumPy, nibabel, SciPy, Matplotlib)

## Data
This repository does **not** include the BraTS 2021 dataset. Access requires
registration and acceptance of the official BraTS Data Usage Agreement.
See the official BraTS 2021 data access and registration procedure:
https://www.med.upenn.edu/cbica/brats2021/

## Reproducing

### 1. Download and extract the BraTS 2021 dataset
```bash
python src/data_download.py
```
Downloads and extracts the BraTS 2021 dataset via `kagglehub`.

### 2. Train the 3D U-Net
```bash
python src/train.py
```
Batch size 1, 128³ input volumes, Dice loss, Adam optimizer, mixed-precision
training (AMP). Preprocessing (reorientation, resampling, normalization,
WT/TC/ET label derivation) is applied on the fly through `transforms.py` and
`dataset.py` — there is no separate preprocessing script.

### 3. Evaluate the trained model
```bash
python src/evaluate.py
```
Reprojects predictions to native BraTS resolution and computes DSC and HD95
for WT, TC, and ET (`--split` can be set to `test`, `val`, `train`, or `all`;
defaults to `test`).

### 4. Generate visualizations
```bash
python src/visualize.py --patient-id <n>
```
Produces a qualitative FLAIR / ground-truth / prediction overlay and the
Dice/HD95 scores for the specified test-set patient (`--patient-id` is
1-indexed and required).

`visualize.py` also defines `plot_per_patient_metric()`, which plots a
per-patient Dice or HD95 curve (with mean/median lines) from the evaluation
CSV. This function is not wired to a CLI flag — it must be called manually,
e.g. from a Python shell:
```python
from visualize import plot_per_patient_metric
plot_per_patient_metric("outputs/native_space_test.csv", metric="Patient_Dice")
```
## Expected results
| Region | DSC | HD95 (mm) |
| --- | --- | --- |
| WT | 0.8813 ± 0.0739 | not individually reported (see Fig. 5B; median ≈2–4 mm range) |
| TC | 0.8014 ± 0.2077 | not individually reported (see Fig. 5B; median ≈2–4 mm range) |
| ET | 0.7401 ± 0.1759 | not individually reported (see Fig. 5B; median ≈2–4 mm range) |
| **Mean (overall, per-patient)** | **0.8076 (95% CI 0.7890–0.8248), median 0.8504** | **9.04 mm (95% CI 6.19–12.95), median 3.59 mm (IQR 2.54–7.72)** |

*HD95 penalty (373.13 mm, native-space image diagonal) applied to 1 TC case and 3 ET cases; no WT case penalized.*

## Citation
If you use this code, please cite both the article and the software:
- Article: The full article citation will be added after acceptance and publication.
- Software: DOI 10.5281/zenodo.22011395

## License
This project is licensed under the MIT License. See `LICENSE` for details.
