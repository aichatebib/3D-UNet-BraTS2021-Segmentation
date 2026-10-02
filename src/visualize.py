"""Publication-quality figures for the 3D U-Net BraTS 2021 pipeline.

Qualitative figures (per test patient, one figure per plane):
    Axial / Coronal / Sagittal  ->  MRI | Ground Truth | 3D U-Net Prediction
    600 dpi PNG + vector PDF, black background, no patient/DSC/HD95 header.

Quantitative figures (from the per-patient results CSV):
    Fig 3 - per-patient Dice   (12 x 5.5 in, 600 dpi PNG + PDF)
    Fig 4 - per-patient HD95   (13 x 5.5 in, 600 dpi PNG + PDF)

Usage:
    python visualize.py --patient-id 75 53 173
    python visualize.py --summary --results-csv Native_Space_HD95_DSC_188_patients.csv
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from config import BEST_MODEL_PATH, DEVICE, OUTPUT_DIR
from dataset import build_dataloaders
from model import build_model

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
DPI = 600
QUAL_FIGSIZE = (12.0, 4.7)      # qualitative figures
FIG3_FIGSIZE = (12.0, 5.5)      # per-patient Dice
FIG4_FIGSIZE = (13.0, 5.5)      # per-patient HD95
THRESHOLD = 0.5
HD95_PENALTY = 373.13           # empty-mask HD95 penalty (mm)

# Input channel order is [T1, T1ce, T2, FLAIR] -> FLAIR is index 3.
FLAIR_CHANNEL = 3

# The volume is resized 240x240x155 -> 128^3, so one voxel of the 128^3 grid
# is 1.875 mm in x/y but only ~1.211 mm in z. Coronal/sagittal views need this
# ratio to keep real brain proportions.
Z_ASPECT = 155.0 / 240.0

MASK_CMAP = ListedColormap([
    (0.0, 0.0, 0.0, 0.0),     # 0 background
    (1.0, 0.0, 0.0, 0.38),    # 1 WT
    (1.0, 0.55, 0.0, 0.45),   # 2 TC
    (0.0, 1.0, 0.0, 0.55),    # 3 ET
])

LEGEND_ELEMENTS = [
    Patch(facecolor="#FF0000", edgecolor="#FF0000", alpha=0.75, label="WT"),
    Patch(facecolor="#FF8C00", edgecolor="#FF8C00", alpha=0.75, label="TC"),
    Patch(facecolor="#00FF00", edgecolor="#00FF00", alpha=0.75, label="ET"),
]


def set_publication_style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams["font.size"] = 10


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def make_label_map(mask, spatial_shape):
    """(3, H, W, D) binary [WT, TC, ET] -> one map: 0 bg, 1 WT, 2 TC, 3 ET."""
    label_map = np.zeros(spatial_shape, dtype=np.uint8)
    label_map[mask[2] > 0] = 3
    label_map[(mask[1] > 0) & (label_map == 0)] = 2
    label_map[(mask[0] > 0) & (label_map == 0)] = 1
    return label_map


def crop_to_brain(brain, views, margin=4):
    """Crop every 2D view to the bounding box of the brain mask."""
    ys, xs = np.where(brain)
    if len(xs) == 0:
        return views
    y0 = max(0, ys.min() - margin)
    y1 = min(brain.shape[0], ys.max() + margin + 1)
    x0 = max(0, xs.min() - margin)
    x1 = min(brain.shape[1], xs.max() + margin + 1)
    return tuple(v[y0:y1, x0:x1] for v in views)


@torch.no_grad()
def predict_patient(model, dataset, patient_id, channel=FLAIR_CHANNEL):
    """Run the model on test patient `patient_id` (1-indexed)."""
    n = len(dataset)
    if not 1 <= patient_id <= n:
        raise ValueError(f"patient_id must be in [1, {n}], got {patient_id}")

    sample = dataset[patient_id - 1]
    image = sample["image"].unsqueeze(0).to(DEVICE)

    probs = torch.sigmoid(model(image))
    pred = (probs > THRESHOLD)[0].cpu().numpy().astype(np.uint8)
    gt = torch.as_tensor(sample["label"]).cpu().numpy()

    mri = image[0, channel].detach().cpu().numpy()
    brain = mri != 0            # NormalizeIntensityd(nonzero=True) keeps background at 0
    return mri, brain, gt, pred


# ----------------------------------------------------------------------
# Qualitative figures
# ----------------------------------------------------------------------
def create_view(mri, gt, pred, brain, plane, patient_id, out_dir,
                aspect=1.0, show=True):
    mri, gt, pred, brain = crop_to_brain(brain, (mri, gt, pred, brain))

    if brain.any():
        vmin, vmax = np.percentile(mri[brain], (1, 99))
    else:
        vmin, vmax = float(mri.min()), float(mri.max())
    mri = np.where(brain, mri, vmin)        # background -> black

    H, W = mri.shape
    extent = [0, W, H, 0]

    fig, axes = plt.subplots(1, 3, figsize=QUAL_FIGSIZE, dpi=DPI, facecolor="black")
    titles = ["FLAIR MRI", "Ground Truth", "3D U-Net Prediction"]
    overlays = [None, gt, pred]

    for ax, title, overlay in zip(axes, titles, overlays):
        ax.set_facecolor("black")
        ax.imshow(mri, cmap="gray", vmin=vmin, vmax=vmax,
                  interpolation="nearest", extent=extent, aspect=aspect)
        if overlay is not None:
            ax.imshow(overlay, cmap=MASK_CMAP, vmin=0, vmax=3,
                      interpolation="nearest", extent=extent, aspect=aspect)
        ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=8)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect(aspect, adjustable="box")
        ax.axis("off")

    legend = axes[2].legend(handles=LEGEND_ELEMENTS, loc="lower right", fontsize=9,
                            frameon=True, facecolor="black", edgecolor="white",
                            framealpha=0.75, borderpad=0.6)
    for text in legend.get_texts():
        text.set_color("white")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.04)

    png_path = os.path.join(out_dir, f"Patient_{patient_id}_{plane}_600dpi.png")
    pdf_path = os.path.join(out_dir, f"Patient_{patient_id}_{plane}.pdf")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor="black", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="black", pad_inches=0.02)
    print(f"Saved: {os.path.abspath(png_path)}")
    print(f"Saved: {os.path.abspath(pdf_path)}")

    if show:
        plt.show()
    plt.close(fig)


def plot_patient(model, dataset, patient_id, out_dir, channel=FLAIR_CHANNEL,
                 equal_aspect=False, show=True):
    """Axial + coronal + sagittal figures for one test patient."""
    print("=" * 44)
    print(f"PATIENT {patient_id}")
    print("=" * 44)

    mri, brain, gt, pred = predict_patient(model, dataset, patient_id, channel)
    gt_label = make_label_map(gt, mri.shape)
    pred_label = make_label_map(pred, mri.shape)
    combined = np.maximum(gt_label, pred_label)

    z_aspect = 1.0 if equal_aspect else Z_ASPECT
    #        name        axes summed   slicing                aspect
    planes = [
        ("Axial",    (0, 1), lambda a, i: a[:, :, i], 1.0),
        ("Coronal",  (0, 2), lambda a, i: a[:, i, :], z_aspect),
        ("Sagittal", (1, 2), lambda a, i: a[i, :, :], z_aspect),
    ]

    for plane, sum_axes, take, aspect in planes:
        idx = int(np.argmax(combined.sum(axis=sum_axes)))
        print(f"Best {plane} slice: {idx}")
        views = [np.rot90(take(a, idx)) for a in (mri, gt_label, pred_label, brain)]
        create_view(*views, plane=plane, patient_id=patient_id, out_dir=out_dir,
                    aspect=aspect, show=show)


# ----------------------------------------------------------------------
# Quantitative figures (Fig 3 / Fig 4)
# ----------------------------------------------------------------------
def _load_metric(csv_file, column):
    df = pd.read_csv(csv_file)
    missing = [c for c in ("Patient", column) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    patients = pd.to_numeric(df["Patient"], errors="coerce")
    values = pd.to_numeric(df[column], errors="coerce")
    ok = patients.notna() & values.notna()
    patients = patients[ok].astype(int).to_numpy()
    values = values[ok].astype(float).to_numpy()
    order = np.argsort(patients)
    return patients[order], values[order]


def _save_figure(fig, out_dir, stem, show):
    png_path = os.path.join(out_dir, f"{stem}.png")
    pdf_path = os.path.join(out_dir, f"{stem}.pdf")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"PNG: {os.path.abspath(png_path)}  ({DPI} dpi)")
    print(f"PDF: {os.path.abspath(pdf_path)}")
    if show:
        plt.show()
    plt.close(fig)


def plot_per_patient_dice(csv_file, out_dir, show=True):
    """Fig 3 - per-patient Dice (no internal title)."""
    set_publication_style()
    patients, dice = _load_metric(csv_file, "Patient_Dice")
    mean, median, std = dice.mean(), np.median(dice), dice.std(ddof=1)
    print(f"Fig 3 | N={len(dice)} | mean={mean:.4f} median={median:.4f} std={std:.4f}")

    fig, ax = plt.subplots(figsize=FIG3_FIGSIZE)
    ax.plot(patients, dice, linewidth=1.4, marker="o", markersize=2.2,
            color="#1f77b4", label="Patient-level Dice")
    ax.axhline(mean, linestyle="--", linewidth=1.4, color="#d62728",
               label=f"Mean Dice = {mean:.4f}")
    ax.axhline(median, linestyle=":", linewidth=1.6, color="#2ca02c",
               label=f"Median Dice = {median:.4f}")
    ax.axhline(0.90, linestyle="-.", linewidth=1.2, color="#9467bd", label="Dice = 0.90")

    ax.set_xlabel("Patient", fontsize=10)
    ax.set_ylabel("Dice Similarity Coefficient (DSC)", fontsize=10)
    ax.set_xlim(patients.min(), patients.max())
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()

    _save_figure(fig, out_dir, "Fig3_Per_Patient_Dice_Native_Space", show)


def plot_per_patient_hd95(csv_file, out_dir, show=True):
    """Fig 4 - per-patient HD95 with the empty-mask penalty level (no internal title)."""
    set_publication_style()
    patients, hd95 = _load_metric(csv_file, "Patient_HD95_mm")
    mean, median = hd95.mean(), np.median(hd95)
    q1, q3 = np.quantile(hd95, [0.25, 0.75])
    print(f"Fig 4 | N={len(hd95)} | mean={mean:.4f} median={median:.4f} "
          f"Q1={q1:.4f} Q3={q3:.4f} IQR={q3 - q1:.4f} mm")

    fig, ax = plt.subplots(figsize=FIG4_FIGSIZE)
    ax.plot(patients, hd95, linewidth=1.4, marker="o", markersize=2.2,
            color="#1f77b4", label="Patient-level HD95")
    ax.axhline(mean, linestyle="--", linewidth=1.4, color="#d62728",
               label=f"Mean HD95 = {mean:.2f} mm")
    ax.axhline(median, linestyle=":", linewidth=1.6, color="#2ca02c",
               label=f"Median HD95 = {median:.2f} mm")
    ax.axhline(HD95_PENALTY, linestyle="-.", linewidth=1.2, color="#9467bd",
               label=f"Penalty = {HD95_PENALTY:.2f} mm")

    ax.set_xlabel("Patient", fontsize=10)
    ax.set_ylabel("HD95 (mm)", fontsize=10)
    ax.set_xlim(patients.min(), patients.max())
    ax.set_ylim(0, max(HD95_PENALTY, hd95.max()) * 1.05)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()

    _save_figure(fig, out_dir, "Fig4_Per_Patient_HD95_Native_Space", show)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize BraTS segmentation results")
    parser.add_argument("--patient-id", type=int, nargs="*", default=[],
                        help="1-indexed test-set patients for the qualitative figures")
    parser.add_argument("--summary", action="store_true",
                        help="also draw Fig 3 (Dice) and Fig 4 (HD95) from the CSV")
    parser.add_argument("--checkpoint", default=BEST_MODEL_PATH)
    parser.add_argument("--results-csv", default=os.path.join(OUTPUT_DIR, "native_space_test.csv"))
    parser.add_argument("--out-dir", default=OUTPUT_DIR)
    parser.add_argument("--channel", type=int, default=FLAIR_CHANNEL,
                        help="input channel shown as MRI: 0=T1, 1=T1ce, 2=T2, 3=FLAIR")
    parser.add_argument("--equal-aspect", action="store_true",
                        help="do not correct the z-axis voxel aspect in coronal/sagittal views")
    parser.add_argument("--no-show", action="store_true",
                        help="save only (headless servers)")
    args = parser.parse_args()

    if not args.patient_id and not args.summary:
        parser.error("nothing to do: pass --patient-id and/or --summary")

    if args.no_show:
        plt.switch_backend("Agg")
    show = not args.no_show
    os.makedirs(args.out_dir, exist_ok=True)

    if args.patient_id:
        model = build_model().to(DEVICE)
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=True))
        model.eval()
        _, _, test_loader = build_dataloaders()
        for pid in args.patient_id:
            plot_patient(model, test_loader.dataset, pid, args.out_dir,
                         channel=args.channel, equal_aspect=args.equal_aspect, show=show)

    if args.summary:
        plot_per_patient_dice(args.results_csv, args.out_dir, show=show)
        plot_per_patient_hd95(args.results_csv, args.out_dir, show=show)


if __name__ == "__main__":
    main()
