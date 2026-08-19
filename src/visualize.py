"""Qualitative visualization of model predictions vs. ground truth,
and per-patient Dice / HD95 summary plots from a results CSV."""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config import DEVICE, BEST_MODEL_PATH, OUTPUT_DIR
from dataset import build_dataloaders
from model import build_model

REGION_COLORS = {
    "WT": {"fill": "red", "contour": "#FF3333"},
    "TC": {"fill": "orange", "contour": "#FFA500"},
    "ET": {"fill": "lime", "contour": "#00FF00"},
}


def plot_patient_result(model, test_loader, patient_id: int, dice_score=None, hd95_score=None,
                         save_path: str = None):
    """Plot FLAIR / Ground Truth / Prediction for a single test-set patient.

    `patient_id` is 1-indexed to match the evaluation CSV's "Patient" column.
    """
    model.eval()

    sample = test_loader.dataset[patient_id - 1]

    image = sample["image"].unsqueeze(0).to(DEVICE)
    label = sample["label"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)

    prediction = (torch.sigmoid(output) > 0.5).float()
    pred_np = prediction[0].cpu().numpy()
    gt_np = label[0].cpu().numpy()

    flair = image[0, 0].cpu().numpy()
    flair = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)

    # Slice with the largest predicted tumor area
    z_pred = pred_np.sum(axis=0).sum(axis=(0, 1))
    z = int(np.argmax(z_pred))

    fig, ax = plt.subplots(1, 3, figsize=(18, 6), facecolor="black", dpi=150)
    for a in ax:
        a.set_facecolor("black")

    ax[0].imshow(flair[:, :, z], cmap="gray")
    ax[0].set_title("Original FLAIR MRI", color="white", fontsize=13, fontweight="bold", pad=10)
    ax[0].axis("off")

    ax[1].imshow(flair[:, :, z], cmap="gray")
    for idx, (name, col) in enumerate(REGION_COLORS.items()):
        mask = gt_np[idx, :, :, z]
        if mask.sum() > 0:
            ax[1].contour(mask, colors=[col["contour"]], levels=[0.5], linewidths=1.5)
            ax[1].imshow(
                np.ma.masked_where(mask == 0, mask),
                cmap=plt.cm.colors.ListedColormap([col["fill"]]),
                alpha=0.2,
            )
    ax[1].set_title("Ground Truth (Contours)", color="cyan", fontsize=13, fontweight="bold", pad=10)
    ax[1].axis("off")

    ax[2].imshow(flair[:, :, z], cmap="gray")
    for idx, (name, col) in enumerate(REGION_COLORS.items()):
        mask = pred_np[idx, :, :, z]
        if mask.sum() > 0:
            ax[2].contour(mask, colors=[col["contour"]], levels=[0.5], linewidths=1.5)
            ax[2].imshow(
                np.ma.masked_where(mask == 0, mask),
                cmap=plt.cm.colors.ListedColormap([col["fill"]]),
                alpha=0.2,
            )
    ax[2].set_title("Model Prediction", color="magenta", fontsize=13, fontweight="bold", pad=10)
    ax[2].axis("off")

    legend_elements = [
        Patch(facecolor=REGION_COLORS["WT"]["fill"], edgecolor=REGION_COLORS["WT"]["contour"], alpha=0.6, label="WT (Whole Tumor)"),
        Patch(facecolor=REGION_COLORS["TC"]["fill"], edgecolor=REGION_COLORS["TC"]["contour"], alpha=0.6, label="TC (Tumor Core)"),
        Patch(facecolor=REGION_COLORS["ET"]["fill"], edgecolor=REGION_COLORS["ET"]["contour"], alpha=0.6, label="ET (Enhancing Tumor)"),
    ]
    leg = ax[2].legend(handles=legend_elements, loc="lower right", fontsize=9, facecolor="#111111", edgecolor="none")
    for text in leg.get_texts():
        text.set_color("white")

    title = f"Patient {patient_id}"
    if dice_score is not None and hd95_score is not None:
        title += f"   |   Dice Coefficient = {dice_score:.4f}   |   HD95 = {hd95_score:.4f}"

    plt.suptitle(title, fontsize=16, color="white", fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        print("Saved:", save_path)

    plt.show()


def plot_per_patient_metric(csv_file: str, metric: str = "Patient_Dice", save_prefix: str = None):
    """Plot a per-patient metric curve (Dice or HD95) with mean/median lines."""
    df = pd.read_csv(csv_file)

    patients = df["Patient"].astype(int)
    values = df[metric].astype(float)

    mean_val = values.mean()
    median_val = values.median()

    plt.figure(figsize=(13, 5.5))
    plt.plot(patients, values, color="royalblue", linewidth=1.3, marker="o", markersize=2.5, label=metric)
    plt.axhline(mean_val, color="red", linestyle="--", linewidth=2.0, label=f"Mean = {mean_val:.4f}")
    plt.axhline(median_val, color="green", linestyle=":", linewidth=2.2, label=f"Median = {median_val:.4f}")

    plt.xlabel("Patient", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.title(f"Per-patient {metric} (N={len(values)})", fontsize=13)
    plt.xlim(1, len(values))
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    if save_prefix:
        plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
        print(f"Saved: {save_prefix}.png / .pdf")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize BraTS segmentation results")
    parser.add_argument("--patient-id", type=int, required=True, help="1-indexed patient in the test set")
    parser.add_argument("--checkpoint", default=BEST_MODEL_PATH)
    parser.add_argument("--results-csv", default=os.path.join(OUTPUT_DIR, "native_space_test.csv"))
    args = parser.parse_args()

    model = build_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))

    _, _, test_loader = build_dataloaders()

    dice_score, hd95_score = None, None
    if os.path.exists(args.results_csv):
        df = pd.read_csv(args.results_csv)
        row = df[df["Patient"] == args.patient_id]
        if not row.empty:
            dice_score = float(row["Patient_Dice"].iloc[0])
            hd95_score = float(row["Patient_HD95_mm"].iloc[0])

    save_path = os.path.join(OUTPUT_DIR, f"patient_{args.patient_id}.png")
    plot_patient_result(model, test_loader, args.patient_id, dice_score, hd95_score, save_path=save_path)


if __name__ == "__main__":
    main()
