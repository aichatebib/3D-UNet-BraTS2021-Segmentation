"""Evaluate the trained model in native BraTS space (240x240x155).

Predictions and ground-truth masks (produced at 128^3) are resized back
to native resolution with nearest-neighbor interpolation before computing
per-region Dice and HD95 (95th percentile Hausdorff Distance).
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt, zoom
from tqdm.auto import tqdm

from config import DEVICE, NATIVE_SHAPE, EMPTY_MASK_PENALTY, BEST_MODEL_PATH, OUTPUT_DIR
from dataset import build_dataloaders
from model import build_model

REGION_NAMES = ["WT", "TC", "ET"]


def resize_to_native(mask, native_shape=NATIVE_SHAPE):
    """Resize a binary mask from 128^3 to native space (nearest-neighbor)."""
    mask = np.asarray(mask).astype(np.uint8)
    zoom_factors = [native_shape[i] / mask.shape[i] for i in range(3)]
    native = zoom(mask, zoom_factors, order=0)
    return native.astype(bool)


def dice_binary(pred, target):
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)

    pred_sum = pred.sum()
    target_sum = target.sum()

    if pred_sum == 0 and target_sum == 0:
        return 1.0
    if pred_sum == 0 or target_sum == 0:
        return 0.0

    intersection = np.logical_and(pred, target).sum()
    return (2.0 * intersection) / (pred_sum + target_sum)


def hd95_binary(pred, target, penalty=EMPTY_MASK_PENALTY):
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)

    pred_empty = not pred.any()
    target_empty = not target.any()

    if pred_empty and target_empty:
        return 0.0
    if pred_empty or target_empty:
        return penalty

    dt_target = distance_transform_edt(~target)
    dt_pred = distance_transform_edt(~pred)

    pred_distances = dt_target[pred]
    target_distances = dt_pred[target]

    all_distances = np.concatenate([pred_distances, target_distances])
    return float(np.percentile(all_distances, 95))


def evaluate_loader(model, loader, dataset_name: str, output_csv: str) -> pd.DataFrame:
    results = []

    print()
    print("=" * 70)
    print(f"NATIVE-SPACE EVALUATION — {dataset_name}")
    print("=" * 70)

    for patient_idx, batch in enumerate(tqdm(loader, desc=f"{dataset_name} evaluation")):
        image = batch["image"].to(DEVICE)
        label = batch["label"].to(DEVICE)

        with torch.no_grad():
            output = model(image)
            prediction = (torch.sigmoid(output) > 0.5).float()

        prediction = prediction[0].detach().cpu().numpy()
        label_np = label[0].detach().cpu().numpy()

        patient_result = {"Patient": patient_idx + 1}
        region_dice, region_hd95 = [], []

        for channel, region_name in enumerate(REGION_NAMES):
            pred_128 = prediction[channel] >= 0.5
            target_128 = label_np[channel] >= 0.5

            pred_native = resize_to_native(pred_128)
            target_native = resize_to_native(target_128)

            dice = dice_binary(pred_native, target_native)
            hd95 = hd95_binary(pred_native, target_native)

            patient_result[f"{region_name}_Dice"] = dice
            patient_result[f"{region_name}_HD95_mm"] = hd95

            region_dice.append(dice)
            region_hd95.append(hd95)

        patient_result["Patient_Dice"] = np.mean(region_dice)
        patient_result["Patient_HD95_mm"] = np.mean(region_hd95)

        results.append(patient_result)

        if len(results) % 25 == 0:
            pd.DataFrame(results).to_csv(output_csv, index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    print()
    print(f"{dataset_name} complete. Patients evaluated: {len(results_df)}")
    print("Saved:", output_csv)

    return results_df


def print_summary(name: str, df: pd.DataFrame):
    print(f"\n===== {name} =====")
    print("N:", len(df))
    print(f"Mean Dice   : {df['Patient_Dice'].mean():.4f}")
    print(f"Median Dice : {df['Patient_Dice'].median():.4f}")
    print(f"Std Dice    : {df['Patient_Dice'].std():.4f}")
    print(f"Mean HD95   : {df['Patient_HD95_mm'].mean():.4f} mm")
    print(f"Median HD95 : {df['Patient_HD95_mm'].median():.4f} mm")
    print(f"Q1 HD95     : {df['Patient_HD95_mm'].quantile(0.25):.4f} mm")
    print(f"Q3 HD95     : {df['Patient_HD95_mm'].quantile(0.75):.4f} mm")


def main():
    parser = argparse.ArgumentParser(description="Native-space BraTS evaluation")
    parser.add_argument("--checkpoint", default=BEST_MODEL_PATH)
    parser.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    args = parser.parse_args()

    model = build_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()
    print("Loaded checkpoint:", args.checkpoint)

    train_loader, val_loader, test_loader = build_dataloaders()

    loaders = {
        "train": ("TRAINING", train_loader, os.path.join(OUTPUT_DIR, "native_space_train.csv")),
        "val": ("VALIDATION", val_loader, os.path.join(OUTPUT_DIR, "native_space_val.csv")),
        "test": ("TEST", test_loader, os.path.join(OUTPUT_DIR, "native_space_test.csv")),
    }

    to_run = loaders.keys() if args.split == "all" else [args.split]

    summaries = {}
    for split in to_run:
        name, loader, out_csv = loaders[split]
        df = evaluate_loader(model, loader, name, out_csv)
        summaries[name] = df

    print("\n" + "=" * 70)
    print("FINAL NATIVE-SPACE SUMMARY")
    print("=" * 70)
    for name, df in summaries.items():
        print_summary(name, df)


if __name__ == "__main__":
    main()
