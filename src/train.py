"""Train the 3D U-Net on BraTS 2021 with AMP, checkpointing and resume."""

import os
import time

import torch
from torch import amp
from monai.losses import DiceLoss

from config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LOG_EVERY_N_BATCHES,
    BEST_MODEL_PATH,
    CHECKPOINT_PATH,
)
from dataset import build_dataloaders
from model import build_model


def train():
    train_loader, val_loader, _ = build_dataloaders()

    model = build_model()

    criterion = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = amp.GradScaler("cuda")

    start_epoch = 0
    best_val_loss = float("inf")

    # ----------------------------------------------------------------
    # Resume from checkpoint if available
    # ----------------------------------------------------------------
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

        print("=" * 60)
        print(f"Resuming training from epoch {start_epoch + 1}")
        print("=" * 60)

    elif os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        print("=" * 60)
        print("Loaded best_model.pth. Training will continue from this model.")
        print("=" * 60)

    else:
        print("=" * 60)
        print("No checkpoint found. Training from scratch.")
        print("=" * 60)

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()

        # ---------------- TRAIN ----------------
        model.train()
        running_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()

            if (batch_idx + 1) % LOG_EVERY_N_BATCHES == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_train_loss = running_train_loss / len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE, non_blocking=True)
                labels = batch["label"].to(DEVICE, non_blocking=True)

                with amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)

        # ---------------- SAVE BEST MODEL ----------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("Best model saved.")

        # ---------------- SAVE CHECKPOINT ----------------
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_loss": best_val_loss,
            },
            CHECKPOINT_PATH,
        )

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Epoch           : {epoch + 1}/{EPOCHS}")
        print(f"Train Loss      : {avg_train_loss:.4f}")
        print(f"Validation Loss : {avg_val_loss:.4f}")
        print(f"Best Val Loss   : {best_val_loss:.4f}")
        print(f"Time            : {elapsed:.1f} seconds")
        print("Checkpoint saved.")
        print("=" * 60 + "\n")

    print("Training finished.")


if __name__ == "__main__":
    train()
