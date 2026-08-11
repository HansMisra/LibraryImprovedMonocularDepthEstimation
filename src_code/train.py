import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data_utils import KITTIDataset, eval_transforms, train_transforms
from evaluate import evaluate_model
from model import DepthNet
from split_utils import load_or_create_split


script_dir = os.path.dirname(os.path.abspath(__file__))


def masked_mse(predictions, targets):
    valid = (
        (targets > 0)
        & torch.isfinite(targets)
        & torch.isfinite(predictions)
    )

    if not valid.any():
        return None

    return ((predictions - targets) ** 2)[valid].mean()


def train(
    image_dir,
    disparity_dir,
    epochs=10,
    batch_size=8,
    save_path=None,
    split_path=None,
    val_fraction=0.2,
    seed=42,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    if split_path is None:
        split_path = os.path.join(script_dir, "split.json")

    train_full = KITTIDataset(
        image_dir,
        disparity_dir,
        transform=train_transforms,
    )
    val_full = KITTIDataset(
        image_dir,
        disparity_dir,
        transform=eval_transforms,
    )

    train_indices, val_indices = load_or_create_split(
        train_full,
        split_path,
        val_fraction=val_fraction,
        seed=seed,
    )

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, min(batch_size, 4)),
        shuffle=False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Split file: {split_path}")

    model = DepthNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_d1 = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        valid_batches = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            leave=False,
        )

        for batch_idx, batch in enumerate(progress, start=1):
            images = batch["image"].to(device)
            targets = batch["disparity"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            outputs = F.interpolate(
                outputs,
                size=targets.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

            loss = masked_mse(outputs, targets)
            if loss is None:
                continue

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            valid_batches += 1
            progress.set_postfix(
                batch=f"{batch_idx}/{len(train_loader)}",
                loss=f"{loss.item():.4f}",
            )

        avg_loss = running_loss / max(valid_batches, 1)
        val_metrics, _ = evaluate_model(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train MSE {avg_loss:.4f} | "
            f"val EPE {val_metrics['epe']:.4f} | "
            f"val RMSE {val_metrics['rmse']:.4f} | "
            f"val D1 {val_metrics['d1'] * 100:.2f}%"
        )

        if save_path and val_metrics["d1"] < best_d1:
            best_d1 = val_metrics["d1"]
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    return model


if __name__ == "__main__":
    image_dir = os.path.join(
        "kitti_data", "data_scene_flow", "training", "image_2"
    )
    disparity_dir = os.path.join(
        "kitti_data", "data_scene_flow", "training", "disp_occ_0"
    )
    save_path = "model_weights.pth"

    train(image_dir, disparity_dir, save_path=save_path)
