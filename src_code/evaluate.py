import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def disparity_metrics(pred, target, valid_mask=None):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    valid = (
        (target > 0)
        & np.isfinite(target)
        & np.isfinite(pred)
    )

    if valid_mask is not None:
        valid &= np.asarray(valid_mask).astype(bool)

    if not np.any(valid):
        return None

    pred = pred[valid]
    target = target[valid]
    error = np.abs(pred - target)
    relative_error = error / np.maximum(target, 1e-6)

    return {
        "valid_pixels": int(valid.sum()),
        "epe": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bad_1": float(np.mean(error > 1.0)),
        "bad_3": float(np.mean(error > 3.0)),
        "d1": float(np.mean((error > 3.0) & (relative_error > 0.05))),
    }


def evaluate_model(model, data_loader, device):
    model.eval()

    abs_error_sum = 0.0
    sq_error_sum = 0.0
    bad_1_count = 0
    bad_3_count = 0
    d1_count = 0
    valid_count = 0
    image_infos = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", unit="batch", leave=False):
            images = batch["image"].to(device)
            targets = batch["disparity"].to(device)

            predictions = model(images)
            if predictions.shape != targets.shape:
                predictions = F.interpolate(
                    predictions,
                    size=targets.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

            valid = (
                (targets > 0)
                & torch.isfinite(targets)
                & torch.isfinite(predictions)
            )

            if valid.any():
                error = torch.abs(predictions - targets)
                relative_error = error / torch.clamp(targets, min=1e-6)

                abs_error_sum += error[valid].sum().item()
                sq_error_sum += (error[valid] ** 2).sum().item()
                bad_1_count += (error[valid] > 1.0).sum().item()
                bad_3_count += (error[valid] > 3.0).sum().item()
                d1_count += (
                    (error[valid] > 3.0)
                    & (relative_error[valid] > 0.05)
                ).sum().item()
                valid_count += valid.sum().item()

            for i in range(images.shape[0]):
                sample_valid = valid[i]
                if sample_valid.any():
                    sample_error = torch.abs(predictions[i] - targets[i])
                    sample_rmse = torch.sqrt(
                        (sample_error[sample_valid] ** 2).mean()
                    ).item()
                else:
                    sample_rmse = math.inf

                image_infos.append(
                    (
                        images[i].cpu(),
                        targets[i].cpu(),
                        predictions[i].cpu(),
                        sample_rmse,
                    )
                )

    if valid_count == 0:
        raise ValueError("No valid disparity pixels were found during evaluation.")

    metrics = {
        "valid_pixels": int(valid_count),
        "epe": abs_error_sum / valid_count,
        "rmse": math.sqrt(sq_error_sum / valid_count),
        "bad_1": bad_1_count / valid_count,
        "bad_3": bad_3_count / valid_count,
        "d1": d1_count / valid_count,
    }

    return metrics, image_infos


def display_image_results(image_infos):
    if not image_infos:
        print("No image results to display.")
        return

    fig, axs = plt.subplots(len(image_infos), 3, figsize=(15, 5 * len(image_infos)))

    if len(image_infos) == 1:
        axs = np.expand_dims(axs, axis=0)

    for idx, (image, true_disp, pred_disp, error) in enumerate(image_infos):
        axs[idx, 0].imshow(image.permute(1, 2, 0))
        axs[idx, 0].set_title(f"Original - RMSE: {error:.2f}")
        axs[idx, 0].axis("off")

        axs[idx, 1].imshow(true_disp.squeeze(), cmap="plasma")
        axs[idx, 1].set_title("Target Disparity")
        axs[idx, 1].axis("off")

        axs[idx, 2].imshow(pred_disp.squeeze(), cmap="plasma")
        axs[idx, 2].set_title("Predicted Disparity")
        axs[idx, 2].axis("off")

    plt.tight_layout()
    plt.show()
