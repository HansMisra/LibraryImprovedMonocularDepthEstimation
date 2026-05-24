# evaluate module

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def apply_transformation(predicted, transformation):
    """
    Apply a simple linear transformation to model predictions.

    Args:
        predicted: torch.Tensor of predicted disparity values.
        transformation: np.array or list like [scale, bias].

    Returns:
        torch.Tensor with transformed predictions.
    """
    if transformation is None or len(transformation) == 0:
        return predicted

    transformation = torch.tensor(
        transformation,
        dtype=torch.float32,
        device=predicted.device
    )

    scale, bias = transformation[0], transformation[1]
    return scale * predicted + bias


def percentile_accuracy(y_true, y_pred, tolerance=0.25):
    """
    Calculates the fraction of predictions within a fixed absolute tolerance.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.
        tolerance: Maximum absolute error counted as correct.

    Returns:
        Float between 0 and 1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    return np.mean(np.abs(y_true - y_pred) <= tolerance)


def evaluate_model(model, data_loader, device, transformation_matrix=None, evaluate_transform=False):
    model.eval()

    errors = []
    accuracies = []
    precisions = []
    percentile_accuracies = []
    image_infos = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            true_disparities = batch["disparity"].to(device)

            predictions = model(images)

            if predictions.shape != true_disparities.shape:
                predictions = torch.nn.functional.interpolate(
                    predictions,
                    size=true_disparities.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )

            if evaluate_transform and transformation_matrix is not None:
                predictions = apply_transformation(predictions, transformation_matrix)

            true_flat = true_disparities.view(-1).cpu().numpy()
            pred_flat = predictions.view(-1).cpu().numpy()

            rmse_error = mean_squared_error(true_flat, pred_flat, squared=False)
            binary_true = true_flat > 0.5
            binary_pred = pred_flat > 0.5

            accuracy = accuracy_score(binary_true, binary_pred)
            precision = precision_score(
                binary_true,
                binary_pred,
                average="macro",
                zero_division=1
            )
            pct_accuracy = percentile_accuracy(true_flat, pred_flat, tolerance=0.25)

            errors.append(rmse_error)
            accuracies.append(accuracy)
            precisions.append(precision)
            percentile_accuracies.append(pct_accuracy)

            image_infos.append(
                (
                    images.cpu(),
                    true_disparities.cpu(),
                    predictions.cpu(),
                    rmse_error
                )
            )

    return errors, accuracies, precisions, percentile_accuracies, image_infos


def display_image_results(image_infos):
    if not image_infos:
        print("No image results to display.")
        return

    fig, axs = plt.subplots(len(image_infos), 3, figsize=(15, 5 * len(image_infos)))

    if len(image_infos) == 1:
        axs = np.expand_dims(axs, axis=0)

    for idx, (images, true_disp, pred_disp, error) in enumerate(image_infos):
        axs[idx, 0].imshow(images[0].permute(1, 2, 0))
        axs[idx, 0].set_title(f"Original Image - RMSE: {error:.2f}")
        axs[idx, 0].axis("off")

        axs[idx, 1].imshow(true_disp[0].squeeze(), cmap="plasma")
        axs[idx, 1].set_title("Target Disparity")
        axs[idx, 1].axis("off")

        axs[idx, 2].imshow(pred_disp[0].squeeze(), cmap="plasma")
        axs[idx, 2].set_title("Predicted Disparity")
        axs[idx, 2].axis("off")

    plt.tight_layout()
    plt.show()