import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from data_utils import KITTIDataset
from load_model import load_model
from evaluate import evaluate_model, display_image_results


def run_evaluation():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(
        script_dir,
        "kitti_data",
        "data_scene_flow",
        "testing",
        "image_2"
    )

    disparity_dir = os.path.join(
        script_dir,
        "kitti_data",
        "data_scene_flow",
        "testing",
        "test_disp"
    )

    model_path = os.path.join(script_dir, "model_weights.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    if not os.path.exists(data_dir):
        print(f"Image directory not found: {data_dir}")
        return

    if not os.path.exists(disparity_dir):
        print(f"Disparity directory not found: {disparity_dir}")
        return

    if not os.path.exists(model_path):
        print(f"Model weights not found: {model_path}")
        print("Train the model first, or place model_weights.pth in src_code.")
        return

    model = load_model(model_path, "depthnet", device=device)

    dataset = KITTIDataset(
        data_dir,
        disparity_dir,
        transform=ToTensor()
    )

    data_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False
    )

    errors, accuracies, precisions, percentile_accuracies, image_infos = evaluate_model(
        model,
        data_loader,
        device
    )

    if not errors:
        print("No evaluation results were produced.")
        return

    max_error_idx = int(np.argmax(errors))
    min_error_idx = int(np.argmin(errors))
    median_error_idx = int(np.argsort(errors)[len(errors) // 2])

    selected_infos = [
        image_infos[max_error_idx],
        image_infos[min_error_idx],
        image_infos[median_error_idx]
    ]

    display_image_results(selected_infos)

    print(f"Average RMSE: {np.mean(errors):.4f}")
    print(f"Average Accuracy: {np.mean(accuracies):.4f}")
    print(f"Average Precision: {np.mean(precisions):.4f}")
    print(f"Percentile Accuracy within tolerance 0.25: {np.mean(percentile_accuracies):.4f}")
    print(f"Max RMSE: {errors[max_error_idx]:.4f}")
    print(f"Min RMSE: {errors[min_error_idx]:.4f}")
    print(f"Median RMSE: {errors[median_error_idx]:.4f}")


if __name__ == "__main__":
    run_evaluation()