import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data_utils import KITTIDataset, eval_transforms
from evaluate import display_image_results, evaluate_model
from load_model import load_model
from split_utils import load_or_create_split


def run_evaluation():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    image_dir = os.path.join(
        script_dir,
        "kitti_data",
        "data_scene_flow",
        "training",
        "image_2",
    )
    disparity_dir = os.path.join(
        script_dir,
        "kitti_data",
        "data_scene_flow",
        "training",
        "disp_occ_0",
    )
    model_path = os.path.join(script_dir, "model_weights.pth")
    split_path = os.path.join(script_dir, "split.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    dataset = KITTIDataset(
        image_dir,
        disparity_dir,
        transform=eval_transforms,
    )

    _, val_indices = load_or_create_split(dataset, split_path)
    val_dataset = Subset(dataset, val_indices)
    data_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = load_model(model_path, "depthnet", device=device)
    metrics, image_infos = evaluate_model(model, data_loader, device)

    finite_infos = [info for info in image_infos if np.isfinite(info[3])]
    if finite_infos:
        finite_infos.sort(key=lambda x: x[3])
        selected = [
            finite_infos[-1],
            finite_infos[0],
            finite_infos[len(finite_infos) // 2],
        ]
        display_image_results(selected)

    print(f"Valid pixels: {metrics['valid_pixels']:,}")
    print(f"EPE: {metrics['epe']:.4f} px")
    print(f"RMSE: {metrics['rmse']:.4f} px")
    print(f"Bad-1: {metrics['bad_1'] * 100:.2f}%")
    print(f"Bad-3: {metrics['bad_3'] * 100:.2f}%")
    print(f"D1: {metrics['d1'] * 100:.2f}%")


if __name__ == "__main__":
    run_evaluation()
