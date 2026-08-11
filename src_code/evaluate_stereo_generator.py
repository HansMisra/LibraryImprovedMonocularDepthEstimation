import os

import cv2
import numpy as np
from tqdm import tqdm

from create_test_disp import generate_disparity_map, preprocess_image
from split_utils import load_split_names


def load_kitti_disparity(path):
    disparity = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if disparity is None:
        raise FileNotFoundError(path)
    return disparity.astype(np.float32) / 256.0


def evaluate_stereo_generator(
    left_dir,
    right_dir,
    gt_dir,
    split_path,
    split_name="val",
):
    filenames = load_split_names(split_path, split_name)

    total_gt_valid = 0
    total_covered = 0
    abs_sum = 0.0
    sq_sum = 0.0
    bad_1_count = 0
    bad_3_count = 0
    d1_count = 0

    for filename in tqdm(filenames, desc="Evaluating stereo teacher", unit="pair"):
        left = preprocess_image(os.path.join(left_dir, filename))
        right = preprocess_image(os.path.join(right_dir, filename))
        gt = load_kitti_disparity(os.path.join(gt_dir, filename))

        pred, confidence, pred_valid = generate_disparity_map(left, right)

        gt_valid = np.isfinite(gt) & (gt > 0)
        compare_valid = gt_valid & pred_valid & np.isfinite(pred)

        total_gt_valid += int(gt_valid.sum())
        total_covered += int(compare_valid.sum())

        if not np.any(compare_valid):
            continue

        error = np.abs(pred[compare_valid] - gt[compare_valid])
        relative_error = error / np.maximum(gt[compare_valid], 1e-6)

        abs_sum += float(error.sum())
        sq_sum += float((error ** 2).sum())
        bad_1_count += int((error > 1.0).sum())
        bad_3_count += int((error > 3.0).sum())
        d1_count += int(((error > 3.0) & (relative_error > 0.05)).sum())

    if total_covered == 0:
        raise ValueError("Stereo generator produced no valid comparable pixels.")

    print(f"GT valid pixels: {total_gt_valid:,}")
    print(f"Teacher covered pixels: {total_covered:,}")
    print(f"Coverage: {100 * total_covered / total_gt_valid:.2f}%")
    print(f"EPE: {abs_sum / total_covered:.4f} px")
    print(f"RMSE: {np.sqrt(sq_sum / total_covered):.4f} px")
    print(f"Bad-1: {100 * bad_1_count / total_covered:.2f}%")
    print(f"Bad-3: {100 * bad_3_count / total_covered:.2f}%")
    print(f"D1: {100 * d1_count / total_covered:.2f}%")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(script_dir, "kitti_data", "data_scene_flow", "training")

    evaluate_stereo_generator(
        left_dir=os.path.join(root, "image_2"),
        right_dir=os.path.join(root, "image_3"),
        gt_dir=os.path.join(root, "disp_occ_0"),
        split_path=os.path.join(script_dir, "split.json"),
    )
