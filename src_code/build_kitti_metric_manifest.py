import argparse
import json
import os
from pathlib import Path

import numpy as np

from depth_manifest import write_manifest
from split_utils import load_split_names


def parse_calibration(path):
    values = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            nums = raw.strip().split()
            if nums:
                values[key.strip()] = np.array([float(x) for x in nums], dtype=np.float64)

    left_key = next((k for k in ("P_rect_02", "P2", "P_rect_2") if k in values), None)
    right_key = next((k for k in ("P_rect_03", "P3", "P_rect_3") if k in values), None)
    if left_key is None or right_key is None:
        raise ValueError(f"Could not find left/right projection matrices in {path}")

    p2 = values[left_key].reshape(3, 4)
    p3 = values[right_key].reshape(3, 4)
    fx = float(p2[0, 0])
    fy = float(p2[1, 1])
    cx = float(p2[0, 2])
    cy = float(p2[1, 2])
    center2_x = -float(p2[0, 3]) / fx
    center3_x = -float(p3[0, 3]) / float(p3[0, 0])
    baseline_m = abs(center3_x - center2_x)

    return fx, fy, cx, cy, baseline_m


def calibration_filename(frame_name):
    return frame_name.replace("_10.png", ".txt")


def build_manifest(args):
    allowed = None
    if args.split_file:
        allowed = set(load_split_names(args.split_file, args.split))

    image_files = sorted(f for f in os.listdir(args.image_dir) if f.endswith("_10.png"))
    samples = []

    for frame in image_files:
        if allowed is not None and frame not in allowed:
            continue

        image_path = Path(args.image_dir) / frame
        disparity_path = Path(args.disparity_dir) / frame
        calib_path = Path(args.calib_dir) / calibration_filename(frame)
        if not disparity_path.exists() or not calib_path.exists():
            continue

        fx, fy, cx, cy, baseline_m = parse_calibration(calib_path)
        samples.append(
            {
                "id": f"kitti2015:{frame}",
                "dataset": "kitti2015",
                "image_path": str(image_path.resolve()),
                "target_path": str(disparity_path.resolve()),
                "target_encoding": "kitti_disparity_png256",
                "target_source": "kitti_ground_truth_disparity",
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "baseline_m": baseline_m,
                "max_depth_m": args.max_depth,
            }
        )

    if not samples:
        raise ValueError("No KITTI samples were emitted. Check image, disparity, calibration, and split paths.")

    write_manifest(samples, args.output)
    print(f"Saved {len(samples)} KITTI samples to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--disparity-dir", required=True)
    parser.add_argument("--calib-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--max-depth", type=float, default=100.0)
    build_manifest(parser.parse_args())
