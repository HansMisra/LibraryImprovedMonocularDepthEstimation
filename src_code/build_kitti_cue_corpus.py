import argparse
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)

from split_utils import load_split_names


def load_rgb(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_disparity(path):
    disparity = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if disparity is None:
        raise FileNotFoundError(path)
    return disparity.astype(np.float32) / 256.0


def mask_touches_border(mask):
    return bool(
        mask[0].any()
        or mask[-1].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )


def entropy_from_gray(gray, mask):
    values = gray[mask]
    if values.size < 2:
        return None

    hist, _ = np.histogram(values, bins=32, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    hist = hist[hist > 0]
    if hist.size == 0:
        return None

    p = hist / hist.sum()
    return float(-(p * np.log2(p)).sum())


def sharpness_features(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = mask.astype(np.uint8)
    eroded_1 = cv2.erode(mask_u8, kernel, iterations=1).astype(bool)
    eroded_2 = cv2.erode(mask_u8, kernel, iterations=2).astype(bool)

    interior = eroded_2 if eroded_2.sum() >= 25 else mask
    boundary = mask & ~eroded_1

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
    high_frequency = gray - blurred

    interior_values = gray[interior]
    gradient_values = gradient[interior]
    laplacian_values = laplacian[interior]
    high_frequency_values = high_frequency[interior]
    boundary_gradient = gradient[boundary]

    if interior_values.size == 0:
        return {}

    contrast = float(
        np.quantile(interior_values, 0.95)
        - np.quantile(interior_values, 0.05)
    )

    gradient_p90 = (
        float(np.quantile(gradient_values, 0.90))
        if gradient_values.size
        else 0.0
    )
    boundary_gradient_p90 = (
        float(np.quantile(boundary_gradient, 0.90))
        if boundary_gradient.size
        else 0.0
    )

    return {
        "sharp_laplacian_variance": (
            float(np.var(laplacian_values))
            if laplacian_values.size
            else None
        ),
        "sharp_tenengrad_mean": (
            float(np.mean(gradient_values ** 2))
            if gradient_values.size
            else None
        ),
        "sharp_gradient_p90": gradient_p90,
        "sharp_high_frequency_rms": (
            float(np.sqrt(np.mean(high_frequency_values ** 2)))
            if high_frequency_values.size
            else None
        ),
        "sharp_local_contrast": contrast,
        "sharp_boundary_gradient_p90": boundary_gradient_p90,
        "sharp_line_tightness": float(
            boundary_gradient_p90 / max(contrast, 1e-6)
        ),
        "texture_entropy": entropy_from_gray(gray, interior),
    }


def make_instance_predictions(image, model, transform, device):
    tensor = transform(Image.fromarray(image)).to(device)
    with torch.no_grad():
        output = model([tensor])[0]
    return {key: value.detach().cpu() for key, value in output.items()}


def build_corpus(args):
    os.makedirs(args.output_dir, exist_ok=True)
    records_path = os.path.join(args.output_dir, "instances.jsonl")

    allowed = set(load_split_names(args.split_file, args.split))
    image_files = sorted(
        filename
        for filename in os.listdir(args.image_dir)
        if filename.endswith("_10.png")
        and filename in allowed
        and os.path.exists(os.path.join(args.disparity_dir, filename))
    )

    if not image_files:
        raise ValueError(
            f"No matching {args.split} images found. Check image/disparity paths and split.json."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    print(f"Split: {args.split} | frames: {len(image_files)}")

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
    transform = weights.transforms()
    categories = weights.meta["categories"]

    total_instances = 0

    with open(records_path, "w", encoding="utf-8") as records_file:
        for frame in tqdm(
            image_files,
            desc=f"Building KITTI {args.split} cue corpus",
            unit="frame",
        ):
            image_path = os.path.join(args.image_dir, frame)
            disparity_path = os.path.join(args.disparity_dir, frame)

            image = load_rgb(image_path)
            disparity = load_disparity(disparity_path)
            height, width = image.shape[:2]

            output = make_instance_predictions(image, model, transform, device)

            instance_id = 0
            for index, score in enumerate(output["scores"].numpy()):
                if score < args.score_threshold:
                    continue

                mask = output["masks"][index, 0].numpy() >= args.mask_threshold
                area_pixels = int(mask.sum())
                if area_pixels < args.min_mask_pixels:
                    continue

                x1, y1, x2, y2 = [
                    float(value)
                    for value in output["boxes"][index].numpy()
                ]
                label_id = int(output["labels"][index].item())
                class_name = categories[label_id]

                valid = mask & np.isfinite(disparity) & (disparity > 0)
                disparity_values = disparity[valid]
                valid_fraction = float(valid.sum() / max(area_pixels, 1))

                bbox_width = max(0.0, x2 - x1)
                bbox_height = max(0.0, y2 - y1)

                record = {
                    "dataset": "kitti2015",
                    "split": args.split,
                    "frame": frame,
                    "instance_id": instance_id,
                    "instance_class_id": label_id,
                    "instance_class_name": class_name,
                    "instance_score": float(score),
                    "area_pixels": area_pixels,
                    "area_fraction": float(area_pixels / (height * width)),
                    "bbox_width_fraction": float(bbox_width / width),
                    "bbox_height_fraction": float(bbox_height / height),
                    "bbox_area_fraction": float(
                        bbox_width * bbox_height / (height * width)
                    ),
                    "bbox_aspect_ratio": float(
                        bbox_width / max(bbox_height, 1e-6)
                    ),
                    "center_x_fraction": float(((x1 + x2) / 2.0) / width),
                    "center_y_fraction": float(((y1 + y2) / 2.0) / height),
                    "valid_depth_fraction": valid_fraction,
                    "touches_border": mask_touches_border(mask),
                    "median_disparity": (
                        float(np.median(disparity_values))
                        if disparity_values.size
                        else None
                    ),
                    "disparity_q25": (
                        float(np.quantile(disparity_values, 0.25))
                        if disparity_values.size
                        else None
                    ),
                    "disparity_q75": (
                        float(np.quantile(disparity_values, 0.75))
                        if disparity_values.size
                        else None
                    ),
                }
                record.update(sharpness_features(image, mask))

                records_file.write(json.dumps(record) + "\n")
                instance_id += 1
                total_instances += 1

    print(f"Saved {total_instances} object records to {records_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build KITTI-only instance size + focus/texture cue records using the existing split.json."
        )
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--disparity-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--score-threshold", type=float, default=0.7)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-mask-pixels", type=int, default=50)
    build_corpus(parser.parse_args())
