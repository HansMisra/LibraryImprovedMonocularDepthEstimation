import argparse
import json
import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)

from depth_manifest import load_rgb, load_target, read_manifest, safe_sample_id


SEGFORMER_ID = "nvidia/segformer-b0-finetuned-ade-512-512"


def mask_touches_border(mask):
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def make_semantic_map(image, processor, model, device):
    inputs = processor(images=Image.fromarray(image), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    logits = F.interpolate(logits, size=image.shape[:2], mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint16)


def make_instance_predictions(image, model, transform, device):
    tensor = transform(Image.fromarray(image)).to(device)
    with torch.no_grad():
        output = model([tensor])[0]
    return {k: v.detach().cpu() for k, v in output.items()}


def masked_mode(values):
    if values.size == 0:
        return None
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[np.argmax(counts)])


def entropy_from_gray(gray, mask):
    values = gray[mask]
    if values.size < 2:
        return None
    hist, _ = np.histogram(values, bins=32, range=(0.0, 1.0), density=False)
    p = hist.astype(np.float64)
    p = p[p > 0]
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def sharpness_features(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=2).astype(bool)
    interior = eroded if eroded.sum() >= 25 else mask
    boundary = mask & ~cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
    high_frequency = gray - blurred

    values = gray[interior]
    grad_values = gradient[interior]
    lap_values = laplacian[interior]
    hf_values = high_frequency[interior]
    boundary_grad = gradient[boundary]

    if values.size == 0:
        return {}

    contrast = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
    grad_p90 = float(np.quantile(grad_values, 0.90)) if grad_values.size else 0.0
    boundary_p90 = float(np.quantile(boundary_grad, 0.90)) if boundary_grad.size else 0.0

    return {
        "sharp_laplacian_variance": float(np.var(lap_values)) if lap_values.size else None,
        "sharp_tenengrad_mean": float(np.mean(grad_values ** 2)) if grad_values.size else None,
        "sharp_gradient_p90": grad_p90,
        "sharp_high_frequency_rms": float(np.sqrt(np.mean(hf_values ** 2))) if hf_values.size else None,
        "sharp_local_contrast": contrast,
        "sharp_boundary_gradient_p90": boundary_p90,
        "sharp_line_tightness": float(boundary_p90 / max(contrast, 1e-6)),
        "texture_entropy": entropy_from_gray(gray, interior),
    }


def angular_features(box, area_pixels, fx, fy):
    if fx is None or fy is None or fx <= 0 or fy <= 0:
        return {
            "bbox_angular_width_rad": None,
            "bbox_angular_height_rad": None,
            "angular_area_proxy": None,
        }

    x1, y1, x2, y2 = box
    width_px = max(0.0, x2 - x1)
    height_px = max(0.0, y2 - y1)
    return {
        "bbox_angular_width_rad": float(2.0 * math.atan(width_px / (2.0 * fx))),
        "bbox_angular_height_rad": float(2.0 * math.atan(height_px / (2.0 * fy))),
        "angular_area_proxy": float(area_pixels / (fx * fy)),
    }


def build(args):
    samples = read_manifest(args.manifest)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    semantic_dir = os.path.join(output_dir, "semantic_maps")
    os.makedirs(semantic_dir, exist_ok=True)
    records_path = os.path.join(output_dir, "instances.jsonl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    semantic_processor = AutoImageProcessor.from_pretrained(SEGFORMER_ID)
    semantic_model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_ID).to(device).eval()
    ade_labels = {int(k): v for k, v in semantic_model.config.id2label.items()}

    instance_weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    instance_model = maskrcnn_resnet50_fpn_v2(weights=instance_weights).to(device).eval()
    instance_transform = instance_weights.transforms()
    coco_categories = instance_weights.meta["categories"]

    with open(os.path.join(output_dir, "semantic_labels.json"), "w", encoding="utf-8") as f:
        json.dump(ade_labels, f, indent=2)

    with open(records_path, "w", encoding="utf-8") as records_file:
        for sample in tqdm(samples, desc="Building cue corpus", unit="frame"):
            image = load_rgb(sample)
            target = load_target(sample)
            depth = target["depth_m"]
            valid_depth = target["valid_mask"]
            disparity = target["disparity_px"]

            if depth.shape != image.shape[:2]:
                depth = cv2.resize(depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                valid_depth = cv2.resize(valid_depth.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                if disparity is not None:
                    disparity = cv2.resize(disparity, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            height, width = image.shape[:2]
            semantic_map = make_semantic_map(image, semantic_processor, semantic_model, device)
            safe_id = safe_sample_id(sample.get("id", "sample"))
            np.save(os.path.join(semantic_dir, safe_id + ".npy"), semantic_map)

            output = make_instance_predictions(image, instance_model, instance_transform, device)
            instance_id = 0
            for idx, score in enumerate(output["scores"].numpy()):
                if score < args.score_threshold:
                    continue

                mask = output["masks"][idx, 0].numpy() >= args.mask_threshold
                area_pixels = int(mask.sum())
                if area_pixels < args.min_mask_pixels:
                    continue

                box = [float(v) for v in output["boxes"][idx].numpy()]
                x1, y1, x2, y2 = box
                label_id = int(output["labels"][idx].item())
                class_name = coco_categories[label_id]

                object_valid = mask & valid_depth & np.isfinite(depth) & (depth > 0)
                depth_values = depth[object_valid]
                valid_fraction = float(object_valid.sum() / area_pixels)
                disparity_values = disparity[object_valid] if disparity is not None else np.array([], dtype=np.float32)

                semantic_id = masked_mode(semantic_map[mask])
                semantic_name = ade_labels.get(semantic_id, "unknown") if semantic_id is not None else "unknown"

                record = {
                    "dataset": sample.get("dataset", "unknown"),
                    "sample_id": sample.get("id"),
                    "frame": os.path.basename(sample.get("image_path", "")),
                    "instance_id": instance_id,
                    "instance_class_id": label_id,
                    "instance_class_name": class_name,
                    "instance_score": float(score),
                    "semantic_class_id": semantic_id,
                    "semantic_class_name": semantic_name,
                    "area_pixels": area_pixels,
                    "area_fraction": float(area_pixels / (height * width)),
                    "bbox_width_fraction": float(max(0.0, x2 - x1) / width),
                    "bbox_height_fraction": float(max(0.0, y2 - y1) / height),
                    "bbox_area_fraction": float(max(0.0, x2 - x1) * max(0.0, y2 - y1) / (height * width)),
                    "center_x_fraction": float(((x1 + x2) / 2.0) / width),
                    "center_y_fraction": float(((y1 + y2) / 2.0) / height),
                    "valid_depth_fraction": valid_fraction,
                    "touches_border": mask_touches_border(mask),
                    "fx": sample.get("fx"),
                    "fy": sample.get("fy"),
                    "median_depth_m": float(np.median(depth_values)) if depth_values.size else None,
                    "depth_q25_m": float(np.quantile(depth_values, 0.25)) if depth_values.size else None,
                    "depth_q75_m": float(np.quantile(depth_values, 0.75)) if depth_values.size else None,
                    "median_inverse_depth": float(np.median(1.0 / depth_values)) if depth_values.size else None,
                    "median_disparity": float(np.median(disparity_values)) if disparity_values.size else None,
                }
                record.update(angular_features(box, area_pixels, sample.get("fx"), sample.get("fy")))
                record.update(sharpness_features(image, mask))
                records_file.write(json.dumps(record) + "\n")
                instance_id += 1

    print(f"Saved cue records to {records_path}")
    print(f"Saved semantic maps to {semantic_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.7)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-mask-pixels", type=int, default=50)
    build(parser.parse_args())
