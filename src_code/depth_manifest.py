import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np


def read_manifest(path):
    path = Path(path)
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sample["_manifest_dir"] = str(path.parent.resolve())
            sample["_manifest_path"] = str(path.resolve())
            sample["_manifest_line"] = line_number
            samples.append(sample)
    return samples


def write_manifest(samples, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            clean = {k: v for k, v in sample.items() if not k.startswith("_manifest_")}
            f.write(json.dumps(clean) + "\n")


def resolve_path(sample, key):
    value = sample.get(key)
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    base = Path(sample.get("_manifest_dir", "."))
    return str((base / path).resolve())


def safe_sample_id(value):
    text = str(value)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    stem = stem[:80].strip("_") or "sample"
    return f"{stem}_{digest}"


def load_rgb(sample):
    path = resolve_path(sample, "image_path")
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"RGB image not found: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_mask(sample, shape):
    path = resolve_path(sample, "valid_mask_path")
    if path is None:
        return np.ones(shape, dtype=bool)

    if path.lower().endswith(".npy"):
        mask = np.load(path)
    else:
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Valid mask not found: {path}")

    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != shape:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def _load_confidence(sample, shape):
    path = resolve_path(sample, "confidence_path")
    if path is None:
        return None

    if path.lower().endswith(".npy"):
        confidence = np.load(path).astype(np.float32)
    else:
        confidence = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if confidence is None:
            raise FileNotFoundError(f"Confidence map not found: {path}")
        confidence = confidence.astype(np.float32)

    if confidence.ndim == 3:
        confidence = confidence[..., 0]
    if confidence.shape != shape:
        confidence = cv2.resize(confidence, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return confidence


def load_target(sample):
    path = resolve_path(sample, "target_path")
    encoding = sample.get("target_encoding")

    if path is None or encoding is None:
        raise ValueError("Manifest sample must contain target_path and target_encoding.")

    disparity_px = None

    if encoding == "depth_npy_m":
        depth_m = np.load(path).astype(np.float32)

    elif encoding == "depth_png_cm":
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(path)
        depth_m = raw.astype(np.float32) / 100.0

    elif encoding == "depth_png_mm":
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(path)
        depth_m = raw.astype(np.float32) / 1000.0

    elif encoding == "depth_png_scaled":
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(path)
        scale_to_m = float(sample["target_scale_to_m"])
        depth_m = raw.astype(np.float32) * scale_to_m

    elif encoding == "kitti_disparity_png256":
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(path)
        disparity_px = raw.astype(np.float32) / 256.0
        fx = float(sample["fx"])
        baseline_m = float(sample["baseline_m"])
        depth_m = np.zeros_like(disparity_px, dtype=np.float32)
        valid_disp = np.isfinite(disparity_px) & (disparity_px > 0)
        depth_m[valid_disp] = (fx * baseline_m) / disparity_px[valid_disp]

    else:
        raise ValueError(f"Unsupported target_encoding: {encoding}")

    if depth_m.ndim == 3:
        depth_m = np.squeeze(depth_m)
    depth_m = depth_m.astype(np.float32)

    valid = np.isfinite(depth_m) & (depth_m > 0)
    valid &= _load_mask(sample, depth_m.shape)

    min_depth_m = sample.get("min_depth_m")
    max_depth_m = sample.get("max_depth_m")
    if min_depth_m is not None:
        valid &= depth_m >= float(min_depth_m)
    if max_depth_m is not None:
        valid &= depth_m <= float(max_depth_m)

    confidence = _load_confidence(sample, depth_m.shape)

    return {
        "depth_m": depth_m,
        "valid_mask": valid,
        "confidence": confidence,
        "disparity_px": disparity_px,
    }
