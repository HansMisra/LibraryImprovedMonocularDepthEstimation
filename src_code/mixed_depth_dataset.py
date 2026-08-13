import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from depth_manifest import load_rgb, load_target, read_manifest


class DepthLetterboxTransform:
    def __init__(self, target_size=384, random_horizontal_flip=False):
        self.target_size = int(target_size)
        self.random_horizontal_flip = bool(random_horizontal_flip)

    def __call__(self, image, depth, valid, intrinsics=None):
        height, width = image.shape[:2]
        scale = self.target_size / max(height, width)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        valid = cv2.resize(valid.astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_NEAREST).astype(bool)

        pad_x = self.target_size - new_width
        pad_y = self.target_size - new_height
        left = pad_x // 2
        right = pad_x - left
        top = pad_y // 2
        bottom = pad_y - top

        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        depth = cv2.copyMakeBorder(depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        valid = cv2.copyMakeBorder(valid.astype(np.uint8), top, bottom, left, right, cv2.BORDER_CONSTANT, value=0).astype(bool)

        adjusted = None
        if intrinsics is not None:
            adjusted = dict(intrinsics)
            if adjusted.get("fx") is not None:
                adjusted["fx"] = float(adjusted["fx"]) * scale
            if adjusted.get("fy") is not None:
                adjusted["fy"] = float(adjusted["fy"]) * scale
            if adjusted.get("cx") is not None:
                adjusted["cx"] = float(adjusted["cx"]) * scale + left
            if adjusted.get("cy") is not None:
                adjusted["cy"] = float(adjusted["cy"]) * scale + top

        if self.random_horizontal_flip and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            depth = np.ascontiguousarray(depth[:, ::-1])
            valid = np.ascontiguousarray(valid[:, ::-1])
            if adjusted is not None and adjusted.get("cx") is not None:
                adjusted["cx"] = (self.target_size - 1) - adjusted["cx"]

        return image, depth, valid, adjusted


class MixedManifestDepthDataset(Dataset):
    def __init__(self, manifest_paths, transform=None, max_depth_m=None):
        self.samples = []
        for manifest_path in manifest_paths:
            self.samples.extend(read_manifest(manifest_path))
        if not self.samples:
            raise ValueError("No samples were found in the supplied manifests.")

        self.transform = transform
        self.max_depth_m = max_depth_m

    def __len__(self):
        return len(self.samples)

    def source_counts(self):
        counts = {}
        for sample in self.samples:
            source = sample.get("dataset", "unknown")
            counts[source] = counts.get(source, 0) + 1
        return counts

    def __getitem__(self, index):
        sample = self.samples[index]
        image = load_rgb(sample)
        target = load_target(sample)
        depth = target["depth_m"]
        valid = target["valid_mask"]

        if depth.shape != image.shape[:2]:
            depth = cv2.resize(depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            valid = cv2.resize(valid.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

        if self.max_depth_m is not None:
            valid &= depth <= float(self.max_depth_m)

        intrinsics = {
            "fx": sample.get("fx"),
            "fy": sample.get("fy"),
            "cx": sample.get("cx"),
            "cy": sample.get("cy"),
        }

        if self.transform is not None:
            image, depth, valid, intrinsics = self.transform(
                image,
                depth,
                valid,
                intrinsics=intrinsics,
            )

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        depth_tensor = torch.from_numpy(depth).float().unsqueeze(0)
        valid_tensor = torch.from_numpy(valid).bool().unsqueeze(0)

        return {
            "image": image_tensor,
            "depth": depth_tensor,
            "valid_mask": valid_tensor,
            "dataset": sample.get("dataset", "unknown"),
            "sample_id": sample.get("id", str(index)),
            "fx": float(intrinsics["fx"]) if intrinsics and intrinsics.get("fx") is not None else float("nan"),
            "fy": float(intrinsics["fy"]) if intrinsics and intrinsics.get("fy") is not None else float("nan"),
            "cx": float(intrinsics["cx"]) if intrinsics and intrinsics.get("cx") is not None else float("nan"),
            "cy": float(intrinsics["cy"]) if intrinsics and intrinsics.get("cy") is not None else float("nan"),
        }
