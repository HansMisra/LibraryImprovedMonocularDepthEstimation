import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {file_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def load_disparity(file_path):
    disparity = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if disparity is None:
        raise FileNotFoundError(f"Disparity file not found: {file_path}")

    disparity = disparity.astype(np.float32) / 256.0
    return Image.fromarray(disparity)


class PairedSquareTransform:
    def __init__(self, target_size=1300, rotation_range=0.0, rotation_probability=0.0):
        self.target_size = target_size
        self.rotation_range = float(rotation_range)
        self.rotation_probability = float(rotation_probability)

    def _resize_and_pad(self, img, is_disparity=False):
        width, height = img.size
        scale = self.target_size / max(width, height)

        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        resample = Image.Resampling.NEAREST if is_disparity else Image.Resampling.BILINEAR
        img = img.resize((new_width, new_height), resample=resample)

        if is_disparity:
            disparity = np.asarray(img, dtype=np.float32) * scale
            img = Image.fromarray(disparity)

        pad_w = self.target_size - new_width
        pad_h = self.target_size - new_height
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        img = transforms.Pad((left, top, right, bottom), fill=0)(img)
        return img

    def __call__(self, image, disparity):
        image = self._resize_and_pad(image, is_disparity=False)
        disparity = self._resize_and_pad(disparity, is_disparity=True)

        angle = 0.0
        if self.rotation_range > 0 and random.random() < self.rotation_probability:
            angle = random.uniform(-self.rotation_range, self.rotation_range)

        if angle != 0.0:
            image = image.rotate(
                angle,
                resample=Image.Resampling.BILINEAR,
                fillcolor=0
            )
            disparity = disparity.rotate(
                angle,
                resample=Image.Resampling.NEAREST,
                fillcolor=0
            )

        return image, disparity


train_transforms = PairedSquareTransform(
    target_size=1300,
    rotation_range=10.0,
    rotation_probability=0.5
)

eval_transforms = PairedSquareTransform(
    target_size=1300,
    rotation_range=0.0,
    rotation_probability=0.0
)

image_transforms = train_transforms


class KITTIDataset(Dataset):
    def __init__(self, image_dir, disparity_dir, transform=None):
        self.image_dir = image_dir
        self.disparity_dir = disparity_dir
        self.transform = transform if transform is not None else eval_transforms
        self.to_tensor = transforms.ToTensor()

        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        if not os.path.exists(disparity_dir):
            raise FileNotFoundError(f"Disparity directory not found: {disparity_dir}")

        image_files = sorted(
            f for f in os.listdir(image_dir)
            if f.endswith("_10.png")
        )

        self.samples = []

        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            disparity_path = os.path.join(disparity_dir, image_file)

            if os.path.exists(disparity_path):
                self.samples.append((image_path, disparity_path))

        if not self.samples:
            raise ValueError(
                "No matching image/disparity pairs found. "
                "Expected matching *_10.png files in both directories."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, disparity_path = self.samples[idx]

        image = load_image(image_path)
        disparity = load_disparity(disparity_path)

        if self.transform:
            image, disparity = self.transform(image, disparity)

        image_tensor = image if torch.is_tensor(image) else self.to_tensor(image)

        if torch.is_tensor(disparity):
            disparity_tensor = disparity
            if disparity_tensor.ndim == 2:
                disparity_tensor = disparity_tensor.unsqueeze(0)
        else:
            disparity_array = np.array(disparity, dtype=np.float32)
            disparity_tensor = torch.from_numpy(disparity_array).unsqueeze(0)

        return {
            "image": image_tensor.float(),
            "disparity": disparity_tensor.float()
        }
