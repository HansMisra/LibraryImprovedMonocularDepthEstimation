import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


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


class HypotenuseSquareTransform:
    def __init__(self, target_size=1300, rotate_degrees=0):
        self.target_size = target_size
        self.rotate_degrees = rotate_degrees

    def __call__(self, img):
        width, height = img.size
        scaling_factor = self.target_size / max(width, height)

        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)

        img = img.resize((new_width, new_height), Image.NEAREST)

        pad_horizontal = max((self.target_size - new_width) // 2, 0)
        pad_vertical = max((self.target_size - new_height) // 2, 0)

        img = transforms.Pad((pad_horizontal, pad_vertical), fill=0)(img)

        if self.rotate_degrees != 0:
            img = img.rotate(self.rotate_degrees, fillcolor=0)

        img = img.resize((self.target_size, self.target_size), Image.NEAREST)
        return img


image_transforms = HypotenuseSquareTransform(rotate_degrees=45)


class KITTIDataset(Dataset):
    def __init__(self, image_dir, disparity_dir, transform=None):
        self.image_dir = image_dir
        self.disparity_dir = disparity_dir
        self.transform = transform if transform is not None else image_transforms
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
            image = self.transform(image)
            disparity = self.transform(disparity)

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