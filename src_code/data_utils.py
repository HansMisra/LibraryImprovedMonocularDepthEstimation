import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import math


def load_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {file_path}")
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def load_disparity(file_path):
    disparity = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if disparity is None:
        raise FileNotFoundError(f"Disparity file not found: {file_path}")
    return disparity.astype(np.float32) / 256.0  # Normalize disparity





class HypotenuseSquareTransform:
    def __init__(self, target_size=1300, rotate_degrees=0):
        self.target_size = target_size  # A fixed size that is known to be larger than any image dimension after rotation
        self.rotate_degrees = rotate_degrees

    def __call__(self, img):
        # Calculate scaling factor to ensure the image covers the target size
        width, height = img.size
        scaling_factor = self.target_size / max(width, height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        img = img.resize((new_width, new_height), Image.NEAREST)

        # Center the image in a target_size x target_size square
        pad_horizontal = (self.target_size - new_width) // 2
        pad_vertical = (self.target_size - new_height) // 2
        img = transforms.Pad((pad_horizontal, pad_vertical), fill=0)(img)

        # Rotate the image if necessary
        if self.rotate_degrees != 0:
            img = img.rotate(self.rotate_degrees, fillcolor=0)

        # Ensure the image is exactly the target size by target size
        img = img.resize((self.target_size, self.target_size), Image.NEAREST)
        return img

# Apply the transformation
image_transforms = HypotenuseSquareTransform(rotate_degrees=45)



class KITTIDataset(Dataset):
    def __init__(self, image_dir, disparity_dir, transform=None):
        self.image_dir = image_dir
        self.disparity_dir = disparity_dir
        self.transform = image_transforms
        self.image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if f.endswith("_10.png")]
        self.disparity_paths = [os.path.join(disparity_dir, f.replace('_10.png', '_10.png')) for f in sorted(os.listdir(image_dir)) if f.endswith("_10.png")]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        disparity_path = self.disparity_paths[idx]
        image = load_image(image_path)
        disparity = load_disparity(disparity_path)
        if self.transform:
            image = self.transform(image)
            disparity = self.transform(Image.fromarray(disparity))
        return {'image': transforms.ToTensor()(image), 'disparity': torch.from_numpy(np.array(disparity)).unsqueeze(0)}

# Define the transformations
image_transforms = HypotenuseSquareTransform(rotate_degrees=45)

