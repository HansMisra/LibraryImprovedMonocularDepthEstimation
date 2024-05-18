import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

def load_image(file_path):
    """Load an image from a file."""
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {file_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def load_disparity(file_path):
    """Load a disparity map from a file and convert to float type."""
    disparity = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if disparity is None:
        raise FileNotFoundError(f"Disparity file not found: {file_path}")
    return disparity.astype(np.float32) / 256.0

def calculate_depth_map(disparity, baseline, focal_length):
    """Convert disparity to depth map using the given baseline and focal length."""
    with np.errstate(divide='ignore', invalid='ignore'):
        depth_map = (focal_length * baseline) / disparity
        depth_map[disparity == 0] = np.inf
    return depth_map

def normalize_depth_map(depth_map):
    """Normalize the depth map for better visualization."""
    valid_depths = depth_map[depth_map != np.inf]
    max_depth = np.max(valid_depths)
    min_depth = np.min(valid_depths)
    normalized_depth_map = (depth_map - min_depth) / (max_depth - min_depth)
    normalized_depth_map[depth_map == np.inf] = 0
    return normalized_depth_map

def display_depth_map(depth_map):
    """Display the normalized depth map."""
    normalized_depth_map = normalize_depth_map(depth_map)
    plt.imshow(normalized_depth_map, cmap='plasma')
    plt.colorbar()
    plt.title('Normalized Depth Map')
    plt.show()

class KITTIDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = [os.path.join(data_dir, 'image_2', f) for f in sorted(os.listdir(os.path.join(data_dir, 'image_2')))]
        self.disparity_paths = [os.path.join(data_dir, 'disp_occ_0', f.replace('.png', '_10.png')) for f in sorted(os.listdir(os.path.join(data_dir, 'disp_occ_0')))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_image(self.image_paths[idx])
        disparity = load_disparity(self.disparity_paths[idx])
        sample = {'image': image, 'disparity': disparity}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, disparity = sample['image'], sample['disparity']
        return {'image': transforms.functional.to_tensor(image),
                'disparity': torch.from_numpy(disparity).unsqueeze(0)}
