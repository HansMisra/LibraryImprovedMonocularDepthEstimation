import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from data_utils import KITTIDataset, ToTensor
from model import DepthNet
import torch.nn.functional as F
import numpy as np
import os

def ssim_loss(pred, target):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = F.avg_pool2d(pred, 3, 1, padding=1)
    mu_y = F.avg_pool2d(target, 3, 1, padding=1)
    sigma_x = F.avg_pool2d(pred**2, 3, 1, padding=1) - mu_x**2
    sigma_y = F.avg_pool2d(target**2, 3, 1, padding=1) - mu_y**2
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, padding=1) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    return torch.clamp((1 - ssim) / 2, 0, 1).mean()

def train(data_dir, epochs=10, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = KITTIDataset(data_dir, transform=transforms.Compose([ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = DepthNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = ssim_loss  # Using structural similarity index for loss calculation

    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            images, true_disparities = batch['image'].to(device), batch['disparity'].to(device)
            optimizer.zero_grad()
            predicted_disparities = model(images)
            loss = criterion(predicted_disparities, true_disparities)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

if __name__ == '__main__':
    train('path/to/kitti_data/data_scene_flow/training')
