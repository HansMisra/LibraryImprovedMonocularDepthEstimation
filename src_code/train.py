import torch
from torch.utils.data import DataLoader
from data_utils import KITTIDataset, image_transforms
from model import DepthNet
import torch.optim as optim
import os
import torch.nn.functional as F
import sys

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def train(image_dir, disparity_dir, epochs=10, batch_size=8, save_path=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = KITTIDataset(image_dir, disparity_dir, transform=image_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = DepthNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss().to(device)

    for epoch in range(epochs):
        for batch in dataloader:
            images, targets = batch['image'].to(device), batch['disparity'].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs_resized = F.interpolate(outputs, size=targets.shape[2:], mode='bilinear', align_corners=True)
            loss = criterion(outputs_resized, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        if save_path:
            torch.save(model.state_dict(), save_path)
            print(f"Model weights saved to {save_path}")

if __name__ == '__main__':
    image_dir = os.path.join('kitti_data', 'data_scene_flow', 'training', 'image_2')
    disparity_dir = os.path.join('kitti_data', 'data_scene_flow', 'training', 'disp_occ_0')
    model_save_path = os.path.join('model_weights.pth')

    train(image_dir, disparity_dir, save_path=model_save_path)
