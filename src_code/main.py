#main module

import os
import load_model, kitti_data, data_utils, evaluate, train
from evaluate import evaluate_model, display_image_results, device
from load_model import load_model
from data_utils import KITTIDataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
import torch


def main():
    data_dir = r'C:\Users\Hans Kirtan Misra\Documents\Professional\UMD\MSML\MDE_v2\LibraryImprovedMonocularDepthEstimation\src_code\kitti_data\data_scene_flow\testing\image_2'
    disparity_dir = r'C:\Users\Hans Kirtan Misra\Documents\Professional\UMD\MSML\MDE_v2\LibraryImprovedMonocularDepthEstimation\src_code\kitti_data\data_scene_flow\testing\test_disp'
    model_path = r'C:\Users\Hans Kirtan Misra\Documents\Professional\UMD\MSML\MDE_v2\LibraryImprovedMonocularDepthEstimation\src_code\model_weights.pth'

    # Check if model weights file exists
    if not os.path.exists(model_path):
        print("Model weights not found. Please train the model first.")
        return

    model = load_model(model_path, 'depthnet', device=device)
    dataset = KITTIDataset(data_dir, disparity_dir, transform=ToTensor())
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    errors, accuracies, precisions, image_infos = evaluate_model(model, data_loader, device)
    
    max_error_idx = np.argmax(errors)
    min_error_idx = np.argmin(errors)
    median_error_idx = np.argsort(errors)[len(errors)//2]

    display_image_results([image_infos[max_error_idx], image_infos[min_error_idx], image_infos[median_error_idx]])

    print(f'Average Accuracy: {np.mean(accuracies):.2f}, Average Precision: {np.mean(precisions):.2f}')
    print(f'Max Error: {errors[max_error_idx]}, Min Error: {errors[min_error_idx]}, Median Error: {errors[median_error_idx]}')

if __name__ == '__main__':
    main()

