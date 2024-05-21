import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import os
# Assuming all necessary modules are in the same directory or properly installed as packages
from data_utils import KITTIDataset, display_image_results
from load_model import load_model
from evaluate import device, apply_transformation, mean_squared_error, accuracy_score, precision_score

def evaluate_model(model, data_loader, device, transformation_matrix=None, evaluate_transform=False):
    model.eval()
    errors = []
    accuracies = []
    precisions = []
    percentile_accuracies = []
    image_infos = []

    with torch.no_grad():
        for batch in data_loader:
            images, true_disparities = batch['image'].to(device), batch['disparity'].to(device)
            predictions = model(images)

            # Ensure predictions match the dimension of the ground truth
            if predictions.shape != true_disparities.shape:
                predictions = torch.nn.functional.interpolate(predictions, size=true_disparities.shape[2:], mode='bilinear', align_corners=False)

            if evaluate_transform and transformation_matrix is not None:
                predictions = apply_transformation(predictions, transformation_matrix)

            # Flatten the tensors for MSE and other metrics calculation
            true_disparities_flat = true_disparities.view(-1).cpu().numpy()
            predictions_flat = predictions.view(-1).cpu().numpy()

            mse_error = mean_squared_error(true_disparities_flat, predictions_flat, squared=False)
            errors.append(mse_error)
            accuracies.append(accuracy_score(true_disparities_flat > 0.5, predictions_flat > 0.5))
            precisions.append(precision_score(true_disparities_flat > 0.5, predictions_flat > 0.5, average='macro', zero_division=1))
            percentile_acc = percentile_accuracy(true_disparities_flat, predictions_flat, tolerance=0.25)
            percentile_accuracies.append(percentile_acc)
           
            print(f"Appending to image_infos: {images.shape}, {true_disparities.shape}, {predictions.shape}, {mse_error}, {percentile_acc}")

            image_infos.append((images.cpu(), true_disparities.cpu(), predictions.cpu(), mse_error))

    return errors, accuracies, precisions, percentile_accuracies, image_infos

def run_evaluation():
    data_dir = r'C:\Users\Hans Kirtan Misra\Documents\Professional\UMD\MSML\MDE_v2\LibraryImprovedMonocularDepthEstimation\src_code\kitti_data\data_scene_flow\testing\image_2'
    disparity_dir = r'C:\Users\Hans Kirtan Misra\Documents\Professional\UMD\MSML\MDE_v2\LibraryImprovedMonocularDepthEstimation\src_code\kitti_data\data_scene_flow\testing\test_disp'
    model_path = r'C:\Users\Hans Kirtan Misra\Documents\Professional\UMD\MSML\MDE_v2\LibraryImprovedMonocularDepthEstimation\src_code\model_weights.pth'


    # Set device to CPU explicitly if CUDA is not available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    if not os.path.exists(model_path):
        print("Model weights not found. Please train the model first.")
        return

    model = load_model(model_path, 'depthnet', device=device)
    dataset = KITTIDataset(data_dir, disparity_dir, transform=ToTensor())
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    errors, accuracies, precisions, percentile_accuracies, image_infos = evaluate_model(model, data_loader, device)
    
    max_error_idx = np.argmax(errors)
    min_error_idx = np.argmin(errors)
    median_error_idx = np.argsort(errors)[len(errors)//2]

    display_image_results([image_infos[max_error_idx], image_infos[min_error_idx], image_infos[median_error_idx]])

    print(f'Average Accuracy: {np.mean(accuracies):.2f}, Average Precision: {np.mean(precisions):.2f}')
    print(f'Max Error: {errors[max_error_idx]}, Min Error: {errors[min_error_idx]}, Median Error: {errors[median_error_idx]}')
    print(f'Percentile Accuracy (within tolerance): {np.mean(percentile_accuracies):.2f}')

if __name__ == '__main__':
    run_evaluation()
