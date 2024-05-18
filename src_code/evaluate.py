#evaluate module

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_utils import KITTIDataset
from torchvision.transforms import ToTensor
from model import DepthNet, create_dnn  # Make sure these are correctly defined in model.py
from load_model import load_model
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def apply_transformation(predicted, transformation):
    """
    Apply a simple linear transformation to the predictions.
    
    Args:
    predicted (torch.Tensor): The predicted outputs from the model.
    transformation (np.array): A 2-element array [scale, bias] for transformation.
    
    Returns:
    torch.Tensor: The transformed predictions.
    """
    if transformation is None or not transformation.size:
        return predicted  # Return as is if no transformation is provided.
    
    # Ensure the transformation is a tensor and on the same device as the predictions.
    transformation = torch.tensor(transformation, dtype=torch.float32, device=predicted.device)
    
    # Apply the transformation: scale * predicted + bias
    scale, bias = transformation[0], transformation[1]
    transformed_predictions = scale * predicted + bias
    
    return transformed_predictions

from sklearn.metrics import mean_squared_error, accuracy_score, precision_score

def evaluate_model(model, data_loader, device, transformation_matrix=None, evaluate_transform=False):
    model.eval()
    errors = []
    accuracies = []
    precisions = []
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
            image_infos.append((images.cpu(), true_disparities.cpu(), predictions.cpu(), mse_error))

    return errors, accuracies, precisions, image_infos



def display_image_results(image_infos):
    fig, axs = plt.subplots(len(image_infos), 3, figsize=(15, 5 * len(image_infos)))
    for idx, (images, true_disp, pred_disp, error) in enumerate(image_infos):
        axs[idx, 0].imshow(images[0].permute(1, 2, 0))
        axs[idx, 0].set_title(f'Original Image - MSE Error: {error:.2f}')
        axs[idx, 1].imshow(true_disp[0].squeeze(), cmap='plasma')
        axs[idx, 1].set_title('Ground Truth Disparity')
        axs[idx, 2].imshow(pred_disp[0].squeeze(), cmap='plasma')
        axs[idx, 2].set_title('Predicted Disparity')
    plt.show()
