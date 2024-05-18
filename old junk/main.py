import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from data_utils import KITTIDataset, ToTensor
from model import create_dnn
import numpy as np
import os
from sklearn.metrics import mean_squared_error

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_transformation(predicted, true):
    # Placeholder for transformation logic
    return torch.eye(3).to(predicted.device)  # Example: identity matrix

def apply_transformation(predicted, transformation):
    # Placeholder for applying transformation
    return predicted  # No change as placeholder

def load_transformation_matrix(path):
    return np.load(path)

def evaluate_model(model, data_loader, device, transformation_matrix, evaluate_transform=False):
    model.eval()
    errors = []
    images_info = []

    with torch.no_grad():
        for batch in data_loader:
            images, true_disparities = batch['image'].to(device), batch['disparity'].to(device)
            predictions = model(images)

            if evaluate_transform:
                predictions = apply_transformation(predictions, transformation_matrix)

            error = mean_squared_error(true_disparities.cpu().numpy(), predictions.cpu().numpy(), squared=False)
            errors.append(error)
            images_info.append((images.cpu(), error))

    return errors, images_info

def main():
    data_dir = 'path/to/kitti_data/data_scene_flow/training'
    transform = transforms.Compose([ToTensor()])
    dataset = KITTIDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = create_dnn().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epoch_transforms = []

    # Training loop
    for epoch in range(10):
        model.train()
        for batch in dataloader:
            images, true_disparities = batch['image'].to(device), batch['disparity'].to(device)
            optimizer.zero_grad()
            predicted_disparities = model(images)
            loss = criterion(predicted_disparities, true_disparities)
            loss.backward()
            optimizer.step()

            # Compute and apply transformation (to be implemented)
            transformation_matrix = calculate_transformation(predicted_disparities, true_disparities)
            epoch_transforms.append(transformation_matrix.cpu().numpy())

        print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save model and transformation
    final_transform = np.mean(epoch_transforms, axis=0)
    np.save(os.path.join(data_dir, 'final_transformation.npy'), final_transform)
    torch.save(model.state_dict(), os.path.join(data_dir, 'weights.pth'))

    # Evaluation
    test_data_dir = 'path/to/kitti_data/data_scene_flow/testing'
    test_dataset = KITTIDataset(test_data_dir, transform=transform)
    test_size = int(0.85 * len(test_dataset))
    test_indices = np.random.choice(len(test_dataset), size=test_size, replace=False)
    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = DataLoader(test_dataset, batch_size=4, sampler=test_sampler)

    errors_before, _ = evaluate_model(model, test_loader, device, None)
    print(f"Mean Error before applying transformation: {np.mean(errors_before)}")

    transformation_matrix = load_transformation_matrix(os.path.join(data_dir, 'final_transformation.npy'))
    errors_after, _ = evaluate_model(model, test_loader, device, transformation_matrix, evaluate_transform=True)
    print(f"Mean Error after applying transformation: {np.mean(errors_after)}")

if __name__ == '__main__':
    main()
