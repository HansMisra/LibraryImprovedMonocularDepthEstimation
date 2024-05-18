#model module

import torch.nn as nn
from torchvision import models

class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),  # maintains dimensions
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),  # maintains dimensions
            nn.ReLU(),
            nn.MaxPool2d(2)  # reduces dimensions by half
        )
        self.decoder = nn.Sequential(
            # Increase output padding to adjust the final layer's height and width
            nn.ConvTranspose2d(32, 16, 2, stride=2, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, stride=1, padding=1)  # maintains dimensions
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
def create_dnn():
    """Create a deep neural network model."""
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)  # Modify the final layer for regression
    return model
