import torch.nn as nn
import torch.nn.functional as F


class TinyMetricDepthNet(nn.Module):
    def __init__(self, min_depth_m=1e-3):
        super().__init__()
        self.min_depth_m = float(min_depth_m)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return F.softplus(x) + self.min_depth_m
