import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, zs_dim=128, zc_dim=128):
        super().__init__()

        self.flat_dim = 1024 * 4 * 3  # 12288

        self.fc = nn.Sequential(
            nn.Linear(zs_dim + zc_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.flat_dim)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1), # (B, 512, 8, 6)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # (B, 256, 16, 12)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # (B, 128, 32, 24)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # (B, 64, 64, 48)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),     # (B, 1, 128, 96)
        )

    def forward(self, zs, zc):
        z = torch.cat([zs, zc], dim=1)
        x = self.fc(z)
        x = x.view(x.size(0), 1024, 4, 3)
        x = self.deconv(x)
        x = F.interpolate(x, size=(128, 126))
        x = x.squeeze(1)
        return x
