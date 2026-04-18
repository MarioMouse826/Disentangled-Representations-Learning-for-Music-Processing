import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, zs_dim=32, zc_dim=32):
        super().__init__()

        self.flat_dim = 256 * 8 * 7  # 14336

        self.fc = nn.Linear(zs_dim + zc_dim, self.flat_dim)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # (B, 128, 16, 14)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # (B, 64, 32, 28)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # (B, 32, 64, 56)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),    # (B, 1, 128, 112)
        )

    def forward(self, zs, zc):
        z = torch.cat([zs, zc], dim=1)       # (B, 64)
        x = self.fc(z)                         # (B, 14336)
        x = x.view(x.size(0), 256, 8, 7)      # (B, 256, 8, 7)
        x = self.deconv(x)                     # (B, 1, 128, 112)
        x = F.interpolate(x, size=(128, 126))  # (B, 1, 128, 125)
        x = x.squeeze(1)                       # (B, 128, 125)
        return x
