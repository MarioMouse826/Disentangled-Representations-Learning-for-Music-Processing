import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, zs_dim=32, zc_dim=32):
        super().__init__()
        self.zs_dim = zs_dim
        self.zc_dim = zc_dim

        # shared CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),   # (B, 32, 64, 62)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # (B, 64, 32, 31)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # (B, 128, 16, 15)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),# (B, 256, 8, 7)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.flat_dim = 256 * 8 * 7  # 14336

        # style head -> timbre/identity
        self.style_head = nn.Linear(self.flat_dim, zs_dim * 2)

        # content head -> pitch
        self.content_head = nn.Linear(self.flat_dim, zc_dim * 2)

    def forward(self, mel):
        # mel: (B, 128, 125) -> add channel dim
        x = mel.unsqueeze(1)           # (B, 1, 128, 125)
        x = self.backbone(x)           # (B, 256, 8, 7)
        x = x.view(x.size(0), -1)     # (B, 14336)

        zs = self.style_head(x)        # (B, zs_dim*2)
        zc = self.content_head(x)      # (B, zc_dim*2)

        zs_mean, zs_logvar = zs.chunk(2, dim=1)
        zc_mean, zc_logvar = zc.chunk(2, dim=1)

        return zs_mean, zs_logvar, zc_mean, zc_logvar

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean