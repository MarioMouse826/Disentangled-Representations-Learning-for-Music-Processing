import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, zs_dim=128, zc_dim=128):
        super().__init__()
        self.zs_dim = zs_dim
        self.zc_dim = zc_dim

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, 4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2),
        )

        self.flat_dim = 1024 * 4 * 3  # 12288

        self.style_head = nn.Sequential(
            nn.Linear(self.flat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, zs_dim * 2)
        )

        self.content_head = nn.Sequential(
            nn.Linear(self.flat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, zc_dim * 2)
        )

    def forward(self, mel):
        x = mel.unsqueeze(1)
        x = self.backbone(x)
        x = x.view(x.size(0), -1)

        zs = self.content_head(x)
        zc = self.content_head(x)

        zs_mean, zs_logvar = zs.chunk(2, dim=1)
        zc_mean, zc_logvar = zc.chunk(2, dim=1)

        return zs_mean, zs_logvar, zc_mean, zc_logvar

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean
