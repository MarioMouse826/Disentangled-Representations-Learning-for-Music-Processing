import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalEncoder(nn.Module):
    def __init__(self, zg_dim=128, zl_dim=128):
        super().__init__()
        self.zg_dim = zg_dim
        self.zl_dim = zl_dim

        # shared backbone
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
        )

        # top level — global style from fully compressed representation
        self.top_backbone = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, 4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2),
        )

        self.flat_dim_top = 1024 * 4 * 3
        self.flat_dim_bot = 256 * 16 * 15

        # global head
        self.global_head = nn.Sequential(
            nn.Linear(self.flat_dim_top, 512),
            nn.ReLU(),
            nn.Linear(512, zg_dim * 2)
        )

        # local head — from mid-level features
        self.local_head = nn.Sequential(
            nn.Linear(self.flat_dim_bot, 512),
            nn.ReLU(),
            nn.Linear(512, zl_dim * 2)
        )

    def forward(self, mel):
        x = mel.unsqueeze(1)
        x_mid = self.backbone(x)              # (B, 256, 16, 15)
        x_top = self.top_backbone(x_mid)      # (B, 1024, 4, 3)

        # global latent
        zg = self.global_head(x_top.view(x_top.size(0), -1))
        zg_mean, zg_logvar = zg.chunk(2, dim=1)

        # local latent
        zl = self.local_head(x_mid.view(x_mid.size(0), -1))
        zl_mean, zl_logvar = zl.chunk(2, dim=1)

        return zg_mean, zg_logvar, zl_mean, zl_logvar

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean


class HierarchicalDecoder(nn.Module):
    def __init__(self, zg_dim=128, zl_dim=128):
        super().__init__()
        self.flat_dim = 1024 * 4 * 3

        self.fc = nn.Sequential(
            nn.Linear(zg_dim + zl_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.flat_dim)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
        )

    def forward(self, zg, zl):
        z = torch.cat([zg, zl], dim=1)
        x = self.fc(z)
        x = x.view(x.size(0), 1024, 4, 3)
        x = self.deconv(x)
        x = F.interpolate(x, size=(128, 126))
        x = x.squeeze(1)
        return x


class HierarchicalVAE(nn.Module):
    def __init__(self, zg_dim=128, zl_dim=128, beta=4.0):
        super().__init__()
        self.encoder = HierarchicalEncoder(zg_dim, zl_dim)
        self.decoder = HierarchicalDecoder(zg_dim, zl_dim)
        self.beta = beta

    def forward(self, mel, mel_shifted=None):
        zg_mean, zg_logvar, zl_mean, zl_logvar = self.encoder(mel)
        zg = self.encoder.reparameterize(zg_mean, zg_logvar)
        zl = self.encoder.reparameterize(zl_mean, zl_logvar)

        mel_recon = self.decoder(zg, zl)

        loss, components = self._loss(
            mel, mel_recon,
            zg_mean, zg_logvar,
            zl_mean, zl_logvar
        )

        return mel_recon, loss, components

    def _loss(self, mel, mel_recon, zg_mean, zg_logvar, zl_mean, zl_logvar):
        lrecon = F.mse_loss(mel_recon, mel)

        kl_zg = -0.5 * torch.mean(1 + zg_logvar - zg_mean.pow(2) - zg_logvar.exp())
        kl_zl = -0.5 * torch.mean(1 + zl_logvar - zl_mean.pow(2) - zl_logvar.exp())

        loss = lrecon + self.beta * (kl_zg + kl_zl)

        components = {
            "lrecon": lrecon.item(),
            "kl_zg":  kl_zg.item(),
            "kl_zl":  kl_zl.item(),
            "lsym":   0.0,
            "total":  loss.item(),
        }

        return loss, components

    def encode(self, mel):
        zg_mean, _, zl_mean, _ = self.encoder(mel)
        return zg_mean, zl_mean

    def decode(self, zg, zl):
        return self.decoder(zg, zl)
