import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.encoder import Encoder
from src.models.decoder import Decoder


class SymmetryVAE(nn.Module):
    def __init__(self, zs_dim=32, zc_dim=32, beta=1.0, lambda_sym=1.0):
        super().__init__()
        self.encoder = Encoder(zs_dim, zc_dim)
        self.decoder = Decoder(zs_dim, zc_dim)
        self.beta = beta
        self.lambda_sym = lambda_sym

    def forward(self, mel, mel_shifted):
        # encode original
        zs_mean, zs_logvar, zc_mean, zc_logvar = self.encoder(mel)
        zs = self.encoder.reparameterize(zs_mean, zs_logvar)
        zc = self.encoder.reparameterize(zc_mean, zc_logvar)

        # decode
        mel_recon = self.decoder(zs, zc)

        # encode shifted — only need zs_mean for symmetry loss
        zs_shifted_mean, _, _, _ = self.encoder(mel_shifted)

        loss, components = self._loss(
            mel, mel_recon,
            zs_mean, zs_logvar,
            zc_mean, zc_logvar,
            zs_shifted_mean
        )

        return mel_recon, loss, components

    def _loss(self, mel, mel_recon, zs_mean, zs_logvar, zc_mean, zc_logvar, zs_shifted_mean):
        lrecon = F.mse_loss(mel_recon, mel)

        kl_zs = -0.5 * torch.mean(1 + zs_logvar - zs_mean.pow(2) - zs_logvar.exp())
        kl_zc = -0.5 * torch.mean(1 + zc_logvar - zc_mean.pow(2) - zc_logvar.exp())

        lsym = F.mse_loss(zs_mean, zs_shifted_mean)

        loss = lrecon + self.beta * (kl_zs + kl_zc) + self.lambda_sym * lsym

        components = {
            "lrecon":   lrecon.item(),
            "kl_zs":    kl_zs.item(),
            "kl_zc":    kl_zc.item(),
            "lsym":     lsym.item(),
            "total":    loss.item(),
        }

        return loss, components

    def encode(self, mel):
        zs_mean, _, zc_mean, _ = self.encoder(mel)
        return zs_mean, zc_mean

    def decode(self, zs, zc):
        return self.decoder(zs, zc)