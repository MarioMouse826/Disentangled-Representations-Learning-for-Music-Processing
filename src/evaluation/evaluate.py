import numpy as np
import torch
from src.evaluation.mig import mig
from src.evaluation.dci import dci
from src.evaluation.srr import srr


@torch.no_grad()
def evaluate(model, nsynth_loader, moisesdb_loader, device="cpu"):
    model.eval()

    zs_all, zc_all = [], []
    pitch_all, identity_all = [], []
    mel_orig_all, mel_recon_all = [], []

    # --- NSynth pass ---
    for batch in nsynth_loader:
        mel = batch["mel"].to(device)
        zs, zc = model.encode(mel)
        mel_recon = model.decode(zs, zc)

        zs_all.append(zs.cpu().numpy())
        zc_all.append(zc.cpu().numpy())
        pitch_all.append(batch["pitch"].numpy())
        identity_all.append(np.full(len(batch["mel"]), -1))
        mel_orig_all.append(mel.cpu().numpy())
        mel_recon_all.append(mel_recon.cpu().numpy())

    # --- MoisesDB pass ---
    for batch in moisesdb_loader:
        mel = batch["mel"].to(device)
        zs, zc = model.encode(mel)
        mel_recon = model.decode(zs, zc)

        zs_all.append(zs.cpu().numpy())
        zc_all.append(zc.cpu().numpy())
        pitch_all.append(np.full(len(batch["mel"]), -1))
        identity_all.append(batch["identity"].numpy())
        mel_orig_all.append(mel.cpu().numpy())
        mel_recon_all.append(mel_recon.cpu().numpy())

    zs_all = np.concatenate(zs_all)
    zc_all = np.concatenate(zc_all)
    pitch_all = np.concatenate(pitch_all)
    identity_all = np.concatenate(identity_all)
    mel_orig_all = np.concatenate(mel_orig_all)
    mel_recon_all = np.concatenate(mel_recon_all)

    mig_pitch    = mig(zc_all, pitch_all)
    mig_identity = mig(zs_all, identity_all)
    dci_scores   = dci(np.concatenate([zs_all, zc_all], axis=1),
                       {"pitch": pitch_all, "identity": identity_all})
    srr_score    = srr(mel_orig_all, mel_recon_all) 

    results = {
        "mig_pitch":    round(mig_pitch, 4),
        "mig_identity": round(mig_identity, 4),
        "dci_d":        round(dci_scores["disentanglement"], 4),
        "dci_c":        round(dci_scores["completeness"], 4),
        "dci_i":        round(dci_scores["informativeness"], 4),
        "srr_db":       round(srr_score, 2),
    }

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k:<15} {v}")

    return results
