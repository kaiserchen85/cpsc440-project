from __future__ import annotations

import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MustardMelDataset
from utils import handle, main
from vae import VAE


def cvae_loss(x, x_hat, mu, logvar, beta=0.1, l1_weight=1.0, l2_weight=1.0):
    l1_loss = F.l1_loss(x_hat, x)
    l2_loss = F.mse_loss(x_hat, x)
    recon_loss = (l1_weight * l1_loss) + (l2_weight * l2_loss)

    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss, kl_loss, recon_loss + (beta * kl_loss)


def _project_root() -> Path:
    return Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Tunable training hyperparameters (see docs/MODEL.md for guidance)
# ---------------------------------------------------------------------------
VAE_BATCH_SIZE = 16
VAE_EPOCHS = 25
VAE_LR = 1e-3
VAE_TARGET_BETA = 0.1
VAE_LATENT_DIM = 64
# Linear KL warmup: beta ramps from 0 -> VAE_TARGET_BETA over this many optimizer steps
VAE_KL_WARMUP_STEPS = None  # None = 2 * len(train_loader) per current epoch-1 length
EXPORT_LATENTS_AFTER_TRAIN = True
VAE_SEED = 440
LATENT_EXPORT_SEED = 440


def seed_everything(seed: int) -> None:
    """Best-effort reproducibility for this project (CPU/GPU)."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic (may reduce performance).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def export_cvae_latents(
    model: VAE,
    device: torch.device,
    npz_path: Path,
    out_npz: Path,
    *,
    seed: int = 440,
) -> None:
    """
    Save encoder means `mu` and one stochastic `z` per row (reparameterization with
    fixed RNG) for train/val/test, aligned with `id_*` / `label_*` row order in the
    merged dataset (same order as `MustardMelDataset` sequential access).
    """
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    bundle: dict[str, np.ndarray] = {}
    meta = {
        "latent_dim": model.latent_dim,
        "seed": seed,
        "npz_path": str(npz_path),
        "description": (
            "mu_*: posterior mean q(z|x,c) per row. z_*: single reparameterized sample "
            "mu + std*epsilon with batch-ordered RNG (use mu_* for stable t-SNE/UMAP)."
        ),
    }
    for split in ("train", "val", "test"):
        ds = MustardMelDataset(npz_path, split)  # type: ignore[arg-type]
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        mus, zs, labs, ids_list = [], [], [], []
        with torch.no_grad():
            for x, text, y, clip_ids in loader:
                x = x.to(device)
                text = text.to(device)
                y = y.to(device)
                mu, logvar = model.encode(x, text, y)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn(std.shape, device=device, dtype=std.dtype, generator=gen)
                z = mu + std * eps
                mus.append(mu.cpu().numpy().astype(np.float32, copy=False))
                zs.append(z.cpu().numpy().astype(np.float32, copy=False))
                labs.append(y.cpu().numpy().astype(np.int64, copy=False))
                ids_list.extend(str(i) for i in clip_ids)

        bundle[f"mu_{split}"] = np.concatenate(mus, axis=0)
        bundle[f"z_{split}"] = np.concatenate(zs, axis=0)
        bundle[f"labels_{split}"] = np.concatenate(labs, axis=0)
        bundle[f"ids_{split}"] = np.asarray(ids_list, dtype=object)

    bundle["meta_json"] = np.array([json.dumps(meta)], dtype=object)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **bundle)
    print(f"saved latents {out_npz}")


@handle("vae-test")
def vae_test():
    """Short sanity run on synthetic mel matching dataset spec_shape when available."""
    root = _project_root()
    npz_path = root / "data/mustard_processed/mustard_logmel.npz"
    if npz_path.is_file():
        spec_shape = MustardMelDataset.spec_shape_from_npz(npz_path, "train")
    else:
        # Fallback shape for quick checks before preprocessing exists.
        spec_shape = (1, 80, 130)

    _, mel_bins, frames = spec_shape
    mel = torch.zeros(1, *spec_shape)
    stripe_mod = max(1, frames // 8)
    for i in range(mel_bins):
        for j in range(frames):
            if j % stripe_mod == i % stripe_mod:
                mel[0, 0, i, j] = 1.0

    text = torch.randint(0, 10000, (1, 64))
    label = torch.tensor([1])

    model = VAE(spec_shape=spec_shape, latent_dim=VAE_LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=VAE_LR)

    total_steps = 200
    target_beta = VAE_TARGET_BETA

    for step in range(total_steps):
        current_beta = target_beta * min(1.0, step / 100.0)
        x_hat, mu, logvar = model(mel, text, label)
        recon, kl, total_loss = cvae_loss(mel, x_hat, mu, logvar, beta=current_beta)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(
                f"Step {step:3d} | Total: {total_loss.item():.4f} | "
                f"Recon: {recon.item():.4f} | KL: {kl.item():.4f} | Beta: {current_beta:.4f}"
            )

    with torch.no_grad():
        x_hat, _, _ = model(mel, text, label)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(mel[0, 0].numpy(), aspect="auto", origin="lower")
    plt.title("Input (mel)")

    plt.subplot(1, 2, 2)
    plt.imshow(x_hat[0, 0].numpy(), aspect="auto", origin="lower")
    plt.title("Reconstruction")
    plt.tight_layout()
    plt.show()


@handle("vae-train")
def vae_train(args=None):
    """Train CVAE on real MUStARD mels + BPE tokens from data/mustard_processed/."""
    root = _project_root()
    npz_path = root / "data/mustard_processed/mustard_logmel.npz"
    vocab_path = root / "data/mustard_processed/vocab.json"
    if not npz_path.is_file():
        raise FileNotFoundError(f"missing {npz_path}")
    if not vocab_path.is_file():
        raise FileNotFoundError(f"missing {vocab_path} (run export_text_tokens.py)")

    spec_shape = MustardMelDataset.spec_shape_from_npz(npz_path, "train")
    vocab_size = MustardMelDataset.text_vocab_size_from_json(vocab_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(VAE_SEED)
    batch_size = VAE_BATCH_SIZE
    epochs = VAE_EPOCHS
    lr = VAE_LR
    target_beta = VAE_TARGET_BETA
    latent_dim = VAE_LATENT_DIM
    kl_warmup_steps = VAE_KL_WARMUP_STEPS

    train_ds = MustardMelDataset(npz_path, "train")
    val_ds = MustardMelDataset(npz_path, "val")
    dl_gen = torch.Generator()
    dl_gen.manual_seed(VAE_SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=dl_gen,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    if kl_warmup_steps is None:
        kl_warmup_steps = max(1, len(train_loader) * 2)

    model = VAE(
        spec_shape=spec_shape, text_vocab_size=vocab_size, latent_dim=latent_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    hist = {
        "train_recon": [],
        "train_kl": [],
        "val_recon": [],
        "val_kl": [],
    }
    for epoch in range(epochs):
        model.train()
        train_recon = 0.0
        train_kl = 0.0
        n_tr = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs} train")
        for x, text, y, _ in pbar:
            x = x.to(device)
            text = text.to(device)
            y = y.to(device)

            current_beta = target_beta * min(1.0, global_step / float(kl_warmup_steps))
            x_hat, mu, logvar = model(x, text, y)
            recon, kl, loss = cvae_loss(x, x_hat, mu, logvar, beta=current_beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            bs = x.shape[0]
            train_recon += recon.item() * bs
            train_kl += kl.item() * bs
            n_tr += bs
            pbar.set_postfix(recon=f"{recon.item():.4f}", kl=f"{kl.item():.4f}", beta=f"{current_beta:.3f}")

        model.eval()
        val_recon = 0.0
        val_kl = 0.0
        n_va = 0
        with torch.no_grad():
            for x, text, y, _ in val_loader:
                x = x.to(device)
                text = text.to(device)
                y = y.to(device)
                x_hat, mu, logvar = model(x, text, y)
                recon, kl, _ = cvae_loss(x, x_hat, mu, logvar, beta=target_beta)
                bs = x.shape[0]
                val_recon += recon.item() * bs
                val_kl += kl.item() * bs
                n_va += bs

        print(
            f"epoch {epoch+1}: train recon {train_recon/n_tr:.4f} kl {train_kl/n_tr:.4f} | "
            f"val recon {val_recon/max(1,n_va):.4f} kl {val_kl/max(1,n_va):.4f}"
        )
        hist["train_recon"].append(train_recon / n_tr)
        hist["train_kl"].append(train_kl / n_tr)
        hist["val_recon"].append(val_recon / max(1, n_va))
        hist["val_kl"].append(val_kl / max(1, n_va))

    ckpt = root / "checkpoints" / "cvae_last.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    train_hparams = {
        "seed": VAE_SEED,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "target_beta": target_beta,
        "latent_dim": latent_dim,
        "kl_warmup_steps": kl_warmup_steps,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "spec_shape": spec_shape,
            "text_vocab_size": vocab_size,
            "latent_dim": latent_dim,
            "train_hparams": train_hparams,
        },
        ckpt,
    )
    print(f"saved {ckpt}")

    plot_path = None
    if args is not None and getattr(args, "plot", None):
        plot_path = Path(str(args.plot))
    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt

        epochs_axis = list(range(1, epochs + 1))
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(epochs_axis, hist["train_recon"], label="train recon")
        ax1.plot(epochs_axis, hist["val_recon"], label="val recon")
        ax1.set_title("CVAE training curves")
        ax1.set_ylabel("recon (L1 + MSE)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(epochs_axis, hist["train_kl"], label="train KL")
        ax2.plot(epochs_axis, hist["val_kl"], label="val KL")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("KL")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"saved training plot {plot_path}")

    if EXPORT_LATENTS_AFTER_TRAIN:
        lat_path = root / "checkpoints" / "cvae_latents.npz"
        export_cvae_latents(
            model,
            device,
            npz_path,
            lat_path,
            seed=LATENT_EXPORT_SEED,
        )


@handle("vae-export-latents")
def vae_export_latents_cmd():
    """Load `checkpoints/cvae_last.pt` and write `checkpoints/cvae_latents.npz` (no training)."""
    root = _project_root()
    npz_path = root / "data/mustard_processed/mustard_logmel.npz"
    ckpt_path = root / "checkpoints" / "cvae_last.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing {ckpt_path} — run `python main.py vae-train` first")
    if not npz_path.is_file():
        raise FileNotFoundError(f"missing {npz_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    spec_shape = tuple(ckpt["spec_shape"])
    text_vocab_size = int(ckpt["text_vocab_size"])
    latent_dim = int(ckpt.get("latent_dim", 64))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(VAE_SEED)
    model = VAE(
        spec_shape=spec_shape, text_vocab_size=text_vocab_size, latent_dim=latent_dim
    ).to(device)
    model.load_state_dict(ckpt["model"])

    out_npz = root / "checkpoints" / "cvae_latents.npz"
    export_cvae_latents(
        model,
        device,
        npz_path,
        out_npz,
        seed=LATENT_EXPORT_SEED,
    )


################################################################################

if __name__ == "__main__":
    main()
