#!/usr/bin/env python3
"""
Reconstruction check: dataset mel -> (encoder+decoder) -> vocode both input and recon.

This is for evaluating **reconstruction quality** (A), not prior-sampled generation (B).
For generation from text + sampled z, use cvae_generate.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import torch

from dataset import MustardMelDataset
from vae import VAE
from vocode import load_norm_stats, mel01_to_db, vocode_griffin


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _write_wav_pcm16(path: Path, wav: np.ndarray, sr: int) -> None:
    import wave
    from typing import cast

    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f = cast("wave.Wave_write", f)
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(int(sr))
        f.writeframes(pcm.tobytes())


def main() -> None:
    root = _project_root()
    p = argparse.ArgumentParser(description="CVAE reconstruction check (input vs recon wav)")
    p.add_argument("--split", choices=("train", "val", "test"), default="train")
    p.add_argument("--index", type=int, default=0)
    p.add_argument(
        "--npz",
        type=Path,
        default=root / "data/mustard_processed/mustard_logmel.npz",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "checkpoints" / "cvae_last.pt",
    )
    p.add_argument(
        "--out-in-wav",
        type=Path,
        required=True,
        help="Output wav for vocoded INPUT mel (dataset row)",
    )
    p.add_argument(
        "--out-recon-wav",
        type=Path,
        required=True,
        help="Output wav for vocoded RECON mel (encoder->decoder output)",
    )
    p.add_argument(
        "--out-raw-wav",
        type=Path,
        default=None,
        help="Optional: if data/mustard_raw/audio/{id}.wav exists, copy it here for comparison",
    )
    p.add_argument(
        "--out-plot",
        type=Path,
        default=None,
        help="Output path for the Mel-Spectrogram comparison plot (PNG)",
    )
    p.add_argument("--n-griffin", type=int, default=64)
    args = p.parse_args()

    if not args.npz.is_file():
        raise SystemExit(f"missing {args.npz}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing {args.checkpoint} (train with: python main.py vae-train)")

    ds = MustardMelDataset(args.npz, args.split)  # type: ignore[arg-type]
    x, tokens, y, clip_id = ds[args.index]

    # Optional: copy raw wav cache if present
    if args.out_raw_wav is not None:
        raw = root / "data" / "mustard_raw" / "audio" / f"{clip_id}.wav"
        if raw.is_file():
            args.out_raw_wav.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(raw, args.out_raw_wav)

    # Load utterance text (nice for sanity checks)
    json_path = root / "data" / "mustard_raw" / "sarcasm_data.json"
    utt = None
    if json_path.is_file():
        with json_path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        utt = str(d.get(clip_id, {}).get("utterance", "")).strip() or None

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    spec_shape = tuple(ckpt["spec_shape"])
    text_vocab_size = int(ckpt["text_vocab_size"])
    latent_dim = int(ckpt.get("latent_dim", 64))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(spec_shape=spec_shape, text_vocab_size=text_vocab_size, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    xb = x.unsqueeze(0).to(device)  # (1,1,80,T)
    tb = tokens.unsqueeze(0).to(device)  # (1,T)
    yb = y.unsqueeze(0).to(device)  # (1,)
    with torch.no_grad():
        x_hat, _, _ = model(xb, tb, yb)

    x_np = x.cpu().numpy().astype(np.float32)
    xh_np = x_hat[0].cpu().numpy().astype(np.float32)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(x_np[0], aspect="auto", origin="lower")
    plt.title("Original Mel")

    plt.subplot(1, 2, 2)
    plt.imshow(xh_np[0], aspect="auto", origin="lower")
    plt.title("Reconstructed Mel")

    if args.out_plot is not None:
        save_path = args.out_plot
    else:
        # Fallback to the original logic if no path is provided
        save_path = root / "out" / f"recon_check_{clip_id}.png"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()
    
    print(f"wrote plot      -> {save_path}")

    stats = load_norm_stats(args.npz)
    sr = int(stats["sr"])

    # Vocode input mel and recon mel with the same pipeline
    wav_in = vocode_griffin(mel01_to_db(x_np[0], stats), stats, n_iter=args.n_griffin)
    wav_rec = vocode_griffin(mel01_to_db(xh_np[0], stats), stats, n_iter=args.n_griffin)

    _write_wav_pcm16(args.out_in_wav, wav_in, sr)
    _write_wav_pcm16(args.out_recon_wav, wav_rec, sr)

    print(f"id={clip_id} label={int(y.item())} split={args.split} index={args.index}")
    if utt:
        print(f"utterance: {utt}")
    print(f"wrote input wav -> {args.out_in_wav}")
    print(f"wrote recon wav -> {args.out_recon_wav}")


if __name__ == "__main__":
    main()

