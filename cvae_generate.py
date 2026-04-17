#!/usr/bin/env python3
"""
Decode a random latent + text + sarcasm label into a mel, then optionally vocode to .wav.

Uses the same BPE tokenizer as training (see export_text_tokens.py). Generation uses z ~ N(0, I)
and the decoder; this is an approximate prior (see docs/MODEL.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from vae import VAE
from vocode import load_norm_stats, mel01_to_db, vocode_griffin


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _encode_text(tokenizer_path: Path, text: str, max_len: int, pad_id: int) -> torch.Tensor:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_path))
    enc = tok.encode(text.strip())
    ids = enc.ids[:max_len]
    if len(ids) < max_len:
        ids = ids + [pad_id] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def main() -> None:
    root = _project_root()
    p = argparse.ArgumentParser(description="CVAE: raw string + label -> mel and/or wav")
    p.add_argument("--text", type=str, required=True, help="Utterance text (same style as MUStARD)")
    p.add_argument(
        "--label",
        type=int,
        choices=(0, 1),
        default=0,
        help="0 = non-sarcastic, 1 = sarcastic",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "checkpoints" / "cvae_last.pt",
    )
    p.add_argument(
        "--tokenizer",
        type=Path,
        default=root / "data/mustard_processed/tokenizer.json",
    )
    p.add_argument(
        "--vocab",
        type=Path,
        default=root / "data/mustard_processed/vocab.json",
    )
    p.add_argument(
        "--npz",
        type=Path,
        default=root / "data/mustard_processed/mustard_logmel.npz",
        help="Used for norm_stats sibling path",
    )
    p.add_argument("--out-mel", type=Path, help="Write float32 mel (1,80,130) .npy")
    p.add_argument("--out-wav", type=Path, help="Write waveform via Griffin–Lim")
    p.add_argument("--seed", type=int, default=440)
    p.add_argument(
        "--z-scale",
        type=float,
        default=1.0,
        help="Scale for z ~ N(0, z_scale^2 I); try 0.5–1.5 if output is too noisy/flat",
    )
    p.add_argument("--n-griffin", type=int, default=64)
    args = p.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint {args.checkpoint} (train with: python main.py vae-train)")
    if not args.tokenizer.is_file():
        raise SystemExit(f"missing {args.tokenizer}")
    if not args.vocab.is_file():
        raise SystemExit(f"missing {args.vocab}")

    with args.vocab.open("r", encoding="utf-8") as f:
        vocab_doc = json.load(f)
    meta = vocab_doc["meta"]
    max_len = int(meta["max_seq_len"])
    pad_id = int(meta["pad_id"])

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    spec_shape = tuple(ckpt["spec_shape"])
    text_vocab_size = int(ckpt["text_vocab_size"])
    latent_dim = int(ckpt.get("latent_dim", 64))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(spec_shape=spec_shape, text_vocab_size=text_vocab_size, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokens = _encode_text(args.tokenizer, args.text, max_len, pad_id).to(device)
    y = torch.tensor([args.label], dtype=torch.long, device=device)

    g = torch.Generator(device=device)
    g.manual_seed(args.seed)
    z = torch.randn(1, latent_dim, device=device, generator=g) * args.z_scale

    with torch.no_grad():
        mel = model.decode(z, tokens, y)
    mel = mel.clamp(0.0, 1.0)

    mel_np = mel[0].cpu().numpy().astype(np.float32)

    if args.out_mel:
        args.out_mel.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out_mel, mel_np)
        print(f"wrote mel {mel_np.shape} -> {args.out_mel}")

    if args.out_wav:
        stats = load_norm_stats(args.npz)
        mel_db = mel01_to_db(mel_np, stats)
        wav = vocode_griffin(mel_db, stats, n_iter=args.n_griffin)
        args.out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(args.out_wav), wav, int(stats["sr"]))
        print(f"wrote wav sr={stats['sr']} -> {args.out_wav}")

    if not args.out_mel and not args.out_wav:
        raise SystemExit("provide at least one of --out-mel or --out-wav")


if __name__ == "__main__":
    main()
