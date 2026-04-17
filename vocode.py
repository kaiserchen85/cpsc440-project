"""Invert stored [0,1] log-mel tensors to waveform (Griffin–Lim or optional HiFi-GAN)."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np

Backend = Literal["griffin", "hifigan"]


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _configure_numba_cache() -> None:
    """
    Librosa pulls in numba; some environments error when numba tries to cache compiled
    functions. Disable caching by default for robustness.
    """
    os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(_project_root() / ".numba_cache"))
    try:
        Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def load_norm_stats(npz_path: Path) -> Dict[str, Any]:
    # mustard_logmel.npz -> mustard_logmel.norm_stats.json
    stats_path = npz_path.with_name(f"{npz_path.stem}.norm_stats.json")
    if not stats_path.is_file():
        raise FileNotFoundError(f"missing {stats_path}")
    with stats_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mel01_to_db(mel: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    """Undo train-fitted min–max: mel in [0,1] -> log-mel in dB (same scale as preprocess)."""
    lo = float(stats["lo"])
    hi = float(stats["hi"])
    return mel.astype(np.float64) * (hi - lo) + lo


def vocode_griffin(mel_db: np.ndarray, stats: Dict[str, Any], n_iter: int = 64) -> np.ndarray:
    _configure_numba_cache()
    import librosa

    # Per preprocess: power mel -> power_to_db(..., ref=np.max). Inversion is approximate without per-frame ref.
    mel_power = librosa.db_to_power(mel_db, ref=1.0)
    sr = int(stats["sr"])
    n_fft = int(stats["n_fft"])
    hop_length = int(stats["hop_length"])
    wav = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        power=1.0,
        n_iter=n_iter,
    )
    return wav.astype(np.float32, copy=False)


def vocode_hifigan(mel_db: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    """SpeechBrain pretrained HiFi-GAN (LJS). Mel domain may differ from MUStARD; listen before trusting."""
    _configure_numba_cache()
    import librosa
    import torch
    import torchaudio

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]

    from speechbrain.inference.vocoders import HIFIGAN

    mel_lin = librosa.db_to_power(mel_db.astype(np.float64), ref=1.0).astype(np.float32)
    mel_log = np.log(np.maximum(mel_lin, 1e-5))
    t = torch.from_numpy(mel_log)[None, :, :]  # (1, n_mels, T)

    savedir = tempfile.mkdtemp(prefix="sb_hifigan_")
    vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir=savedir)
    vocoder.eval()
    with torch.no_grad():
        wavs = vocoder.decode_batch(t)
    w = wavs[0]
    if hasattr(w, "cpu"):
        w = w.cpu().numpy()
    return np.asarray(w, dtype=np.float32).reshape(-1)


def load_mel_from_npz(npz_path: Path, split: str, index: int) -> np.ndarray:
    z = np.load(npz_path, allow_pickle=True)
    key = f"specs_{split}"
    if key not in z:
        raise KeyError(f"{npz_path} has no {key}")
    x = z[key][index]
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"expected mel (1, H, W), got {x.shape}")
    return x[0].astype(np.float32, copy=False)


def main() -> None:
    root = _project_root()
    p = argparse.ArgumentParser(description="Vocode a single row from mustard_logmel.npz")
    p.add_argument(
        "--npz",
        type=Path,
        default=root / "data/mustard_processed/mustard_logmel.npz",
        help="Path to merged .npz",
    )
    p.add_argument("--split", choices=("train", "val", "test"), default="train")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="Output .wav path")
    p.add_argument("--backend", choices=("griffin", "hifigan"), default="griffin")
    p.add_argument("--n-griffin", type=int, default=64, help="Griffin–Lim iterations")
    args = p.parse_args()

    stats = load_norm_stats(args.npz)
    mel01 = load_mel_from_npz(args.npz, args.split, args.index)
    mel_db = mel01_to_db(mel01, stats)

    if args.backend == "griffin":
        wav = vocode_griffin(mel_db, stats, n_iter=args.n_griffin)
    else:
        try:
            wav = vocode_hifigan(mel_db, stats)
        except TypeError as e:
            if "use_auth_token" in str(e) or "unexpected keyword" in str(e):
                raise SystemExit(
                    "HiFi-GAN load failed (often old speechbrain vs huggingface_hub). "
                    "Upgrade: pip install -U 'speechbrain>=1.1.0' 'huggingface_hub>=0.25.0' torchaudio"
                ) from e
            raise

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    sf.write(str(args.out), wav, int(stats["sr"]))
    print(f"wrote {args.out} sr={stats['sr']} samples={wav.shape[0]} backend={args.backend}")


if __name__ == "__main__":
    main()
