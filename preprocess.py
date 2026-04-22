import argparse
import json
import math
import os
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from mustard_split import split_ids_random as _split_ids_random

@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for waveform → log-mel preprocessing."""

    sr: int
    n_fft: int
    hop_length: int
    n_mels: int
    target_seconds: float
    trim_top_db: float
    norm_mode: str  # "minmax" or "percentile"
    norm_low: float
    norm_high: float
    scale_range: str  # "0_1" or "neg1_1"
    seed: int


def _ensure_dir(p: Path) -> None:
    """Create a directory if it does not already exist."""
    p.mkdir(parents=True, exist_ok=True)


def _run_ffmpeg_extract_wav(mp4_path: Path, wav_path: Path, sr: int) -> None:
    """
    Extract mono PCM wav at fixed sample rate.
    Requires ffmpeg on PATH.
    """
    _ensure_dir(wav_path.parent)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp4_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "wav",
        str(wav_path),
    ]
    # Keep ffmpeg quiet unless there's an error
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _load_wav_mono(wav_path: Path) -> np.ndarray:
    """Load a wav file as a mono float32 waveform."""
    import soundfile as sf

    y, sr = sf.read(str(wav_path), always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    # soundfile may return float64
    y = y.astype(np.float32, copy=False)
    return y


def _trim_silence(y: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Trim leading/trailing silence using an energy-based heuristic."""
    import librosa

    yt, _ = librosa.effects.trim(y, top_db=cfg.trim_top_db)
    return yt.astype(np.float32, copy=False)


def _fix_length_center(y: np.ndarray, target_len: int) -> np.ndarray:
    """Center-crop or zero-pad a waveform to a fixed number of samples."""
    if len(y) == target_len:
        return y
    if len(y) > target_len:
        start = (len(y) - target_len) // 2
        return y[start : start + target_len]
    # pad
    pad_total = target_len - len(y)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(y, (pad_left, pad_right), mode="constant").astype(np.float32, copy=False)


def _logmel(y: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Compute a log-mel spectrogram in dB (shape: n_mels × n_frames)."""
    import librosa

    m = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        power=2.0,
    )
    db = librosa.power_to_db(m, ref=np.max)
    return db.astype(np.float32, copy=False)


def _scale_spec(
    spec: np.ndarray,
    stats: Dict[str, float],
    cfg: PreprocessConfig,
) -> np.ndarray:
    """Scale a dB log-mel spectrogram to [0,1] (or [-1,1]) using train-fit stats."""
    lo = stats["lo"]
    hi = stats["hi"]
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        raise ValueError(f"bad normalization stats lo={lo} hi={hi}")
    x = (spec - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    if cfg.scale_range == "neg1_1":
        x = (x * 2.0) - 1.0
    return x.astype(np.float32, copy=False)


def _compute_norm_stats(
    specs_db: Sequence[np.ndarray],
    cfg: PreprocessConfig,
) -> Dict[str, float]:
    """Compute normalization constants from training dB log-mel spectrograms."""
    flat = np.concatenate([s.reshape(-1) for s in specs_db]).astype(np.float32, copy=False)
    if cfg.norm_mode == "minmax":
        lo = float(np.min(flat))
        hi = float(np.max(flat))
    elif cfg.norm_mode == "percentile":
        lo = float(np.percentile(flat, cfg.norm_low))
        hi = float(np.percentile(flat, cfg.norm_high))
    else:
        raise ValueError(f"unknown norm_mode {cfg.norm_mode}")
    return {"lo": lo, "hi": hi}


def _load_mustard_json(json_path: Path) -> Dict[str, Any]:
    """Load MUStARD annotations (ID → dict with 'sarcasm' label, etc.)."""
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_split_indices_pkl(pkl_path: Path) -> Any:
    """Load a pickle file (used for optional MUStARD split indices, if present)."""
    with pkl_path.open("rb") as f:
        return pickle.load(f)


def _relpath(path: Path, base_dir: Path) -> str:
    """Return a stable, relative path string for manifests/metadata."""
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except Exception:
        # Fall back to os.path.relpath for paths outside base_dir
        return os.path.relpath(str(path), str(base_dir))


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Preprocess MUStARD utterance MP4s into log-mel tensors.")
    p.add_argument("--json-path", type=Path, default=Path("data/mustard_raw/sarcasm_data.json"))
    p.add_argument(
        "--mp4-dir",
        type=Path,
        default=Path("data/mustard_raw/videos/utterances_final"),
        help="Directory containing utterance mp4s named {id}.mp4",
    )
    p.add_argument(
        "--wav-cache-dir",
        type=Path,
        default=Path("data/mustard_raw/audio"),
        help="Where to write extracted wavs (cached).",
    )
    p.add_argument("--output-dir", type=Path, default=Path("data/mustard_processed"))
    p.add_argument(
        "--output-name",
        type=str,
        default="mustard_logmel",
        help="Base name for output files under output-dir.",
    )
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--n-fft", type=int, default=4096)
    p.add_argument("--hop-length", type=int, default=512)
    p.add_argument("--n-mels", type=int, default=256)
    p.add_argument("--target-seconds", type=float, default=5.0)
    p.add_argument("--trim-top-db", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=440)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to export. 'all' exports train/val/test in one npz.",
    )
    p.add_argument(
        "--norm-mode",
        choices=["minmax", "percentile"],
        default="minmax",
    )
    p.add_argument("--norm-low", type=float, default=1.0, help="Percentile low (if norm-mode=percentile).")
    p.add_argument("--norm-high", type=float, default=99.0, help="Percentile high (if norm-mode=percentile).")
    p.add_argument("--scale-range", choices=["0_1", "neg1_1"], default="0_1")
    p.add_argument(
        "--split-indices-pkl",
        type=Path,
        default=None,
        help="Optional: MUStARD split indices pickle (if you have it) for reproducible folds. "
        "If provided, we will store it in outputs for later use (and still do a random split unless extended).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Sanity-check a small subset (decode → log-mel) without writing outputs.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only process the first N IDs in each selected split (useful for quick tests).",
    )
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent

    cfg = PreprocessConfig(
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        target_seconds=args.target_seconds,
        trim_top_db=args.trim_top_db,
        norm_mode=args.norm_mode,
        norm_low=args.norm_low,
        norm_high=args.norm_high,
        scale_range=args.scale_range,
        seed=args.seed,
    )

    _ensure_dir(args.output_dir)
    _ensure_dir(args.wav_cache_dir)

    data = _load_mustard_json(args.json_path)
    all_ids = sorted(data.keys())
    splits = _split_ids_random(all_ids, seed=cfg.seed, train_frac=args.train_frac, val_frac=args.val_frac)

    if args.split == "all":
        export_splits = ["train", "val", "test"]
    else:
        export_splits = [args.split]

    def _take_limit(xs: List[str]) -> List[str]:
        return xs if args.limit <= 0 else xs[: args.limit]

    # Precompute log-mels for train split to fit normalization (train-only).
    train_specs_db: List[np.ndarray] = []
    train_ok_ids: List[str] = []
    manifest_path = args.output_dir / f"{args.output_name}.manifest.jsonl"
    norm_path = args.output_dir / f"{args.output_name}.norm_stats.json"
    npz_path = args.output_dir / f"{args.output_name}.npz"

    # Build manifest while doing extraction; write streaming for debuggability
    with manifest_path.open("w", encoding="utf-8") as mf:
        for clip_id in _take_limit(splits["train"]):
            mp4_path = args.mp4_dir / f"{clip_id}.mp4"
            wav_path = args.wav_cache_dir / f"{clip_id}.wav"
            rec: Dict[str, Any] = {
                "id": clip_id,
                "mp4_path": _relpath(mp4_path, project_root),
                "wav_path": _relpath(wav_path, project_root),
            }
            if not mp4_path.exists():
                rec.update({"ok": False, "error": "missing_mp4"})
                mf.write(json.dumps(rec) + "\n")
                continue
            try:
                if not wav_path.exists():
                    _run_ffmpeg_extract_wav(mp4_path, wav_path, sr=cfg.sr)
                y = _load_wav_mono(wav_path)
                y = _trim_silence(y, cfg)
                target_len = int(round(cfg.target_seconds * cfg.sr))
                y = _fix_length_center(y, target_len)
                spec_db = _logmel(y, cfg)
                train_specs_db.append(spec_db)
                train_ok_ids.append(clip_id)
                rec.update({"ok": True})
            except Exception as e:
                rec.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
            mf.write(json.dumps(rec) + "\n")

    if args.dry_run:
        # Quick sanity report, then exit.
        ok = len(train_specs_db)
        total = len(_take_limit(splits["train"]))
        ex_shape = train_specs_db[0].shape
        ex_min = float(np.min(train_specs_db[0]))
        ex_max = float(np.max(train_specs_db[0]))
        print(f"[dry-run] train_ok={ok}/{total} example_shape={ex_shape} example_db_range=({ex_min:.2f},{ex_max:.2f})")
        return

    if not train_specs_db:
        raise RuntimeError("no usable training examples found; check mp4-dir/json-path and ffmpeg availability")

    stats = _compute_norm_stats(train_specs_db, cfg)
    with norm_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "norm_mode": cfg.norm_mode,
                "scale_range": cfg.scale_range,
                "lo": stats["lo"],
                "hi": stats["hi"],
                "sr": cfg.sr,
                "n_fft": cfg.n_fft,
                "hop_length": cfg.hop_length,
                "n_mels": cfg.n_mels,
                "target_seconds": cfg.target_seconds,
                "trim_top_db": cfg.trim_top_db,
                "seed": cfg.seed,
                "train_ids_used_for_stats": train_ok_ids,
            },
            f,
            indent=2,
        )

    def process_split(split_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        specs: List[np.ndarray] = []
        labels: List[int] = []
        ids_out: List[str] = []
        for clip_id in _take_limit(splits[split_name]):
            mp4_path = args.mp4_dir / f"{clip_id}.mp4"
            wav_path = args.wav_cache_dir / f"{clip_id}.wav"
            if not mp4_path.exists():
                continue
            try:
                if not wav_path.exists():
                    _run_ffmpeg_extract_wav(mp4_path, wav_path, sr=cfg.sr)
                y = _load_wav_mono(wav_path)
                y = _trim_silence(y, cfg)
                target_len = int(round(cfg.target_seconds * cfg.sr))
                y = _fix_length_center(y, target_len)
                spec_db = _logmel(y, cfg)
                spec = _scale_spec(spec_db, stats, cfg)
            except Exception:
                continue

            label = int(bool(data[clip_id].get("sarcasm", False)))
            specs.append(spec[None, ...])  # (1, n_mels, T)
            labels.append(label)
            ids_out.append(clip_id)

        if not specs:
            return (
                np.zeros((0, 1, cfg.n_mels, 0), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.array([], dtype=object),
            )
        x = np.stack(specs, axis=0).astype(np.float32, copy=False)  # (N, 1, n_mels, T)
        y = np.asarray(labels, dtype=np.int64)
        z = np.asarray(ids_out, dtype=object)
        return x, y, z

    out: Dict[str, Any] = {}
    for s in export_splits:
        x, y, z = process_split(s)
        out[f"specs_{s}"] = x
        out[f"labels_{s}"] = y
        out[f"ids_{s}"] = z

    # Also store config to make downstream usage unambiguous
    out["meta"] = np.array(
        [
            json.dumps(
                {
                    "sr": cfg.sr,
                    "n_fft": cfg.n_fft,
                    "hop_length": cfg.hop_length,
                    "n_mels": cfg.n_mels,
                    "target_seconds": cfg.target_seconds,
                    "scale_range": cfg.scale_range,
                    "norm_stats_path": _relpath(norm_path, project_root),
                    "manifest_path": _relpath(manifest_path, project_root),
                    "mp4_dir": _relpath(args.mp4_dir, project_root),
                    "wav_cache_dir": _relpath(args.wav_cache_dir, project_root),
                }
            )
        ],
        dtype=object,
    )

    if args.split_indices_pkl is not None and args.split_indices_pkl.exists():
        # We don't interpret folds yet; we just package it for reproducible future work.
        try:
            _ = _load_split_indices_pkl(args.split_indices_pkl)
            out["split_indices_pkl_path"] = np.array([_relpath(args.split_indices_pkl, project_root)], dtype=object)
        except Exception:
            pass

    np.savez_compressed(npz_path, **out)
    print(f"Wrote: {npz_path}")
    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {norm_path}")


if __name__ == "__main__":
    main()
