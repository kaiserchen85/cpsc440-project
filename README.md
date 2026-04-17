## CPSC 440 Project

### What this repo contains

- **MUStARD audio dataset (preprocessed)**: `data/mustard_processed/mustard_logmel.npz`
- **Entry points**: `python main.py <command>` (see `python main.py -h`)

### Repository layout

| File | Role |
|------|------|
| [`docs/DATA.md`](docs/DATA.md) | Preprocessed tensors, regeneration, mel visualization |
| [`docs/MODEL.md`](docs/MODEL.md) | CVAE architecture, training, latent export, generation |
| [`docs/SETUP.md`](docs/SETUP.md) | Virtualenv and dependencies |
| [`main.py`](main.py) | CLI: `vae-train`, `vae-export-latents`, `vae-test` |
| [`vae.py`](vae.py) | `VAE` module |
| [`dataset.py`](dataset.py) | `MustardMelDataset`; helpers for `spec_shape` / vocab size |
| [`cvae_generate.py`](cvae_generate.py) | Text + sarcasm label + sampled `z` → mel or `.wav` |
| [`vocode.py`](vocode.py) | Mel → waveform (Griffin–Lim; optional HiFi-GAN) |
| [`export_text_tokens.py`](export_text_tokens.py) | Train BPE, write `tokenizer.json` / `vocab.json`, merge `tokens_*` into `.npz` |
| [`preprocess.py`](preprocess.py) | Raw clips → `mustard_logmel.npz` |
| [`mustard_split.py`](mustard_split.py) | Deterministic train/val/test split (shared with preprocessing / tokens) |
| [`spectrogram.py`](spectrogram.py) | Plot or save a mel row from the merged `.npz` |
| [`utils.py`](utils.py) | `@handle` registry and `main()` for `main.py` commands |

### Quick start (run examples)

```bash
cd cpsc440-project
python main.py vae-test
python main.py vae-train          # also writes checkpoints/cvae_latents.npz by default
python main.py vae-export-latents # re-export latents from checkpoints/cvae_last.pt only
python vocode.py --split train --index 0 --out /tmp/sample.wav --backend griffin
python cvae_generate.py --text "Great, just great." --label 1 --out-wav /tmp/gen.wav
```
