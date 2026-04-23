## Usage (after training/tuning)

This document assumes you already have:

- a trained model checkpoint: `checkpoints/cvae_last.pt`
- tokenization artifacts: `data/mustard_processed/{vocab.json,tokenizer.json}`
- a merged dataset (for reconstruction / latent export): `data/mustard_processed/mustard_logmel.npz`

If you still need to train or tune, go to [`TUNING.md`](TUNING.md). For model architecture only, see [`MODEL.md`](MODEL.md).

### Best model (current)

The repo is currently set up to use the **best CVAE hyperparameters we found** (saved as constants at the top of [`main.py`](../main.py)). Training with `python main.py vae-train` will write `checkpoints/cvae_last.pt`, and the checkpoint also stores these values under `train_hparams` for reproducibility.

- **Training command**: `python main.py vae-train`
- **Checkpoint path**: `checkpoints/cvae_last.pt`
- **Processed dataset path**: `data/mustard_processed/mustard_logmel.npz` (shape is inferred at runtime via `spec_shape_from_npz`, so it’s OK if preprocessing produced a larger `.npz`)

Hyperparameters:

- **batch size**: 32
- **epochs**: 50
- **optimizer**: Adam
- **learning rate**: 1e-3
- **latent dim**: 256
- **KL target beta**: 0.1
- **KL warmup**: linear ramp over 100 optimizer steps
- **seed**: 440 (training) / 440 (latent export)

### Contents

- [What artifacts matter](#what-artifacts-matter)
- [Generate augmented audio (text + label → wav)](#generate-augmented-audio-text--label--wav)
- [Latent space export and visualization](#latent-space-export-and-visualization)

---

## What artifacts matter

- **Trained model**: `checkpoints/cvae_last.pt`
  - contains `model` weights + `spec_shape`, `text_vocab_size`, `latent_dim`, `train_hparams`
- **Latent export (optional)**: `checkpoints/cvae_latents.npz`
  - contains `mu_*` and `z_*` arrays aligned with `ids_*`/`labels_*`
- **Tokenizer**: `data/mustard_processed/tokenizer.json` and `vocab.json`
  - required to encode **raw strings** consistently with training
- **Vocoder stats**: `data/mustard_processed/mustard_logmel.norm_stats.json`
  - required to denormalize mel `[0,1]` back to dB scale for vocoding

---

## Generate augmented audio (text + label → wav)

Goal: generate a new mel (and optionally a `.wav`) from **only** a raw string + chosen sarcasm label, for augmentation/analysis.

Use [`cvae_generate.py`](../cvae_generate.py). It:

1. encodes `--text` using `tokenizer.json`
2. samples `z ~ Normal(0, z_scale² I)` (seeded)
3. decodes mel with `model.decode(z, tokens, label)`
4. optionally vocodes to `.wav`

Example:

```bash
python cvae_generate.py \
  --text "Well that's just great." \
  --label 1 \
  --out-wav out/gen.wav \
  --seed 123
```

Useful knobs:

- `--z-scale`:
  - try smaller (e.g. `0.5`) if output is mostly noise
  - try larger (e.g. `1.2`) if output is too flat/repetitive
- `--out-mel out/gen.npy` if you only want the generated mel tensor.

If you want to *evaluate* whether tuning helped (reconstruction/generation checks), follow the evaluation section in [`TUNING.md`](TUNING.md#how-to-evaluate-changes-minimal--practical).

---

## Latent space export and visualization

To export per-utterance latents aligned with `ids_*` / `labels_*`, run:

```bash
python main.py vae-export-latents
```

This writes `checkpoints/cvae_latents.npz` (see `meta_json` inside the file for details).

Recommended for embeddings:

- Use **`mu_*`** for stable t-SNE/UMAP (no sampling noise).
- Use `labels_*` to color points by sarcasm label.

The exact arrays inside are documented in `checkpoints/cvae_latents.npz` output itself (`meta_json`) and in [`main.py`](../main.py) (`export_cvae_latents`).

