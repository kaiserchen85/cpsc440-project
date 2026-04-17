# Model, training, and inference

This document follows [`DATA.md`](DATA.md): it describes **what the codebase implements**—inputs, [`vae.py`](../vae.py) architecture, [`main.py`](../main.py) training and exports, and how to run [`vocode.py`](../vocode.py) / [`cvae_generate.py`](../cvae_generate.py).

### Contents

- [Inputs (from the merged dataset)](#inputs-from-the-merged-dataset)
- [Model: conditional VAE (`VAE` in `vae.py`)](#model-conditional-vae-vae-in-vaepy)
  - [Encoder](#encoder)
  - [Latent and decoder](#latent-and-decoder)
  - [Loss (`cvae_loss` in `main.py`)](#loss-cvae_loss-in-mainpy)
- [Training](#training)
  - [Constants (`main.py`)](#constants-mainpy)
  - [Outputs](#outputs)
  - [Other `main.py` commands](#other-mainpy-commands)
- [Latent export (`cvae_latents.npz`)](#latent-export-cvae_latentsnpz)
- [Evaluation and splits](#evaluation-and-splits)
- [Generation: text + label to mel / wav (`cvae_generate.py`)](#generation-text--label-to-mel--wav-cvae_generatepy)

---

## Inputs (from the merged dataset)

| Artifact | Role |
|----------|------|
| [`data/mustard_processed/mustard_logmel.npz`](../data/mustard_processed/mustard_logmel.npz) | `specs_*` `(N,1,80,130)` float mel in `[0,1]`, `labels_*`, `ids_*`, `tokens_*` |
| [`data/mustard_processed/vocab.json`](../data/mustard_processed/vocab.json) | BPE `meta`: `vocab_size`, `max_seq_len`, pad/unk ids |
| [`data/mustard_processed/tokenizer.json`](../data/mustard_processed/tokenizer.json) | Same BPE as training; used to encode raw text in [`cvae_generate.py`](../cvae_generate.py) |
| [`data/mustard_processed/mustard_logmel.norm_stats.json`](../data/mustard_processed/mustard_logmel.norm_stats.json) | `lo`/`hi` dB range, `sr`, `n_fft`, `hop_length` for vocoding |

**Code map:** [`vae.py`](../vae.py) defines `VAE`. [`main.py`](../main.py) runs `vae-train`, `vae-export-latents`, `vae-test`, and `cvae_loss` / `export_cvae_latents`. [`dataset.py`](../dataset.py) loads batches from the `.npz`. [`vocode.py`](../vocode.py) turns a mel tensor into audio; [`cvae_generate.py`](../cvae_generate.py) runs the decoder from text + label + sampled `z`.

---

## Model: conditional VAE (`VAE` in `vae.py`)

The network maps:

- **Mel** `x`: `(batch, 1, 80, 130)` — same layout as `specs_*`.
- **Tokens** `(batch, T)` — BPE ids, padded length `T` = `max_seq_len` from `vocab.json` (typically 64).
- **Label** `y`: `(batch,)` int64, values `0` or `1` (same encoding as `labels_*`).

to a reconstruction `x̂` with the same shape as `x`, and (during training) a KL term on a Gaussian latent.

### Encoder

1. **Mel branch:** `Conv2d` stack: `1→32→64→128` channels, kernel 4, stride 2, padding 1 (three blocks). Output is flattened; size of that vector is computed at init from `spec_shape` (see `enc_shape` in code).
2. **Text branch:** `nn.Embedding(text_vocab_size, text_embed_dim)` → mean over sequence → `Linear` + ReLU → **128-d** `h_text`.
3. **Label branch:** `F.one_hot(y, 2)` → `Linear` + ReLU → **32-d** `h_label`.
4. Concatenate `[flat_mel, h_text, h_label]` → two linear heads → **`mu`** and **`logvar`**, each `(batch, latent_dim)` (default `latent_dim=64`).

### Latent and decoder

- **Reparameterization:** `z = mu + exp(0.5 * logvar) * epsilon`, `epsilon ~ Normal(0, I)` (see `reparameterize` in [`vae.py`](../vae.py)).
- **Decoder:** concat `[z, h_text, h_label]` → linear to vector of length `flattened CNN dim` → reshape to `enc_shape` → three `ConvTranspose2d` blocks mirroring the encoder. If output spatial size ≠ `(80, 130)`, **`F.interpolate`** bilinearly to `spec_shape[1:]`.

### Loss (`cvae_loss` in `main.py`)

- Reconstruction: `L1(x̂, x) + MSE(x̂, x)` (equal weights).
- KL: batch mean of `-0.5 * mean(1 + logvar - mu² - exp(logvar))` over latent dims.
- Total: `recon + beta * kl`, with `beta` ramped from `0` to `VAE_TARGET_BETA` over `VAE_KL_WARMUP_STEPS` steps (default `None` → `2 * len(train_loader)`).

```mermaid
flowchart TB
  subgraph in[Inputs]
    M["x (B,1,80,130)"]
    TOK["tokens (B,T)"]
    LAB["y (B,)"]
  end
  M --> CNN["3× Conv2d stride 2"]
  CNN --> FL["flatten"]
  TOK --> EMB["Embedding → mean → Linear+ReLU → 128d"]
  LAB --> OH["one-hot → Linear+ReLU → 32d"]
  FL --> C1["concat"]
  EMB --> C1
  OH --> C1
  C1 --> MU["mu"]
  C1 --> LV["logvar"]
  MU --> Z["z = reparam(mu, logvar)"]
  LV --> Z
  Z --> C2["concat z, h_text, h_label"]
  EMB --> C2
  OH --> C2
  C2 --> DEC["Linear → reshape → 3× ConvTranspose2d → interpolate if needed"]
  DEC --> XH["x̂"]
```

---

## Training

```bash
cd cpsc440-project
python main.py vae-train
```

Requires `tokens_*` in the `.npz` and [`vocab.json`](../data/mustard_processed/vocab.json) (from [`export_text_tokens.py`](../export_text_tokens.py)).

### Constants (`main.py`)

| Name | Default | Effect |
|------|---------|--------|
| `VAE_BATCH_SIZE` | 16 | DataLoader batch size |
| `VAE_EPOCHS` | 5 | Training epochs |
| `VAE_LR` | 1e-3 | Adam learning rate |
| `VAE_TARGET_BETA` | 0.1 | Target weight on KL after warmup |
| `VAE_LATENT_DIM` | 64 | Passed to `VAE(...)`; stored in checkpoint |
| `VAE_KL_WARMUP_STEPS` | `None` | If `None`, uses `max(1, 2 * len(train_loader))` |
| `EXPORT_LATENTS_AFTER_TRAIN` | `True` | Calls `export_cvae_latents` after the last epoch |
| `LATENT_EXPORT_SEED` | 440 | Torch generator seed for `z_*` in the export |

Each epoch prints train/val reconstruction and KL means. Checkpoints include `train_hparams` with the same values used for that run.

### Outputs

| Path | Contents |
|------|----------|
| `checkpoints/cvae_last.pt` | `model`, `spec_shape`, `text_vocab_size`, `latent_dim`, `train_hparams` |
| `checkpoints/cvae_latents.npz` | Written when `EXPORT_LATENTS_AFTER_TRAIN` or when you run `vae-export-latents` |

`checkpoints/` is gitignored (see [`.gitignore`](../.gitignore)).

### Other `main.py` commands

| Command | Behavior |
|---------|----------|
| `python main.py vae-test` | Synthetic mel + random tokens; short Adam loop; matplotlib recon plot |
| `python main.py vae-export-latents` | Loads `cvae_last.pt`, runs encoder on train/val/test in order, writes/overwrites `cvae_latents.npz` |

---

## Latent export (`cvae_latents.npz`)

Load with `numpy.load(..., allow_pickle=True)` (string `ids_*`).

| Key | Description |
|-----|-------------|
| `mu_train`, `mu_val`, `mu_test` | Encoder output `mu`, shape `(N, latent_dim)` |
| `z_train`, `z_val`, `z_test` | One reparameterized sample per row (RNG seeded by `LATENT_EXPORT_SEED`, batch order as in `MustardMelDataset`) |
| `labels_*` | Same as `labels_*` in the `.npz` |
| `ids_*` | Same MUStARD ids as in the `.npz`, row-aligned |
| `meta_json` | JSON string: `latent_dim`, `seed`, `npz_path`, short note |

Row order matches sequential indexing of [`MustardMelDataset`](../dataset.py) for each split (same as `.npz` row order).

---

## Evaluation and splits

- Train/val/test partitions follow the same deterministic split as preprocessing ([`DATA.md`](DATA.md), [`mustard_split.py`](../mustard_split.py)).
- For reconstruction on **real** clips: load `specs_test` / `tokens_test` / `labels_test`, run `model(x, tokens, labels)`, compare `x` and `x̂` (e.g. MSE or plots). `vae-test` only exercises **synthetic** data.
- Training objective is **reconstruction + KL**, not classification accuracy on sarcasm.

---

## Generation: text + label to mel / wav (`cvae_generate.py`)

This path **does not** use a reference mel. It:

1. Encodes `--text` with `tokenizer.json`, pads/truncates to `max_seq_len` from `vocab.json` `meta`.
2. Samples `z ~ Normal(0, z_scale² I)` (default `z_scale=1`, `--seed` for reproducibility).
3. Runs `model.decode(z, tokens, label)`.
4. Clamps mel to `[0, 1]`.
5. If `--out-mel`: saves `.npy` `(1, 80, 130)`. If `--out-wav`: denormalizes with `mustard_logmel.norm_stats.json` and Griffin–Lim via [`vocode.py`](../vocode.py) helpers.

```bash
python cvae_generate.py --text "..." --label 0 --out-mel gen.npy
python cvae_generate.py --text "..." --label 1 --out-wav gen.wav --seed 123
```

Optional: `--checkpoint`, `--tokenizer`, `--vocab`, `--npz`, `--z-scale`. The decoder was trained with `z` from the **encoder**; sampling `z` from a spherical Gaussian is an approximate inference-time use of the same `decode` (as in standard VAE sampling).

A **repository-wide file map** (including preprocessing and docs) is in the [`README`](../README.md). Tensor shapes and layer definitions are authoritative in [`vae.py`](../vae.py); step counts and checkpoints are authoritative in [`main.py`](../main.py).
