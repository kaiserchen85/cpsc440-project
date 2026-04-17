## Tuning guide (CVAE training + generation)

This guide is intentionally **instructional**: it summarizes the current baseline training behavior and gives concrete, reversible tuning steps to improve the model—especially **generation quality for data augmentation**.

Prereqs:

- Read [`DATA.md`](DATA.md) for how `mustard_logmel.npz` + `tokens_*` are produced.
- Read [`MODEL.md`](MODEL.md) for what the model/loss are and what artifacts are saved.

### Contents

- [Current baseline (what we have now)](#current-baseline-what-we-have-now)
  - [How to reproduce the baseline](#how-to-reproduce-the-baseline)
  - [What “good” looks like in logs](#what-good-looks-like-in-logs)
- [What we’re optimizing](#what-were-optimizing)
- [Quick checklist before tuning](#quick-checklist-before-tuning)
- [Tuning for better generation (priority)](#tuning-for-better-generation-priority)
  - [1) Train longer (most reliable)](#1-train-longer-most-reliable)
  - [2) Beta and KL warmup (controls “sampleability”)](#2-beta-and-kl-warmup-controls-sampleability)
  - [3) Latent dimensionality](#3-latent-dimensionality)
  - [4) Learning rate and batch size](#4-learning-rate-and-batch-size)
  - [5) Sampling controls for augmentation](#5-sampling-controls-for-augmentation)
- [Tuning for better reconstruction (secondary)](#tuning-for-better-reconstruction-secondary)
- [How to evaluate changes (minimal + practical)](#how-to-evaluate-changes-minimal--practical)
  - [A) Reconstruction sanity checks](#a-reconstruction-sanity-checks)
  - [B) Generation sanity checks](#b-generation-sanity-checks)
  - [C) Latent-space sanity checks](#c-latent-space-sanity-checks)
- [Rollback / revert strategy](#rollback--revert-strategy)

---

## Current baseline (what we have now)

The current training loop is `python main.py vae-train` (see [`main.py`](../main.py)). It trains for **5 epochs** with:

- `VAE_BATCH_SIZE = 16`
- `VAE_LR = 1e-3`
- `VAE_TARGET_BETA = 0.1`
- `VAE_LATENT_DIM = 64`
- `VAE_SEED = 440` (best-effort reproducibility)
- KL warmup: `VAE_KL_WARMUP_STEPS = None` → computed as `2 * len(train_loader)`

An example run (your numbers will match closely when the seed + environment match) looked like:

```text
epoch 1: train recon 0.2336 kl 0.2843 | val recon 0.1548 kl 0.1509
epoch 2: train recon 0.1563 kl 0.1123 | val recon 0.1567 kl 0.0924
epoch 3: train recon 0.1495 kl 0.0952 | val recon 0.1427 kl 0.0930
epoch 4: train recon 0.1460 kl 0.1080 | val recon 0.1472 kl 0.1224
epoch 5: train recon 0.1437 kl 0.1106 | val recon 0.1326 kl 0.1316
```

Interpretation:

- **Recon decreases and stabilizes** → training is healthy.
- **KL stays nonzero** → the latent is being used (not obviously collapsed).
- **Val recon ~ train recon** → no obvious overfitting in 5 epochs.

### How to reproduce the baseline

From `cpsc440-project/`:

```bash
# Ensure tokens exist (required for training)
python export_text_tokens.py

# Train (overwrites checkpoints/cvae_last.pt and usually checkpoints/cvae_latents.npz)
python main.py vae-train
```

### What “good” looks like in logs

During tuning, it’s normal for recon/KL to trade off. For generation, you typically want:

- KL **not** ≈ 0 for all epochs (collapse risk).
- Recon decreasing, even if it’s slightly worse than a pure autoencoder.
- Stable val curves (no explosion).

---

## What we’re optimizing

We care about both:

- **A) Reconstruction quality**: given `(x, tokens, label)`, the model reconstructs `x̂` close to `x`.
- **B) Generation quality for augmentation (priority)**: given **only** `(raw text, desired label)`, we sample `z ~ Normal(0, I)` (or scaled) and decode a **plausible** mel, then optionally vocode to `.wav`.

For B, the key is: **training must produce latents that are “sampleable.”** That is exactly what the KL term and warmup influence.

---

## Quick checklist before tuning

1. **Confirm your data is tokenized**: `mustard_logmel.npz` should contain `tokens_train/val/test`. See [`DATA.md`](DATA.md).
2. **Don’t lose previous results**: before a new run, copy/rename your checkpoints:

```bash
mkdir -p checkpoints/runs
cp checkpoints/cvae_last.pt checkpoints/runs/cvae_last_baseline.pt
cp checkpoints/cvae_latents.npz checkpoints/runs/cvae_latents_baseline.npz
```

3. **Keep generation tests fixed**: pick 3–5 fixed prompts and a fixed `--seed` for `cvae_generate.py` so you can compare runs.

---

## Tuning for better generation (priority)

All knobs below are constants near the top of [`main.py`](../main.py). The edits are small and reversible.

### 1) Train longer (most reliable)

**What to change**

- Increase `VAE_EPOCHS` from `5` → `10`, then `20`, then `30` if still improving.

**Why**

- With small datasets, 5 epochs often isn’t enough to learn a decent decoder.

**How**

- Edit `VAE_EPOCHS` in [`main.py`](../main.py), then re-run:

```bash
python main.py vae-train
```

**Rollback**

- Set `VAE_EPOCHS` back to `5` and restore the baseline checkpoint from `checkpoints/runs/`.

### 2) Beta and KL warmup (controls “sampleability”)

This is the biggest lever for B.

**What to try**

1. **Slightly higher beta**: `VAE_TARGET_BETA = 0.2`  
   - pushes posteriors closer to `Normal(0, I)` → can improve sampling from `z~N(0,I)`, but may worsen recon.
2. **Longer warmup**: set `VAE_KL_WARMUP_STEPS` explicitly, e.g. `5 * len(train_loader)`  
   - lets decoder learn reconstruction first, then gradually enforces sampleability.

**How**

- In `main.py`, set:
  - `VAE_TARGET_BETA = 0.2`
  - `VAE_KL_WARMUP_STEPS = 5 * 34` (or just `170`) if you want a concrete number

Then retrain.

**What to watch**

- If KL becomes extremely large early and recon degrades badly → warmup is too short or LR too high.
- If KL becomes ~0 and stays there → beta too low or decoder overpowering encoder (collapse).

**Rollback**

- Restore `VAE_TARGET_BETA = 0.1`, `VAE_KL_WARMUP_STEPS = None`, and restore checkpoint.

### 3) Latent dimensionality

**What to try**

- `VAE_LATENT_DIM = 32` (more bottleneck → sometimes better behaved sampling)
- or `VAE_LATENT_DIM = 128` (more capacity → sometimes better realism, but can overfit)

**Why**

- For B, too-small latents can underfit; too-large can create a complicated posterior that’s harder to sample from.

**How**

- Change `VAE_LATENT_DIM` and retrain.

**Rollback**

- Keep separate checkpoints per latent dim (file names), because you must match latent dim at generation time.

### 4) Learning rate and batch size

**What to try**

- If training feels noisy/unstable: `VAE_LR = 3e-4`
- If training is stable but slow: try `VAE_LR = 2e-3` (cautiously)
- If you have GPU memory: increase `VAE_BATCH_SIZE` to `32` (then re-check recon/KL)

**Why**

- For B, we want a stable, smooth posterior. Lower LR often helps.

**Rollback**

- Revert constants and retrain; restore checkpoints if needed.

### 5) Sampling controls for augmentation

Generation is performed by [`cvae_generate.py`](../cvae_generate.py):

```bash
python cvae_generate.py --text "..." --label 1 --out-wav out.wav --seed 123
```

**Key knob:** `--z-scale`

- Try `--z-scale 0.7` if outputs are very noisy.
- Try `--z-scale 1.2` if outputs are too flat / repetitive.

This is a *post-training* knob and is easy to tune without retraining.

---

## Tuning for better reconstruction (secondary)

If recon is the focus, you generally do the opposite:

- Lower `VAE_TARGET_BETA` (e.g. `0.05`) to prioritize reconstruction.
- Increase `VAE_LATENT_DIM`.
- Train longer.

But note: these changes can make sampling from `z~Normal(0,I)` worse, which hurts augmentation.

---

## How to evaluate changes (minimal + practical)

### A) Reconstruction sanity checks

On a few fixed examples:

- compare mel plots (input vs recon)
- compare `recon` value trend across epochs

### B) Generation sanity checks

Pick a fixed prompt list and evaluate per run:

```bash
python cvae_generate.py --text "Well that's just great." --label 1 --out-wav out/a.wav --seed 123
python cvae_generate.py --text "Thanks for your help." --label 0 --out-wav out/b.wav --seed 123
```

Listen for:

- gross artifacts / noise (reduce `--z-scale` or tune beta/warmup)
- mode collapse (same-sounding outputs for different prompts; try higher beta or longer warmup)

### C) Latent-space sanity checks

After training:

```bash
python main.py vae-export-latents
```

Use `mu_*` for stable t-SNE/UMAP (see [`MODEL.md`](MODEL.md#latent-export-cvae_latentsnpz)).

---

## Rollback / revert strategy

Because training overwrites `checkpoints/cvae_last.pt` and `checkpoints/cvae_latents.npz`, treat each run as a “versioned artifact”:

1. **Before** retraining, copy baseline to `checkpoints/runs/`.
2. Use descriptive names:
   - `cvae_last_beta0p2_warmup170_lat32.pt`
   - `cvae_latents_beta0p2_warmup170_lat32.npz`
3. If a tuning attempt is worse, restore by copying the previous “best” back to the default names:

```bash
cp checkpoints/runs/cvae_last_baseline.pt checkpoints/cvae_last.pt
cp checkpoints/runs/cvae_latents_baseline.npz checkpoints/cvae_latents.npz
```

If you are using git commits for code changes, the safest pattern is:

- commit hyperparameter changes separately
- keep artifacts (`cvae_last.pt`, `cvae_latents.npz`) in a commit only when you want teammates to use that exact model
