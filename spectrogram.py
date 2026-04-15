#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize labeled mel-spectrograms from mustard_logmel.npz.")
    p.add_argument("--npz", type=Path, default=Path("data/mustard_processed/mustard_logmel.npz"))
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--index", type=int, default=0, help="Index within the chosen split.")
    p.add_argument("--save", type=Path, default=None, help="If set, save the figure instead of showing it.")
    args = p.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    x = d[f"specs_{args.split}"]  # (N, 1, n_mels, n_frames)
    y = d[f"labels_{args.split}"]
    ids = d[f"ids_{args.split}"]

    if x.shape[0] == 0:
        raise SystemExit(f"No examples in split={args.split}")

    i = args.index % x.shape[0]
    spec = x[i, 0]

    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect="auto", origin="lower")
    plt.title(f"id={ids[i]} label={int(y[i])} (1=sarcastic)")
    plt.xlabel("frame")
    plt.ylabel("mel bin")
    plt.colorbar()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=160, bbox_inches="tight")
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

