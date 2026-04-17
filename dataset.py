"""Load preprocessed MUStARD tensors for training."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


SplitName = Literal["train", "val", "test"]


class MustardMelDataset(Dataset):
    """Rows aligned: specs[i], labels[i], tokens[i], ids[i]."""

    def __init__(
        self,
        npz_path: Path | str,
        split: SplitName,
    ) -> None:
        self._path = Path(npz_path)
        self._split = split
        self._z = np.load(self._path, allow_pickle=True)
        self.specs = self._z[f"specs_{split}"]
        self.labels = self._z[f"labels_{split}"]
        self.tokens = self._z[f"tokens_{split}"]
        self.ids = self._z[f"ids_{split}"]

        if self.specs.shape[0] != self.labels.shape[0] or self.specs.shape[0] != self.tokens.shape[0]:
            raise ValueError(
                f"length mismatch: specs={self.specs.shape[0]} labels={self.labels.shape[0]} tokens={self.tokens.shape[0]}"
            )

    def __len__(self) -> int:
        return int(self.specs.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        x = torch.from_numpy(self.specs[idx].astype(np.float32, copy=False))
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        t = torch.from_numpy(self.tokens[idx].astype(np.int64, copy=False))
        clip_id = str(self.ids[idx])
        return x, t, y, clip_id

    @staticmethod
    def spec_shape_from_npz(npz_path: Path | str, split: SplitName = "train") -> Tuple[int, int, int]:
        z = np.load(npz_path, allow_pickle=True)
        s = z[f"specs_{split}"]
        if s.ndim != 4 or s.shape[1] != 1:
            raise ValueError(f"expected specs (N,1,H,W), got {s.shape}")
        return int(s.shape[1]), int(s.shape[2]), int(s.shape[3])

    @staticmethod
    def text_vocab_size_from_json(vocab_path: Path | str) -> int:
        import json

        with open(vocab_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return int(d["meta"]["vocab_size"])
