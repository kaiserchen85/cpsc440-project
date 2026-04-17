"""Shared deterministic train/val/test ID splits for MUStARD preprocessing and token export."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence


def split_ids_random(
    ids: Sequence[str],
    seed: int,
    train_frac: float,
    val_frac: float,
) -> Dict[str, List[str]]:
    """Match preprocess.py: shuffle then split by fractions."""
    if train_frac <= 0 or val_frac < 0 or train_frac + val_frac >= 1:
        raise ValueError("invalid split fracs: require train>0, val>=0, train+val<1")
    ids = list(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(max(n_train, 1), n)
    n_val = min(max(n_val, 0), n - n_train)
    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def load_sarcasm_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
