#!/usr/bin/env python3
"""
Train a small BPE tokenizer on MUStARD train utterances and merge token matrices into mustard_logmel.npz.

Does not re-run audio extraction. Requires existing specs/ids in the .npz (same row order as ids_*).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from mustard_split import load_sarcasm_json, split_ids_random


def _train_bpe(
    train_texts: List[str],
    vocab_size: int,
    pad_token: str,
    unk_token: str,
) -> Any:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = Whitespace()
    # Reserve room for special tokens inside vocab_size
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=[pad_token, unk_token])
    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id(pad_token), pad_token=pad_token)
    return tokenizer


def _encode_padded(
    tokenizer: Any,
    texts: List[str],
    max_len: int,
    pad_id: int,
) -> np.ndarray:
    rows: List[List[int]] = []
    for t in texts:
        enc = tokenizer.encode(t)
        ids = enc.ids[:max_len]
        if len(ids) < max_len:
            ids = ids + [pad_id] * (max_len - len(ids))
        rows.append(ids)
    return np.asarray(rows, dtype=np.int32)


def main() -> None:
    p = argparse.ArgumentParser(description="Export BPE tokens aligned to mustard_logmel.npz rows.")
    p.add_argument("--json-path", type=Path, default=Path("data/mustard_raw/sarcasm_data.json"))
    p.add_argument("--npz-path", type=Path, default=Path("data/mustard_processed/mustard_logmel.npz"))
    p.add_argument("--vocab-out", type=Path, default=Path("data/mustard_processed/vocab.json"))
    p.add_argument("--tokenizer-out", type=Path, default=Path("data/mustard_processed/tokenizer.json"))
    p.add_argument("--seed", type=int, default=440)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--vocab-size", type=int, default=4096, help="Must be <= VAE text_vocab_size (10000).")
    p.add_argument("--max-seq-len", type=int, default=64, help="Pad/truncate encoded token length.")
    p.add_argument("--pad-token", type=str, default="<pad>")
    p.add_argument("--unk-token", type=str, default="<unk>")
    args = p.parse_args()

    if args.vocab_size > 10000:
        raise SystemExit("--vocab-size must be <= 10000 to match nn.Embedding(10000) in vae.py")

    data = load_sarcasm_json(args.json_path)
    all_ids = sorted(data.keys())
    splits = split_ids_random(all_ids, seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac)

    z = np.load(args.npz_path, allow_pickle=True)

    train_texts: List[str] = []
    for clip_id in splits["train"]:
        if clip_id not in data:
            continue
        u = str(data[clip_id].get("utterance", "")).strip()
        if u:
            train_texts.append(u)

    tokenizer = _train_bpe(train_texts, args.vocab_size, args.pad_token, args.unk_token)
    tokenizer.save(str(args.tokenizer_out))

    vocab = tokenizer.get_vocab()
    if len(vocab) > 10000:
        raise SystemExit(f"trained vocab has {len(vocab)} tokens; must be <= 10000")

    pad_id = tokenizer.token_to_id(args.pad_token)
    unk_id = tokenizer.token_to_id(args.unk_token)
    if pad_id is None or unk_id is None:
        raise SystemExit("PAD/UNK must exist in tokenizer special tokens")

    tokenizer.enable_padding(pad_id=int(pad_id), pad_token=args.pad_token)

    meta_vocab = {
        "pad_token": args.pad_token,
        "unk_token": args.unk_token,
        "pad_id": int(pad_id),
        "unk_id": int(unk_id),
        "vocab_size": len(vocab),
        "max_seq_len": int(args.max_seq_len),
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "json_path": str(args.json_path),
        "npz_path": str(args.npz_path),
        "tokenizer_path": str(args.tokenizer_out),
    }

    with args.vocab_out.open("w", encoding="utf-8") as f:
        json.dump({"token2id": vocab, "meta": meta_vocab}, f, indent=2)

    out_arrays: Dict[str, Any] = dict(z)
    for split in ("train", "val", "test"):
        ids_arr = z[f"ids_{split}"]
        texts: List[str] = []
        for clip_id in ids_arr.tolist():
            cid = str(clip_id)
            if cid not in data:
                raise KeyError(f"id {cid} missing from sarcasm_data.json")
            texts.append(str(data[cid].get("utterance", "")))
        tok_mat = _encode_padded(tokenizer, texts, args.max_seq_len, int(pad_id))
        if tok_mat.shape[0] != z[f"specs_{split}"].shape[0]:
            raise ValueError(
                f"token rows {tok_mat.shape[0]} != specs rows {z[f'specs_{split}'].shape[0]} for split={split}"
            )
        out_arrays[f"tokens_{split}"] = tok_mat

    meta_obj = json.loads(str(out_arrays["meta"][0]))
    meta_obj["text_max_len"] = int(args.max_seq_len)
    meta_obj["vocab_json"] = str(args.vocab_out)
    meta_obj["tokenizer_json"] = str(args.tokenizer_out)
    meta_obj["text_vocab_size"] = len(vocab)
    out_arrays["meta"] = np.array([json.dumps(meta_obj)], dtype=object)

    np.savez_compressed(args.npz_path, **out_arrays)
    print(f"Updated {args.npz_path} with tokens_* arrays")
    print(f"Wrote {args.vocab_out}")
    print(f"Wrote {args.tokenizer_out}")


if __name__ == "__main__":
    main()
