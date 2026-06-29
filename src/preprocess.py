from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
import re

import numpy as np
import pandas as pd

from .utils import SplitConfig, ensure_dir, stratified_split, write_json


TEST_NAME_PATTERN = re.compile(r"-test(?:\b|-|\s|\(|$)", re.IGNORECASE)


def compute_feature_standardization(sequence_paths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sums = None
    sq_sums = None
    count = 0
    for p in sequence_paths:
        seq = np.load(p)
        flat = seq.reshape(-1, seq.shape[-1]).astype(np.float64)
        if sums is None:
            sums = flat.sum(axis=0)
            sq_sums = (flat**2).sum(axis=0)
        else:
            sums += flat.sum(axis=0)
            sq_sums += (flat**2).sum(axis=0)
        count += flat.shape[0]
    assert sums is not None and sq_sums is not None
    mean = sums / max(count, 1)
    var = (sq_sums / max(count, 1)) - mean**2
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def small_sample_split(labels: pd.Series, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx_train: list[int] = []
    idx_val: list[int] = []
    idx_test: list[int] = []

    grouped: dict[str, list[int]] = {}
    for idx, label in enumerate(labels.astype(str).tolist()):
        grouped.setdefault(label, []).append(idx)

    for label in sorted(grouped):
        indices = grouped[label]
        shuffled = list(rng.permutation(indices))
        n = len(shuffled)
        if n == 1:
            idx_train.extend(shuffled)
        elif n == 2:
            idx_train.append(shuffled[0])
            idx_test.append(shuffled[1])
        elif n == 3:
            idx_train.append(shuffled[0])
            idx_val.append(shuffled[1])
            idx_test.append(shuffled[2])
        else:
            n_train = max(1, int(round(n * 0.7)))
            n_val = max(1, int(round(n * 0.15)))
            if n_train + n_val >= n:
                n_val = 1
            n_test = n - n_train - n_val
            if n_test <= 0:
                n_test = 1
                if n_train > 1:
                    n_train -= 1
                else:
                    n_val -= 1

            idx_train.extend(shuffled[:n_train])
            idx_val.extend(shuffled[n_train : n_train + n_val])
            idx_test.extend(shuffled[n_train + n_val : n_train + n_val + n_test])

    return (
        np.array(sorted(idx_train), dtype=np.int64),
        np.array(sorted(idx_val), dtype=np.int64),
        np.array(sorted(idx_test), dtype=np.int64),
    )


def filename_test_mask(df: pd.DataFrame) -> pd.Series:
    if "path" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df["path"].astype(str).map(lambda p: bool(TEST_NAME_PATTERN.search(Path(p).stem)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out_dir", type=Path, default=Path("models"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--signer_independent", action="store_true")
    ap.add_argument("--signer_col", type=str, default="signer_id")
    ap.add_argument("--filename_test_split", action="store_true")
    args = ap.parse_args()

    labels_path = args.processed_dir / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing: {labels_path}")

    df = pd.read_csv(labels_path)

    test_mask = filename_test_mask(df) if args.filename_test_split else pd.Series([False] * len(df), index=df.index)

    if args.filename_test_split and test_mask.any():
        df["split"] = "train"
        df.loc[test_mask, "split"] = "test"
        train_val_df = df[~test_mask].copy()
        if train_val_df.empty:
            raise RuntimeError("Filename test split left no train/val samples.")
        try:
            idx_train_local, idx_val_local, _ = stratified_split(
                train_val_df["label"].astype(str).tolist(),
                split=SplitConfig(train=0.85, val=0.15, test=0.0),
                seed=args.seed,
            )
            val_indices = train_val_df.iloc[idx_val_local].index
        except ValueError:
            rng = np.random.default_rng(args.seed)
            val_indices = []
            for label in sorted(train_val_df["label"].astype(str).unique()):
                label_indices = train_val_df[train_val_df["label"].astype(str) == label].index.to_numpy()
                if len(label_indices) >= 2:
                    n_val = max(1, int(round(len(label_indices) * args.val)))
                    picked = rng.choice(label_indices, size=min(n_val, len(label_indices) - 1), replace=False)
                    val_indices.extend(picked.tolist())
            val_indices = pd.Index(val_indices)
        df.loc[val_indices, "split"] = "val"
    elif args.signer_independent and args.signer_col in df.columns:
        signer_ids = df[args.signer_col].astype(str).values
        unique = np.unique(signer_ids)
        rng = np.random.default_rng(args.seed)
        rng.shuffle(unique)
        n = len(unique)
        n_train = int(round(n * args.train))
        n_val = int(round(n * args.val))
        train_ids = set(unique[:n_train])
        val_ids = set(unique[n_train : n_train + n_val])
        test_ids = set(unique[n_train + n_val :])
        df["split"] = [
            "train" if sid in train_ids else "val" if sid in val_ids else "test" for sid in signer_ids
        ]
    else:
        try:
            idx_train, idx_val, idx_test = stratified_split(
                df["label"].astype(str).tolist(),
                split=SplitConfig(train=args.train, val=args.val, test=args.test),
                seed=args.seed,
            )
        except ValueError:
            idx_train, idx_val, idx_test = small_sample_split(df["label"], seed=args.seed)
        df["split"] = "train"
        df.loc[idx_val, "split"] = "val"
        df.loc[idx_test, "split"] = "test"

    ensure_dir(args.out_dir)

    train_paths = df[df["split"] == "train"]["sequence_path"].astype(str).values
    mean, std = compute_feature_standardization(train_paths)
    np.savez(args.out_dir / "feature_norm.npz", mean=mean, std=std)

    label_list = sorted(df["label"].astype(str).unique().tolist())
    label_to_idx = {lab: i for i, lab in enumerate(label_list)}
    write_json(args.out_dir / "labels.json", {"labels": label_list, "label_to_idx": label_to_idx})

    df.to_csv(labels_path, index=False)
    print(f"Wrote splits to {labels_path} and feature stats to {args.out_dir/'feature_norm.npz'}")


if __name__ == "__main__":
    main()
