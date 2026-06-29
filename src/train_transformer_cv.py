from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .train_transformer import (
    SequenceDataset,
    TransformerClassifier,
    compute_feature_standardization_for_df,
    eval_epoch,
)
from .utils import ensure_dir, set_seed, write_json


def make_loader(
    df: pd.DataFrame,
    label_to_idx: Dict[str, int],
    feature_norm_path: Path,
    batch_size: int,
    keypoint_preprocess: bool,
    augment: bool,
    temporal_crop: bool,
    balanced_sampler: bool,
    seed: int,
) -> DataLoader:
    sampler = None
    shuffle = augment
    if balanced_sampler:
        labels = df["label"].astype(str).tolist()
        counts = pd.Series(labels).value_counts().to_dict()
        weights = [1.0 / float(counts[label]) for label in labels]
        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)
        shuffle = False

    return DataLoader(
        SequenceDataset(
            df,
            label_to_idx=label_to_idx,
            feature_norm_path=feature_norm_path,
            augment=augment,
            noise_sigma=0.01 if augment else 0.0,
            scale_min=0.95 if augment else 1.0,
            scale_max=1.05 if augment else 1.0,
            temporal_crop=temporal_crop if augment else False,
            min_crop_ratio=0.8,
            keypoint_preprocess=keypoint_preprocess,
            seed=seed,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
    )


def train_one_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_to_idx: Dict[str, int],
    feature_dim: int,
    args: argparse.Namespace,
    fold_dir: Path,
    fold: int,
) -> Tuple[float, float, Path]:
    ensure_dir(fold_dir)
    norm_path = fold_dir / "feature_norm.npz"
    mean, std = compute_feature_standardization_for_df(train_df, keypoint_preprocess=args.keypoint_preprocess)
    np.savez(norm_path, mean=mean, std=std)

    train_loader = make_loader(
        train_df,
        label_to_idx,
        norm_path,
        batch_size=args.batch_size,
        keypoint_preprocess=args.keypoint_preprocess,
        augment=args.augment,
        temporal_crop=args.temporal_crop,
        balanced_sampler=args.balanced_sampler,
        seed=args.seed + fold,
    )
    val_loader = make_loader(
        val_df,
        label_to_idx,
        norm_path,
        batch_size=args.batch_size,
        keypoint_preprocess=args.keypoint_preprocess,
        augment=False,
        temporal_crop=False,
        balanced_sampler=False,
        seed=args.seed + fold,
    )

    device = torch.device(args.device)
    model = TransformerClassifier(
        feature_dim=feature_dim,
        num_classes=len(label_to_idx),
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pooling=args.pooling,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best_f1 = -1.0
    best_acc = 0.0
    best_path = fold_dir / "best_transformer.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))
        val_acc, val_f1 = eval_epoch(model, val_loader, device=device)
        print(f"[fold {fold}] epoch={epoch} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_dim": feature_dim,
                    "num_classes": len(label_to_idx),
                    "label_to_idx": label_to_idx,
                    "arch": "transformer",
                    "params": {
                        "d_model": args.d_model,
                        "num_layers": args.num_layers,
                        "nhead": args.nhead,
                        "dim_feedforward": args.dim_feedforward,
                        "dropout": args.dropout,
                        "pooling": args.pooling,
                        "keypoint_preprocess": args.keypoint_preprocess,
                    },
                },
                best_path,
            )
    return best_acc, best_f1, best_path


def train_final_all(df: pd.DataFrame, label_to_idx: Dict[str, int], feature_dim: int, args: argparse.Namespace) -> Path:
    final_dir = args.out_dir / "final_all"
    ensure_dir(final_dir)
    norm_path = final_dir / "feature_norm.npz"
    mean, std = compute_feature_standardization_for_df(df, keypoint_preprocess=args.keypoint_preprocess)
    np.savez(norm_path, mean=mean, std=std)
    loader = make_loader(
        df,
        label_to_idx,
        norm_path,
        batch_size=args.batch_size,
        keypoint_preprocess=args.keypoint_preprocess,
        augment=args.augment,
        temporal_crop=args.temporal_crop,
        balanced_sampler=args.balanced_sampler,
        seed=args.seed,
    )
    device = torch.device(args.device)
    model = TransformerClassifier(
        feature_dim=feature_dim,
        num_classes=len(label_to_idx),
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pooling=args.pooling,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, args.final_epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Final all-data Epoch {epoch}/{args.final_epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

    final_path = final_dir / "best_transformer.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_dim": feature_dim,
            "num_classes": len(label_to_idx),
            "label_to_idx": label_to_idx,
            "arch": "transformer",
            "params": {
                "d_model": args.d_model,
                "num_layers": args.num_layers,
                "nhead": args.nhead,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
                "pooling": args.pooling,
                "keypoint_preprocess": args.keypoint_preprocess,
            },
        },
        final_path,
    )
    return final_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out_dir", type=Path, default=Path("models_transformer_cv7_all"))
    ap.add_argument("--reports_dir", type=Path, default=Path("reports_transformer_cv7_all"))
    ap.add_argument("--folds", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--final_epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--dim_feedforward", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"])
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--temporal_crop", action="store_true")
    ap.add_argument("--keypoint_preprocess", action="store_true")
    ap.add_argument("--balanced_sampler", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)
    ensure_dir(args.reports_dir)

    df = pd.read_csv(args.processed_dir / "labels.csv")
    y = df["label"].astype(str).values
    labels = sorted(np.unique(y).tolist())
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    label_info = {"labels": labels, "label_to_idx": label_to_idx}
    write_json(args.out_dir / "labels.json", label_info)

    example_seq = np.load(df.iloc[0]["sequence_path"]).astype(np.float32)
    feature_dim = int(example_seq.shape[1])
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows: List[dict] = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(df, y), start=1):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        acc, macro_f1, best_path = train_one_fold(
            train_df=train_df,
            val_df=val_df,
            label_to_idx=label_to_idx,
            feature_dim=feature_dim,
            args=args,
            fold_dir=args.out_dir / f"fold_{fold}",
            fold=fold,
        )
        rows.append({"fold": fold, "accuracy": acc, "macro_f1": macro_f1, "model_path": str(best_path).replace("\\", "/")})
        print(f"[fold {fold}] best_acc={acc:.4f} best_macro_f1={macro_f1:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(args.reports_dir / "cv_results.csv", index=False)
    summary = {
        "folds": args.folds,
        "accuracy_mean": float(results["accuracy"].mean()),
        "accuracy_std": float(results["accuracy"].std(ddof=1)),
        "macro_f1_mean": float(results["macro_f1"].mean()),
        "macro_f1_std": float(results["macro_f1"].std(ddof=1)),
    }
    final_path = train_final_all(df, label_to_idx, feature_dim, args)
    summary["final_model_path"] = str(final_path).replace("\\", "/")
    write_json(args.reports_dir / "cv_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
