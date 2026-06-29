from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .utils import ensure_dir, read_json, set_seed


class SequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_to_idx: Dict[str, int],
        feature_norm_path: Path | None,
        augment: bool,
        noise_sigma: float,
        scale_min: float,
        scale_max: float,
        temporal_crop: bool,
        min_crop_ratio: float,
        seed: int,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.augment = augment
        self.noise_sigma = noise_sigma
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.temporal_crop = temporal_crop
        self.min_crop_ratio = min_crop_ratio
        self.rng = np.random.default_rng(seed)

        self.mean = None
        self.std = None
        if feature_norm_path is not None and feature_norm_path.exists():
            stats = np.load(feature_norm_path)
            self.mean = stats["mean"].astype(np.float32)
            self.std = stats["std"].astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, seq: np.ndarray) -> np.ndarray:
        if not self.augment:
            return seq
        if self.temporal_crop:
            t = seq.shape[0]
            min_len = max(1, int(round(t * self.min_crop_ratio)))
            crop_len = int(self.rng.integers(min_len, t + 1))
            start = int(self.rng.integers(0, t - crop_len + 1))
            cropped = seq[start : start + crop_len]
            if crop_len < t:
                out = np.zeros_like(seq)
                out[:crop_len] = cropped
                seq = out
            else:
                seq = cropped
        if self.scale_min != 1.0 or self.scale_max != 1.0:
            scale = float(self.rng.uniform(self.scale_min, self.scale_max))
            seq = seq * scale
        if self.noise_sigma > 0:
            seq = seq + self.rng.normal(0.0, self.noise_sigma, size=seq.shape).astype(np.float32)
        return seq

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        seq = np.load(row["sequence_path"]).astype(np.float32)
        seq = self._augment(seq)
        if self.mean is not None and self.std is not None:
            seq = (seq - self.mean[None, :]) / self.std[None, :]
        y = self.label_to_idx[str(row["label"])]
        return torch.from_numpy(seq), torch.tensor(y, dtype=torch.long)


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        proj_dim: int = 256,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(feature_dim, proj_dim)
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    ys: List[int] = []
    preds: List[int] = []
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        p = logits.argmax(dim=-1)
        ys.extend(y.cpu().tolist())
        preds.extend(p.cpu().tolist())
        total += y.numel()
        correct += int((p == y).sum().item())
    acc = correct / max(total, 1)
    f1 = f1_score(ys, preds, average="macro") if len(set(ys)) > 1 else 0.0
    return float(acc), float(f1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out_dir", type=Path, default=Path("models"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--proj_dim", type=int, default=256)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--noise_sigma", type=float, default=0.01)
    ap.add_argument("--scale_min", type=float, default=0.95)
    ap.add_argument("--scale_max", type=float, default=1.05)
    ap.add_argument("--temporal_crop", action="store_true")
    ap.add_argument("--min_crop_ratio", type=float, default=0.8)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    df = pd.read_csv(args.processed_dir / "labels.csv")
    if "split" not in df.columns:
        raise RuntimeError("Missing split column. Run: python -m src.preprocess")

    labels_json = args.out_dir / "labels.json"
    if not labels_json.exists():
        raise RuntimeError("Missing labels.json. Run: python -m src.preprocess --out_dir models")
    label_info = read_json(labels_json)
    label_to_idx: Dict[str, int] = {k: int(v) for k, v in label_info["label_to_idx"].items()}
    num_classes = len(label_to_idx)

    example_seq = np.load(df.iloc[0]["sequence_path"]).astype(np.float32)
    _, feature_dim = example_seq.shape

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if len(val_df) == 0:
        val_df = test_df

    feature_norm_path = args.out_dir / "feature_norm.npz"

    train_loader = DataLoader(
        SequenceDataset(
            train_df,
            label_to_idx=label_to_idx,
            feature_norm_path=feature_norm_path,
            augment=args.augment,
            noise_sigma=args.noise_sigma,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            temporal_crop=args.temporal_crop,
            min_crop_ratio=args.min_crop_ratio,
            seed=args.seed,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        SequenceDataset(
            val_df,
            label_to_idx=label_to_idx,
            feature_norm_path=feature_norm_path,
            augment=False,
            noise_sigma=0.0,
            scale_min=1.0,
            scale_max=1.0,
            temporal_crop=False,
            min_crop_ratio=args.min_crop_ratio,
            seed=args.seed,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        SequenceDataset(
            test_df,
            label_to_idx=label_to_idx,
            feature_norm_path=feature_norm_path,
            augment=False,
            noise_sigma=0.0,
            scale_min=1.0,
            scale_max=1.0,
            temporal_crop=False,
            min_crop_ratio=args.min_crop_ratio,
            seed=args.seed,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(args.device)
    model = LSTMClassifier(
        feature_dim=feature_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        proj_dim=args.proj_dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_path = args.out_dir / "best_lstm.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        val_acc, val_f1 = eval_epoch(model, val_loader, device=device)
        print(f"Val acc={val_acc:.4f} macro_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_dim": feature_dim,
                    "num_classes": num_classes,
                    "label_to_idx": label_to_idx,
                    "arch": "lstm",
                    "params": {
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "proj_dim": args.proj_dim,
                    },
                },
                best_path,
            )
            print(f"Saved best model to {best_path} (macro_f1={best_f1:.4f})")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_acc, test_f1 = eval_epoch(model, test_loader, device=device)
    print(f"Test acc={test_acc:.4f} macro_f1={test_f1:.4f}")


if __name__ == "__main__":
    main()
