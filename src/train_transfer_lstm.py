from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .train_lstm import LSTMClassifier, SequenceDataset, eval_epoch
from .preprocess import compute_feature_standardization
from .utils import ensure_dir, read_json, set_seed, write_json


def _infer_repo_root(processed_dir: Path) -> Path:
    resolved = processed_dir.resolve()
    for parent in [resolved] + list(resolved.parents):
        if (parent / "src").exists() and (parent / "requirements.txt").exists():
            return parent
    return Path.cwd()


def _absolutize_sequence_paths(df: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    out = df.copy()
    paths = []
    for raw in out["sequence_path"].astype(str):
        p = Path(raw)
        if not p.is_absolute():
            p = repo_root / p
        paths.append(str(p.resolve()))
    out["sequence_path"] = paths
    return out


def _resolve_feature_dim(labels_df: pd.DataFrame) -> int:
    example_seq = np.load(labels_df.iloc[0]["sequence_path"]).astype(np.float32)
    _, feature_dim = example_seq.shape
    return int(feature_dim)


def _load_target_labels(out_dir: Path) -> Dict[str, int]:
    labels_json = out_dir / "labels.json"
    if not labels_json.exists():
        raise RuntimeError(f"Missing labels.json at {labels_json}. Run preprocess on the NZSL target set first.")
    label_info = read_json(labels_json)
    return {k: int(v) for k, v in label_info["label_to_idx"].items()}


def _resolve_mapping_columns(mapping_df: pd.DataFrame) -> Tuple[str, str]:
    if "nzsl_label" in mapping_df.columns and "aux_label" in mapping_df.columns:
        return "aux_label", "nzsl_label"
    if "nzsl_label" in mapping_df.columns and "auslan_gloss" in mapping_df.columns:
        return "auslan_gloss", "nzsl_label"
    if "label" in mapping_df.columns:
        return "label", "label"
    raise RuntimeError(
        "Mapping CSV must contain either ['aux_label', 'nzsl_label'] or a single ['label'] column."
    )


def load_auxiliary_dataframe(
    aux_processed_dir: Path,
    mapping_csv: Path,
    target_label_to_idx: Dict[str, int],
) -> pd.DataFrame:
    aux_labels_path = aux_processed_dir / "labels.csv"
    if not aux_labels_path.exists():
        raise FileNotFoundError(f"Missing auxiliary labels.csv: {aux_labels_path}")

    aux_df = pd.read_csv(aux_labels_path)
    if "label" not in aux_df.columns:
        raise RuntimeError("Auxiliary labels.csv must contain a 'label' column.")

    mapping_df = pd.read_csv(mapping_csv)
    aux_col, nzsl_col = _resolve_mapping_columns(mapping_df)

    if "recommended_use" in mapping_df.columns:
        allowed = {"pretrain", "template_or_pretrain", "template_or_rule_reference", "template_reference"}
        mapping_df = mapping_df[mapping_df["recommended_use"].astype(str).isin(allowed)]
    if "auslan_auxiliary_status" in mapping_df.columns:
        mapping_df = mapping_df[mapping_df["auslan_auxiliary_status"].astype(str) == "candidate"]
    if "bsl_auxiliary_status" in mapping_df.columns:
        mapping_df = mapping_df[mapping_df["bsl_auxiliary_status"].astype(str) == "candidate"]

    mapping_df = mapping_df.dropna(subset=[aux_col, nzsl_col]).copy()
    label_map = {
        str(row[aux_col]).strip(): str(row[nzsl_col]).strip()
        for _, row in mapping_df.iterrows()
        if str(row[nzsl_col]).strip() in target_label_to_idx
    }
    if not label_map:
        raise RuntimeError(f"No usable auxiliary-to-target label mappings found in {mapping_csv}")

    source_label_col = "aux_label" if "aux_label" in aux_df.columns else "label"
    aux_df = aux_df[aux_df[source_label_col].astype(str).isin(label_map)].copy()
    if aux_df.empty:
        raise RuntimeError("Auxiliary labels.csv has no rows matching the mapping file.")

    aux_df["aux_label"] = aux_df[source_label_col].astype(str)
    aux_df["label"] = aux_df["aux_label"].map(label_map)
    if "split" not in aux_df.columns:
        aux_df["split"] = "train"

    train_like = aux_df["split"].astype(str).isin({"train", "val"})
    aux_df.loc[train_like, "split"] = "train"
    aux_df.loc[~train_like, "split"] = "holdout"
    return aux_df.reset_index(drop=True)


def _build_loader(
    df: pd.DataFrame,
    label_to_idx: Dict[str, int],
    feature_norm_path: Path | None,
    batch_size: int,
    augment: bool,
    noise_sigma: float,
    scale_min: float,
    scale_max: float,
    temporal_crop: bool,
    min_crop_ratio: float,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        SequenceDataset(
            df,
            label_to_idx=label_to_idx,
            feature_norm_path=feature_norm_path,
            augment=augment,
            noise_sigma=noise_sigma,
            scale_min=scale_min,
            scale_max=scale_max,
            temporal_crop=temporal_crop,
            min_crop_ratio=min_crop_ratio,
            seed=seed,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def _train_phase(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    phase_name: str,
    epoch: int,
    total_epochs: int,
) -> None:
    model.train()
    pbar = tqdm(loader, desc=f"{phase_name} {epoch}/{total_epochs}")
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=float(loss.item()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_processed_dir", type=Path, required=True)
    ap.add_argument("--target_out_dir", type=Path, required=True)
    ap.add_argument("--aux_processed_dir", type=Path, required=True)
    ap.add_argument("--mapping_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrain_epochs", type=int, default=10)
    ap.add_argument("--finetune_epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--pretrain_lr", type=float, default=1e-3)
    ap.add_argument("--finetune_lr", type=float, default=5e-4)
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
    ap.add_argument("--freeze_proj", action="store_true")
    ap.add_argument("--freeze_lstm", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)
    target_labels_json = args.target_out_dir / "labels.json"
    target_feature_norm_path = args.target_out_dir / "feature_norm.npz"
    if target_labels_json.exists():
        write_json(args.out_dir / "labels.json", read_json(target_labels_json))
    if target_feature_norm_path.exists():
        target_stats = np.load(target_feature_norm_path)
        np.savez(args.out_dir / "feature_norm.npz", mean=target_stats["mean"], std=target_stats["std"])

    target_df = pd.read_csv(args.target_processed_dir / "labels.csv")
    if "split" not in target_df.columns:
        raise RuntimeError("Target labels.csv is missing a split column. Run preprocess on the NZSL target set first.")
    repo_root = _infer_repo_root(args.target_processed_dir)
    target_df = _absolutize_sequence_paths(target_df, repo_root=repo_root)

    label_to_idx = _load_target_labels(args.target_out_dir)
    aux_df = load_auxiliary_dataframe(
        aux_processed_dir=args.aux_processed_dir,
        mapping_csv=args.mapping_csv,
        target_label_to_idx=label_to_idx,
    )
    aux_df = _absolutize_sequence_paths(aux_df, repo_root=repo_root)

    feature_dim = _resolve_feature_dim(target_df)
    aux_feature_dim = _resolve_feature_dim(aux_df)
    num_classes = len(label_to_idx)
    feature_norm_path = args.target_out_dir / "feature_norm.npz"
    aux_feature_norm_path = args.out_dir / "aux_feature_norm.npz"
    aux_mean, aux_std = compute_feature_standardization(aux_df[aux_df["split"] == "train"]["sequence_path"].astype(str).values)
    np.savez(aux_feature_norm_path, mean=aux_mean, std=aux_std)

    aux_train_df = aux_df[aux_df["split"] == "train"].copy()
    if aux_train_df.empty:
        raise RuntimeError("No auxiliary training rows remained after applying mapping and split filters.")

    target_train_df = target_df[target_df["split"] == "train"].copy()
    target_val_df = target_df[target_df["split"] == "val"].copy()
    target_test_df = target_df[target_df["split"] == "test"].copy()
    if target_val_df.empty:
        target_val_df = target_test_df

    aux_train_loader = _build_loader(
        aux_train_df,
        label_to_idx=label_to_idx,
        feature_norm_path=aux_feature_norm_path,
        batch_size=args.batch_size,
        augment=args.augment,
        noise_sigma=args.noise_sigma,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        temporal_crop=args.temporal_crop,
        min_crop_ratio=args.min_crop_ratio,
        seed=args.seed,
        shuffle=True,
    )
    target_train_loader = _build_loader(
        target_train_df,
        label_to_idx=label_to_idx,
        feature_norm_path=feature_norm_path,
        batch_size=args.batch_size,
        augment=args.augment,
        noise_sigma=args.noise_sigma,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        temporal_crop=args.temporal_crop,
        min_crop_ratio=args.min_crop_ratio,
        seed=args.seed,
        shuffle=True,
    )
    target_val_loader = _build_loader(
        target_val_df,
        label_to_idx=label_to_idx,
        feature_norm_path=feature_norm_path,
        batch_size=args.batch_size,
        augment=False,
        noise_sigma=0.0,
        scale_min=1.0,
        scale_max=1.0,
        temporal_crop=False,
        min_crop_ratio=args.min_crop_ratio,
        seed=args.seed,
        shuffle=False,
    )
    target_test_loader = _build_loader(
        target_test_df,
        label_to_idx=label_to_idx,
        feature_norm_path=feature_norm_path,
        batch_size=args.batch_size,
        augment=False,
        noise_sigma=0.0,
        scale_min=1.0,
        scale_max=1.0,
        temporal_crop=False,
        min_crop_ratio=args.min_crop_ratio,
        seed=args.seed,
        shuffle=False,
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
    aux_model = model
    if aux_feature_dim != feature_dim:
        aux_model = LSTMClassifier(
            feature_dim=aux_feature_dim,
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            proj_dim=args.proj_dim,
        ).to(device)

    loss_fn = nn.CrossEntropyLoss()

    pretrain_opt = torch.optim.AdamW(aux_model.parameters(), lr=args.pretrain_lr)
    for epoch in range(1, args.pretrain_epochs + 1):
        _train_phase(
            model=aux_model,
            loader=aux_train_loader,
            device=device,
            optimizer=pretrain_opt,
            loss_fn=loss_fn,
            phase_name="Aux pretrain",
            epoch=epoch,
            total_epochs=args.pretrain_epochs,
        )
        aux_acc, aux_f1 = eval_epoch(aux_model, aux_train_loader, device=device)
        print(f"Aux train acc={aux_acc:.4f} macro_f1={aux_f1:.4f}")

    if aux_model is not model:
        model.lstm.load_state_dict(aux_model.lstm.state_dict())
        model.classifier.load_state_dict(aux_model.classifier.state_dict())

    if args.freeze_proj:
        for param in model.proj.parameters():
            param.requires_grad = False
    if args.freeze_lstm:
        for param in model.lstm.parameters():
            param.requires_grad = False

    finetune_params = [p for p in model.parameters() if p.requires_grad]
    finetune_opt = torch.optim.AdamW(finetune_params, lr=args.finetune_lr)
    best_f1 = -1.0
    best_path = args.out_dir / "best_transfer_lstm.pt"

    for epoch in range(1, args.finetune_epochs + 1):
        _train_phase(
            model=model,
            loader=target_train_loader,
            device=device,
            optimizer=finetune_opt,
            loss_fn=loss_fn,
            phase_name="NZSL finetune",
            epoch=epoch,
            total_epochs=args.finetune_epochs,
        )
        val_acc, val_f1 = eval_epoch(model, target_val_loader, device=device)
        print(f"Target val acc={val_acc:.4f} macro_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_dim": feature_dim,
                    "num_classes": num_classes,
                    "label_to_idx": label_to_idx,
                    "arch": "transfer_lstm",
                    "params": {
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "proj_dim": args.proj_dim,
                        "pretrain_epochs": args.pretrain_epochs,
                        "finetune_epochs": args.finetune_epochs,
                        "mapping_csv": str(args.mapping_csv),
                    },
                },
                best_path,
            )
            print(f"Saved best transfer model to {best_path} (macro_f1={best_f1:.4f})")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_acc, test_f1 = eval_epoch(model, target_test_loader, device=device)
    print(f"Target test acc={test_acc:.4f} macro_f1={test_f1:.4f}")

    write_json(
        args.out_dir / "transfer_summary.json",
        {
            "target_processed_dir": str(args.target_processed_dir),
            "target_out_dir": str(args.target_out_dir),
            "aux_processed_dir": str(args.aux_processed_dir),
            "mapping_csv": str(args.mapping_csv),
            "target_num_classes": num_classes,
            "target_feature_dim": feature_dim,
            "aux_feature_dim": aux_feature_dim,
            "aux_rows_used": int(len(aux_train_df)),
            "target_train_rows": int(len(target_train_df)),
            "target_val_rows": int(len(target_val_df)),
            "target_test_rows": int(len(target_test_df)),
            "best_val_macro_f1": best_f1,
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
        },
    )


if __name__ == "__main__":
    main()
