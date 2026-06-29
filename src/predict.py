from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .extract_keypoints import HolisticConfig, extract_sequence_from_image, extract_sequence_from_video
from .hybrid_recognition import blend_probabilities, predict_sequence_template, rule_based_scores
from .keypoint_features import prepare_sequence_for_features
from .utils import IMAGE_EXTS, VIDEO_EXTS, pad_or_sample_sequence, read_json, softmax


def _load_feature_norm(models_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    norm_path = models_dir / "feature_norm.npz"
    if not norm_path.exists():
        return None
    stats = np.load(norm_path)
    return stats["mean"].astype(np.float32), stats["std"].astype(np.float32)


def _apply_feature_norm(seq: np.ndarray, norm: Optional[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    if norm is None:
        return seq
    mean, std = norm
    return (seq - mean[None, :]) / std[None, :]


def load_labels(models_dir: Path) -> List[str]:
    p = models_dir / "labels.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run: python -m src.preprocess --out_dir {models_dir}")
    info = read_json(p)
    return list(info["labels"])


def predict_sequence_sklearn(model_path: Path, seq: np.ndarray, labels: List[str]) -> Tuple[str, np.ndarray]:
    import joblib

    from .train_sklearn import aggregate_features

    model = joblib.load(model_path)
    x = aggregate_features(seq)[None, :]
    if hasattr(model, "predict_proba"):
        raw_proba = model.predict_proba(x)[0]
        classes = [str(c) for c in getattr(model, "classes_", labels)]
        proba = np.zeros((len(labels),), dtype=np.float32)
        for class_label, value in zip(classes, raw_proba):
            if class_label in labels:
                proba[labels.index(class_label)] = float(value)
    else:
        pred = str(model.predict(x)[0])
        proba = np.zeros((len(labels),), dtype=np.float32)
        proba[labels.index(pred)] = 1.0
    pred_i = int(np.argmax(proba))
    return labels[pred_i], proba.astype(np.float32)


def _torch_predict_logits(model, seq: np.ndarray, device: str) -> np.ndarray:
    import torch

    model.eval()
    x = torch.from_numpy(seq[None, :, :]).to(device)
    with torch.no_grad():
        logits = model(x).detach().cpu().numpy()[0]
    return logits.astype(np.float32)


def predict_sequence_torch(
    model_type: str,
    model_path: Path,
    seq: np.ndarray,
    labels: List[str],
    device: str,
) -> Tuple[str, np.ndarray]:
    import torch

    ckpt = torch.load(model_path, map_location=device)
    label_to_idx: Dict[str, int] = {k: int(v) for k, v in ckpt["label_to_idx"].items()}
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    params = ckpt.get("params", {})
    if bool(params.get("keypoint_preprocess", False)):
        target_len = seq.shape[0]
        seq = prepare_sequence_for_features(seq, trim=True)
        seq = pad_or_sample_sequence(seq, num_frames=target_len)
    seq = _apply_feature_norm(seq, _load_feature_norm(model_path.parent))

    if model_type == "lstm":
        from .train_lstm import LSTMClassifier

        model = LSTMClassifier(
            feature_dim=int(ckpt["feature_dim"]),
            num_classes=int(ckpt["num_classes"]),
            hidden_dim=int(params.get("hidden_dim", 256)),
            num_layers=int(params.get("num_layers", 2)),
            dropout=float(params.get("dropout", 0.2)),
            proj_dim=int(params.get("proj_dim", 256)),
        ).to(device)
    else:
        from .train_transformer import TransformerClassifier

        model = TransformerClassifier(
            feature_dim=int(ckpt["feature_dim"]),
            num_classes=int(ckpt["num_classes"]),
            d_model=int(params.get("d_model", 256)),
            num_layers=int(params.get("num_layers", 3)),
            nhead=int(params.get("nhead", 8)),
            dim_feedforward=int(params.get("dim_feedforward", 512)),
            dropout=float(params.get("dropout", 0.2)),
            pooling=str(params.get("pooling", "mean")),
        ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    logits = _torch_predict_logits(model, seq, device=device)
    proba = softmax(logits)
    pred_i = int(np.argmax(proba))
    pred_label = idx_to_label.get(pred_i, labels[pred_i] if pred_i < len(labels) else str(pred_i))

    out = np.zeros((len(labels),), dtype=np.float32)
    for lab, i in label_to_idx.items():
        if lab in labels and i < len(proba):
            out[labels.index(lab)] = float(proba[i])
    if out.sum() <= 0:
        out = proba[: len(labels)]

    return pred_label, out.astype(np.float32)


def extract_sequence_for_path(path: Path, num_frames: int = 60) -> np.ndarray:
    cfg = HolisticConfig()
    if path.suffix.lower() in VIDEO_EXTS:
        return extract_sequence_from_video(path, holistic_cfg=cfg, num_frames=num_frames)
    if path.suffix.lower() in IMAGE_EXTS:
        return extract_sequence_from_image(path, holistic_cfg=cfg, num_frames=num_frames)
    raise ValueError(f"Unsupported file type: {path}")


def load_model_and_predict_df(
    model_type: str,
    model_path: Path,
    df: pd.DataFrame,
    device: str = "cpu",
    processed_dir: Optional[Path] = None,
    hybrid_base_model: str = "lstm",
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
    models_dir = model_path.parent
    labels = load_labels(models_dir)
    norm = _load_feature_norm(models_dir)

    y_true = df["label"].astype(str).values
    y_pred: List[str] = []
    y_proba: List[np.ndarray] = []

    for p in df["sequence_path"].astype(str).tolist():
        seq = np.load(p).astype(np.float32)
        if model_type == "sklearn":
            pred, proba = predict_sequence_sklearn(model_path, seq, labels=labels)
        elif model_type in {"lstm", "transformer"}:
            pred, proba = predict_sequence_torch(model_type, model_path, seq, labels=labels, device=device)
        elif model_type == "template":
            if processed_dir is None:
                raise ValueError("processed_dir is required for template prediction")
            pred, proba = predict_sequence_template(seq, labels=labels, processed_dir=processed_dir)
        elif model_type == "hybrid":
            if processed_dir is None:
                raise ValueError("processed_dir is required for hybrid prediction")
            if hybrid_base_model == "sklearn":
                _, ml_proba = predict_sequence_sklearn(model_path, seq, labels=labels)
            else:
                _, ml_proba = predict_sequence_torch(
                    hybrid_base_model,
                    model_path,
                    seq,
                    labels=labels,
                    device=device,
                )
            _, template_proba = predict_sequence_template(seq, labels=labels, processed_dir=processed_dir)
            rule_proba = rule_based_scores(seq, labels=labels)
            pred, proba, _ = blend_probabilities(
                labels=labels,
                ml_proba=ml_proba,
                template_proba=template_proba,
                rule_proba=rule_proba,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        y_pred.append(pred)
        y_proba.append(proba)

    proba_arr = np.stack(y_proba, axis=0) if len(y_proba) else None
    return y_true, np.array(y_pred), proba_arr, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", type=Path, required=True)
    ap.add_argument("--model_type", type=str, required=True, choices=["sklearn", "lstm", "transformer", "template", "hybrid"])
    ap.add_argument("--model_path", type=Path, required=True)
    ap.add_argument("--num_frames", type=int, default=60)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--hybrid_base_model", type=str, default="lstm", choices=["sklearn", "lstm", "transformer"])
    args = ap.parse_args()

    labels = load_labels(args.model_path.parent)
    seq = extract_sequence_for_path(args.input_path, num_frames=args.num_frames)

    if args.model_type == "sklearn":
        pred, proba = predict_sequence_sklearn(args.model_path, seq, labels=labels)
    elif args.model_type in {"lstm", "transformer"}:
        pred, proba = predict_sequence_torch(args.model_type, args.model_path, seq, labels=labels, device=args.device)
    elif args.model_type == "template":
        pred, proba = predict_sequence_template(seq, labels=labels, processed_dir=args.processed_dir)
    else:
        if args.hybrid_base_model == "sklearn":
            _, ml_proba = predict_sequence_sklearn(args.model_path, seq, labels=labels)
        else:
            _, ml_proba = predict_sequence_torch(
                args.hybrid_base_model,
                args.model_path,
                seq,
                labels=labels,
                device=args.device,
            )
        _, template_proba = predict_sequence_template(seq, labels=labels, processed_dir=args.processed_dir)
        rule_proba = rule_based_scores(seq, labels=labels)
        pred, proba, _ = blend_probabilities(
            labels=labels,
            ml_proba=ml_proba,
            template_proba=template_proba,
            rule_proba=rule_proba,
        )

    topk = np.argsort(-proba)[:5]
    print(f"Predicted: {pred}")
    for i in topk:
        print(f"  {labels[i]}: {float(proba[i]):.4f}")


if __name__ == "__main__":
    main()
