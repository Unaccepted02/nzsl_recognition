from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .utils import ensure_dir, write_json


def top_k_accuracy(y_true_idx: np.ndarray, proba: np.ndarray, k: int) -> float:
    topk = np.argsort(-proba, axis=1)[:, :k]
    return float(np.mean([yt in row for yt, row in zip(y_true_idx, topk)]))


def compute_and_save_reports(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_proba: Optional[np.ndarray],
    labels: Sequence[Any],
    out_dir: Path,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ensure_dir(out_dir)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro")),
        "classification_report": classification_report(y_true_arr, y_pred_arr, output_dict=True, zero_division=0),
        "labels": list(labels),
    }

    if y_proba is not None:
        label_to_i = {lab: i for i, lab in enumerate(labels)}
        y_true_i = np.array([label_to_i[v] for v in y_true_arr])
        metrics["top3_accuracy"] = top_k_accuracy(y_true_i, y_proba, k=3)
        metrics["top5_accuracy"] = top_k_accuracy(y_true_i, y_proba, k=5)

    if extra_metrics:
        metrics.update(extra_metrics)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(labels))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    write_json(out_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--model_type", type=str, required=True, choices=["sklearn", "lstm", "transformer", "template", "hybrid"])
    ap.add_argument("--model_path", type=Path, required=True)
    ap.add_argument("--reports_dir", type=Path, default=Path("reports"))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--hybrid_base_model", type=str, default="lstm", choices=["sklearn", "lstm", "transformer"])
    args = ap.parse_args()

    import pandas as pd

    from .predict import load_model_and_predict_df

    df = pd.read_csv(args.processed_dir / "labels.csv")
    if "split" not in df.columns:
        raise RuntimeError("Missing split column. Run: python -m src.preprocess")
    test_df = df[df["split"] == "test"].copy()

    y_true, y_pred, y_proba, labels = load_model_and_predict_df(
        model_type=args.model_type,
        model_path=args.model_path,
        df=test_df,
        device=args.device,
        processed_dir=args.processed_dir,
        hybrid_base_model=args.hybrid_base_model,
    )

    compute_and_save_reports(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        labels=labels,
        out_dir=args.reports_dir,
        extra_metrics={"model_type": args.model_type, "model_path": str(args.model_path).replace("\\", "/")},
    )
    print(f"Wrote reports to {args.reports_dir}")


if __name__ == "__main__":
    main()
