from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

STARTER_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = STARTER_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.utils import ensure_dir, write_json  # noqa: E402


# Maps a human-readable model variant name to its evaluate.py report directory.
MODEL_REPORT_DIRS = {
    "lstm": "reports",
    "transformer": "reports_transformer_optimized",
    "sklearn": "reports_sklearn",
}


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compute_dataset_stats(processed_dir: Path, vocab_csv: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    df = pd.read_csv(processed_dir / "labels.csv")
    vocab = pd.read_csv(vocab_csv)

    per_class = df.groupby("label").size()
    per_class_split = df.groupby(["label", "split"]).size().unstack(fill_value=0)

    covered_labels = set(df["label"].unique())
    vocab_labels = set(vocab["label"].unique())

    stats = {
        "total_samples": int(len(df)),
        "total_classes": int(df["label"].nunique()),
        "target_vocab_size": int(len(vocab)),
        "missing_vocab_labels": sorted(vocab_labels - covered_labels),
        "split_counts": {k: int(v) for k, v in df.groupby("split").size().items()},
        "split_class_counts": {k: int(v) for k, v in df.groupby("split")["label"].nunique().items()},
        "per_class_count_min": int(per_class.min()),
        "per_class_count_max": int(per_class.max()),
        "per_class_count_mean": float(per_class.mean()),
        "per_class_counts": {k: int(v) for k, v in per_class.sort_values(ascending=False).items()},
    }
    return stats, per_class_split


def plot_class_distribution(per_class_split: pd.DataFrame, out_path: Path) -> None:
    order = per_class_split.sum(axis=1).sort_values(ascending=False).index
    df = per_class_split.loc[order]
    splits = [c for c in ["train", "val", "test"] if c in df.columns]

    fig, ax = plt.subplots(figsize=(16, 7))
    df[splits].plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_xlabel("Label")
    ax.set_ylabel("Sample count")
    ax.set_title("Verified NZSL samples per class, by split")
    ax.legend(title="Split")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def aggregate_model_comparison(starter_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for variant, dirname in MODEL_REPORT_DIRS.items():
        metrics = load_json(starter_dir / dirname / "metrics.json")
        if metrics is None:
            continue
        rows.append({
            "variant": variant,
            "model_type": metrics.get("model_type"),
            "report_dir": dirname,
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "top3_accuracy": metrics.get("top3_accuracy"),
            "top5_accuracy": metrics.get("top5_accuracy"),
        })
    return rows


def plot_model_comparison(rows: List[Dict[str, Any]], out_path: Path) -> None:
    df = pd.DataFrame(rows).set_index("variant")
    metric_cols = [c for c in ["accuracy", "macro_f1", "top3_accuracy", "top5_accuracy"] if c in df.columns]

    fig, ax = plt.subplots(figsize=(10, 6))
    df[metric_cols].plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on NZSL test split (36 classes, 1 sample/class)")
    ax.legend(title="Metric")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate dataset statistics and model comparison assets for the feasibility report.")
    ap.add_argument("--starter_dir", type=Path, default=STARTER_DIR)
    ap.add_argument("--out_dir", type=Path, default=STARTER_DIR / "reports_analysis")
    args = ap.parse_args()

    starter_dir = args.starter_dir
    out_dir = args.out_dir
    ensure_dir(out_dir)

    stats, per_class_split = compute_dataset_stats(
        starter_dir / "data" / "processed",
        starter_dir / "config" / "verified_nzsl_vocab_30plus.csv",
    )
    write_json(out_dir / "dataset_stats.json", stats)
    per_class_split.to_csv(out_dir / "per_class_split_counts.csv")
    plot_class_distribution(per_class_split, out_dir / "class_distribution.png")
    print(f"Wrote dataset stats to {out_dir / 'dataset_stats.json'}")

    comparison = aggregate_model_comparison(starter_dir)
    write_json(out_dir / "model_comparison.json", {"models": comparison})
    pd.DataFrame(comparison).to_csv(out_dir / "model_comparison.csv", index=False)
    if comparison:
        plot_model_comparison(comparison, out_dir / "model_comparison.png")
    print(f"Wrote model comparison to {out_dir / 'model_comparison.json'}")


if __name__ == "__main__":
    main()
