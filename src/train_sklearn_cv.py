from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .train_sklearn import aggregate_features
from .utils import ensure_dir, set_seed, write_json


def load_features(processed_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(processed_dir / "labels.csv")
    feats = []
    for p in df["sequence_path"].astype(str).tolist():
        seq = np.load(p)
        feats.append(aggregate_features(seq))
    return df, np.stack(feats, axis=0), df["label"].astype(str).values


def build_models(seed: int) -> Dict[str, object]:
    return {
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out_dir", type=Path, default=Path("models_cv"))
    ap.add_argument("--reports_dir", type=Path, default=Path("reports_cv"))
    ap.add_argument("--folds", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)
    ensure_dir(args.reports_dir)

    _, X, y = load_features(args.processed_dir)
    labels = sorted(np.unique(y).tolist())
    write_json(args.out_dir / "labels.json", {"labels": labels, "label_to_idx": {lab: i for i, lab in enumerate(labels)}})

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    model_defs = build_models(args.seed)
    rows: List[dict] = []

    for model_name, model in model_defs.items():
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
            clf = clone(model)
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[val_idx])
            acc = float(accuracy_score(y[val_idx], pred))
            macro_f1 = float(f1_score(y[val_idx], pred, average="macro"))
            rows.append({"model": model_name, "fold": fold, "accuracy": acc, "macro_f1": macro_f1})
            print(f"[{model_name}] fold={fold}/{args.folds} acc={acc:.4f} macro_f1={macro_f1:.4f}")

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(args.reports_dir / "cv_results.csv", index=False)
    summary = (
        cv_df.groupby("model")[["accuracy", "macro_f1"]]
        .agg(["mean", "std"])
        .sort_values(("macro_f1", "mean"), ascending=False)
    )
    summary.to_csv(args.reports_dir / "cv_summary.csv")
    print(summary)

    best_model_name = str(summary.index[0])
    final_model = clone(model_defs[best_model_name])
    final_model.fit(X, y)
    final_path = args.out_dir / f"sklearn_{best_model_name}_cv{args.folds}_all.joblib"
    joblib.dump(final_model, final_path)
    write_json(
        args.reports_dir / "cv_summary.json",
        {
            "folds": args.folds,
            "best_model": best_model_name,
            "final_model_path": str(final_path).replace("\\", "/"),
            "models": {
                model_name: {
                    "accuracy_mean": float(group["accuracy"].mean()),
                    "accuracy_std": float(group["accuracy"].std(ddof=1)),
                    "macro_f1_mean": float(group["macro_f1"].mean()),
                    "macro_f1_std": float(group["macro_f1"].std(ddof=1)),
                }
                for model_name, group in cv_df.groupby("model")
            },
        },
    )
    print(f"Trained final {best_model_name} on all samples: {final_path}")


if __name__ == "__main__":
    main()
