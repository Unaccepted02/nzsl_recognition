from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .evaluate import compute_and_save_reports
from .keypoint_features import prepare_sequence_for_features
from .utils import ensure_dir, set_seed


def aggregate_features(seq: np.ndarray) -> np.ndarray:
    """Aggregate a [T, D] sequence into a fixed vector for classical models."""
    seq = prepare_sequence_for_features(seq, trim=True)
    vel = np.diff(seq, axis=0, prepend=seq[:1])
    acc = np.diff(vel, axis=0, prepend=vel[:1])

    global_features = [
        seq.mean(axis=0),
        seq.std(axis=0),
        seq.min(axis=0),
        seq.max(axis=0),
        np.abs(vel).mean(axis=0),
        np.abs(vel).max(axis=0),
        np.abs(acc).mean(axis=0),
        seq[-1] - seq[0],
    ]

    segment_features = []
    for segment in np.array_split(seq, 3, axis=0):
        if len(segment) == 0:
            segment = seq
        segment_features.extend([segment.mean(axis=0), segment.std(axis=0)])

    return np.concatenate(global_features + segment_features, axis=0).astype(np.float32)


def load_split(processed_dir: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(processed_dir / "labels.csv")
    feats = []
    for p in df["sequence_path"].astype(str).tolist():
        seq = np.load(p)
        feats.append(aggregate_features(seq))
    X = np.stack(feats, axis=0)
    y = df["label"].astype(str).values
    split = df["split"].astype(str).values if "split" in df.columns else np.array(["train"] * len(df))
    return df, X, y, split


def maybe_xgboost():
    try:
        from xgboost import XGBClassifier  # type: ignore

        return XGBClassifier
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out_dir", type=Path, default=Path("models"))
    ap.add_argument("--reports_dir", type=Path, default=Path("reports"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", type=str, default="all", choices=["all", "svm", "rf", "xgb"])
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)
    ensure_dir(args.reports_dir)

    _, X, y, split = load_split(args.processed_dir)
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    if not val_mask.any():
        val_mask = test_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    to_train = ["svm", "rf", "xgb"] if args.model == "all" else [args.model]
    labels = sorted(np.unique(y).tolist())

    best_name = None
    best_f1 = -1.0
    best_path = None

    for name in to_train:
        if name == "svm":
            clf = Pipeline(
                [("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced"))]
            )
        elif name == "rf":
            clf = RandomForestClassifier(
                n_estimators=500,
                random_state=args.seed,
                class_weight="balanced_subsample",
                n_jobs=-1,
            )
        else:
            XGBClassifier = maybe_xgboost()
            if XGBClassifier is None:
                print("Skipping xgb (xgboost not installed).")
                continue
            clf = XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=args.seed,
            )

        clf.fit(X_train, y_train)
        val_pred = clf.predict(X_val)
        val_f1 = float(f1_score(y_val, val_pred, average="macro"))
        print(f"[{name}] Val macro F1: {val_f1:.4f}")

        model_path = args.out_dir / f"sklearn_{name}.joblib"
        joblib.dump(clf, model_path)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_name = name
            best_path = model_path

    if best_path is None or best_name is None:
        raise RuntimeError("No sklearn models trained successfully.")

    clf = joblib.load(best_path)
    test_pred = clf.predict(X_test)
    print(f"Best model: {best_name} (val_macro_f1={best_f1:.4f})")
    print(f"Test acc: {accuracy_score(y_test, test_pred):.4f}")
    print(f"Test macro F1: {f1_score(y_test, test_pred, average='macro'):.4f}")
    print(classification_report(y_test, test_pred))

    y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
    compute_and_save_reports(
        y_true=y_test,
        y_pred=test_pred,
        y_proba=y_proba,
        labels=labels,
        out_dir=args.reports_dir,
        extra_metrics={"model_type": f"sklearn_{best_name}", "val_macro_f1": float(best_f1)},
    )
    print(f"Saved models to {args.out_dir} (best: {best_path})")


if __name__ == "__main__":
    main()
