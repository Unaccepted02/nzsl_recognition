from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def uniform_sample_indices(length: int, num: int) -> np.ndarray:
    if num <= 0:
        raise ValueError("num must be > 0")
    if length <= 0:
        return np.zeros((num,), dtype=np.int64)
    if length == num:
        return np.arange(length, dtype=np.int64)
    if length < num:
        idx = np.arange(length, dtype=np.int64)
        pad = np.full((num - length,), idx[-1], dtype=np.int64)
        return np.concatenate([idx, pad], axis=0)
    lin = np.linspace(0, length - 1, num=num)
    return np.round(lin).astype(np.int64)


def pad_or_sample_sequence(seq: np.ndarray, num_frames: int) -> np.ndarray:
    if seq.ndim != 2:
        raise ValueError(f"Expected [T, D], got {seq.shape}")
    t, d = seq.shape
    if t == num_frames:
        return seq
    if t == 0:
        return np.zeros((num_frames, d), dtype=seq.dtype)
    if t < num_frames:
        out = np.zeros((num_frames, d), dtype=seq.dtype)
        out[:t] = seq
        return out
    idx = uniform_sample_indices(t, num_frames)
    return seq[idx]


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15


def stratified_split(
    labels: Sequence[str],
    split: SplitConfig,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(labels))
    idx_train, idx_tmp, y_train, y_tmp = train_test_split(
        idx,
        np.array(labels),
        test_size=(1.0 - split.train),
        random_state=seed,
        stratify=np.array(labels),
    )
    val_ratio = split.val / (split.val + split.test)
    idx_val, idx_test = train_test_split(
        idx_tmp,
        test_size=(1.0 - val_ratio),
        random_state=seed,
        stratify=y_tmp,
    )
    return idx_train, idx_val, idx_test
