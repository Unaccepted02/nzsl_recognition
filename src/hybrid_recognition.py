from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .utils import softmax


POSE_DIM = 33 * 4
HAND_DIM = 21 * 3
LEFT_HAND_OFFSET = POSE_DIM
RIGHT_HAND_OFFSET = POSE_DIM + HAND_DIM

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
NOSE = 0


def _reshape_pose(seq: np.ndarray) -> np.ndarray:
    return seq[:, :POSE_DIM].reshape(seq.shape[0], 33, 4)


def _reshape_left_hand(seq: np.ndarray) -> np.ndarray:
    return seq[:, LEFT_HAND_OFFSET:RIGHT_HAND_OFFSET].reshape(seq.shape[0], 21, 3)


def _reshape_right_hand(seq: np.ndarray) -> np.ndarray:
    return seq[:, RIGHT_HAND_OFFSET:RIGHT_HAND_OFFSET + HAND_DIM].reshape(seq.shape[0], 21, 3)


def _hand_present(hand_seq: np.ndarray) -> np.ndarray:
    return (np.abs(hand_seq).sum(axis=(1, 2)) > 1e-6).astype(np.float32)


def _dominant_hand(seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    left = _reshape_left_hand(seq)
    right = _reshape_right_hand(seq)
    left_present = _hand_present(left).mean()
    right_present = _hand_present(right).mean()
    left_motion = np.linalg.norm(np.diff(left[:, WRIST, :], axis=0), axis=1).mean() if len(left) > 1 else 0.0
    right_motion = np.linalg.norm(np.diff(right[:, WRIST, :], axis=0), axis=1).mean() if len(right) > 1 else 0.0
    left_score = left_present + left_motion
    right_score = right_present + right_motion
    if left_score >= right_score:
        return left, right, "left"
    return right, left, "right"


def _safe_norm(value: float, scale: float) -> float:
    if scale <= 1e-6:
        return 0.0
    return float(np.clip(value / scale, 0.0, 1.0))


def _prob_from_scores(labels: Sequence[str], score_map: Dict[str, float]) -> np.ndarray:
    raw = np.array([max(0.0, float(score_map.get(label, 0.0))) for label in labels], dtype=np.float32)
    if raw.sum() <= 1e-6:
        return np.full((len(labels),), 1.0 / max(len(labels), 1), dtype=np.float32)
    return raw / raw.sum()


def template_feature_sequence(seq: np.ndarray) -> np.ndarray:
    pose = _reshape_pose(seq)
    left = _reshape_left_hand(seq)
    right = _reshape_right_hand(seq)

    left_wrist = left[:, WRIST, :]
    right_wrist = right[:, WRIST, :]
    left_index = left[:, INDEX_TIP, :]
    right_index = right[:, INDEX_TIP, :]
    left_middle = left[:, MIDDLE_TIP, :]
    right_middle = right[:, MIDDLE_TIP, :]
    shoulder_center = (pose[:, LEFT_SHOULDER, :3] + pose[:, RIGHT_SHOULDER, :3]) / 2.0

    return np.concatenate(
        [
            left_wrist,
            right_wrist,
            left_index,
            right_index,
            left_middle,
            right_middle,
            shoulder_center,
        ],
        axis=1,
    ).astype(np.float32)


@lru_cache(maxsize=8)
def _load_template_bank(processed_dir_str: str) -> Dict[str, np.ndarray]:
    processed_dir = Path(processed_dir_str)
    labels_path = processed_dir / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing {labels_path}")

    df = pd.read_csv(labels_path)
    if "split" in df.columns:
        df = df[df["split"] == "train"].copy()
    if df.empty:
        raise RuntimeError(f"No train samples found in {labels_path}")

    bank: Dict[str, List[np.ndarray]] = {}
    for _, row in df.iterrows():
        seq = np.load(row["sequence_path"]).astype(np.float32)
        bank.setdefault(str(row["label"]), []).append(template_feature_sequence(seq))

    prototypes: Dict[str, np.ndarray] = {}
    for label, items in bank.items():
        stacked = np.stack(items, axis=0)
        prototypes[label] = stacked.mean(axis=0).astype(np.float32)
    return prototypes


def predict_sequence_template(
    seq: np.ndarray,
    labels: Sequence[str],
    processed_dir: Path,
) -> Tuple[str, np.ndarray]:
    prototypes = _load_template_bank(str(processed_dir.resolve()))
    feat = template_feature_sequence(seq)

    dists: List[float] = []
    for label in labels:
        proto = prototypes.get(label)
        if proto is None:
            dists.append(1e6)
            continue
        dists.append(float(np.mean((feat - proto) ** 2)))

    logits = -np.array(dists, dtype=np.float32)
    proba = softmax(logits)
    pred_i = int(np.argmax(proba))
    return str(labels[pred_i]), proba.astype(np.float32)


def rule_based_scores(seq: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    pose = _reshape_pose(seq)
    dom, nondom, _ = _dominant_hand(seq)

    dom_present = _hand_present(dom)
    nondom_present = _hand_present(nondom)
    dom_wrist = dom[:, WRIST, :]
    nondom_wrist = nondom[:, WRIST, :]
    dom_index = dom[:, INDEX_TIP, :]
    dom_middle = dom[:, MIDDLE_TIP, :]
    dom_thumb = dom[:, THUMB_TIP, :]
    dom_ring = dom[:, RING_TIP, :]
    dom_pinky = dom[:, PINKY_TIP, :]
    shoulder_y = ((pose[:, LEFT_SHOULDER, 1] + pose[:, RIGHT_SHOULDER, 1]) / 2.0).astype(np.float32)
    nose = pose[:, NOSE, :3]

    tip_stack = np.stack([dom_thumb, dom_index, dom_middle, dom_ring, dom_pinky], axis=1)
    finger_spread = np.linalg.norm(tip_stack - dom_wrist[:, None, :], axis=2).mean(axis=1)
    avg_spread = float(np.mean(finger_spread * dom_present))

    wrist_speed = np.linalg.norm(np.diff(dom_wrist, axis=0), axis=1) if len(dom_wrist) > 1 else np.zeros((0,), dtype=np.float32)
    avg_speed = float(wrist_speed.mean()) if len(wrist_speed) else 0.0
    horiz_disp = float(dom_wrist[-1, 0] - dom_wrist[0, 0])
    depth_disp = float(dom_wrist[-1, 2] - dom_wrist[0, 2])
    face_start_dist = float(np.linalg.norm(dom_wrist[: max(1, len(dom_wrist) // 3)] - nose[: max(1, len(nose) // 3)], axis=1).mean())
    face_end_dist = float(np.linalg.norm(dom_wrist[-max(1, len(dom_wrist) // 3) :] - nose[-max(1, len(nose) // 3) :], axis=1).mean())
    above_shoulder = float(np.mean((dom_wrist[:, 1] < shoulder_y).astype(np.float32) * dom_present))
    wrist_gap = float(np.linalg.norm(dom_wrist - nondom_wrist, axis=1).mean())
    both_hands_present = float(min(dom_present.mean(), nondom_present.mean()))

    score_map: Dict[str, float] = {}
    if "stop" in labels:
        stop_score = _safe_norm(avg_spread, 0.25) * _safe_norm(above_shoulder, 0.6) * (1.0 - _safe_norm(avg_speed, 0.08))
        score_map["stop"] = stop_score
    if "hello" in labels:
        hello_score = _safe_norm(avg_spread, 0.22) * _safe_norm(abs(horiz_disp), 0.18) * _safe_norm(avg_speed, 0.04)
        score_map["hello"] = hello_score
    if "thank_you" in labels:
        thank_score = _safe_norm(avg_spread, 0.2) * _safe_norm(face_end_dist - face_start_dist, 0.18)
        score_map["thank_you"] = thank_score
    if "wait" in labels:
        wait_score = _safe_norm(0.18 - wrist_gap, 0.18) * _safe_norm(both_hands_present, 0.5) * (
            1.0 - _safe_norm(avg_speed, 0.06)
        )
        score_map["wait"] = wait_score

    return _prob_from_scores(labels, score_map)


def blend_probabilities(
    labels: Sequence[str],
    ml_proba: np.ndarray,
    template_proba: Optional[np.ndarray],
    rule_proba: Optional[np.ndarray],
    ml_weight: float = 0.85,
    template_weight: float = 0.1,
    rule_weight: float = 0.05,
) -> Tuple[str, np.ndarray, Dict[str, np.ndarray]]:
    parts: Dict[str, np.ndarray] = {"ml": ml_proba.astype(np.float32)}
    combined = ml_weight * ml_proba.astype(np.float32)
    total_weight = ml_weight

    if template_proba is not None:
        parts["template"] = template_proba.astype(np.float32)
        combined += template_weight * template_proba.astype(np.float32)
        total_weight += template_weight
    if rule_proba is not None:
        parts["rule"] = rule_proba.astype(np.float32)
        combined += rule_weight * rule_proba.astype(np.float32)
        total_weight += rule_weight

    combined /= max(total_weight, 1e-6)
    pred_i = int(np.argmax(combined))
    return str(labels[pred_i]), combined.astype(np.float32), parts
