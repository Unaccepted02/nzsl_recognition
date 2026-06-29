from __future__ import annotations

import numpy as np


POSE_POINTS = 33
POSE_STRIDE = 4
HAND_POINTS = 21
FACE_POINTS = 19

POSE_DIM = POSE_POINTS * POSE_STRIDE
HAND_DIM = HAND_POINTS * 3
FACE_DIM = FACE_POINTS * 3

POSE_START = 0
LEFT_HAND_START = POSE_START + POSE_DIM
RIGHT_HAND_START = LEFT_HAND_START + HAND_DIM
FACE_START = RIGHT_HAND_START + HAND_DIM

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def _reshape_pose(seq: np.ndarray) -> np.ndarray:
    return seq[:, POSE_START : POSE_START + POSE_DIM].reshape(seq.shape[0], POSE_POINTS, POSE_STRIDE)


def _reshape_block(seq: np.ndarray, start: int, points: int) -> np.ndarray:
    return seq[:, start : start + points * 3].reshape(seq.shape[0], points, 3)


def _flatten_block(block: np.ndarray) -> np.ndarray:
    return block.reshape(block.shape[0], -1)


def _missing_landmark_frames(block: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # Missing hands/face in older extracted arrays can appear as the same translated
    # coordinate repeated for every landmark in a frame. Real detections vary by point.
    return np.nanmax(np.nanstd(block, axis=1), axis=1) < eps


def repair_missing_landmark_blocks(seq: np.ndarray) -> np.ndarray:
    out = seq.astype(np.float32, copy=True)
    for start, points in ((LEFT_HAND_START, HAND_POINTS), (RIGHT_HAND_START, HAND_POINTS), (FACE_START, FACE_POINTS)):
        block = _reshape_block(out, start, points)
        missing = _missing_landmark_frames(block)
        block[missing] = 0.0
        out[:, start : start + points * 3] = _flatten_block(block)
    return out


def normalize_pose_scale(seq: np.ndarray) -> np.ndarray:
    out = repair_missing_landmark_blocks(seq)
    pose = _reshape_pose(out)
    left = pose[:, LEFT_SHOULDER, :3]
    right = pose[:, RIGHT_SHOULDER, :3]
    shoulder_center = (left + right) / 2.0
    shoulder_dist = np.linalg.norm(left - right, axis=1)
    valid = shoulder_dist > 1e-5
    if valid.any():
        scale = float(np.median(shoulder_dist[valid]))
    else:
        scale = 1.0
    scale = max(scale, 1e-5)

    pose[:, :, :3] = (pose[:, :, :3] - shoulder_center[:, None, :]) / scale
    out[:, POSE_START : POSE_START + POSE_DIM] = pose.reshape(out.shape[0], -1)

    for start, points in ((LEFT_HAND_START, HAND_POINTS), (RIGHT_HAND_START, HAND_POINTS), (FACE_START, FACE_POINTS)):
        block = _reshape_block(out, start, points)
        present = ~_missing_landmark_frames(block)
        block[present] = (block[present] - shoulder_center[present, None, :]) / scale
        block[~present] = 0.0
        out[:, start : start + points * 3] = _flatten_block(block)

    return out.astype(np.float32)


def trim_static_edges(seq: np.ndarray, min_keep_ratio: float = 0.45, motion_quantile: float = 0.55) -> np.ndarray:
    if seq.shape[0] < 4:
        return seq
    hands = np.concatenate(
        [
            seq[:, LEFT_HAND_START : LEFT_HAND_START + HAND_DIM],
            seq[:, RIGHT_HAND_START : RIGHT_HAND_START + HAND_DIM],
        ],
        axis=1,
    )
    motion = np.linalg.norm(np.diff(hands, axis=0, prepend=hands[:1]), axis=1)
    if not np.isfinite(motion).any() or float(motion.max()) <= 1e-8:
        return seq

    threshold = float(np.quantile(motion, motion_quantile))
    active = np.flatnonzero(motion >= threshold)
    if active.size == 0:
        return seq

    start = int(active[0])
    end = int(active[-1]) + 1
    min_len = max(2, int(round(seq.shape[0] * min_keep_ratio)))
    if end - start < min_len:
        center = (start + end) // 2
        start = max(0, center - min_len // 2)
        end = min(seq.shape[0], start + min_len)
        start = max(0, end - min_len)
    return seq[start:end]


def prepare_sequence_for_features(seq: np.ndarray, trim: bool = True) -> np.ndarray:
    out = normalize_pose_scale(seq)
    if trim:
        out = trim_static_edges(out)
    return out.astype(np.float32)
