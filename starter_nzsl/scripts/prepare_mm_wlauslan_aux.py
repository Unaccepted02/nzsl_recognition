from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import ensure_dir, pad_or_sample_sequence


POSE_DIM = 33 * 4
HAND_DIM = 21 * 3
FACE_DIM = 19 * 3
FEATURE_DIM = POSE_DIM + HAND_DIM * 2 + FACE_DIM


def load_selected_glosses(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("recommended_use") != "pretrain":
                continue
            nzsl_label = str(row["nzsl_label"]).strip()
            auslan_gloss = str(row["auslan_gloss"]).strip()
            if nzsl_label and auslan_gloss:
                mapping[auslan_gloss] = nzsl_label
    if not mapping:
        raise RuntimeError(f"No pretrain glosses found in {path}")
    return mapping


def iter_selected_ids(split_json: Path, gloss_to_label: Dict[str, str], split_name: str) -> Iterable[dict]:
    rows = json.loads(split_json.read_text(encoding="utf-8"))
    for video_id, gloss in rows.items():
        if gloss in gloss_to_label:
            yield {
                "video_id": str(video_id),
                "auslan_gloss": str(gloss),
                "label": gloss_to_label[str(gloss)],
                "split": split_name,
            }


def halpe136_to_mediapipe_like(seq: np.ndarray) -> np.ndarray:
    """Convert MM-WLAuslan Halpe-136 [T, 136, 3] to the local 315-D feature layout.

    Halpe-136 is treated as:
    - 0:26 body/foot keypoints
    - 26:94 face keypoints
    - 94:115 left hand keypoints
    - 115:136 right hand keypoints
    """
    arr = np.asarray(seq, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (136, 3):
        raise ValueError(f"Expected [T, 136, 3], got {arr.shape}")

    xy = arr[..., :2].copy()
    conf = arr[..., 2:3].copy()

    # COCO/Halpe shoulders are normally indices 5 and 6.
    left_shoulder = xy[:, 5, :]
    right_shoulder = xy[:, 6, :]
    center = (left_shoulder + right_shoulder) / 2.0
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1, keepdims=True)
    scale = np.maximum(shoulder_width, 1.0)
    xy = (xy - center[:, None, :]) / scale[:, None, :]

    t = arr.shape[0]
    pose = np.zeros((t, 33, 4), dtype=np.float32)
    body_count = min(26, 33)
    pose[:, :body_count, 0:2] = xy[:, :body_count, :]
    pose[:, :body_count, 3:4] = conf[:, :body_count, :]

    left_hand = np.zeros((t, 21, 3), dtype=np.float32)
    right_hand = np.zeros((t, 21, 3), dtype=np.float32)
    left_hand[:, :, 0:2] = xy[:, 94:115, :]
    right_hand[:, :, 0:2] = xy[:, 115:136, :]

    face = np.zeros((t, 19, 3), dtype=np.float32)
    face[:, :, 0:2] = xy[:, 26:45, :]

    return np.concatenate(
        [
            pose.reshape(t, -1),
            left_hand.reshape(t, -1),
            right_hand.reshape(t, -1),
            face.reshape(t, -1),
        ],
        axis=1,
    ).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations_dir", type=Path, default=Path("aux_auslan/mm_wlauslan/annotations"))
    ap.add_argument("--pose_pkl", type=Path, default=Path("aux_auslan/mm_wlauslan/pose/pose_train_valid_cam1.pkl"))
    ap.add_argument("--selected_glosses", type=Path, default=Path("starter_nzsl/config/mm_wlauslan_selected_glosses.csv"))
    ap.add_argument("--out_dir", type=Path, default=Path("aux_auslan/data/processed"))
    ap.add_argument("--num_frames", type=int, default=60)
    ap.add_argument("--max_train_per_label", type=int, default=12)
    ap.add_argument("--max_val_per_label", type=int, default=2)
    args = ap.parse_args()

    seq_dir = args.out_dir / "sequences"
    ensure_dir(seq_dir)

    gloss_to_label = load_selected_glosses(args.selected_glosses)
    candidates = list(iter_selected_ids(args.annotations_dir / "Train.json", gloss_to_label, "train"))
    candidates += list(iter_selected_ids(args.annotations_dir / "Valid.json", gloss_to_label, "val"))

    kept: List[dict] = []
    by_label_split: Dict[tuple[str, str], int] = {}
    for row in candidates:
        key = (row["label"], row["split"])
        limit = args.max_train_per_label if row["split"] == "train" else args.max_val_per_label
        if by_label_split.get(key, 0) >= limit:
            continue
        by_label_split[key] = by_label_split.get(key, 0) + 1
        kept.append(row)

    wanted_ids = {row["video_id"] for row in kept}
    if not wanted_ids:
        raise RuntimeError("No selected MM-WLAuslan samples matched the provided gloss mapping.")

    with gzip.open(args.pose_pkl, "rb") as f:
        pose_data = pickle.load(f)

    rows: List[dict] = []
    for row in tqdm(kept, desc="Converting MM-WLAuslan pose"):
        video_id = row["video_id"]
        if video_id not in pose_data:
            continue
        seq = halpe136_to_mediapipe_like(pose_data[video_id])
        seq = pad_or_sample_sequence(seq, num_frames=args.num_frames).astype(np.float32)
        sample_id = f"mm_wlauslan_{row['label']}_{video_id}"
        out_path = seq_dir / f"{sample_id}.npy"
        np.save(out_path, seq)
        rows.append(
            {
                "sample_id": sample_id,
                "label": row["label"],
                "aux_label": row["auslan_gloss"],
                "video_id": video_id,
                "path": str(args.pose_pkl),
                "sequence_path": str(out_path),
                "modality": "mm_wlauslan_halpe136_pose",
                "split": row["split"],
            }
        )

    if not rows:
        raise RuntimeError("No pose samples were converted.")

    df = pd.DataFrame(rows).sort_values(["label", "split", "sample_id"])
    df.to_csv(args.out_dir / "labels.csv", index=False)
    summary = df.groupby(["label", "split"]).size().unstack(fill_value=0)
    print(summary)
    print(f"Wrote {len(df)} samples to {args.out_dir}")


if __name__ == "__main__":
    main()
