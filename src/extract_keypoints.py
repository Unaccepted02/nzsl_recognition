from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import IMAGE_EXTS, VIDEO_EXTS, ensure_dir, pad_or_sample_sequence, uniform_sample_indices


# Small subset to keep the feature vector manageable for a starter prototype.
# You can expand this later if face movement is important for your sign set.
FACE_SUBSET_IDX = [1, 2, 4, 5, 6, 9, 10, 13, 14, 17, 18, 19, 20, 61, 291, 0, 11, 12, 152]


def _lazy_import_cv2():
    import cv2  # type: ignore

    return cv2


def _lazy_import_mediapipe():
    import mediapipe as mp  # type: ignore

    return mp

@dataclass
class HolisticConfig:
    model_complexity: int = 1
    smooth_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


def _landmarks_to_np(landmarks, include_visibility: bool) -> np.ndarray:
    if landmarks is None:
        return np.zeros((0,), dtype=np.float32)
    pts: List[float] = []
    for lm in landmarks.landmark:
        pts.extend([lm.x, lm.y, lm.z])
        if include_visibility:
            pts.append(float(getattr(lm, "visibility", 0.0)))
    return np.array(pts, dtype=np.float32)


def _face_subset_to_np(face_landmarks, subset_idx: Sequence[int]) -> np.ndarray:
    if face_landmarks is None:
        return np.zeros((len(subset_idx) * 3,), dtype=np.float32)
    pts: List[float] = []
    lms = face_landmarks.landmark
    for i in subset_idx:
        if i < 0 or i >= len(lms):
            pts.extend([0.0, 0.0, 0.0])
        else:
            lm = lms[i]
            pts.extend([lm.x, lm.y, lm.z])
    return np.array(pts, dtype=np.float32)


def _shoulder_center_from_pose(pose_landmarks) -> Optional[np.ndarray]:
    if pose_landmarks is None:
        return None
    lms = pose_landmarks.landmark
    if len(lms) <= 12:
        return None
    left = lms[11]
    right = lms[12]
    return np.array([(left.x + right.x) / 2.0, (left.y + right.y) / 2.0, (left.z + right.z) / 2.0], dtype=np.float32)


def extract_frame_features(
    results,
    face_subset_idx: Sequence[int] = FACE_SUBSET_IDX,
    normalize: bool = True,
) -> np.ndarray:
    pose = _landmarks_to_np(results.pose_landmarks, include_visibility=True)  # 33*(x,y,z,vis)
    lh = _landmarks_to_np(results.left_hand_landmarks, include_visibility=False)  # 21*(x,y,z)
    rh = _landmarks_to_np(results.right_hand_landmarks, include_visibility=False)
    face = _face_subset_to_np(results.face_landmarks, subset_idx=face_subset_idx)

    # Fill missing parts with zeros of expected size
    if pose.size == 0:
        pose = np.zeros((33 * 4,), dtype=np.float32)
    if lh.size == 0:
        lh = np.zeros((21 * 3,), dtype=np.float32)
    if rh.size == 0:
        rh = np.zeros((21 * 3,), dtype=np.float32)

    feat = np.concatenate([pose, lh, rh, face], axis=0).astype(np.float32)
    if not normalize:
        return feat

    center = _shoulder_center_from_pose(results.pose_landmarks)
    if center is None:
        return feat

    def _translate_xyz(vec: np.ndarray, stride: int, xyz: int = 3) -> np.ndarray:
        out = vec.copy()
        for j in range(0, len(out), stride):
            out[j : j + xyz] -= center[:xyz]
        return out

    pose_t = _translate_xyz(pose, stride=4, xyz=3)
    lh_t = _translate_xyz(lh, stride=3, xyz=3)
    rh_t = _translate_xyz(rh, stride=3, xyz=3)
    face_t = _translate_xyz(face, stride=3, xyz=3)
    return np.concatenate([pose_t, lh_t, rh_t, face_t], axis=0).astype(np.float32)


def extract_sequence_from_video(
    video_path: Path,
    holistic_cfg: HolisticConfig,
    num_frames: int,
    face_subset_idx: Sequence[int] = FACE_SUBSET_IDX,
    normalize: bool = True,
) -> np.ndarray:
    """Extract a fixed-length [num_frames, feature_dim] sequence from a video.

    - If the video has fewer than `num_frames`, pad with zeros.
    - If it has more, sample uniformly before running MediaPipe (faster).
    """
    cv2 = _lazy_import_cv2()
    mp = _lazy_import_mediapipe()
    if not getattr(mp, "solutions", None) or not hasattr(mp.solutions, "holistic"):  # type: ignore[attr-defined]
        raise RuntimeError(
            "This script requires the classic MediaPipe Solutions API (mp.solutions.holistic), "
            "but your installed 'mediapipe' package does not expose it."
        )
    mp_holistic = mp.solutions.holistic  # type: ignore[attr-defined]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from: {video_path}")

    if len(frames) > num_frames:
        idx = uniform_sample_indices(len(frames), num_frames)
        frames = [frames[i] for i in idx]

    feats: List[np.ndarray] = []
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=holistic_cfg.model_complexity,
        smooth_landmarks=holistic_cfg.smooth_landmarks,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=holistic_cfg.min_detection_confidence,
        min_tracking_confidence=holistic_cfg.min_tracking_confidence,
    ) as holistic:
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            feats.append(extract_frame_features(results, face_subset_idx=face_subset_idx, normalize=normalize))

    seq = np.stack(feats, axis=0).astype(np.float32)
    return pad_or_sample_sequence(seq, num_frames=num_frames)


def extract_sequence_from_image(
    image_path: Path,
    holistic_cfg: HolisticConfig,
    num_frames: int,
    face_subset_idx: Sequence[int] = FACE_SUBSET_IDX,
    normalize: bool = True,
) -> np.ndarray:
    """Extract a fixed-length [num_frames, feature_dim] sequence from an image.

    For static signs, this repeats the single-frame feature vector across time so
    it matches the video feature format.
    """
    cv2 = _lazy_import_cv2()
    mp = _lazy_import_mediapipe()
    if not getattr(mp, "solutions", None) or not hasattr(mp.solutions, "holistic"):  # type: ignore[attr-defined]
        raise RuntimeError(
            "This script requires the classic MediaPipe Solutions API (mp.solutions.holistic), "
            "but your installed 'mediapipe' package does not expose it."
        )
    mp_holistic = mp.solutions.holistic  # type: ignore[attr-defined]

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=holistic_cfg.model_complexity,
        smooth_landmarks=holistic_cfg.smooth_landmarks,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=holistic_cfg.min_detection_confidence,
        min_tracking_confidence=holistic_cfg.min_tracking_confidence,
    ) as holistic:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        feat = extract_frame_features(results, face_subset_idx=face_subset_idx, normalize=normalize)

    return np.repeat(feat[None, :], repeats=num_frames, axis=0).astype(np.float32)


def iter_raw_samples(raw_dir: Path) -> Iterable[Tuple[str, Path]]:
    for label_dir in sorted([p for p in raw_dir.iterdir() if p.is_dir()]):
        label = label_dir.name
        for path in sorted(label_dir.rglob("*")):
            if path.suffix.lower() in VIDEO_EXTS or path.suffix.lower() in IMAGE_EXTS:
                yield label, path


def sample_id_for_path(path: Path) -> str:
    h = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
    return f"{path.stem}_{h}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--num_frames", type=int, default=60)
    ap.add_argument("--disable_normalize", action="store_true")
    ap.add_argument("--model_complexity", type=int, default=1)
    ap.add_argument("--min_detection_conf", type=float, default=0.5)
    ap.add_argument("--min_tracking_conf", type=float, default=0.5)
    args = ap.parse_args()

    seq_dir = args.out_dir / "sequences"
    ensure_dir(seq_dir)

    holistic_cfg = HolisticConfig(
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_conf,
        min_tracking_confidence=args.min_tracking_conf,
    )

    rows: List[Dict[str, str]] = []
    errors: List[str] = []
    samples = list(iter_raw_samples(args.raw_dir))
    if len(samples) == 0:
        raise RuntimeError(f"No samples found under: {args.raw_dir}")

    for label, path in tqdm(samples, desc="Extracting"):
        sid = sample_id_for_path(path)
        out_path = seq_dir / f"{sid}.npy"
        try:
            if path.suffix.lower() in VIDEO_EXTS:
                seq = extract_sequence_from_video(
                    path,
                    holistic_cfg=holistic_cfg,
                    num_frames=args.num_frames,
                    normalize=not args.disable_normalize,
                )
            else:
                seq = extract_sequence_from_image(
                    path,
                    holistic_cfg=holistic_cfg,
                    num_frames=args.num_frames,
                    normalize=not args.disable_normalize,
                )
            np.save(out_path, seq.astype(np.float32))
            rows.append(
                {
                    "sample_id": sid,
                    "label": label,
                    "path": str(path).replace("\\", "/"),
                    "sequence_path": str(out_path).replace("\\", "/"),
                    "modality": "video" if path.suffix.lower() in VIDEO_EXTS else "image",
                }
            )
        except Exception as e:  # noqa: BLE001
            errors.append(f"{path}: {e}")

    if not rows:
        ensure_dir(args.out_dir)
        if errors:
            (args.out_dir / "extraction_errors.txt").write_text("\n".join(errors), encoding="utf-8")
            raise RuntimeError(
                f"All {len(samples)} samples failed during extraction. "
                f"See: {args.out_dir/'extraction_errors.txt'}"
            )
        raise RuntimeError("No sequences extracted and no errors were captured.")

    df = pd.DataFrame(rows).sort_values(["label", "sample_id"])
    ensure_dir(args.out_dir)
    df.to_csv(args.out_dir / "labels.csv", index=False)

    if errors:
        (args.out_dir / "extraction_errors.txt").write_text("\n".join(errors), encoding="utf-8")
        print(f"Extraction completed with {len(errors)} errors. See: {args.out_dir/'extraction_errors.txt'}")
    else:
        print("Extraction completed successfully.")


if __name__ == "__main__":
    main()
