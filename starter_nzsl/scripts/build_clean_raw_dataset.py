"""Build a cleaned raw-video workspace for NZSL training.

The source ``data/raw`` folder is left untouched. Files are copied into
``data/raw_cleaned`` with a small manifest describing source hints.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import stat
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


NON_MUTUAL_LABELS = {
    "bathroom",
    "bicycle",
    "delay",
    "hello",
    "lift",
    "platform",
    "sorry",
    "thank_you",
}


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}
A00_PATTERN = re.compile(r"-a00(?:\b|\s|\(|$)", re.IGNORECASE)
TEST_PATTERN = re.compile(r"-test(?:\b|-|\s|\(|$)", re.IGNORECASE)


def filename_role(path: Path) -> str:
    stem = path.stem.lower()
    if TEST_PATTERN.search(stem):
        return "test"
    if A00_PATTERN.search(stem):
        return "train_val"
    return "ignored_name_format"


def build_clean_dataset(raw_dir: Path, out_dir: Path, manifest_path: Path) -> None:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    kept_counts: dict[str, int] = defaultdict(int)
    excluded_counts: dict[str, int] = defaultdict(int)
    entries = []

    for label_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        label = label_dir.name
        target_dir = out_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)

        for video_path in sorted(p for p in label_dir.iterdir() if p.is_file()):
            if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            role = filename_role(video_path)
            source_hint = "nzsl"
            kept = role in {"train_val", "test"}

            entry = {
                "label": label,
                "file": video_path.name,
                "source_hint": source_hint,
                "kept": kept,
                "reason": "" if kept else "filename_does_not_match_a00_or_test_pattern",
                "non_mutual_label": label in NON_MUTUAL_LABELS,
                "split_hint": role,
            }

            if kept:
                target_path = target_dir / video_path.name
                if target_path.exists():
                    target_path.chmod(stat.S_IWRITE)
                    target_path.unlink()
                shutil.copy2(video_path, target_path)
                kept_counts[label] += 1
            else:
                excluded_counts[label] += 1

            entries.append(entry)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "non_mutual_labels": sorted(NON_MUTUAL_LABELS),
        "total_kept": int(sum(kept_counts.values())),
        "total_excluded": int(sum(excluded_counts.values())),
        "kept_counts": dict(sorted(kept_counts.items())),
        "excluded_counts": dict(sorted(excluded_counts.items())),
        "entries": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, default=Path("starter_nzsl/data/raw"))
    parser.add_argument("--out_dir", type=Path, default=Path("starter_nzsl/data/raw_cleaned"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("starter_nzsl/reports_analysis/cleaning_manifest.json"),
    )
    args = parser.parse_args()
    build_clean_dataset(args.raw_dir, args.out_dir, args.manifest)


if __name__ == "__main__":
    main()
