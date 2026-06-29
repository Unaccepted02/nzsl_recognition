from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


STARTER_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = STARTER_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.utils import ensure_dir, write_json  # noqa: E402


def count_vocab_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def main() -> None:
    ap = argparse.ArgumentParser(description="Initialize a clean NZSL-only starter workspace.")
    ap.add_argument("--starter_dir", type=Path, default=STARTER_DIR)
    args = ap.parse_args()

    starter_dir = args.starter_dir
    raw_dir = starter_dir / "data" / "raw"
    processed_dir = starter_dir / "data" / "processed"
    metadata_dir = starter_dir / "data" / "metadata"
    models_dir = starter_dir / "models"
    reports_dir = starter_dir / "reports"

    for path in [raw_dir, processed_dir, metadata_dir, models_dir, reports_dir]:
        ensure_dir(path)

    summary = {
        "workspace": str(starter_dir),
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "metadata_dir": str(metadata_dir),
        "models_dir": str(models_dir),
        "reports_dir": str(reports_dir),
        "source_registry": str(starter_dir / "config" / "verified_nzsl_sources.csv"),
        "vocabulary_file": str(starter_dir / "config" / "verified_nzsl_vocab_30plus.csv"),
        "vocabulary_size": count_vocab_rows(starter_dir / "config" / "verified_nzsl_vocab_30plus.csv"),
        "sample_template": str(metadata_dir / "verified_samples_template.csv"),
        "status": "ready_for_verified_nzsl_ingest"
    }
    write_json(reports_dir / "workspace_status.json", summary)
    print(f"Wrote {reports_dir / 'workspace_status.json'}")


if __name__ == "__main__":
    main()
