from __future__ import annotations

import argparse
import csv
import urllib.request
from pathlib import Path


STARTER_DIR = Path(__file__).resolve().parents[1]
USER_AGENT = "Mozilla/5.0"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download NZSL seed samples listed in a metadata CSV.")
    ap.add_argument(
        "--seed_csv",
        type=Path,
        default=STARTER_DIR / "data" / "metadata" / "batch_01_nzsl_online_seed_samples.csv",
    )
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    with args.seed_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    downloaded = 0
    skipped = 0
    for row in rows:
        relpath = row["local_relpath"].replace("/", "\\")
        target = STARTER_DIR.parent / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and target.exists():
            skipped += 1
            continue

        req = urllib.request.Request(row["remote_video_url"], headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as response:
            target.write_bytes(response.read())
        downloaded += 1
        print(f"downloaded {row['sample_id']} -> {target}")

    print(f"Downloaded: {downloaded}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
