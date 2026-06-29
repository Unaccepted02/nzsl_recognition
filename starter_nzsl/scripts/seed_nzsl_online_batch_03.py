from __future__ import annotations

import argparse
import csv
import html
import re
import urllib.request
from pathlib import Path
from typing import Iterable


STARTER_DIR = Path(__file__).resolve().parents[1]
METADATA_DIR = STARTER_DIR / "data" / "metadata"
USER_AGENT = "Mozilla/5.0"


CANDIDATE_MAP: dict[str, list[dict[str, str | int]]] = {
    "boat": [
        {"sign_id": 859, "status": "seed", "rationale": "Exact match available as broader gloss: boat, sail"},
        {"sign_id": 6295, "status": "reject", "rationale": "Different action: row"},
    ],
    "bridge": [
        {"sign_id": 5424, "status": "seed", "rationale": "Exact match: bridge"},
        {"sign_id": 4483, "status": "seed", "rationale": "Exact match: bridge"},
    ],
    "path": [
        {"sign_id": 2804, "status": "seed", "rationale": "Exact match available as broader gloss: path, road"},
    ],
    "crossing": [
        {"sign_id": 4538, "status": "seed", "rationale": "Exact transport match: pedestrian crossing"},
        {"sign_id": 1974, "status": "candidate", "rationale": "Related infrastructure term: level crossing barrier"},
    ],
    "motorbike": [
        {"sign_id": 3358, "status": "seed", "rationale": "Exact match: motorbike"},
    ],
    "ferry": [
        {"sign_id": 859, "status": "candidate", "rationale": "No exact ferry entry found; boat, sail remains proxy only"},
    ],
}


LABEL_META = {
    "boat": ("boat / waka", "Boat", "Waka"),
    "bridge": ("bridge / piriti", "Bridge", "Piriti"),
    "path": ("path / ara", "Path", "Ara"),
    "crossing": ("crossing / ara whakawhiti", "Crossing", "Ara whakawhiti"),
    "motorbike": ("motorbike / motopaika", "Motorbike", "Motopaika"),
    "ferry": ("ferry / waka whakawhiti", "Ferry", "Waka whakawhiti"),
}


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8", "ignore")


def clean_text(value: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(value)).split())


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def parse_sign_page(sign_id: int) -> tuple[str, list[str]]:
    url = f"https://www.nzsl.nz/signs/{sign_id}"
    text = fetch_text(url)
    title_match = re.search(r"<h1[^>]*>(.*?)</h1>", text, flags=re.I | re.S)
    title = clean_text(title_match.group(1) if title_match else "")
    mp4s = unique_preserve_order(re.findall(r"https://[^\"']+\.mp4[^\"']*", text))
    return title, mp4s


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed batch-03 NZSL Online candidate and sample manifests.")
    ap.add_argument("--metadata_dir", type=Path, default=METADATA_DIR)
    args = ap.parse_args()

    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    candidates_csv = args.metadata_dir / "batch_03_nzsl_online_candidates.csv"
    samples_csv = args.metadata_dir / "batch_03_nzsl_online_seed_samples.csv"

    candidate_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []

    for label, entries in CANDIDATE_MAP.items():
        bilingual_gloss, english, maori = LABEL_META[label]
        for entry in entries:
            sign_id = int(entry["sign_id"])
            source_url = f"https://www.nzsl.nz/signs/{sign_id}"
            title, mp4s = parse_sign_page(sign_id)
            candidate_rows.append(
                {
                    "label": label,
                    "bilingual_gloss": bilingual_gloss,
                    "english": english,
                    "maori": maori,
                    "source_id": "nzsl_online",
                    "source_url": source_url,
                    "sign_id": sign_id,
                    "sign_title": title,
                    "candidate_status": entry["status"],
                    "candidate_rationale": entry["rationale"],
                    "video_count": len(mp4s),
                }
            )
            if entry["status"] != "seed":
                continue
            for idx, video_url in enumerate(mp4s, start=1):
                suffix = video_url.rsplit("/", 1)[-1]
                sample_rows.append(
                    {
                        "sample_id": f"nzsl_online_{label}_{sign_id}_{idx:02d}",
                        "label": label,
                        "bilingual_gloss": bilingual_gloss,
                        "english": english,
                        "maori": maori,
                        "source_id": "nzsl_online",
                        "source_url": source_url,
                        "license": "CC BY-NC-SA 4.0",
                        "signer_id": "",
                        "variant_id": f"{sign_id}_{idx:02d}",
                        "recorded_date": "",
                        "local_relpath": f"starter_nzsl/data/raw/{label}/{suffix}",
                        "verified_nzsl": "true",
                        "verified_by": "codex_seed",
                        "verification_notes": f"Official NZSL Online entry '{title}'",
                        "split_notes": "batch_03_seed",
                        "remote_video_url": video_url,
                    }
                )

    candidate_fields = [
        "label",
        "bilingual_gloss",
        "english",
        "maori",
        "source_id",
        "source_url",
        "sign_id",
        "sign_title",
        "candidate_status",
        "candidate_rationale",
        "video_count",
    ]
    with candidates_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_fields)
        writer.writeheader()
        writer.writerows(candidate_rows)

    sample_fields = [
        "sample_id",
        "label",
        "bilingual_gloss",
        "english",
        "maori",
        "source_id",
        "source_url",
        "license",
        "signer_id",
        "variant_id",
        "recorded_date",
        "local_relpath",
        "verified_nzsl",
        "verified_by",
        "verification_notes",
        "split_notes",
        "remote_video_url",
    ]
    with samples_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fields)
        writer.writeheader()
        writer.writerows(sample_rows)

    print(f"Wrote {candidates_csv}")
    print(f"Wrote {samples_csv}")
    print(f"Seed samples: {len(sample_rows)}")


if __name__ == "__main__":
    main()
