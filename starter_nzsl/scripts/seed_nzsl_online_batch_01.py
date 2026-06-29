from __future__ import annotations

import argparse
import csv
import html
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable


STARTER_DIR = Path(__file__).resolve().parents[1]
METADATA_DIR = STARTER_DIR / "data" / "metadata"
USER_AGENT = "Mozilla/5.0"


CANDIDATE_MAP: dict[str, list[dict[str, str | int]]] = {
    "bus": [
        {"sign_id": 3587, "status": "seed", "rationale": "Exact match: bus, truck"},
        {"sign_id": 2823, "status": "candidate", "rationale": "Subtype: trolley bus"},
        {"sign_id": 4216, "status": "candidate", "rationale": "Subtype: shuttle bus, van"},
    ],
    "train": [
        {"sign_id": 5934, "status": "seed", "rationale": "Exact match: train"},
        {"sign_id": 2827, "status": "seed", "rationale": "Exact match: train"},
        {"sign_id": 2926, "status": "seed", "rationale": "Exact match: train"},
        {"sign_id": 3592, "status": "seed", "rationale": "Exact match: train"},
    ],
    "ferry": [
        {"sign_id": 859, "status": "candidate", "rationale": "Semantic proxy only: boat, sail"},
    ],
    "taxi": [
        {"sign_id": 3810, "status": "seed", "rationale": "Exact match: taxi"},
    ],
    "car": [
        {"sign_id": 3462, "status": "seed", "rationale": "Exact match: car, drive"},
        {"sign_id": 8001, "status": "reject", "rationale": "Different concept: cable car"},
        {"sign_id": 3386, "status": "reject", "rationale": "Different concept: car body"},
        {"sign_id": 8970, "status": "reject", "rationale": "Different concept: park (car)"},
        {"sign_id": 603, "status": "reject", "rationale": "Different concept: park (car)"},
    ],
    "bicycle": [
        {"sign_id": 7050, "status": "seed", "rationale": "Exact match: bicycle, ride a bicycle"},
        {"sign_id": 3521, "status": "seed", "rationale": "Exact match: bicycle, ride a bicycle"},
    ],
    "walk": [
        {"sign_id": 2883, "status": "seed", "rationale": "Exact match: walk"},
        {"sign_id": 4993, "status": "seed", "rationale": "Exact match: walk"},
        {"sign_id": 2942, "status": "candidate", "rationale": "Related but narrower: walk across"},
    ],
    "wheelchair": [
        {"sign_id": 3972, "status": "seed", "rationale": "Exact match: wheelchair"},
        {"sign_id": 5990, "status": "seed", "rationale": "Exact match: wheelchair"},
    ],
    "ramp": [
        {"sign_id": 6140, "status": "seed", "rationale": "Exact match: ramp, (go) uphill"},
    ],
    "lift": [
        {"sign_id": 934, "status": "seed", "rationale": "Exact match: lift"},
        {"sign_id": 4694, "status": "seed", "rationale": "Exact match: lift"},
        {"sign_id": 508, "status": "seed", "rationale": "Exact match: lift"},
        {"sign_id": 234, "status": "seed", "rationale": "Exact match: lift"},
    ],
    "station": [
        {"sign_id": 2916, "status": "seed", "rationale": "Exact match: railway station"},
        {"sign_id": 845, "status": "reject", "rationale": "Ambiguous and broader: house, station"},
    ],
    "platform": [
        {"sign_id": 6002, "status": "seed", "rationale": "Exact match: platform"},
    ],
    "stop": [
        {"sign_id": 425, "status": "seed", "rationale": "Exact match: stop"},
        {"sign_id": 5971, "status": "seed", "rationale": "Exact match: stop"},
        {"sign_id": 6376, "status": "seed", "rationale": "Exact match for requested gloss stop / bus stop"},
        {"sign_id": 3677, "status": "reject", "rationale": "Different function: stop it!"},
    ],
    "help": [
        {"sign_id": 1123, "status": "seed", "rationale": "Exact match: help"},
        {"sign_id": 5091, "status": "seed", "rationale": "Exact match: help"},
        {"sign_id": 384, "status": "seed", "rationale": "Exact match: help"},
    ],
    "hello": [
        {"sign_id": 6351, "status": "seed", "rationale": "Exact match: kia ora"},
        {"sign_id": 1301, "status": "candidate", "rationale": "Related but broader: goodbye, hello"},
        {"sign_id": 5243, "status": "candidate", "rationale": "Related but broader: goodbye, hello"},
        {"sign_id": 321, "status": "candidate", "rationale": "Related but broader: hello, salute"},
    ],
    "thank_you": [
        {"sign_id": 1015, "status": "seed", "rationale": "Exact match: thank"},
    ],
}


LABEL_META = {
    "bus": ("bus / pahi", "Bus", "Pahi"),
    "train": ("train / tereina", "Train", "Tereina"),
    "ferry": ("ferry / waka whakawhiti", "Ferry", "Waka whakawhiti"),
    "taxi": ("taxi / tekihi", "Taxi", "Tekihi"),
    "car": ("car / motoka", "Car", "Motoka"),
    "bicycle": ("bicycle / paihikara", "Bicycle", "Paihikara"),
    "walk": ("walk / hikoi", "Walk", "Hikoi"),
    "wheelchair": ("wheelchair / turu wira", "Wheelchair", "Turu wira"),
    "ramp": ("ramp / arawhata", "Ramp", "Arawhata"),
    "lift": ("lift / hiki", "Lift", "Hiki"),
    "station": ("station / teihana", "Station", "Teihana"),
    "platform": ("platform / papa waka", "Platform", "Papa waka"),
    "stop": ("stop / tauranga pahi", "Stop / Bus stop", "Tauranga pahi"),
    "help": ("help / awhina", "Help", "Awhina"),
    "hello": ("hello / kia ora", "Hello", "Kia ora"),
    "thank_you": ("thank you / mihi", "Thank you", "Mihi"),
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


def search_term(label: str) -> str:
    if label == "hello":
        return "kia ora"
    if label == "thank_you":
        return "thank you"
    return label


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed batch-01 NZSL Online candidate and sample manifests.")
    ap.add_argument("--metadata_dir", type=Path, default=METADATA_DIR)
    args = ap.parse_args()

    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    candidates_csv = args.metadata_dir / "batch_01_nzsl_online_candidates.csv"
    samples_csv = args.metadata_dir / "batch_01_nzsl_online_seed_samples.csv"

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
                    "search_term": search_term(label),
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
                        "split_notes": "batch_01_seed",
                        "remote_video_url": video_url,
                    }
                )

    candidate_fields = [
        "label",
        "bilingual_gloss",
        "english",
        "maori",
        "search_term",
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
