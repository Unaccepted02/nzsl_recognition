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
    "coach": [
        {"sign_id": 744, "status": "seed", "rationale": "Exact match: coach"},
        {"sign_id": 4216, "status": "candidate", "rationale": "Related transport subtype: shuttle bus, van"},
    ],
    "truck": [
        {"sign_id": 8995, "status": "seed", "rationale": "Exact match: truck"},
        {"sign_id": 3587, "status": "candidate", "rationale": "Related broader gloss: bus, truck"},
    ],
    "ticket": [
        {"sign_id": 4472, "status": "seed", "rationale": "Exact match: ticket"},
        {"sign_id": 4480, "status": "candidate", "rationale": "Subtype: clip ticket"},
    ],
    "pay": [
        {"sign_id": 3863, "status": "seed", "rationale": "Exact match: pay"},
        {"sign_id": 3850, "status": "seed", "rationale": "Exact match: pay"},
        {"sign_id": 8154, "status": "seed", "rationale": "Exact match: pay"},
        {"sign_id": 282, "status": "reject", "rationale": "Different meaning: pay attention"},
    ],
    "where": [
        {"sign_id": 1544, "status": "seed", "rationale": "Exact match: where"},
        {"sign_id": 4996, "status": "candidate", "rationale": "Broader gloss: what, where, why"},
    ],
    "when": [
        {"sign_id": 1338, "status": "seed", "rationale": "Exact match: when"},
    ],
    "late": [
        {"sign_id": 2758, "status": "seed", "rationale": "Exact match: late"},
        {"sign_id": 419, "status": "seed", "rationale": "Exact match: late"},
        {"sign_id": 2757, "status": "seed", "rationale": "Exact match: late"},
        {"sign_id": 2393, "status": "candidate", "rationale": "Different emphasis: too late"},
        {"sign_id": 5811, "status": "candidate", "rationale": "Related concept: delay"},
    ],
    "wait": [
        {"sign_id": 3518, "status": "seed", "rationale": "Exact match: wait"},
        {"sign_id": 909, "status": "seed", "rationale": "Exact match: wait"},
        {"sign_id": 404, "status": "seed", "rationale": "Exact match: wait"},
        {"sign_id": 7216, "status": "candidate", "rationale": "Narrower phrase: wait a minute"},
        {"sign_id": 4213, "status": "candidate", "rationale": "Related phrase: hold on"},
    ],
    "go": [
        {"sign_id": 2582, "status": "seed", "rationale": "Exact match: go"},
        {"sign_id": 924, "status": "seed", "rationale": "Exact match: go"},
        {"sign_id": 1126, "status": "candidate", "rationale": "Related phrase: go ahead, move on"},
        {"sign_id": 1197, "status": "candidate", "rationale": "Related phrase: go ahead"},
        {"sign_id": 2410, "status": "candidate", "rationale": "Related phrase: go to"},
    ],
    "arrive": [
        {"sign_id": 1031, "status": "seed", "rationale": "Exact match: arrive"},
    ],
    "leave": [
        {"sign_id": 432, "status": "seed", "rationale": "Exact match: leave"},
        {"sign_id": 777, "status": "candidate", "rationale": "Broader gloss: leave, let go"},
        {"sign_id": 1175, "status": "candidate", "rationale": "Related concept: exit"},
        {"sign_id": 5915, "status": "candidate", "rationale": "Related transport phrase: drop off"},
    ],
    "please": [
        {"sign_id": 8974, "status": "seed", "rationale": "Exact match: please"},
        {"sign_id": 385, "status": "seed", "rationale": "Exact match: please"},
    ],
    "yes": [
        {"sign_id": 3236, "status": "seed", "rationale": "Exact match: yes"},
        {"sign_id": 5337, "status": "candidate", "rationale": "Phrase: yes or no"},
    ],
    "no": [
        {"sign_id": 1464, "status": "seed", "rationale": "Exact match: no"},
        {"sign_id": 3272, "status": "seed", "rationale": "Exact match: no"},
        {"sign_id": 5337, "status": "candidate", "rationale": "Phrase: yes or no"},
        {"sign_id": 5089, "status": "candidate", "rationale": "Related phrase: no more"},
    ],
    "sorry": [
        {"sign_id": 1708, "status": "seed", "rationale": "Exact match: sorry"},
        {"sign_id": 3724, "status": "seed", "rationale": "Exact match: sorry"},
        {"sign_id": 1471, "status": "seed", "rationale": "Exact match: sorry"},
        {"sign_id": 769, "status": "candidate", "rationale": "Related concept: sympathise"},
    ],
    "bathroom": [
        {"sign_id": 1439, "status": "seed", "rationale": "Exact match: bathroom"},
        {"sign_id": 1440, "status": "reject", "rationale": "Different concept: bathroom scales"},
        {"sign_id": 3480, "status": "reject", "rationale": "Different concept: bath, bathe"},
        {"sign_id": 6259, "status": "reject", "rationale": "Different concept: bath, bathe"},
    ],
}


LABEL_META = {
    "coach": ("coach / pahi nui", "Coach", "Pahi nui"),
    "truck": ("truck / taraka", "Truck", "Taraka"),
    "ticket": ("ticket / tikiti", "Ticket", "Tikiti"),
    "pay": ("pay / utu", "Pay", "Utu"),
    "where": ("where / kei hea", "Where", "Kei hea"),
    "when": ("when / awhea", "When", "Awhea"),
    "late": ("late / tomuri", "Late", "Tomuri"),
    "wait": ("wait / tatari", "Wait", "Tatari"),
    "go": ("go / haere", "Go", "Haere"),
    "arrive": ("arrive / tae mai", "Arrive", "Tae mai"),
    "leave": ("leave / wehe", "Leave", "Wehe"),
    "please": ("please / tena koa", "Please", "Tena koa"),
    "yes": ("yes / ae", "Yes", "Ae"),
    "no": ("no / kaore", "No", "Kaore"),
    "sorry": ("sorry / aroha mai", "Sorry", "Aroha mai"),
    "bathroom": ("bathroom / wharepaku", "Bathroom", "Wharepaku"),
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
    ap = argparse.ArgumentParser(description="Seed batch-02 NZSL Online candidate and sample manifests.")
    ap.add_argument("--metadata_dir", type=Path, default=METADATA_DIR)
    args = ap.parse_args()

    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    candidates_csv = args.metadata_dir / "batch_02_nzsl_online_candidates.csv"
    samples_csv = args.metadata_dir / "batch_02_nzsl_online_seed_samples.csv"

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
                        "split_notes": "batch_02_seed",
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
