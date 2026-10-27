"""Prepare the 100-candidate pilot. No model loading, APIs, or training."""

import argparse
import hashlib
import json
import random
import sys
import tarfile
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jais_guard.annotations import (
    ROOT,
    benchmark_references,
    import_candidates,
    now,
    quality_issues,
)

PILOT = ROOT / "data/pilot_v1"
RAW = ROOT / "data/raw/pilot_v1"
SOURCES = {
    "tydi": (
        "google-research-datasets/tydiqa",
        "da78f23f9119363459acbaf46bf89426ff26c259",
        "Apache-2.0",
        "TyDi QA contributors / Google Research",
    ),
    "aya": (
        "CohereLabs/aya_dataset",
        "f9ea04583f02a8f86404ff6c58bf75fe637df8a2",
        "Apache-2.0",
        "Aya Open Science contributors / Cohere Labs",
    ),
    "massive": (
        "AmazonScience/massive",
        "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740",
        "CC-BY-4.0",
        "Amazon MASSIVE contributors",
    ),
}


def download(url: str, path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        temporary.replace(path)


def fetch() -> None:
    for name, (repo, revision, _, _) in SOURCES.items():
        download(
            f"https://huggingface.co/datasets/{repo}/raw/{revision}/README.md",
            RAW / f"{name}_README.md",
        )
        download(
            f"https://huggingface.co/api/datasets/{repo}/revision/{revision}",
            RAW / f"{name}_info.json",
        )
    download(
        f"https://huggingface.co/datasets/{SOURCES['aya'][0]}/resolve/"
        f"{SOURCES['aya'][1]}/data/train-00000-of-00001.parquet",
        RAW / "aya_train.parquet",
    )
    download(
        "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz",
        RAW / "massive.tar.gz",
    )
    for offset, filename in [(0, "tydi_rows.json"), (100, "tydi_rows_100.json")]:
        query = urllib.parse.urlencode(
            {
                "dataset": SOURCES["tydi"][0],
                "config": "primary_task",
                "split": "train",
                "offset": offset,
                "length": 100,
            }
        )
        download("https://datasets-server.huggingface.co/rows?" + query, RAW / filename)
    # The viewer does not support revision pinning: reject silently changed source revisions.
    request = urllib.request.urlopen(
        f"https://huggingface.co/api/datasets/{SOURCES['tydi'][0]}", timeout=30
    )
    if json.load(request)["sha"] != SOURCES["tydi"][1]:
        raise ValueError("TyDi viewer revision changed; review source selection before rebuilding")


def record(text, label, method, subtype, dialect, domain, family, provenance, generation=None):
    return {
        "id": "P-" + hashlib.sha256((method + family + text).encode()).hexdigest()[:12],
        "input_text": text,
        "partition": "train",
        "intended_label": label,
        "source_method": method,
        "subtype": subtype,
        "dialect": dialect,
        "dialect_status": "proposed_not_human_confirmed",
        "domain": domain,
        "family_id": family,
        "provenance": provenance,
        "generation": generation or {},
        "created_at": now(),
    }


def public_provenance(name, source_id, upstream_split="train"):
    repo, revision, license_name, attribution = SOURCES[name]
    return {
        "source": repo,
        "url": f"https://huggingface.co/datasets/{repo}",
        "source_id": str(source_id),
        "revision": revision,
        "upstream_split": upstream_split,
        "license": license_name,
        "attribution": attribution,
        "rights_status": "verified",
        "rights_evidence": f"data/pilot_v1/sources/{name}_README.md",
        "modifications": "Input selected verbatim; answers, contexts and demographics omitted.",
    }


def public_candidates():
    import pyarrow.parquet as pq

    result = []
    previews = [
        json.loads((RAW / name).read_text()) for name in ("tydi_rows.json", "tydi_rows_100.json")
    ]
    selected = {18, 22, 26, 33, 52, 59, 66, 72, 74, 92, 95, 99, 117, 164, 199}
    for page in previews:
        for entry in page["rows"]:
            row = entry["row"]
            if entry["row_idx"] in selected:
                if row["language"] != "arabic":
                    raise ValueError("TyDi source snapshot differs from reviewed selection")
                p = public_provenance("tydi", f"primary_task:train:row-{entry['row_idx']}")
                p["original_document_url"] = row["document_url"]
                result.append(
                    record(
                        row["question_text"],
                        "VALID",
                        "public",
                        "question",
                        "MSA",
                        "information_seeking",
                        "tydi-document:" + row["document_url"],
                        p,
                    )
                )
    aya = pq.read_table(RAW / "aya_train.parquet").to_pylist()
    indices = {
        2759: "Egyptian",
        5380: "Egyptian",
        5749: "Egyptian",
        13369: "Egyptian",
        1662: "Gulf",
        4322: "Gulf",
        17600: "Levantine",
        51646: "Levantine",
        349: "MSA",
        2513: "Moroccan",
    }
    for index, dialect in indices.items():
        row = aya[index]
        if row["annotation_type"] != "original-annotations":
            raise ValueError("Only original human Aya annotations are admitted")
        p = public_provenance("aya", f"train:row-{index}")
        p.update(
            {
                "annotation_type": row["annotation_type"],
                "publisher_language_code": row["language_code"],
                "author_group": row["user_id"],
            }
        )
        result.append(
            record(
                row["inputs"],
                "VALID",
                "public",
                "human_instruction",
                dialect,
                "everyday",
                "aya-author:" + row["user_id"],
                p,
            )
        )
    with tarfile.open(RAW / "massive.tar.gz") as archive:
        rows = [json.loads(line) for line in archive.extractfile("1.1/data/ar-SA.jsonl")]
        license_text = archive.extractfile("1.1/LICENSE").read()
    for row in rows:
        if row["id"] in {"1", "96", "4444", "1137", "69"}:
            if row["partition"] != "train":
                raise ValueError("Unexpected MASSIVE upstream split")
            p = public_provenance("massive", f"1.1:ar-SA:{row['id']}")
            p.update(
                {
                    "origin": "human_localized_from_English",
                    "release": "1.1",
                    "release_sha256": hashlib.sha256(
                        (RAW / "massive.tar.gz").read_bytes()
                    ).hexdigest(),
                    "rights_evidence": "data/pilot_v1/sources/massive_LICENSE.txt",
                }
            )
            result.append(
                record(
                    row["utt"],
                    "VALID",
                    "public",
                    "localized_request",
                    "Arabic-ar-SA-unverified",
                    row["scenario"],
                    "massive-original:" + row["id"],
                    p,
                )
            )
    if len(result) != 30:
        raise ValueError(f"Expected 30 public records; found {len(result)}")
    sources_dir = PILOT / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    (sources_dir / "massive_LICENSE.txt").write_bytes(license_text)
    for name in SOURCES:
        (sources_dir / f"{name}_README.md").write_bytes((RAW / f"{name}_README.md").read_bytes())
    manifest = {
        name: {
            "repository": repo,
            "revision": revision,
            "license": license_name,
            "attribution": attribution,
            "rights_status": "verified for selected inputs",
            "upstream_rights_evidence": f"{name}_README.md",
        }
        for name, (repo, revision, license_name, attribution) in SOURCES.items()
    }
    manifest["limitations"] = (
        "TyDi is elicited Arabic; Aya originals are human authored, not production logs; "
        "MASSIVE is localized. Dialect hints need review. Conditional Naous and ArBanking77 "
        "sources are not included. Public redistribution requires compatibility review."
    )
    (sources_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return result


def codex_candidates():
    result = []
    for path in sorted((PILOT / "generation").glob("*.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        for candidate in batch["candidates"]:
            p = {
                "source": "Codex pilot authoring",
                "source_id": batch["batch_id"],
                "license": "generated-content-publication-review-pending",
                "attribution": "Codex / OpenAI; human annotation pending",
                "rights_status": "publication_review_pending",
                "rights_evidence": str(path.relative_to(ROOT)),
            }
            result.append(
                record(
                    candidate["input_text"],
                    candidate["intended_label"],
                    "codex",
                    candidate["subtype"],
                    candidate["dialect"],
                    candidate["domain"],
                    candidate["family_id"],
                    p,
                    {
                        "batch_id": batch["batch_id"],
                        "instructions_file": str(path.relative_to(ROOT)),
                        "model_metadata": batch["available_model_metadata"],
                    },
                )
            )
    return result


def python_candidates(parents):
    rng = random.Random(42)
    result = []
    for i, parent in enumerate(parents[:5]):
        text = parent["input_text"]
        recipes = [
            "remove_hamza",
            "join_first_space",
            "append_limited_noise",
            "repeat_one_letter",
            "drop_one_letter",
        ]
        recipe = recipes[i]
        if i == 0:
            text = text.replace("أ", "ا").replace("إ", "ا")
        elif i == 1:
            text = text.replace(" ", "", 1)
        elif i == 2:
            text += " جج"
        else:
            positions = [j for j, char in enumerate(text) if char.isalpha()]
            j = rng.choice(positions)
            text = text[:j] + (text[j] * 2 if i == 3 else "") + text[j + 1 :]
        p = {
            **parent["provenance"],
            "modifications": recipe,
            "upstream_source_id": parent["provenance"]["source_id"],
            "parent_id": parent["id"],
        }
        result.append(
            record(
                text,
                "VALID",
                "python",
                "meaning_preserving_noise",
                parent["dialect"],
                parent["domain"],
                parent["family_id"],
                p,
                {"seed": 42, "recipe": recipe, "parent_id": parent["id"]},
            )
        )
    keyboard = "ضصثقفغعهخحجدشسيبلاتنمكطئءؤرلاىةوزظ"
    for i in range(10):
        parent = None
        if i < 4:
            start = rng.randrange(0, 12)
            text = keyboard[start : start + rng.randrange(12, 20)]
            recipe = "contiguous_keyboard_run"
        elif i < 7:
            text = rng.choice(keyboard) * rng.randrange(12, 24)
            recipe = "repeated_arabic_character"
        else:
            parent = parents[i - 7]
            words = parent["input_text"].split()
            shuffled = []
            for word in words:
                chars = list(word)
                rng.shuffle(chars)
                shuffled.append("".join(chars))
            text = " ".join(shuffled)
            recipe = "seeded_word_character_shuffle"
        p = {
            "source": "JaisGuard deterministic Python recipes",
            "source_id": recipe,
            "license": "project-original-content",
            "attribution": "JaisGuard contributors",
            "rights_status": "publication_review_pending",
            "rights_evidence": "scripts/prepare_candidates.py",
        }
        if parent:
            p = {**parent["provenance"], "modifications": recipe, "parent_id": parent["id"]}
        result.append(
            record(
                text,
                "GIBBERISH",
                "python",
                recipe,
                "none",
                "noise",
                parent["family_id"] if parent else f"python-keyboard-{i}",
                p,
                {"seed": 42, "recipe": recipe, "parent_id": parent["id"] if parent else None},
            )
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Download admitted source snapshots")
    parser.add_argument(
        "--build", action="store_true", help="Create pilot once from source snapshots"
    )
    parser.add_argument(
        "--import", dest="do_import", action="store_true", help="Import into SQLite"
    )
    args = parser.parse_args()
    path = PILOT / "candidates.jsonl"
    if args.fetch:
        fetch()
    if args.build:
        if path.exists():
            raise FileExistsError("Pilot exists; create a new version instead of overwriting it")
        public = public_candidates()
        rows = public + codex_candidates() + python_candidates(public)
        counts = Counter((r["source_method"], r["intended_label"]) for r in rows)
        if counts != {
            ("public", "VALID"): 30,
            ("codex", "VALID"): 15,
            ("codex", "GIBBERISH"): 40,
            ("python", "VALID"): 5,
            ("python", "GIBBERISH"): 10,
        }:
            raise ValueError(f"Pilot distribution mismatch: {counts}")
        rows.sort(key=lambda r: r["id"])
        issues = quality_issues(rows, benchmark_references())
        if any(i["kind"] != "near_duplicate" or i["cross_partition"] for i in issues):
            raise ValueError(f"Pilot overlap blockers: {issues}")
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    issues = quality_issues(rows, benchmark_references())
    audit = {
        "created_at": now(),
        "candidates": len(rows),
        "partition": "train",
        "intended_counts": dict(Counter(r["intended_label"] for r in rows)),
        "sources": dict(Counter(r["source_method"] for r in rows)),
        "quality_issues": issues,
        "human_review": "pending for every candidate",
        "pilot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (PILOT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    if args.do_import:
        print(f"Imported {import_candidates(rows)} candidates")
    print(f"{len(rows)} candidates; {len(issues)} within-training near-duplicate review hints")
    print("No human labels assigned. Stop here for blind pilot review.")


if __name__ == "__main__":
    main()
