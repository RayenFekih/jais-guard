"""Prepare an explicit training-candidate milestone. No models, paid APIs, or training."""

import argparse
import hashlib
import json
import random
import sqlite3
import sys
import tarfile
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jais_guard.annotations import (
    DEFAULT_DB,
    PROVISIONAL_POLICY_VERSION,
    ROOT,
    benchmark_references,
    import_candidates,
    migration_preview,
    now,
    preview_import,
    quality_issues,
)
from jais_guard.audit_policy import boundary, digest

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
    for offset, filename in [
        (0, "tydi_rows.json"),
        (100, "tydi_rows_100.json"),
        (200, "tydi_rows_200.json"),
    ]:
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


def record(
    text,
    label,
    method,
    subtype,
    dialect,
    domain,
    family,
    provenance,
    generation=None,
    difficulty=None,
):
    result = {
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
    if difficulty is not None:
        result.update(difficulty=difficulty, difficulty_status="proposed_not_human_confirmed")
    return result


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


def codex_candidates(directory=PILOT):
    result = []
    for path in sorted((directory / "generation").glob("*.json")):
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
                    difficulty=candidate.get("difficulty"),
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


def live_candidates():
    """Read current candidates from SQLite, never from a stale annotation export."""
    with sqlite3.connect(f"file:{DEFAULT_DB.resolve()}?mode=ro", uri=True) as db:
        return [json.loads(row[0]) for row in db.execute("SELECT payload FROM candidates")]


def milestone_candidates(directory):
    import pyarrow.parquet as pq

    plan = json.loads((directory / "source_selection.json").read_text())
    public = []
    for page in sorted(RAW.glob("tydi_rows*.json")):
        for entry in json.loads(page.read_text())["rows"]:
            row = entry["row"]
            if entry["row_idx"] in plan["tydi_row_indices"]:
                if row["language"] != "arabic":
                    raise ValueError("Selected TyDi record is not Arabic")
                p = public_provenance("tydi", f"primary_task:train:row-{entry['row_idx']}")
                p["original_document_url"] = row["document_url"]
                public.append(
                    record(
                        row["question_text"],
                        "VALID",
                        "public",
                        "question",
                        "MSA",
                        "information_seeking",
                        "tydi-document:" + row["document_url"],
                        p,
                        difficulty="easy",
                    )
                )
    aya = pq.read_table(RAW / "aya_train.parquet").to_pylist()
    for index, dialect in plan["aya_rows"].items():
        row = aya[int(index)]
        if row["annotation_type"] != "original-annotations":
            raise ValueError("Only original Aya human annotations are admitted")
        p = public_provenance("aya", f"train:row-{index}")
        p.update(
            annotation_type=row["annotation_type"],
            publisher_language_code=row["language_code"],
            author_group=row["user_id"],
        )
        public.append(
            record(
                row["inputs"],
                "VALID",
                "public",
                "human_instruction",
                dialect,
                "everyday",
                "aya-author:" + row["user_id"],
                p,
                difficulty="easy",
            )
        )
    with tarfile.open(RAW / "massive.tar.gz") as archive:
        massive = [json.loads(line) for line in archive.extractfile("1.1/data/ar-SA.jsonl")]
    release_hash = hashlib.sha256((RAW / "massive.tar.gz").read_bytes()).hexdigest()
    for row in massive:
        if row["id"] in plan["massive_ids"]:
            if row["partition"] != "train":
                raise ValueError("Unexpected MASSIVE source split")
            p = public_provenance("massive", f"1.1:ar-SA:{row['id']}")
            p.update(
                origin="human_localized_from_English",
                release="1.1",
                release_sha256=release_hash,
                rights_evidence="data/pilot_v1/sources/massive_LICENSE.txt",
            )
            public.append(
                record(
                    row["utt"],
                    "VALID",
                    "public",
                    "localized_request",
                    "Arabic-ar-SA-unverified",
                    row["scenario"],
                    "massive-original:" + row["id"],
                    p,
                    difficulty="easy",
                )
            )
    if len(public) != 70:
        raise ValueError(f"Expected 70 public records, found {len(public)}")
    rng = random.Random(plan["python_seed"])
    generated = []

    def derived(parent, text, label, subtype, difficulty, recipe):
        p = {**parent["provenance"], "modifications": recipe, "parent_id": parent["id"]}
        return record(
            text,
            label,
            "python",
            subtype,
            parent["dialect"] if label == "VALID" else "none",
            parent["domain"],
            parent["family_id"],
            p,
            {"seed": plan["python_seed"], "recipe": recipe, "parent_id": parent["id"]},
            difficulty=difficulty,
        )

    for i, parent in enumerate(public[:10]):
        text = parent["input_text"]
        recipe = [
            "join_first_space",
            "repeat_one_letter",
            "drop_one_letter",
            "append_limited_noise",
            "join_last_space",
        ][i % 5]
        if recipe == "join_first_space":
            text = text.replace(" ", "", 1)
        elif recipe == "join_last_space":
            position = text.rfind(" ")
            text = text[:position] + text[position + 1 :]
        elif recipe == "append_limited_noise":
            text += " صص"
        else:
            # Internal characters preserve enough of a real input for human assessment.
            positions = [j for j in range(1, len(text) - 1) if text[j].isalpha()]
            position = rng.choice(positions)
            replacement = text[position] * 2 if recipe == "repeat_one_letter" else ""
            text = text[:position] + replacement + text[position + 1 :]
        generated.append(
            derived(parent, text, "VALID", "meaning_preserving_noise", "medium", recipe)
        )
    for parent in public[15:27]:
        # Shuffle characters across the whole input, then restore varied word lengths.
        chars = [c for c in parent["input_text"] if c.isalpha()]
        rng.shuffle(chars)
        words = []
        while chars:
            length = rng.randrange(3, 7)
            words.append("".join(chars[:length]))
            chars = chars[length:]
        generated.append(
            derived(
                parent,
                " ".join(words),
                "GIBBERISH",
                "unrecoverable_scrambling",
                "easy",
                "whole_input_character_shuffle",
            )
        )
    keyboards = ["ضصثقفغعهخحج", "شسيبلاتنمكط", "ئءؤرلاىةوزظ"]
    for i in range(25):
        if i < 15:
            layout = keyboards[i % 3]
            if i % 2:
                layout = layout[::-1]
            start = rng.randrange(0, 4)
            chunk = layout[start : start + rng.randrange(5, 9)]
            other = keyboards[(i + 1) % 3]
            text = chunk + (" " if i % 3 else "") + other[: rng.randrange(4, 8)]
            subtype = "keyboard_noise"
            family = f"m300-keyboard-row-walk-{i % 3}"
            while any(r["input_text"] == text for r in generated):
                text += rng.choice(layout)
        else:
            # Avoid laughter/hesitation fragments and the pilot's repeated characters.
            chars = plan["python_repetition_characters"]
            text = chars[i - 15] * (6 + rng.randrange(0, 18))
            subtype = "repeated_characters"
            family = "m300-repeated-characters"
        p = {
            "source": "JaisGuard deterministic Python recipes",
            "source_id": f"m300:{subtype}:{i}",
            "license": "project-original-content",
            "attribution": "JaisGuard contributors",
            "rights_status": "publication_review_pending",
            "rights_evidence": "scripts/prepare_candidates.py",
        }
        generated.append(
            record(
                text,
                "GIBBERISH",
                "python",
                subtype,
                "none",
                "noise",
                family,
                p,
                {"seed": plan["python_seed"], "recipe": subtype},
                difficulty="easy",
            )
        )
    result = public + codex_candidates(directory) + generated
    actual = Counter(f"{r['source_method']}_{r['intended_label']}" for r in result)
    if actual != plan["planned_counts"]:
        raise ValueError(f"Milestone source distribution mismatch: {actual}")
    difficulties = Counter(r["difficulty"] for r in result if r["intended_label"] == "GIBBERISH")
    if difficulties != plan["gibberish_difficulty_targets"]:
        raise ValueError(f"Gibberish difficulty target mismatch: {difficulties}")
    return result


def expansion_candidates(directory):
    """Rebuild the 400-record collection from inspected source snapshots and saved batches."""
    plan = json.loads((directory / "source_selection.json").read_text())
    sources = [
        json.loads(line) for line in (directory / "selected_sources.jsonl").read_text().splitlines()
    ]
    result = [dict(r) for r in sources if r["source_role"] == "candidate"]
    result.extend(codex_candidates(directory))
    for row in result:
        row.setdefault("collection_batch", row["generation"].get("batch_id"))
    parents = [r for r in sources if r["source_role"] != "candidate"]
    rng = random.Random(plan["python_seed"])

    def add(text, label, subtype, difficulty, recipe, batch, parent=None):
        if parent:
            provenance = {
                **parent["provenance"],
                "parent_id": parent["id"],
                "modifications": recipe,
            }
            dialect, domain, family = parent["dialect"], parent["domain"], parent["family_id"]
        else:
            provenance = {
                "source": "JaisGuard deterministic Python recipes",
                "source_id": batch + ":" + str(len(result)),
                "license": "project-original-content",
                "attribution": "JaisGuard contributors",
                "rights_evidence": "scripts/prepare_candidates.py",
                "rights_status": "publication_review_pending",
            }
            dialect, domain, family = "none", "noise", batch
        row = record(
            text,
            label,
            "python",
            subtype,
            dialect if label == "VALID" else "none",
            domain,
            family,
            provenance,
            {
                "seed": plan["python_seed"],
                "recipe": recipe,
                "batch_id": batch,
                "parent_id": parent["id"] if parent else None,
            },
            difficulty=difficulty,
        )
        row["collection_batch"] = batch
        result.append(row)

    for i, parent in enumerate(
        r for r in parents if r["source_role"] == "valid_transformation_parent"
    ):
        text = parent["input_text"]
        recipe = (
            "join_first_space",
            "repeat_one_letter",
            "drop_one_letter",
            "append_limited_noise",
            "join_last_space",
        )[i % 5]
        if recipe == "join_first_space":
            text = text.replace(" ", "", 1)
        elif recipe == "join_last_space":
            position = text.rfind(" ")
            text = text[:position] + text[position + 1 :]
        elif recipe == "append_limited_noise":
            text += " طظ"
        else:
            positions = [j for j in range(1, len(text) - 1) if text[j].isalpha()]
            position = rng.choice(positions)
            replacement = text[position] * 2 if recipe == "repeat_one_letter" else ""
            text = text[:position] + replacement + text[position + 1 :]
        add(
            text,
            "VALID",
            "meaning_preserving_noise",
            "medium",
            recipe,
            f"m700-python-valid-{i // 5 + 1:02}",
            parent,
        )
    for i, parent in enumerate(r for r in parents if r["source_role"] == "scrambling_parent"):
        words = ["".join(c for c in word if c.isalpha()) for word in parent["input_text"].split()]
        chars = list("".join(words))
        rng.shuffle(chars)
        offset, scrambled = 0, []
        for word in words:
            if word:
                scrambled.append("".join(chars[offset : offset + len(word)]))
                offset += len(word)
        add(
            " ".join(scrambled),
            "GIBBERISH",
            "unrecoverable_scrambling",
            "easy",
            "whole_input_character_shuffle",
            f"m700-python-scramble-{i // 10 + 1:02}",
            parent,
        )
    keyboard = "ضصثقفغعهخحج"
    for i in range(40):
        layout = keyboard if i % 2 else keyboard[::-1]
        start = rng.randrange(4)
        walk = layout[start : start + 8]
        prefix = "".join(rng.choices(keyboard, k=rng.randrange(3, 8)))
        suffix = "".join(rng.choices(keyboard, k=rng.randrange(3, 8)))
        text = prefix + (" " if i % 3 else "") + walk + (" " if i % 4 else "") + suffix
        add(
            text,
            "GIBBERISH",
            "keyboard_noise",
            "easy",
            "keyboard_noise",
            f"m700-python-keyboard-{i // 20 + 1:02}",
        )
    repeated_pairs = set()
    for i in range(16):
        chars = rng.sample("ثظضصقغخجطز", 2)
        while tuple(chars) in repeated_pairs:
            chars = rng.sample("ثظضصقغخجطز", 2)
        repeated_pairs.add(tuple(chars))
        text = chars[0] * rng.randrange(5, 11) + " " + chars[1] * rng.randrange(5, 11)
        add(
            text,
            "GIBBERISH",
            "repeated_characters",
            "easy",
            "two_repeated_character_blocks",
            f"m700-python-repeat-{i // 8 + 1:02}",
        )
    assert len(result) == 400
    assert Counter(r["intended_label"] for r in result) == plan["intended_counts"]
    assert (
        Counter(r["difficulty"] for r in result if r["intended_label"] == "GIBBERISH")
        == plan["gibberish_difficulty_targets"]
    )
    for row in result:
        row["collection_name"] = "milestone700_v1"
    assert not ({r["id"] for r in result} & {p["id"] for p in parents})
    return result


def import_expansion(
    rows, path=DEFAULT_DB, *, policy_version=PROVISIONAL_POLICY_VERSION, batch_id=None
):
    """Keep generation batches separate, even when their source/subtype match."""
    if policy_version == PROVISIONAL_POLICY_VERSION:
        batch_id = batch_id or "collection-v2-" + digest(sorted(r["id"] for r in rows))[:12]
        projection = preview_import(rows, batch_id, 700428, path)
        print(json.dumps(workload_summary(projection, rows), ensure_ascii=False, indent=2))
        return import_candidates(
            rows,
            path,
            provisional_batch=batch_id,
            seed=700428,
            expected_projection=projection["fingerprint"],
        )
    stored = []
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        stored = [json.loads(row[0]) for row in db.execute("SELECT payload FROM candidates")]
        frozen = {r[0] for r in db.execute("SELECT id FROM collection_batches")}
    # Preflight the whole collection before committing any individual batch.
    issues = quality_issues(rows, benchmark_references() + stored)
    if any(i["kind"] != "near_duplicate" or i["cross_partition"] for i in issues):
        raise ValueError(f"Expansion overlap blockers: {issues}")
    batches = {}
    for row in rows:
        batch = row["collection_batch"]
        if batch in frozen:
            raise ValueError(f"Batch already frozen: {batch}")
        if row["source_method"] == "codex" and batch != row["generation"]["batch_id"]:
            raise ValueError("Codex generation batches must not be merged for auditing")
        batches.setdefault(batch, []).append(row)
    for index, (batch_id, members) in enumerate(sorted(batches.items())):
        count = import_candidates(
            members,
            path,
            provisional_batch=batch_id,
            seed=700428 + index,
            policy_version=policy_version,
        )
        print(f"Imported {count} candidates in {batch_id}")


def workload_summary(plan, rows):
    by_id = {r["id"]: r for r in rows}
    breakdown = {}
    for row in rows:
        key = (row["provenance"]["source"], row["source_method"], row["subtype"])
        item = breakdown.setdefault(
            key,
            dict(
                source=key[0],
                method=key[1],
                subtype=key[2],
                candidates=0,
                risk="manual",
                audit_samples=0,
                manual_only=0,
                required_reviews=0,
            ),
        )
        item["candidates"] += 1
    sampled = set()
    for group in plan["groups"]:
        key = (group["source"], group["method"], group["subtype"])
        breakdown[key]["risk"] = group["risk"]
        breakdown[key]["audit_samples"] += len(group["sampled_ids"])
        sampled.update(group["sampled_ids"])
    for record_id in set(plan["review_requests"]) - sampled:
        row = by_id[record_id]
        breakdown[(row["provenance"]["source"], row["source_method"], row["subtype"])][
            "manual_only"
        ] += 1
    for item in breakdown.values():
        item["required_reviews"] = item["audit_samples"] + item["manual_only"]
    return dict(
        policy_version=plan["policy_version"],
        candidates=plan["candidates"],
        audit_groups=len(plan["groups"]),
        audit_samples=plan["audit_samples"],
        manual_only=plan["manual_only"],
        required_reviews=plan["required_reviews"],
        fingerprint=plan["fingerprint"],
        breakdown=list(breakdown.values()),
        reasons=dict(Counter(plan["review_requests"].values())),
    )


def propose_collection_v2(directory, path=DEFAULT_DB):
    """Propose verified source/recipe regrouping; never update the old frozen audits."""
    rows = [json.loads(line) for line in (directory / "candidates.jsonl").read_text().splitlines()]
    source_plan = json.loads((directory / "source_selection.json").read_text())
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        stored = [json.loads(r[0]) for r in db.execute("SELECT payload FROM candidates")]
    live = [r for r in stored if r.get("collection_name") == directory.name]
    if digest(sorted(rows, key=lambda r: r["id"])) != digest(sorted(live, key=lambda r: r["id"])):
        raise ValueError("Saved collection no longer matches the imported immutable records")
    if len(rows) != 400 or directory.name != "milestone700_v1":
        raise ValueError("This evidence-backed regrouping proposal is scoped to the existing 400")
    processes = {}
    for row in rows:
        p, g = row["provenance"], row["generation"]
        if row["source_method"] == "public":
            if row["collection_batch"].split("-")[-1] not in {"01", "02"}:
                raise ValueError("Unexpected public collection boundary")
            processes[row["id"]] = dict(
                run="m700-acquisition:" + p["source"] + ":" + p["revision"], family=row["subtype"]
            )
        elif row["source_method"] == "python":
            if g["seed"] != source_plan["python_seed"]:
                raise ValueError("Python run has inconsistent seeds")
            processes[row["id"]] = dict(
                run="m700-python-seed:" + str(g["seed"]), family=g["recipe"], recipe=g["recipe"]
            )
        else:
            processes[row["id"]] = boundary(row, directory.name)  # Keep Codex chunks separate.
    result = migration_preview(directory.name, boundaries=processes, path=path)
    result["regrouping_evidence"] = {
        "source_selection_sha256": digest(source_plan),
        "candidate_sha256": hashlib.sha256(
            (directory / "candidates.jsonl").read_bytes()
        ).hexdigest(),
        "public": "Same pinned source/revision, acquisition plan and subtype; "
        "40/35 chunks were artificial.",
        "python": "One saved seed and deterministic recipe run; each recipe stays separate.",
        "codex": "Original generation-batch boundaries retained; "
        "no inferred instruction-family merge.",
    }
    result["fingerprint"] = digest({k: v for k, v in result.items() if k != "fingerprint"})
    return result, workload_summary(result, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Download admitted source snapshots")
    parser.add_argument(
        "--project-v2",
        action="store_true",
        help="Read-only proposal for the existing 400; never replace audits",
    )
    parser.add_argument(
        "--build", action="store_true", help="Create pilot once from source snapshots"
    )
    parser.add_argument(
        "--import", dest="do_import", action="store_true", help="Import into SQLite"
    )
    parser.add_argument(
        "--collection",
        choices=("pilot_v1", "milestone300_v1", "milestone700_v1"),
        default="pilot_v1",
        help="Choose an explicit versioned candidate batch",
    )
    args = parser.parse_args()
    directory = ROOT / "data" / args.collection
    path = directory / "candidates.jsonl"
    if args.project_v2:
        if args.fetch or args.build or args.do_import:
            parser.error("--project-v2 cannot be combined with writes, downloads, or imports")
        _, summary = propose_collection_v2(directory)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    stored = live_candidates() if args.collection != "pilot_v1" else []
    if args.fetch:
        fetch()
    if args.build:
        if path.exists():
            raise FileExistsError("Pilot exists; create a new version instead of overwriting it")
        if args.collection == "milestone700_v1":
            rows = expansion_candidates(directory)
        elif args.collection == "milestone300_v1":
            rows = milestone_candidates(directory)
        else:
            public = public_candidates()
            rows = public + codex_candidates() + python_candidates(public)
        counts = Counter((r["source_method"], r["intended_label"]) for r in rows)
        if args.collection == "pilot_v1" and counts != {
            ("public", "VALID"): 30,
            ("codex", "VALID"): 15,
            ("codex", "GIBBERISH"): 40,
            ("python", "VALID"): 5,
            ("python", "GIBBERISH"): 10,
        }:
            raise ValueError(f"Pilot distribution mismatch: {counts}")
        rows.sort(key=lambda r: r["id"])
        issues = quality_issues(rows, benchmark_references() + stored)
        if any(i["kind"] != "near_duplicate" or i["cross_partition"] for i in issues):
            raise ValueError(f"Pilot overlap blockers: {issues}")
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    new_ids = {r["id"] for r in rows}
    references = benchmark_references() + [r for r in stored if r["id"] not in new_ids]
    issues = quality_issues(rows, references)
    audit = {
        "created_at": now(),
        "candidates": len(rows),
        "partition": "train",
        "intended_counts": dict(Counter(r["intended_label"] for r in rows)),
        "sources": dict(Counter(r["source_method"] for r in rows)),
        "public_sources": dict(
            Counter(r["provenance"]["source"] for r in rows if r["source_method"] == "public")
        ),
        "dialect_hints": dict(Counter(r["dialect"] for r in rows)),
        "subtypes": dict(Counter(r["subtype"] for r in rows)),
        "difficulty": dict(Counter(r.get("difficulty", "not_recorded") for r in rows)),
        "gibberish_difficulty": dict(
            Counter(
                r.get("difficulty", "not_recorded")
                for r in rows
                if r["intended_label"] == "GIBBERISH"
            )
        ),
        "families_shared_with_existing_training": len(
            {r["family_id"] for r in rows}
            & {r["family_id"] for r in references if r.get("partition") == "train"}
        ),
        "semantic_overlap": "manual author screening; lexical checks cannot prove independence",
        "quality_issues": issues,
        "human_review": "Candidate intentions only; current human annotations live in SQLite.",
        "pilot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    if args.do_import:
        if args.collection == "milestone700_v1":
            import_expansion(rows)
        else:
            print(f"Imported {import_candidates(rows)} candidates")
    print(f"{len(rows)} candidates; {len(issues)} within-training near-duplicate review hints")
    print("No new human labels assigned. Stop here for blind review before further expansion.")


if __name__ == "__main__":
    main()
