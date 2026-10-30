"""Build a small independent draft from frozen text sources; never import or evaluate."""

import argparse
import csv
import hashlib
import json
import random
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = ROOT / "data/test400_v1/collection_20261029"
RUN = "test400-20261029-independent-collection-v1"


def load_helpers():
    sys.path.insert(0, str(ROOT / "src"))
    from jais_guard.data.annotations import comparison_text, now, validate_candidate
    from jais_guard.data.test_workflow import sha, verify_lock

    return comparison_text, now, validate_candidate, sha, verify_lock


def repeat_key(text):
    """One character repeat with all other words intact; intent still needs human review."""
    positions = [
        i
        for i, c in enumerate(text)
        if c.isalpha() and "ARABIC" in unicodedata.name(c, "") and not unicodedata.combining(c)
    ]
    i = positions[len(positions) // 2]
    return text[:i] + text[i] + text[i:]


def keyboard_displacement(text):
    """Constructed whole-message neighboring-key error, not an observed user accident."""
    mapping = {}
    for row in ("ضصثقفغعهخحجد", "شسيبلاتنمكط", "ئءؤرلىةوزظ"):
        mapping.update({c: row[(i + 1) % len(row)] for i, c in enumerate(row)})
    mapping.update({"أ": "ل", "إ": "ل", "آ": "ل", "ا": "ت", "ڨ": "ف"})
    return "".join(mapping.get(c, c) for c in text)


def scramble_words(text, seed):
    rng = random.Random(seed)
    words = []
    for word in text.split():
        chars = list(word)
        rng.shuffle(chars)
        candidate = "".join(chars)
        if candidate == word:
            candidate = word[::-1]
        words.append(candidate)
    return " ".join(words)


def record(
    text,
    label,
    method,
    dialect,
    subtype,
    domain,
    family,
    provenance,
    generation,
    difficulty,
    timestamp,
):
    opaque = hashlib.sha256((RUN + family + text).encode()).hexdigest()[:12]
    return {
        "id": "T-" + opaque,
        "input_text": text,
        "partition": "test",
        "intended_label": label,
        "intended_label_status": "provisional",
        "review_status": "pending-human-review",
        "source_method": method,
        "subtype": subtype,
        "dialect": dialect,
        "dialect_status": "not_applicable"
        if label == "GIBBERISH"
        else "proposed_not_human_confirmed",
        "domain": domain,
        "family_id": family,
        "provenance": provenance,
        "generation": generation,
        "difficulty": difficulty,
        "difficulty_status": "proposed_not_human_confirmed",
        "collection_name": "test400_v1",
        "collection_batch": RUN,
        "created_at": timestamp,
    }


def build(output, collection=COLLECTION):
    comparison_text, now, validate_candidate, sha, verify_lock = load_helpers()
    verify_lock(ROOT / "data/test400_v1")
    output = Path(output)
    if output.exists():
        raise ValueError("Draft already exists; do not overwrite a reviewed/previous draft")
    source = collection / "sources/derja_mirror_tunsi_english_data.csv"
    with source.open() as stream:
        corpus = list(csv.DictReader(stream))
    revision = json.loads((collection / "sources/derja_mirror_api.json").read_text())["sha"]
    authoring_path = collection / "codex_authoring.json"
    authoring = json.loads(authoring_path.read_text())
    timestamp = now()
    rows, parents = [], []
    selected = {
        "public": [628, 12126, 643, 10577, 12536, 59, 10579, 661, 64, 6766, 9336, 8059],
        "recoverable": [11850, 73, 74, 112, 12776, 10592, 637, 5346, 12474, 95, 10585, 10572],
        "unrecoverable": [
            630,
            11113,
            782,
            10573,
            651,
            3768,
            36,
            659,
            38,
            75,
            65,
            0,
        ],
    }
    assert len(set(sum(selected.values(), []))) == 36
    for category, indices in selected.items():
        for offset, index in enumerate(indices):
            original = corpus[index]["tn"]
            parent_hash = hashlib.sha256(comparison_text(original).encode()).hexdigest()
            family = f"derja-ninja:text:{parent_hash}"
            provenance = {
                "source": "Derja.Ninja",
                "source_id": f"achermiti/derja:csv-row-{index}",
                "url": "https://derja.ninja/",
                "license": "CC-BY-SA-4.0",
                "attribution": "Derja.Ninja; mirror curated by Khawla Jlassi, "
                "Imen Ayari, Jacer Dabbabi",
                "revision": revision,
                "mirror": "achermiti/derja",
                "mirror_file_sha256": sha(source),
                "upstream_text_sha256": parent_hash,
                "rights_status": "verified",
                "rights_evidence": str((collection / "sources/derja_about.txt").relative_to(ROOT)),
                "rights_url": "https://derja.ninja/about/",
                "rights_note": "Upstream text CC-BY-SA-4.0 is used, not mirror's conflicting MIT "
                "placeholder. Audio is all-rights-reserved and is not collected.",
                "origin": "human_authored_dictionary_example_via_cleaned_mirror",
                "author_group": None,
                "original_entry_uuid": None,
                "collection_run": RUN,
            }
            generation = {}
            label, method, text, subtype = "VALID", "public", original, "dictionary_example"
            difficulty = (
                "easy" if offset in (0, 2, 3, 6) else "hard" if offset in (1, 11) else "medium"
            )
            if category != "public":
                seed = 294001 + offset + (100 if category == "unrecoverable" else 0)
                recipe = (
                    "test400-single-key-repeat-v1"
                    if category == "recoverable"
                    else (
                        "test400-neighbor-key-displacement-v1"
                        if offset < 8
                        else "test400-within-token-shuffle-v1"
                    )
                )
                generation = {
                    "run_id": RUN,
                    "batch_id": RUN + "-derja-transformations",
                    "seed": seed,
                    "recipe": recipe,
                    "parent_id": f"external:derja-mirror:{revision}:row-{index}",
                    "parent_text_sha256": parent_hash,
                    "instructions_file": "scripts/collect_test_draft.py",
                }
                method = "python"
                if category == "recoverable":
                    text, subtype = repeat_key(original), "recoverable_key_repeat"
                    difficulty = "hard" if offset % 2 else "medium"
                else:
                    label, subtype = (
                        "GIBBERISH",
                        "constructed_keyboard_displacement"
                        if offset < 8
                        else "constructed_word_scramble",
                    )
                    text = (
                        keyboard_displacement(original)
                        if offset < 8
                        else scramble_words(original, seed)
                    )
                    difficulty = "medium"
                    if offset >= 8:
                        # Source-aware authoring review found recoverable intent.
                        # This is a provisional correction, never a human label.
                        label, subtype, difficulty = "VALID", "recoverable_word_scramble", "hard"
                provenance["modifications"] = (
                    recipe + "; constructed error, not an observed accident"
                )
            else:
                provenance["modifications"] = "Mirror tn field copied verbatim; English omitted"
            dialect = "Tunisian" if label == "VALID" else "not_applicable"
            row = record(
                text,
                label,
                method,
                dialect,
                subtype,
                "everyday",
                family,
                provenance,
                generation,
                difficulty,
                timestamp,
            )
            rows.append(row)
            parents.append(
                {
                    "candidate_id": row["id"],
                    "row_index": index,
                    "text": original,
                    "english_gloss_for_audit_only": corpus[index]["en"],
                    "original_text_sha256": parent_hash,
                    "category": category,
                }
            )
    # Deliberately use a subset, not 400 automatically padded examples. All unused
    # authoring proposals remain reserved to this test-only run, not other partitions.
    author_indices = [
        *range(8),
        16,
        17,
        20,
        21,
        24,
        25,
        28,
        29,
        *range(32, 40),
        *range(48, 52),
        *range(56, 60),
    ]
    for index in author_indices:
        item = authoring["examples"][index]
        provenance = {
            "source": "Codex independent test authoring",
            "source_id": f"{RUN}:item-{index}",
            "license": "generated-content-publication-review-pending",
            "attribution": "Codex / OpenAI; not human-authored or human-labeled",
            "rights_status": "publication_review_pending",
            "rights_evidence": str(authoring_path.relative_to(ROOT)),
            "origin": "assistant_authored_test_only_proposal",
        }
        family = f"test400-independent-authoring:item-{index}"
        generation = {
            "run_id": authoring["run_id"],
            "batch_id": authoring["run_id"],
            "instructions_file": str(authoring_path.relative_to(ROOT)),
            "model_metadata": authoring["model_metadata"],
        }
        difficulty = item["difficulty"]
        if item["intended_label"] == "VALID":
            difficulty = (
                "hard" if item["subtype"] in {"spelling_errors", "incomplete_request"} else "medium"
            )
        rows.append(
            record(
                item["input_text"],
                item["intended_label"],
                "codex",
                item["dialect"],
                item["subtype"],
                item["domain"],
                family,
                provenance,
                generation,
                difficulty,
                timestamp,
            )
        )
    # Eight finite constructed failures: four moderate pseudo-word chains and four
    # easy random-character cases, with length bands declared rather than model-selected.
    alphabet = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
    for index, length in enumerate([32, 39, 46, 54, 18, 25, 33, 45]):
        seed = 294301 + index
        rng = random.Random(seed)
        chunks = ["".join(rng.choices(alphabet, k=5 + (i % 3))) for i in range(8)]
        text = " ".join(chunks)[:length].rstrip()
        subtype = "constructed_pseudoword_chain" if index < 4 else "random_arabic_characters"
        provenance = {
            "source": "JaisGuard independent test failure simulations",
            "source_id": f"{RUN}:random-{index}",
            "license": "project-original",
            "attribution": "JaisGuard deterministic Python recipe; not observed user traffic",
            "rights_evidence": "scripts/collect_test_draft.py",
            "origin": "constructed_failure_simulation",
        }
        recipe = "test400-chunked-noise-v1"
        rows.append(
            record(
                text,
                "GIBBERISH",
                "python",
                "not_applicable",
                subtype,
                "general",
                f"{RUN}:random-run-{seed}",
                provenance,
                {
                    "run_id": RUN,
                    "batch_id": RUN + "-failure-simulations",
                    "seed": seed,
                    "recipe": recipe,
                },
                "medium" if index < 4 else "easy",
                timestamp,
            )
        )
    for row in rows:
        validate_candidate(row)
        if "label" in row or "annotation" in row:
            raise AssertionError("No ground truth in a candidate draft")
    rows.sort(key=lambda r: r["id"])
    parent_path = collection / "selected_parents.jsonl"
    if parent_path.exists():
        previous = [json.loads(line) for line in parent_path.read_text().splitlines()]
        if previous != parents:
            raise ValueError(
                "Parent reservations changed; preserve history and use a new collection"
            )
    with output.open("x") as stream:
        stream.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    if not parent_path.exists():
        with parent_path.open("x") as stream:
            stream.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in parents))
    print(f"Wrote {len(rows)} provisional candidates to {output}; not imported or frozen.")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/test400_v1/draft.jsonl")
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
