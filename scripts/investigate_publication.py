"""Read frozen publication inputs and write local evidence; never approves or uploads."""

import hashlib
import json
import re
import sys
import tarfile
from collections import Counter
from itertools import combinations
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jais_guard.publication import frozen as pub  # noqa: E402

OUT = pub.LOCAL / "investigation"
EXTERNAL = {"AmazonScience/massive", "CohereLabs/aya_dataset", "google-research-datasets/tydiqa"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def strings(value):
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return set().union(set(), *(strings(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(set(), *(strings(v) for v in value))
    return set()


def origin(row):
    p, g = row["provenance"], row.get("generation") or {}
    if p["source"] in EXTERNAL:
        return (
            "external_verbatim"
            if row["source_method"] == "public"
            else "external_python_adaptation"
        )
    if row["source_method"] == "codex":
        return "assistant_direct"
    if p.get("origin_type") == "assistant-authored-scenario":
        return "assistant_scenario_via_python"
    if g.get("parent"):
        return "assistant_seed_via_python"
    if (
        p.get("origin_type") == "simulated-input-failure"
        or p["source"] == "JaisGuard deterministic Python recipes"
    ):
        if g.get("parent_id"):
            return "ambiguous_parent"
        return "standalone_python"
    return "unknown"


def adaptation_matches(raw, text, recipe):
    if recipe == "join_first_space":
        return text == raw.replace(" ", "", 1)
    if recipe == "join_last_space":
        i = raw.rfind(" ")
        return text == raw[:i] + raw[i + 1 :]
    if recipe == "remove_hamza":
        return text == raw.replace("أ", "ا").replace("إ", "ا")
    if recipe == "append_limited_noise":
        return text in {raw + " " + noise for noise in ["جج", "صص", "طظ"]}
    if recipe == "drop_one_letter":
        return any(text == raw[:i] + raw[i + 1 :] for i, c in enumerate(raw) if c.isalpha())
    if recipe == "repeat_one_letter":
        return any(text == raw[:i] + c + raw[i:] for i, c in enumerate(raw) if c.isalpha())
    if recipe == "whole_input_character_shuffle":
        return Counter(c for c in raw if c.isalpha()) == Counter(c for c in text if c.isalpha())
    if recipe == "seeded_word_character_shuffle":
        return [Counter(c for c in w if c.isalpha()) for w in raw.split()] == [
            Counter(c for c in w if c.isalpha()) for w in text.split()
        ]
    return False


def investigate():
    rows, provenance = pub.source_data()
    pub.inventory(pub.STAGE)
    metadata = json.loads((pub.STAGE / "metadata.json").read_text())
    assert metadata["files"] == pub.file_hashes(pub.STAGE)
    assert metadata["sources"] == provenance
    for s, rs in rows.items():
        assert pq.read_table(pub.STAGE / f"data/{s}.parquet").to_pylist() == [
            pub.project(r) for r in rs
        ]
    evidence = set()
    ledger, checks, failures = [], Counter(), []
    cache = ROOT / "data/raw/pilot_v1"
    evidence.update(cache.glob("*info.json"))
    aya_path = cache / "aya_train.parquet"
    evidence.add(aya_path)
    aya = pq.read_table(aya_path, columns=["inputs", "annotation_type", "language_code"])
    tydi = {}
    for path in cache.glob("tydi_rows*.json"):
        evidence.add(path)
        tydi.update({r["row_idx"]: r["row"] for r in json.loads(path.read_text())["rows"]})
    archive = cache / "massive.tar.gz"
    evidence.add(archive)
    with tarfile.open(archive) as tar:
        massive = {
            str(r["id"]): r
            for r in [json.loads(s) for s in tar.extractfile("1.1/data/ar-SA.jsonl")]
        }
    sidecars = {}
    for split, directory in [
        ("train", "train800_iteration2_v1"),
        ("validation", "validation250_iteration2_v1"),
    ]:
        path = ROOT / "data" / directory / "private/corruptions.jsonl"
        parents = path.with_name("parents-final.jsonl" if split == "train" else "parents.jsonl")
        evidence.update([path, parents])
        sidecars[split] = (
            {r["id"]: r for r in read_rows(path)},
            {r["text"] for r in read_rows(parents)},
        )
    batch_cache = {}
    for s, rs in rows.items():
        for i, r in enumerate(rs):
            p, g = r["provenance"], r.get("generation") or {}
            entry = {
                "split": s,
                "row": i,
                "text_sha256": hashlib.sha256(r["input_text"].encode()).hexdigest(),
                "origin": origin(r),
                "source": p["source"],
                "source_method": r["source_method"],
                "source_id": p.get("source_id"),
                "revision": p.get("revision"),
                "modifications": p.get("modifications"),
                "recipe": g.get("recipe"),
                "license_claim": p.get("license"),
            }
            if p["source"] in EXTERNAL:
                sid = p.get("upstream_source_id", p["source_id"])
                if p["source"] == "CohereLabs/aya_dataset":
                    j = int(sid.rsplit("-", 1)[1])
                    raw = aya["inputs"][j].as_py()
                    assert aya["annotation_type"][j].as_py() == "original-annotations"
                    assert p.get("annotation_type") == "original-annotations"
                elif p["source"] == "google-research-datasets/tydiqa":
                    raw = tydi[int(sid.rsplit("-", 1)[1])]["question_text"]
                else:
                    raw = massive[sid.rsplit(":", 1)[1]]["utt"]
                matched = (
                    raw == r["input_text"]
                    if r["source_method"] == "public"
                    else adaptation_matches(raw, r["input_text"], g["recipe"])
                )
                if not matched:
                    failures.append({"record": [s, i], "check": "upstream_text_or_adaptation"})
                else:
                    checks[origin(r)] += 1
                entry["upstream_checked"] = matched
            if r["source_method"] == "codex":
                path = ROOT / p["rights_evidence"]
                evidence.add(path)
                if path not in batch_cache:
                    batch_cache[path] = strings(json.loads(path.read_text()))
                matched = r["input_text"] in batch_cache[path]
                checks["assistant_saved_output_match"] += int(matched)
                if not matched:
                    failures.append({"record": [s, i], "check": "assistant_saved_output"})
            if s in sidecars and p.get("origin_type"):
                record_map, parents = sidecars[s]
                sidecar = record_map[r["id"]]
                original = sidecar.get("original_text")
                checks["sidecar_id_match"] += 1
                if original is not None:
                    matched = (
                        original in parents
                        and hashlib.sha256(original.encode()).hexdigest()
                        == sidecar["parent_sha256"]
                    )
                    checks["assistant_parent_hash_match"] += int(matched)
                    category = g["recipe"].split("/")[-1]
                    if category == "meaningful_request":
                        matched = matched and original == r["input_text"]
                    else:
                        # Pure corruption helper has no filesystem writes.
                        from jais_guard.data.test_collection import corrupt

                        matched = (
                            matched and corrupt(original, g["seed"], category)[0] == r["input_text"]
                        )
                    checks["assistant_parent_recipe_match"] += int(matched)
                    if not matched:
                        failures.append({"record": [s, i], "check": "parent_hash_or_recipe"})
            if g.get("parent"):
                path = ROOT / p["rights_evidence"]
                evidence.add(path)
                matched = g["parent"]["text"] in strings(json.loads(path.read_text()))
                checks["legacy_assistant_seed_match"] += int(matched)
                if not matched:
                    failures.append({"record": [s, i], "check": "legacy_assistant_seed"})
            path = ROOT / p.get("rights_evidence", "")
            if path.is_file():
                evidence.add(path)
            ledger.append(entry)
    flat = [
        (s, i, pub.normalize(r["input_text"])) for s, rs in rows.items() for i, r in enumerate(rs)
    ]
    contained = []
    for (a, i, x), (b, j, y) in combinations(flat, 2):
        if min(len(x), len(y)) >= 20 and x != y and (x in y or y in x):
            contained.append([[a, i], [b, j]])
    patterns = {
        "biography_age": r"عمري\s*[0-9٠-٩۰-۹]+|أنا فارس",
        "named_family_narrative": r"مرتي\s+حنان|بنت زي القمر",
        "email": r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        "phone_or_identifier": r"(?<!\w)(?:\+?\d[\d ()-]{6,}\d)",
        "credential": r"(?i)hf_[a-z0-9]{15,}|sk-[a-z0-9_-]{15,}|AKIA[A-Z0-9]{16}|BEGIN .*PRIVATE KEY|password\s*[:=]",  # noqa: E501
        "harmful_instruction_candidate": r"كيف.*(?:اقتل|أقتل|اصنع قنبلة|أصنع قنبلة|انتحر|أنتحر)|اغتصاب|محتوى جنسي",  # noqa: E501
        "literary_continuation": r"^كمل (?:الحكي|الجملة|الفقرة)",
    }
    hits = {k: [] for k in patterns}
    for s, rs in rows.items():
        for i, r in enumerate(rs):
            for name, pattern in patterns.items():
                if re.search(pattern, r["input_text"]):
                    hits[name].append([s, i])
    staged_metadata = (pub.STAGE / "metadata.json").read_text()
    metadata_hits = {
        k: bool(re.search(v, staged_metadata))
        for k, v in {
            "absolute_workspace_path": r"/home/|/mnt/[a-z]/|[A-Z]:\\",
            "private_metadata_key": r'"(?:author_group|reviewer|user_id|notes|api_key|token|password)"\s*:',  # noqa: E501
            "credential_value": patterns["credential"],
        }.items()
    }
    evidence.update(
        [
            ROOT / "scripts/prepare_candidates.py",
            ROOT / "src/jais_guard/data/test_collection.py",
            ROOT / "scripts/build_test400.py",
            ROOT / "data/pilot_v1/sources/manifest.json",
            ROOT / "data/milestone700_v1/realism_replacements_v1/generation/recipes.py",
            pub.LOCAL / "audit.json",
            pub.STAGE / "metadata.json",
        ]
    )
    for version, _, _, _ in pub.SOURCES.values():
        evidence.add(ROOT / "data/exports" / version / "manifest.json")
    known_parents = {}
    for split, directory in [
        ("train", "train800_iteration2_v1"),
        ("validation", "validation250_iteration2_v1"),
    ]:
        path = ROOT / "data" / directory / "private/corruptions.jsonl"
        known_parents[split] = [
            pub.normalize(r["original_text"]) for r in read_rows(path) if r.get("original_text")
        ]
    known_parents["train"] += [
        pub.normalize(r["generation"]["parent"]["text"])
        for r in rows["train"]
        if r.get("generation", {}).get("parent")
    ]
    known_parents["test"] = [
        pub.normalize(r["input_text"]) for r in rows["test"] if r["subtype"] == "meaningful_request"
    ]
    parent_overlap = {
        f"{a}/{b}": len(set(known_parents[a]) & set(known_parents[b]))
        for a, b in combinations(known_parents, 2)
    }
    expected_revisions = {
        r["provenance"]["source"]: r["provenance"]["revision"]
        for rs in rows.values()
        for r in rs
        if r["provenance"]["source"] in EXTERNAL
    }
    for short, source in [
        ("aya", "CohereLabs/aya_dataset"),
        ("tydi", "google-research-datasets/tydiqa"),
        ("massive", "AmazonScience/massive"),
    ]:
        assert (
            json.loads((cache / f"{short}_info.json").read_text())["sha"]
            == expected_revisions[source]
        )
    missing = [
        p
        for p in [
            "data/test400_collection_v2/private/parents.jsonl",
            "data/test400_collection_v2/private/corruptions.jsonl",
        ]
        if not (ROOT / p).exists()
    ]
    summary = {
        "metadata_sha256": digest(pub.STAGE / "metadata.json"),
        "counts": dict(Counter(x["origin"] for x in ledger)),
        "counts_by_split": {
            s: dict(Counter(x["origin"] for x in ledger if x["split"] == s)) for s in rows
        },
        "source_counts": dict(Counter(x["source"] for x in ledger)),
        "checks": dict(checks),
        "failures": failures,
        "additional_normalized_containment_min_length_20": contained,
        "expanded_content_screening": hits,
        "staged_metadata_screening": metadata_hits,
        "known_parent_counts": {s: len(v) for s, v in known_parents.items()},
        "known_parent_normalized_cross_split_overlap": parent_overlap,
        "missing_test_parent_evidence": missing,
        "test_records_with_unavailable_parent_sidecars": sum(
            r["provenance"].get("origin_type") == "assistant-authored-scenario"
            for r in rows["test"]
        ),
        "original_overlap_counts": {
            k: len(v)
            for k, v in json.loads((pub.LOCAL / "audit.json").read_text())["relationships"].items()
        },
        "input_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sorted(evidence)},
    }
    pub.write_json(OUT / "evidence.json", summary)
    pub.write_json(OUT / "record-provenance.json", ledger)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "input_hashes"}, ensure_ascii=False, indent=2
        )
    )
    if failures:
        raise SystemExit("Evidence mismatches require investigation")


if __name__ == "__main__":
    investigate()
