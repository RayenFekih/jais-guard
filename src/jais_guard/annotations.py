"""Local annotation storage. Human labels and source intentions never share a column."""

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data/annotation/annotations.sqlite3"
LABELS = ("VALID", "GIBBERISH")
PARTITIONS = ("train", "validation", "test")
POLICY_VERSION = "recoverable-meaning-v1"


def now() -> str:
    return datetime.now(UTC).isoformat()


def comparison_text(text: str) -> str:
    # Preserve word boundaries; normalization is only for comparison, never stored text.
    text = unicodedata.normalize("NFKC", text).replace("ـ", "")
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).split())


def arabic_script(text: str) -> bool:
    return (
        bool(text.strip())
        and any("ARABIC" in unicodedata.name(c, "") for c in text)
        and not any(c.isalpha() and "ARABIC" not in unicodedata.name(c, "") for c in text)
    )


def validate_candidate(row: dict) -> None:
    if not isinstance(row, dict):
        raise ValueError("Each candidate must be an object")
    if any(field in row for field in ("human_label", "label", "is_gibberish", "annotation")):
        raise ValueError("Candidate intentions must not be imported as human annotations")
    for field in (
        "id",
        "input_text",
        "partition",
        "source_method",
        "subtype",
        "dialect",
        "domain",
        "family_id",
    ):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"Missing nonempty string: {field}")
    if row["partition"] not in PARTITIONS or row.get("intended_label") not in LABELS:
        raise ValueError("Unknown partition or intended label")
    if row["source_method"] not in {"public", "codex", "python"}:
        raise ValueError("Unknown source method")
    if "difficulty" in row and row["difficulty"] not in {"easy", "medium", "hard"}:
        raise ValueError("Difficulty must be easy, medium or hard when recorded")
    if not arabic_script(row["input_text"]):
        raise ValueError(f"Unsupported script: {row['id']}; no gibberish label assigned")
    provenance = row.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Provenance must be an object")
    for field in ("source", "source_id", "license", "attribution", "rights_evidence"):
        if not isinstance(provenance.get(field), str) or not provenance[field].strip():
            raise ValueError(f"Missing provenance: {field}")
    if row["source_method"] == "public":
        if not provenance.get("revision") or provenance.get("rights_status") != "verified":
            raise ValueError("Public records need a pinned revision and verified upstream rights")
        if provenance["license"] not in {"Apache-2.0", "MIT", "CC-BY-4.0", "CC-BY-SA-4.0"}:
            raise ValueError("Source license requires explicit commercial-redistribution review")
    if row["source_method"] == "codex" and not row.get("generation", {}).get("batch_id"):
        raise ValueError("Codex records need generation batch metadata")
    if row["source_method"] == "python" and "seed" not in row.get("generation", {}):
        raise ValueError("Python records need a deterministic seed")


def quality_issues(rows: list[dict], references: list[dict]) -> list[dict]:
    """Report lexical overlap; semantic independence still needs human review."""
    issues = []
    seen_ids = {r["id"] for r in references}
    prior = list(references)
    for row in rows:
        validate_candidate(row)
        if row["id"] in seen_ids:
            raise ValueError(f"Duplicate ID: {row['id']}")
        seen_ids.add(row["id"])
        a = comparison_text(row["input_text"])
        aa = {a[i : i + 3] for i in range(max(0, len(a) - 2))}
        for other in prior:
            b = comparison_text(other["input_text"])
            kind = None
            if a == b:
                kind = "duplicate_text"
            elif row["family_id"] == other.get("family_id"):
                if row["partition"] != other.get("partition"):
                    kind = "cross_partition_family"
            if kind is None:
                bb = {b[i : i + 3] for i in range(max(0, len(b) - 2))}
                jaccard = len(aa & bb) / len(aa | bb) if aa | bb else 0
                ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
                if jaccard >= 0.8 or ratio >= 0.9:
                    kind = "near_duplicate"
            if kind:
                issues.append(
                    {
                        "id": row["id"],
                        "other_id": other["id"],
                        "kind": kind,
                        "cross_partition": row["partition"] != other.get("partition"),
                    }
                )
        prior.append(row)
    return issues


@contextmanager
def connection(path: Path = DEFAULT_DB):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS candidates (
            id TEXT PRIMARY KEY,
            partition TEXT NOT NULL CHECK(partition IN ('train','validation','test')),
            payload TEXT NOT NULL, imported_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS annotations (
            id TEXT PRIMARY KEY REFERENCES candidates(id),
            human_label TEXT CHECK(human_label IN ('VALID','GIBBERISH')),
            flagged INTEGER NOT NULL DEFAULT 0, notes TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS revisions (
            revision INTEGER PRIMARY KEY, id TEXT NOT NULL REFERENCES candidates(id),
            payload TEXT NOT NULL, timestamp TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS candidate_exclusions (
            id TEXT PRIMARY KEY REFERENCES candidates(id),
            reason TEXT NOT NULL, excluded_at TEXT NOT NULL);
    """)
    try:
        with db:
            yield db
    finally:
        db.close()


def benchmark_references() -> list[dict]:
    path = ROOT / "data/gibberish.jsonl"
    return [
        {"id": r["id"], "input_text": r["input_text"], "partition": "religious_" + r["split"]}
        for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
    ]


def import_candidates(rows: list[dict], path: Path = DEFAULT_DB) -> int:
    with connection(path) as db:
        existing = [json.loads(r[0]) for r in db.execute("SELECT payload FROM candidates")]
        if any(r["partition"] == "test" for r in rows) and not get_state("test_unlocked", path):
            raise ValueError("Test collection locked; record a freeze decision before test import")
        issues = quality_issues(rows, benchmark_references() + existing)
        blockers = [i for i in issues if i["kind"] != "near_duplicate" or i["cross_partition"]]
        if blockers:
            raise ValueError(f"Import blocked by overlap: {blockers[:5]}")
        flagged_ids = {i["id"] for i in issues}
        for row in rows:
            db.execute(
                "INSERT INTO candidates VALUES (?,?,?,?)",
                (row["id"], row["partition"], json.dumps(row, ensure_ascii=False), now()),
            )
            if row["id"] in flagged_ids:
                db.execute(
                    "INSERT INTO annotations VALUES (?,NULL,1,?,?)",
                    (row["id"], "Automatic near-duplicate review required", now()),
                )
        _exclude_transformed_originals(db)
    return len(rows)


def _exclude_transformed_originals(db) -> list[str]:
    """Archive source candidates without deleting their text or human review history."""
    stored = {r[0]: json.loads(r[1]) for r in db.execute("SELECT id,payload FROM candidates")}
    parents = set()
    for row in stored.values():
        parent_id = row.get("generation", {}).get("parent_id")
        if parent_id not in stored:
            continue  # An external source remains provenance, not an active candidate.
        if stored[parent_id]["partition"] != row["partition"]:
            raise ValueError("Transformed candidates must share their parent's partition")
        parents.add(parent_id)
    for parent_id in sorted(parents):
        db.execute(
            "INSERT OR IGNORE INTO candidate_exclusions VALUES (?,?,?)",
            (
                parent_id,
                "Original replaced by transformed candidate; retain provenance/history",
                now(),
            ),
        )
    return sorted(parents)


def exclude_transformed_originals(path: Path = DEFAULT_DB) -> list[str]:
    """Explicitly apply the modified-only policy to an already imported collection."""
    with connection(path) as db:
        return _exclude_transformed_originals(db)


def get_state(key: str, path: Path = DEFAULT_DB) -> str | None:
    with connection(path) as db:
        row = db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return row[0] if row else None


def set_state(key: str, value: str, path: Path = DEFAULT_DB) -> None:
    with connection(path) as db:
        db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, value))


def queue(partition: str, mode: str = "all", path: Path = DEFAULT_DB) -> list[dict]:
    if partition not in PARTITIONS or mode not in {"all", "unreviewed", "flagged"}:
        raise ValueError("Unknown partition or queue")
    with connection(path) as db:
        rows = db.execute(
            """SELECT c.id, c.partition, c.payload, a.human_label,
            COALESCE(a.flagged,0) AS flagged, COALESCE(a.notes,'') AS notes, a.updated_at
            FROM candidates c LEFT JOIN annotations a ON c.id=a.id WHERE c.partition=?
            AND NOT EXISTS (SELECT 1 FROM candidate_exclusions e WHERE e.id=c.id)""",
            (partition,),
        ).fetchall()
    # Opaque IDs plus stable hash ordering avoid grouping candidates by intended label/source.
    result = []
    for row in rows:
        if mode == "unreviewed" and row["human_label"] is not None:
            continue
        if mode == "flagged" and not row["flagged"]:
            continue
        result.append(
            {
                "id": row["id"],
                "partition": row["partition"],
                "input_text": json.loads(row["payload"])["input_text"],
                "human_label": row["human_label"],
                "flagged": bool(row["flagged"]),
                "notes": row["notes"],
                "updated_at": row["updated_at"],
            }
        )
    return sorted(result, key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())


def save_annotation(
    record_id: str, *, label: str | None, flagged: bool, notes: str, path: Path = DEFAULT_DB
) -> None:
    if label is not None and label not in LABELS:
        raise ValueError("Only VALID and GIBBERISH are annotation labels")
    with connection(path) as db:
        stamp = now()
        db.execute(
            """INSERT INTO annotations VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
            human_label=excluded.human_label, flagged=excluded.flagged,
            notes=excluded.notes, updated_at=excluded.updated_at""",
            (record_id, label, int(flagged), notes, stamp),
        )
        db.execute(
            "INSERT INTO revisions(id,payload,timestamp) VALUES (?,?,?)",
            (
                record_id,
                json.dumps({"human_label": label, "flagged": flagged, "notes": notes}),
                stamp,
            ),
        )


def provenance_after_review(record_id: str, path: Path = DEFAULT_DB) -> dict:
    with connection(path) as db:
        row = db.execute(
            """SELECT c.payload FROM candidates c JOIN annotations a ON c.id=a.id
            WHERE c.id=? AND a.human_label IS NOT NULL""",
            (record_id,),
        ).fetchone()
        if row is None:
            raise ValueError("Source metadata is hidden until initial human labeling")
        return json.loads(row[0])


def progress(path: Path = DEFAULT_DB) -> dict:
    result = {}
    for partition in PARTITIONS:
        rows = queue(partition, path=path)
        result[partition] = {
            "imported": len(rows),
            "reviewed": sum(r["human_label"] is not None for r in rows),
            "flagged": sum(r["flagged"] for r in rows),
            "unresolved": sum(r["human_label"] is None or r["flagged"] for r in rows),
            "exportable": sum(r["human_label"] is not None and not r["flagged"] for r in rows),
        }
    return result


def export_dataset(
    partition: str, version: str, destination: Path, *, path: Path = DEFAULT_DB
) -> Path:
    if partition not in PARTITIONS or not re.fullmatch(r"[A-Za-z0-9_-]+", version):
        raise ValueError("Invalid partition or export version")
    with connection(path) as db:
        stored = db.execute(
            """SELECT c.payload, a.human_label, a.flagged, a.notes, a.updated_at
            FROM candidates c LEFT JOIN annotations a ON c.id=a.id
            WHERE c.partition=?
            AND NOT EXISTS (SELECT 1 FROM candidate_exclusions e WHERE e.id=c.id)
            ORDER BY c.id""",
            (partition,),
        ).fetchall()
        archived = [
            dict(r)
            for r in db.execute(
                """SELECT e.* FROM candidate_exclusions e JOIN candidates c ON c.id=e.id
                WHERE c.partition=? ORDER BY e.id""",
                (partition,),
            )
        ]
        snapshots = []
        for row in stored:
            candidate = json.loads(row[0])
            candidate["annotation"] = {
                "label": row[1],
                "flagged": bool(row[2]),
                "notes": row[3],
                "updated_at": row[4],
            }
            snapshots.append(candidate)
    if not snapshots:
        raise ValueError("Partition is empty")
    eligible = []
    for row in snapshots:
        annotation = row["annotation"]
        if annotation["label"] is not None and not annotation["flagged"]:
            eligible.append(
                {
                    **row,
                    "label": annotation["label"],
                    "is_gibberish": annotation["label"] == "GIBBERISH",
                }
            )
    if not eligible:
        raise ValueError("No resolved human annotations to export")
    folder = destination / f"{partition}-{version}"
    folder.mkdir(parents=True, exist_ok=False)
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in eligible)
    data_file = folder / f"{partition}.jsonl"
    data_file.write_text(content, encoding="utf-8")
    sources = {r["provenance"]["source"]: r["provenance"] for r in snapshots}
    manifest = {
        "version": version,
        "partition": partition,
        "policy_version": POLICY_VERSION,
        "created_at": now(),
        "annotation_standard": "single-reviewer",
        "imported": len(snapshots),
        "exported": len(eligible),
        "archived_originals": archived,
        "excluded_ids": [
            r["id"]
            for r in snapshots
            if r["annotation"]["label"] is None or r["annotation"]["flagged"]
        ],
        "complete": len(eligible) == len(snapshots),
        "annotation_complete": len(eligible) == len(snapshots),
        "target_reached": len(eligible)
        >= {"train": 1000, "validation": 200, "test": 400}[partition],
        "target_size": {"train": 1000, "validation": 200, "test": 400}[partition],
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "source_snapshot_sha256": hashlib.sha256(
            json.dumps(snapshots, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "class_counts": dict(Counter(r["label"] for r in eligible)),
        "subtype_counts": dict(Counter(r["subtype"] for r in eligible)),
        "difficulty_counts": dict(Counter(r.get("difficulty", "not_recorded") for r in eligible)),
        "sources": sources,
        "publication_status": "requires source-license compatibility review before publication",
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return folder


def read_export(
    folder: Path, expected_partition: str, *, allow_partial: bool = False
) -> list[dict]:
    """Enforce explicit partition selection for future consumers; no training code here."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if expected_partition not in PARTITIONS or manifest["partition"] != expected_partition:
        raise ValueError("Export partition does not match the requested use")
    if not allow_partial and not (manifest["complete"] and manifest["target_reached"]):
        raise ValueError("Incomplete milestone export; explicit allow_partial is required")
    content = (folder / f"{expected_partition}.jsonl").read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest["sha256"]:
        raise ValueError("Export hash mismatch")
    rows = [json.loads(line) for line in content.decode().splitlines()]
    for row in rows:
        if (
            row["partition"] != expected_partition
            or row["annotation"]["flagged"]
            or row["annotation"]["label"] != row["label"]
            or row["label"] not in LABELS
            or row["is_gibberish"] != (row["label"] == "GIBBERISH")
        ):
            raise ValueError("Export contains unresolved or inconsistent human annotations")
    if len(rows) != manifest["exported"]:
        raise ValueError("Export count mismatch")
    return rows
