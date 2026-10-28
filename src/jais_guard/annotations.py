"""Local annotation storage. Human labels and source intentions never share a column."""

import hashlib
import json
import math
import random
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
PROVISIONAL_POLICY_VERSION = "provisional-v1"
# Source/subset approval is deliberately narrower than license admission.
PUBLIC_RULES = {
    "google-research-datasets/tydiqa": ("da78f23f9119363459acbaf46bf89426ff26c259", "question"),
    "AmazonScience/massive": ("ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740", "localized_request"),
}


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
    _upgrade_audits(db, path)
    try:
        with db:
            yield db
    finally:
        db.close()


def _upgrade_audits(db, path):
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='provisional_labels'").fetchone():
        return
    if db.execute("SELECT 1 FROM candidates LIMIT 1").fetchone():
        backup = path.parent / "backups" / f"before-provisional-{now().replace(':', '-')}.sqlite3"
        backup.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(backup) as destination:
            db.backup(destination)
    # The additive upgrade is atomic; no candidate or human annotation rows are rewritten.
    db.executescript("""
        BEGIN IMMEDIATE;
        CREATE TABLE IF NOT EXISTS collection_batches (
            id TEXT PRIMARY KEY, seed INTEGER NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS audit_groups (
            id TEXT PRIMARY KEY, batch_id TEXT NOT NULL REFERENCES collection_batches(id),
            source TEXT NOT NULL, source_revision TEXT NOT NULL, method TEXT NOT NULL,
            subtype TEXT NOT NULL, rule_id TEXT NOT NULL, policy_version TEXT NOT NULL,
            risk TEXT NOT NULL, seed INTEGER NOT NULL, status TEXT NOT NULL,
            created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS provisional_labels (
            id TEXT PRIMARY KEY REFERENCES candidates(id),
            label TEXT NOT NULL CHECK(label IN ('VALID','GIBBERISH')),
            origin TEXT NOT NULL, group_id TEXT NOT NULL REFERENCES audit_groups(id),
            created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS review_requests (
            id TEXT PRIMARY KEY REFERENCES candidates(id), reason TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS audit_samples (
            id TEXT PRIMARY KEY REFERENCES provisional_labels(id),
            group_id TEXT NOT NULL REFERENCES audit_groups(id), dialect TEXT NOT NULL,
            human_label TEXT, revision INTEGER REFERENCES revisions(revision),
            outcome TEXT NOT NULL, notes TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS audit_events (
            event INTEGER PRIMARY KEY, group_id TEXT NOT NULL REFERENCES audit_groups(id),
            payload TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS disabled_rules (
            rule_id TEXT PRIMARY KEY, group_id TEXT NOT NULL REFERENCES audit_groups(id),
            created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS audited_exports (
            export_path TEXT NOT NULL, group_id TEXT NOT NULL REFERENCES audit_groups(id),
            created_at TEXT NOT NULL, PRIMARY KEY(export_path,group_id));
        COMMIT;
    """)


def benchmark_references() -> list[dict]:
    path = ROOT / "data/gibberish.jsonl"
    return [
        {"id": r["id"], "input_text": r["input_text"], "partition": "religious_" + r["split"]}
        for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
    ]


def import_candidates(
    rows: list[dict],
    path: Path = DEFAULT_DB,
    *,
    provisional_batch: str | None = None,
    seed: int = 0,
) -> int:
    if provisional_batch is not None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", provisional_batch):
            raise ValueError("A unique batch ID is required for provisional import")
        if any(r.get("partition") != "train" for r in rows):
            raise ValueError("Provisional labels are training-only")
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
        if provisional_batch is not None:
            _prepare_provisional_batch(db, rows, provisional_batch, seed)
    return len(rows)


def _proposal(row):
    """Conservative, inspectable heuristics; an intended label is never sufficient alone."""
    p, g = row["provenance"], row.get("generation", {})
    method, subtype, text = row["source_method"], row["subtype"], row["input_text"]
    risk = "higher" if row.get("difficulty") == "hard" else "standard"
    if method == "public":
        approved = PUBLIC_RULES.get(p["source"])
        context_dependent = re.search(r"أكمل|اكمل|كمل|السابق|التالي|أعلاه|الفقرة|ترجم|لخص", text)
        if (
            not approved
            or (p["revision"], subtype) != approved
            or row["intended_label"] != "VALID"
            or context_dependent
            or not 8 <= len(text) <= 300
            or len(text.split()) < 2
            or (subtype == "question" and not text.rstrip().endswith(("؟", "?")))
        ):
            return None
        label, origin = "VALID", "public-source-rule"
    elif method == "python":
        # Only long, contiguous walks on this consonant-heavy keyboard row qualify.
        # Repetition, shuffling, other rows, and ordinary short strings stay manual.
        keyboard = "ضصثقفغعهخحج"
        compact = text.replace(" ", "")
        walks = {layout[i : i + 8] for layout in (keyboard, keyboard[::-1]) for i in range(4)}
        if (
            subtype != "keyboard_noise"
            or g.get("recipe") != "keyboard_noise"
            or g.get("parent_id")
            or row["intended_label"] != "GIBBERISH"
            or not 10 <= len(compact) <= 30
            or not set(compact) <= set(keyboard)
            or not any(walk in compact for walk in walks)
        ):
            return None
        label, origin = "GIBBERISH", "python-rule"
    else:
        if not (g.get("instructions_file") or g.get("instructions")) or not isinstance(
            g.get("model_metadata"), dict
        ):
            return None
        label, origin, risk = row["intended_label"], "codex-proposal", "higher"
    rule = json.dumps(
        [PROVISIONAL_POLICY_VERSION, method, p["source"], p.get("revision", "unknown"), subtype],
        ensure_ascii=False,
    )
    return label, origin, risk, rule


def _prepare_provisional_batch(db, rows, batch, seed):
    if db.execute("SELECT 1 FROM collection_batches WHERE id=?", (batch,)).fetchone():
        raise ValueError("Batch already frozen; use a new batch ID, not a new sample")
    db.execute("INSERT INTO collection_batches VALUES (?,?,?)", (batch, seed, now()))
    groups = {}
    for row in rows:
        if db.execute("SELECT 1 FROM candidate_exclusions WHERE id=?", (row["id"],)).fetchone():
            continue
        proposal = _proposal(row)
        reason = "uncertainty: no approved automatic labeling rule"
        if proposal:
            label, origin, risk, rule = proposal
            disabled = db.execute(
                "SELECT 1 FROM disabled_rules WHERE rule_id=?", (rule,)
            ).fetchone()
            if disabled:
                reason = "failed audit: automatic labeling rule disabled"
            else:
                # Risk is raised for the whole group if any member requires it.
                group_id = hashlib.sha256((batch + rule).encode()).hexdigest()[:20]
                groups.setdefault(group_id, []).append((row, label, origin, risk, rule))
                continue
        db.execute("INSERT INTO review_requests VALUES (?,?)", (row["id"], reason))
    for group_id, members in groups.items():
        first, _, _, _, rule = members[0]
        risk = "higher" if any(m[3] == "higher" for m in members) else "standard"
        p = first["provenance"]
        db.execute(
            "INSERT INTO audit_groups VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                group_id,
                batch,
                p["source"],
                p.get("revision", "unknown"),
                first["source_method"],
                first["subtype"],
                rule,
                PROVISIONAL_POLICY_VERSION,
                risk,
                seed,
                "pending",
                now(),
            ),
        )
        strata = {}
        for row, label, origin, _, _ in members:
            db.execute(
                "INSERT INTO provisional_labels VALUES (?,?,?,?,?)",
                (row["id"], label, origin, group_id, now()),
            )
            strata.setdefault(row.get("dialect") or "unknown", []).append(row["id"])
        sampled = []
        for dialect, ids in sorted(strata.items()):
            rate, minimum = (0.5, 20) if risk == "higher" else (0.2, 10)
            size = min(len(ids), max(minimum, math.ceil(rate * len(ids))))
            rng = random.Random(f"{seed}:{group_id}:{dialect}")
            for record_id in rng.sample(sorted(ids), size):
                db.execute(
                    "INSERT INTO audit_samples VALUES (?,?,?,NULL,NULL,'pending','',?)",
                    (record_id, group_id, dialect, now()),
                )
                sampled.append(record_id)
        _audit_event(
            db,
            group_id,
            {
                "action": "frozen",
                "members": sorted(m[0]["id"] for m in members),
                "sampled_ids": sorted(sampled),
                "strata": {k: len(v) for k, v in strata.items()},
            },
        )


def _audit_event(db, group_id, payload):
    db.execute(
        "INSERT INTO audit_events(group_id,payload,created_at) VALUES (?,?,?)",
        (group_id, json.dumps(payload, ensure_ascii=False), now()),
    )


def _refresh_audit(db, record_id):
    sample = db.execute(
        """SELECT s.*, p.label AS proposed FROM audit_samples s
        JOIN provisional_labels p ON p.id=s.id WHERE s.id=?""",
        (record_id,),
    ).fetchone()
    annotation = db.execute("SELECT * FROM annotations WHERE id=?", (record_id,)).fetchone()
    revision = db.execute(
        "SELECT MAX(revision) FROM revisions WHERE id=?", (record_id,)
    ).fetchone()[0]
    resolved = annotation["human_label"] is not None and not annotation["flagged"]
    if sample is None:
        # A voluntary review can also discover an error; it is not a random audit sample.
        proposal = db.execute(
            "SELECT * FROM provisional_labels WHERE id=?", (record_id,)
        ).fetchone()
        if proposal is not None and resolved:
            group = db.execute(
                "SELECT * FROM audit_groups WHERE id=?", (proposal["group_id"],)
            ).fetchone()
            outcome = "agreement" if annotation["human_label"] == proposal["label"] else "error"
            if outcome == "error":
                db.execute("UPDATE audit_groups SET status='failed' WHERE id=?", (group["id"],))
                db.execute(
                    "INSERT OR IGNORE INTO disabled_rules VALUES (?,?,?)",
                    (group["rule_id"], group["id"], now()),
                )
            _audit_event(
                db,
                group["id"],
                {
                    "action": "manual_review",
                    "id": record_id,
                    "revision": revision,
                    "outcome": outcome,
                },
            )
        return
    outcome = (
        ("agreement" if annotation["human_label"] == sample["proposed"] else "error")
        if resolved
        else "pending"
    )
    db.execute(
        "UPDATE audit_samples SET human_label=?,revision=?,outcome=?,notes=?,updated_at=? "
        "WHERE id=?",
        (annotation["human_label"], revision, outcome, annotation["notes"], now(), record_id),
    )
    group_id = sample["group_id"]
    group = db.execute("SELECT * FROM audit_groups WHERE id=?", (group_id,)).fetchone()
    counts = Counter(
        r[0] for r in db.execute("SELECT outcome FROM audit_samples WHERE group_id=?", (group_id,))
    )
    status = (
        "failed"
        if counts["error"] or group["status"] == "failed"
        else ("pending" if counts["pending"] else "passed")
    )
    db.execute("UPDATE audit_groups SET status=? WHERE id=?", (status, group_id))
    if status == "failed":
        db.execute(
            "INSERT OR IGNORE INTO disabled_rules VALUES (?,?,?)",
            (group["rule_id"], group_id, now()),
        )
    _audit_event(
        db,
        group_id,
        {
            "action": "review",
            "id": record_id,
            "revision": revision,
            "outcome": outcome,
            "status": status,
        },
    )


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


def _audit_reports(db):
    reports = []
    for group in db.execute("SELECT * FROM audit_groups ORDER BY created_at,id").fetchall():
        report = dict(group)
        events = [
            json.loads(r[0])
            for r in db.execute(
                "SELECT payload FROM audit_events WHERE group_id=? ORDER BY event", (group["id"],)
            )
        ]
        samples = [
            dict(r)
            for r in db.execute(
                "SELECT * FROM audit_samples WHERE group_id=? ORDER BY id", (group["id"],)
            )
        ]
        members = [
            json.loads(r[0])
            for r in db.execute(
                "SELECT c.payload FROM candidates c JOIN provisional_labels p ON p.id=c.id "
                "WHERE p.group_id=?",
                (group["id"],),
            )
        ]
        coverage = {}
        for dialect in sorted({r["dialect"] for r in members}):
            selected = [s for s in samples if s["dialect"] == dialect]
            coverage[dialect] = {
                "population": sum(r["dialect"] == dialect for r in members),
                "sampled": len(selected),
                "resolved": sum(s["outcome"] != "pending" for s in selected),
                "errors": sum(s["outcome"] == "error" for s in selected),
            }
        report.update(
            population=len(members),
            sampled=len(samples),
            resolved=sum(s["outcome"] != "pending" for s in samples),
            errors=len({e["id"] for e in events if e.get("outcome") == "error"}),
            sample_errors=sum(s["outcome"] == "error" for s in samples),
            coverage=coverage,
            samples=samples,
            affected_exports=[
                r[0]
                for r in db.execute(
                    "SELECT export_path FROM audited_exports WHERE group_id=? ORDER BY export_path",
                    (group["id"],),
                )
            ]
            if group["status"] != "passed"
            else [],
        )
        reports.append(report)
    return reports


def audit_report(path: Path = DEFAULT_DB) -> list[dict]:
    """Administrative report. Never display source/label metadata in the blind review view."""
    with connection(path) as db:
        return _audit_reports(db)


def _workflow(db, partition):
    reports = {r["id"]: r for r in _audit_reports(db)}
    rows = db.execute(
        """SELECT c.id,c.partition,c.payload,a.human_label,COALESCE(a.flagged,0) AS human_flag,
        COALESCE(a.notes,'') AS notes,a.updated_at,p.label AS provisional_label,p.origin,
        p.group_id,q.reason,s.id AS sampled
        FROM candidates c LEFT JOIN annotations a ON c.id=a.id
        LEFT JOIN provisional_labels p ON c.id=p.id
        LEFT JOIN review_requests q ON c.id=q.id
        LEFT JOIN audit_samples s ON c.id=s.id
        WHERE c.partition=? AND NOT EXISTS(SELECT 1 FROM candidate_exclusions e WHERE e.id=c.id)
        ORDER BY c.id""",
        (partition,),
    ).fetchall()
    result = []
    for stored in rows:
        row = dict(stored)
        row["candidate"] = json.loads(row.pop("payload"))
        group = reports.get(row["group_id"])
        resolved = row["human_label"] is not None and not row["human_flag"]
        reasons = []
        if row["human_flag"]:
            reasons.append("quality flag" if "near-duplicate" in row["notes"] else "uncertainty")
        if not resolved:
            if row["reason"]:
                reasons.append(row["reason"])
            if group and group["status"] == "failed":
                reasons.append("failed audit")
            if row["sampled"]:
                reasons.append("audit sample")
        row["review_reasons"] = reasons
        row["flagged"] = bool(reasons)
        row["status"] = (
            "flagged"
            if reasons
            else (
                "human-reviewed"
                if resolved
                else "provisional"
                if row["provisional_label"]
                else "pending"
            )
        )
        row["audit"] = (
            None
            if not group
            else {
                "group_id": group["id"],
                "batch_id": group["batch_id"],
                "rule_id": group["rule_id"],
                "policy_version": group["policy_version"],
                "status": group["status"],
                "sampled": bool(row["sampled"]),
                "sample_size": group["sampled"],
                "error_count": group["errors"],
                "risk": group["risk"],
            }
        )
        row["exportable"] = not reasons and (
            resolved or (row["provisional_label"] is not None and group["status"] == "passed")
        )
        result.append(row)
    return result


def queue(partition: str, mode: str = "all", path: Path = DEFAULT_DB) -> list[dict]:
    if partition not in PARTITIONS or mode not in {
        "all",
        "unreviewed",
        "flagged",
        "human-reviewed",
        "provisional",
        "manual-review",
    }:
        raise ValueError("Unknown partition or queue")
    with connection(path) as db:
        rows = _workflow(db, partition)
    # Opaque IDs plus stable hash ordering avoid grouping candidates by intended label/source.
    result = []
    for row in rows:
        if mode == "unreviewed" and row["human_label"] is not None:
            continue
        if mode in {"flagged", "manual-review"} and not row["flagged"]:
            continue
        if mode in {"human-reviewed", "provisional"} and row["status"] != mode:
            continue
        result.append(
            {
                "id": row["id"],
                "partition": row["partition"],
                "input_text": row["candidate"]["input_text"],
                "human_label": row["human_label"],
                "flagged": bool(row["flagged"]),
                "notes": row["notes"],
                "updated_at": row["updated_at"],
                "status": row["status"],
                "review_reasons": row["review_reasons"],
                "higher_risk": bool(row["audit"] and row["audit"]["risk"] == "higher"),
                "exportable": row["exportable"],
            }
        )

    def priority(row):
        reasons = " ".join(row["review_reasons"])
        return (
            0
            if "failed audit" in reasons
            else 1
            if "uncertainty" in reasons
            else 2
            if "quality flag" in reasons
            else 3
            if row["higher_risk"]
            else 4
        )

    return sorted(
        result,
        key=lambda row: (
            priority(row) if mode in {"flagged", "manual-review"} else 0,
            hashlib.sha256(row["id"].encode()).hexdigest(),
        ),
    )


def save_annotation(
    record_id: str, *, label: str | None, flagged: bool, notes: str, path: Path = DEFAULT_DB
) -> None:
    if label is not None and label not in LABELS:
        raise ValueError("Only VALID and GIBBERISH are annotation labels")
    with connection(path) as db:
        if (
            label is None
            and not flagged
            and db.execute(
                "SELECT 1 FROM audit_samples WHERE id=? "
                "UNION SELECT 1 FROM review_requests WHERE id=?",
                (record_id, record_id),
            ).fetchone()
        ):
            raise ValueError("Choose a human label before resolving review")
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
        _refresh_audit(db, record_id)


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
            "human_reviewed": sum(r["status"] == "human-reviewed" for r in rows),
            "provisional": sum(r["status"] == "provisional" for r in rows),
            "export_eligible": sum(r["exportable"] for r in rows),
        }
    return result


def export_dataset(
    partition: str,
    version: str,
    destination: Path,
    *,
    path: Path = DEFAULT_DB,
    include_provisional: bool = False,
) -> Path:
    if partition not in PARTITIONS or not re.fullmatch(r"[A-Za-z0-9_-]+", version):
        raise ValueError("Invalid partition or export version")
    if include_provisional and partition != "train":
        raise ValueError("Validation and test exports must be human-reviewed only")
    with connection(path) as db:
        stored = _workflow(db, partition)
        reports = _audit_reports(db)
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
            candidate = dict(row["candidate"])
            candidate.update(
                annotation={
                    "label": row["human_label"],
                    "flagged": row["flagged"],
                    "notes": row["notes"],
                    "updated_at": row["updated_at"],
                },
                review_status=row["status"],
                label_origin="human" if row["human_label"] is not None else row["origin"],
                original_provisional_label=row["provisional_label"],
                audit=row["audit"],
            )
            snapshots.append(candidate)
    if not snapshots:
        raise ValueError("Partition is empty")
    eligible = []
    for row in snapshots:
        annotation = row["annotation"]
        human = row["review_status"] == "human-reviewed"
        automatic = (
            include_provisional
            and row["review_status"] == "provisional"
            and row["audit"]
            and row["audit"]["status"] == "passed"
        )
        if not annotation["flagged"] and (human or automatic):
            label = annotation["label"] if human else row["original_provisional_label"]
            eligible.append({**row, "label": label, "is_gibberish": label == "GIBBERISH"})
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
        "schema_version": 2,
        "include_provisional": include_provisional,
        "partition": partition,
        "policy_version": POLICY_VERSION,
        "created_at": now(),
        "annotation_standard": "single-reviewer with separately identified provisional labels",
        "imported": len(snapshots),
        "exported": len(eligible),
        "archived_originals": archived,
        "excluded_ids": sorted({r["id"] for r in snapshots} - {r["id"] for r in eligible}),
        "review_status_counts": dict(Counter(r["review_status"] for r in eligible)),
        "label_origin_counts": dict(Counter(r["label_origin"] for r in eligible)),
        "audits": reports,
        "complete": len(eligible) == len(snapshots),
        "annotation_complete": all(r["review_status"] == "human-reviewed" for r in snapshots),
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
    with connection(path) as db:
        for group_id in {r["audit"]["group_id"] for r in eligible if r["audit"]}:
            db.execute(
                "INSERT INTO audited_exports VALUES (?,?,?)",
                (str(folder.resolve()), group_id, now()),
            )
    return folder


def read_export(
    folder: Path,
    expected_partition: str,
    *,
    allow_partial: bool = False,
    allow_provisional: bool = False,
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
            or row["label"] not in LABELS
            or row["is_gibberish"] != (row["label"] == "GIBBERISH")
        ):
            raise ValueError("Export contains unresolved or inconsistent human annotations")
        if manifest.get("schema_version", 1) == 1:
            valid = row["annotation"]["label"] == row["label"]
        elif manifest["schema_version"] == 2:
            human = row["review_status"] == "human-reviewed"
            audit = row.get("audit") or {}
            if human:
                valid = (
                    row["label_origin"] == "human" and row["annotation"]["label"] == row["label"]
                )
            else:
                valid = (
                    allow_provisional
                    and expected_partition == "train"
                    and manifest.get("include_provisional")
                    and row["review_status"] == "provisional"
                    and row["label_origin"]
                    in {"public-source-rule", "python-rule", "codex-proposal"}
                    and row["annotation"]["label"] is None
                    and row["original_provisional_label"] == row["label"]
                    and audit.get("status") == "passed"
                    and not audit.get("sampled", True)
                    and audit.get("error_count") == 0
                    and audit.get("sample_size", 0) > 0
                    and all(
                        audit.get(k) for k in ("group_id", "batch_id", "rule_id", "policy_version")
                    )
                )
                group = next(
                    (g for g in manifest.get("audits", []) if g["id"] == audit.get("group_id")),
                    None,
                )
                valid = bool(
                    valid
                    and group
                    and group["status"] == "passed"
                    and group["errors"] == 0
                    and group["sampled"] == audit["sample_size"]
                    and group["rule_id"] == audit["rule_id"]
                    and group["batch_id"] == audit["batch_id"]
                    and row["id"] not in {s["id"] for s in group["samples"]}
                )
        else:
            raise ValueError("Unsupported export schema version")
        if not valid:
            raise ValueError(
                "Export contains inconsistent review metadata or disallowed provisional labels"
            )
    if len(rows) != manifest["exported"]:
        raise ValueError("Export count mismatch")
    if manifest.get("schema_version") == 2 and (
        dict(Counter(r["review_status"] for r in rows)) != manifest["review_status_counts"]
        or dict(Counter(r["label_origin"] for r in rows)) != manifest["label_origin_counts"]
    ):
        raise ValueError("Export review metadata counts do not match the manifest")
    return rows
