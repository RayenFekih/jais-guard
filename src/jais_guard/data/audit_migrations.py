"""Approved audit replacements: additive storage, atomic pointers, immutable predecessors."""

import json
import sqlite3
from pathlib import Path

from jais_guard.data.audit_policy import VERSION, digest


def policy_table(db, name, *, history=False):
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='audit_policy_active'").fetchone():
        return ("all_" if history else "active_") + name
    return name


def _install_schema(db):
    statements = (
        """CREATE TABLE IF NOT EXISTS audit_policy_migrations (
            id TEXT PRIMARY KEY, collection TEXT NOT NULL, policy_version TEXT NOT NULL,
            proposal_fingerprint TEXT NOT NULL, backup_path TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('applying','applied','rolled_back')),
            created_at TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS audit_policy_candidates (
            migration_id TEXT NOT NULL REFERENCES audit_policy_migrations(id),
            id TEXT NOT NULL REFERENCES candidates(id), payload_sha256 TEXT NOT NULL,
            PRIMARY KEY(migration_id,id))""",
        """CREATE TABLE IF NOT EXISTS audit_policy_active (
            id TEXT PRIMARY KEY REFERENCES candidates(id), migration_id TEXT NOT NULL,
            FOREIGN KEY(migration_id,id) REFERENCES audit_policy_candidates(migration_id,id))""",
        """CREATE TABLE IF NOT EXISTS audit_policy_groups (
            migration_id TEXT NOT NULL REFERENCES audit_policy_migrations(id),
            group_id TEXT PRIMARY KEY REFERENCES audit_groups(id),
            role TEXT NOT NULL CHECK(role IN ('predecessor','replacement')))""",
        """CREATE TABLE IF NOT EXISTS versioned_provisional_labels (
            migration_id TEXT NOT NULL, id TEXT NOT NULL, label TEXT NOT NULL
            CHECK(label IN ('VALID','GIBBERISH')),
            origin TEXT NOT NULL, group_id TEXT NOT NULL REFERENCES audit_groups(id),
            created_at TEXT NOT NULL,
            PRIMARY KEY(migration_id,id),
            FOREIGN KEY(migration_id,id) REFERENCES audit_policy_candidates(migration_id,id))""",
        """CREATE TABLE IF NOT EXISTS versioned_audit_samples (
            migration_id TEXT NOT NULL, id TEXT NOT NULL,
            group_id TEXT NOT NULL REFERENCES audit_groups(id),
            dialect TEXT NOT NULL, human_label TEXT,
            revision INTEGER REFERENCES revisions(revision),
            outcome TEXT NOT NULL, notes TEXT NOT NULL, updated_at TEXT NOT NULL,
            PRIMARY KEY(migration_id,id),
            FOREIGN KEY(migration_id,id)
            REFERENCES versioned_provisional_labels(migration_id,id))""",
        """CREATE TABLE IF NOT EXISTS versioned_review_requests (
            migration_id TEXT NOT NULL, id TEXT NOT NULL, reason TEXT NOT NULL,
            PRIMARY KEY(migration_id,id),
            FOREIGN KEY(migration_id,id) REFERENCES audit_policy_candidates(migration_id,id))""",
        """CREATE TABLE IF NOT EXISTS audit_policy_events (
            event INTEGER PRIMARY KEY,
            migration_id TEXT NOT NULL REFERENCES audit_policy_migrations(id),
            action TEXT NOT NULL, details TEXT NOT NULL, created_at TEXT NOT NULL)""",
    )
    for statement in statements:
        db.execute(statement)  # executescript would implicitly commit the migration transaction.
    columns = {
        "provisional_labels": "id,label,origin,group_id,created_at",
        "audit_samples": "id,group_id,dialect,human_label,revision,outcome,notes,updated_at",
        "review_requests": "id,reason",
    }
    for name, fields in columns.items():
        db.execute(f"""CREATE VIEW IF NOT EXISTS all_{name} AS
            SELECT {fields},NULL AS migration_id FROM {name}
            UNION ALL SELECT {fields},migration_id FROM versioned_{name}""")
        db.execute(f"""CREATE VIEW IF NOT EXISTS active_{name} AS
            SELECT p.* FROM all_{name} p LEFT JOIN audit_policy_active a ON a.id=p.id
            WHERE (a.id IS NULL AND p.migration_id IS NULL) OR a.migration_id=p.migration_id""")
    inactive = """EXISTS(SELECT 1 FROM audit_policy_groups v
        JOIN audit_policy_migrations m ON m.id=v.migration_id
        WHERE v.group_id={group} AND
        ((v.role='predecessor' AND m.status='applied') OR
         (v.role='replacement' AND m.status='rolled_back')))"""
    for table, key in (
        ("audit_groups", "id"),
        ("audit_samples", "group_id"),
        ("audit_events", "group_id"),
        ("versioned_audit_samples", "group_id"),
    ):
        for operation in ("UPDATE", "DELETE"):
            condition = inactive.format(group=f"OLD.{key}")
            db.execute(f"""CREATE TRIGGER IF NOT EXISTS guard_{table}_{operation.lower()}
                BEFORE {operation} ON {table} WHEN {condition}
                BEGIN SELECT RAISE(ABORT,'Historical audit policy is immutable'); END""")
    for table in ("audit_samples", "audit_events"):
        condition = inactive.format(group="NEW.group_id")
        db.execute(f"""CREATE TRIGGER IF NOT EXISTS guard_{table}_insert
            BEFORE INSERT ON {table} WHEN {condition}
            BEGIN SELECT RAISE(ABORT,'Historical audit policy is immutable'); END""")
    for table in ("provisional_labels", "review_requests"):
        for operation in ("UPDATE", "DELETE"):
            db.execute(f"""CREATE TRIGGER IF NOT EXISTS guard_{table}_{operation.lower()}
                BEFORE {operation} ON {table}
                WHEN EXISTS(SELECT 1 FROM audit_policy_candidates WHERE id=OLD.id)
                BEGIN SELECT RAISE(ABORT,'Original policy records are immutable'); END""")
    for table in (
        "versioned_provisional_labels",
        "versioned_review_requests",
        "audit_policy_candidates",
        "audit_policy_events",
        "audit_events",
        "audit_policy_groups",
    ):
        for operation in ("UPDATE", "DELETE"):
            db.execute(f"""CREATE TRIGGER IF NOT EXISTS immutable_{table}_{operation.lower()}
                BEFORE {operation} ON {table}
                BEGIN SELECT RAISE(ABORT,'Policy history is append-only'); END""")
    for table in (
        "versioned_provisional_labels",
        "versioned_audit_samples",
        "versioned_review_requests",
    ):
        db.execute(f"""CREATE TRIGGER IF NOT EXISTS freeze_insert_{table}
            BEFORE INSERT ON {table}
            WHEN (SELECT status FROM audit_policy_migrations WHERE id=NEW.migration_id)!='applying'
            BEGIN SELECT RAISE(ABORT,'Frozen policy membership is immutable'); END""")
    for table in ("audit_samples", "versioned_audit_samples"):
        db.execute(f"""CREATE TRIGGER IF NOT EXISTS frozen_membership_{table}
            BEFORE UPDATE OF id,group_id,dialect ON {table}
            BEGIN SELECT RAISE(ABORT,'Frozen audit membership is immutable'); END""")


def _connect(path):
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


def _backup_locked(path, backup_dir, stamp):
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"before-approved-v2-{stamp.replace(':', '-')}.sqlite3"
    if backup.exists():
        raise FileExistsError(backup)
    # A separate reader can back up the committed state while our writer holds BEGIN IMMEDIATE.
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as source:
        with sqlite3.connect(backup) as destination:
            source.backup(destination)
            if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Backup integrity check failed")
    return backup


def _verify(artifact, collection, path):
    from jais_guard.data.annotations import migration_preview

    if artifact["fingerprint"] != digest({k: v for k, v in artifact.items() if k != "fingerprint"}):
        raise ValueError("Migration artifact fingerprint mismatch")
    plan = artifact["proposal"]
    if plan["fingerprint"] != digest({k: v for k, v in plan.items() if k != "fingerprint"}):
        raise ValueError("Policy projection fingerprint mismatch")
    if plan["policy_version"] != VERSION or plan["batch_id"] != collection + "-v2-proposal":
        raise ValueError("Wrong collection or policy version")
    processes = {m["id"]: g["boundary"] for g in plan["groups"] for m in g["members"]}
    try:
        fresh = migration_preview(collection, seed=plan["seed"], boundaries=processes, path=path)
    except KeyError as error:
        raise ValueError("Candidate eligibility changed; regenerate the proposal") from error
    expected = {k: v for k, v in plan.items() if k not in ("fingerprint", "regrouping_evidence")}
    if not fresh["safe_to_propose"] or expected != {
        k: v for k, v in fresh.items() if k != "fingerprint"
    }:
        raise ValueError("Migration proposal is stale or review has started; regenerate it")
    return plan


def _activate(db, migration_id, ids):
    for record_id in sorted(ids):
        db.execute("INSERT INTO audit_policy_active VALUES (?,?)", (record_id, migration_id))


def apply_audit_migration(artifact, collection, *, path, expected_fingerprint, backup_dir=None):
    """Apply an explicitly approved artifact. No candidate or human annotation is rewritten."""
    from jais_guard.data.annotations import _audit_event, _workflow, now

    path = Path(path)
    if expected_fingerprint != artifact["fingerprint"]:
        raise ValueError("Explicit approval does not match this proposal")
    migration_id = "v2-" + artifact["fingerprint"][:20]
    db = _connect(path)
    try:
        db.execute("BEGIN IMMEDIATE")
        if db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='audit_policy_migrations'"
        ).fetchone():
            if db.execute(
                "SELECT 1 FROM audit_policy_migrations WHERE id=?", (migration_id,)
            ).fetchone():
                raise ValueError("Migration already recorded; do not resample or reapply")
        plan = _verify(artifact, collection, path)
        stored = {r[0]: r[1] for r in db.execute("SELECT id,payload FROM candidates")}
        ids = {i for g in plan["groups"] for i in [m["id"] for m in g["members"]]} | set(
            plan["review_requests"]
        )
        if len(ids) != plan["candidates"]:
            raise ValueError("Incomplete record-level mapping")
        before = {
            t: [tuple(r) for r in db.execute(f"SELECT * FROM {t}")]
            for t in (
                "candidates",
                "annotations",
                "revisions",
                "provisional_labels",
                "audit_samples",
                "review_requests",
                "candidate_exclusions",
                "state",
            )
        }
        append_only = {
            t: {tuple(r) for r in db.execute(f"SELECT * FROM {t}")}
            for t in ("audit_groups", "audit_events", "collection_batches")
        }
        stamp = now()
        backup = _backup_locked(
            path, Path(backup_dir) if backup_dir else path.parent / "backups", stamp
        )
        _install_schema(db)
        if any(
            db.execute("SELECT 1 FROM audit_policy_active WHERE id=?", (i,)).fetchone() for i in ids
        ):
            raise ValueError("Candidates already have a replacement policy")
        db.execute(
            "INSERT INTO audit_policy_migrations VALUES (?,?,?,?,?,'applying',?)",
            (
                migration_id,
                collection,
                VERSION,
                artifact["fingerprint"],
                str(backup.resolve()),
                stamp,
            ),
        )
        for record_id in sorted(ids):
            db.execute(
                "INSERT INTO audit_policy_candidates VALUES (?,?,?)",
                (migration_id, record_id, digest(json.loads(stored[record_id]))),
            )
        for group_id in plan["original_group_ids"]:
            db.execute(
                "INSERT INTO audit_policy_groups VALUES (?,?,'predecessor')",
                (migration_id, group_id),
            )
        db.execute(
            "INSERT INTO collection_batches VALUES (?,?,?)", (plan["batch_id"], plan["seed"], stamp)
        )
        for group in plan["groups"]:
            db.execute(
                "INSERT INTO audit_groups VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    group["id"],
                    plan["batch_id"],
                    group["source"],
                    group["source_revision"],
                    group["method"],
                    group["subtype"],
                    group["rule_id"],
                    VERSION,
                    group["risk"],
                    plan["seed"],
                    "pending",
                    stamp,
                ),
            )
            db.execute(
                "INSERT INTO audit_policy_groups VALUES (?,?,'replacement')",
                (migration_id, group["id"]),
            )
            for member in group["members"]:
                db.execute(
                    "INSERT INTO versioned_provisional_labels VALUES (?,?,?,?,?,?)",
                    (
                        migration_id,
                        member["id"],
                        member["label"],
                        member["origin"],
                        group["id"],
                        stamp,
                    ),
                )
                if member["id"] in group["sampled_ids"]:
                    db.execute(
                        "INSERT INTO versioned_audit_samples "
                        "VALUES (?,?,?,?,NULL,NULL,'pending','',?)",
                        (migration_id, member["id"], group["id"], member["dialect"], stamp),
                    )
            _audit_event(
                db,
                group["id"],
                dict(
                    action="frozen",
                    migration_id=migration_id,
                    proposal_fingerprint=artifact["fingerprint"],
                    boundary=group["boundary"],
                    members=[m["id"] for m in group["members"]],
                    sampled_ids=group["sampled_ids"],
                    strata=group["strata"],
                ),
            )
        for record_id, reason in plan["review_requests"].items():
            db.execute(
                "INSERT INTO versioned_review_requests VALUES (?,?,?)",
                (migration_id, record_id, reason),
            )
        _activate(db, migration_id, ids)
        db.execute(
            "UPDATE audit_policy_migrations SET status='applied' WHERE id=?", (migration_id,)
        )
        db.execute(
            "INSERT INTO audit_policy_events(migration_id,action,details,created_at) "
            "VALUES (?,'applied',?,?)",
            (
                migration_id,
                json.dumps({"records": len(ids), "reviews": plan["required_reviews"]}),
                stamp,
            ),
        )
        active = [r for r in _workflow(db, "train") if r["id"] in ids]
        if (
            len(active) != len(ids)
            or any(r["exportable"] or r["human_label"] is not None for r in active)
            or sum(r["flagged"] for r in active) != plan["required_reviews"]
        ):
            raise ValueError("Installed policy failed eligibility/review-count checks")
        for table, values in before.items():
            if [tuple(r) for r in db.execute(f"SELECT * FROM {table}")] != values:
                raise ValueError(f"Migration attempted to rewrite preserved table: {table}")
        for table, values in append_only.items():
            if not values <= {tuple(r) for r in db.execute(f"SELECT * FROM {table}")}:
                raise ValueError(f"Migration attempted to rewrite original audit history: {table}")
        if db.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Migration foreign-key validation failed")
        db.commit()
        return dict(
            migration_id=migration_id,
            backup_path=str(backup.resolve()),
            status="applied",
            records=len(ids),
            audit_groups=len(plan["groups"]),
            audit_samples=plan["audit_samples"],
            manual_only=plan["manual_only"],
            required_reviews=plan["required_reviews"],
        )
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def rollback_audit_migration(migration_id, *, path):
    """Restore v1 pointers before review starts; retain all replacement records as history."""
    from jais_guard.data.annotations import now

    db = _connect(path)
    try:
        db.execute("BEGIN IMMEDIATE")
        migration = db.execute(
            "SELECT * FROM audit_policy_migrations WHERE id=?", (migration_id,)
        ).fetchone()
        if migration is None or migration["status"] != "applied":
            raise ValueError("Migration is not active")
        ids = {
            r[0]
            for r in db.execute(
                "SELECT id FROM audit_policy_candidates WHERE migration_id=?", (migration_id,)
            )
        }
        if any(db.execute("SELECT 1 FROM annotations WHERE id=?", (i,)).fetchone() for i in ids):
            raise ValueError(
                "Review has started; rollback needs a separate history-preserving plan"
            )
        if db.execute(
            "SELECT 1 FROM audit_groups WHERE id IN (SELECT group_id FROM audit_policy_groups "
            "WHERE migration_id=? AND role='replacement') AND status!='pending'",
            (migration_id,),
        ).fetchone():
            raise ValueError("Audit decisions changed; rollback is blocked")
        if db.execute(
            "SELECT 1 FROM audited_exports WHERE group_id IN "
            "(SELECT group_id FROM audit_policy_groups "
            "WHERE migration_id=? AND role='replacement')",
            (migration_id,),
        ).fetchone():
            raise ValueError("Policy has exports; rollback requires explicit export reconciliation")
        count = db.execute(
            "SELECT COUNT(*) FROM audit_policy_active WHERE migration_id=?", (migration_id,)
        ).fetchone()[0]
        if count != len(ids):
            raise ValueError("Active policy pointers changed")
        db.execute("DELETE FROM audit_policy_active WHERE migration_id=?", (migration_id,))
        db.execute(
            "UPDATE audit_policy_migrations SET status='rolled_back' WHERE id=?", (migration_id,)
        )
        db.execute(
            "INSERT INTO audit_policy_events(migration_id,action,details,created_at) "
            "VALUES (?,'rolled_back','{}',?)",
            (migration_id, now()),
        )
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()
