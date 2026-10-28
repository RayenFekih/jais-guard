"""Run locally: .venv/bin/python -m streamlit run apps/annotate.py"""

import hashlib
import html
import json
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jais_guard.annotations import (
    DEFAULT_DB,
    ROOT,
    audit_report,
    export_dataset,
    get_state,
    import_candidates,
    preview_import,
    progress,
    provenance_after_review,
    queue,
    save_annotation,
    set_state,
)

st.set_page_config(page_title="JaisGuard annotation", layout="centered")
# Local paths are configured explicitly, including for isolated application tests.
db_path = Path(
    st.session_state.get("db_path", os.environ.get("JAISGUARD_ANNOTATION_DB", DEFAULT_DB))
)
st.title("JaisGuard · Arabic annotation")
st.caption("Single-reviewer workflow. Provisional answers stay hidden during human review.")
with st.expander("Labeling policy"):
    st.write(
        "VALID: reasonably recoverable meaning or communicative intent. GIBBERISH: no "
        "reasonably recoverable meaning without inventing missing meaning. Accept dialects, "
        "typos, short/incomplete requests, greetings, metaphors, unusual coherent expressions "
        "and factual mistakes. Relevance, safety and answerability do not decide the label. "
        "Prefer VALID when a reasonable interpretation exists. Flag uncertainty for review."
    )

with st.sidebar:
    st.header("Collection")
    if "active_partition" not in st.session_state:
        st.session_state["active_partition"] = get_state("active_partition", db_path) or "train"
    partition = st.selectbox("Partition", ("train", "validation", "test"), key="active_partition")
    view = st.radio("View", ("Review Queue", "Browse All"), key="annotation_view")
    reviewing = view == "Review Queue"
    mode, search = "all", ""
    if not reviewing:
        if "active_queue" not in st.session_state:
            st.session_state["active_queue"] = get_state("browse_filter", db_path) or "all"
        mode = st.selectbox(
            "Queue",
            (
                "all",
                "VALID",
                "GIBBERISH",
                "flagged",
                "unreviewed",
                "human-reviewed",
                "provisional",
                "manual-review",
            ),
            key="active_queue",
        )
        search = (
            st.text_input(
                "Search text or example ID",
                help="Match exact or partial Arabic text, or all or part of an example ID.",
            )
            .strip()
            .casefold()
        )
        set_state("active_queue", mode, db_path)
        set_state("browse_filter", mode, db_path)
    set_state("active_partition", partition, db_path)
    st.caption(
        "Review the current training collection before further collection. "
        "Originals with transformed copies are archived and excluded."
    )
    with st.expander("Import candidates"):
        labeling_mode = st.selectbox(
            "Import labeling", ("Manual review", "Provisional training batch")
        )
        batch_id, audit_seed = None, 0
        if labeling_mode == "Provisional training batch":
            batch_id = st.text_input("New batch ID")
            audit_seed = int(st.number_input("Audit seed", min_value=0, value=0, step=1))
            st.caption(
                "Training only. New imports use provisional-v2. "
                "Existing frozen audits keep their original policy."
            )
        uploaded = st.file_uploader("Candidate JSONL", type="jsonl")
        incoming, projection = None, None
        upload_stamp = (
            None
            if uploaded is None
            else hashlib.sha256(
                uploaded.getvalue() + str((batch_id, audit_seed)).encode()
            ).hexdigest()
        )
        already_imported = upload_stamp and st.session_state.get("imported_upload") == upload_stamp
        if already_imported:
            st.caption("This upload has been imported. Select another file for a new collection.")
        if uploaded is not None and not already_imported:
            try:
                incoming = [
                    json.loads(line)
                    for line in uploaded.getvalue().decode("utf-8").splitlines()
                    if line.strip()
                ]
                if batch_id is not None:
                    projection = preview_import(incoming, batch_id, audit_seed, db_path)
                    st.info(
                        f"Projected manual workload: {projection['required_reviews']}/"
                        f"{projection['candidates']} — "
                        f"{projection['audit_samples']} audit samples, "
                        f"{projection['manual_only']} manual-only cases; "
                        f"{len(projection['groups'])} groups (provisional-v2)."
                    )
                else:
                    st.info(f"Manual import: {len(incoming)} candidates require human review.")
            except (ValueError, OSError) as error:
                incoming = None
                st.error(str(error))
        if st.button("Import uploaded file", disabled=incoming is None):
            try:
                count = import_candidates(
                    incoming,
                    db_path,
                    provisional_batch=batch_id,
                    seed=audit_seed,
                    expected_projection=projection["fingerprint"] if projection else None,
                )
                st.session_state["imported_upload"] = upload_stamp
                st.success(f"Imported {count} candidates")
            except (ValueError, OSError) as error:
                st.error(str(error))
        pilot = ROOT / "data/pilot_v1/candidates.jsonl"
        if pilot.exists() and st.button("Import 100-example pilot"):
            try:
                rows = [json.loads(line) for line in pilot.read_text(encoding="utf-8").splitlines()]
                st.success(f"Imported {import_candidates(rows, db_path)} candidates")
            except (ValueError, OSError) as error:
                st.error(str(error))
    with st.expander("Export reviewed records"):
        version = st.text_input("Export version", value="v1")
        export_mode = st.selectbox(
            "Export labels", ("Human-reviewed only", "Include audited provisional training labels")
        )
        st.caption(
            "Flagged records are excluded. Provisional labels require a passed group audit "
            "and are allowed only for training. Exports are local snapshots, "
            "not automatically cleared for public redistribution."
        )
        if st.button("Export partition"):
            try:
                folder = export_dataset(
                    partition,
                    version,
                    ROOT / "data/exports",
                    path=db_path,
                    include_provisional=export_mode != "Human-reviewed only",
                )
                st.success(f"Saved {folder}")
            except (ValueError, OSError) as error:
                st.error(str(error))


def required_records():
    # Active audit/quality flags come first. Plain manual imports also need labeling,
    # although they have no automatic review_request row and are simply pending.
    flagged = queue(partition, "manual-review", path=db_path)
    pending = [r for r in queue(partition, path=db_path) if r["status"] == "pending"]
    return flagged + pending


required = required_records()
if reviewing:
    st.metric("Remaining reviews", len(required))
    st.caption("Label once to save, resolve and move to the next required example.")
else:
    stats = progress(db_path)
    for column, (name, counts) in zip(st.columns(3), stats.items(), strict=True):
        with column:
            st.metric(name, f"{counts['human_reviewed']} / {counts['imported']} human-reviewed")
            st.caption(
                f"{counts['provisional']} provisional · {counts['flagged']} flagged · "
                f"{counts['export_eligible']} export-eligible"
            )

with st.expander("Audit progress"):
    reports = audit_report(db_path)
    history = [r for r in audit_report(db_path, include_history=True) if not r["active"]]
    if history:
        st.caption(
            f"{len(history)} historical groups retained. Only active policy groups affect review."
        )
    if not reports:
        st.caption("No provisional batches imported. Existing human reviews are unchanged.")
    for report in reports:
        # Administrative source/subtype names and proposed answers are deliberately omitted.
        st.write(
            f"Group {report['id']} · {report['policy_version']} · {report['status']} · "
            f"{report['resolved']}/{report['sampled']} audit samples resolved · "
            f"{report['errors']} errors · {report['population']} group members · "
            f"{len(report['coverage'])} dialect strata"
        )
        if report["retired_ids"]:
            st.caption(
                f"{len(report['retired_ids'])} retired records retained in this frozen audit; "
                f"{report['retired_samples']} retired samples remain historical. "
                f"{report['active_pending_samples']} active samples still pending. "
                "Retirement does not pass or resample an audit."
            )
        if report["affected_exports"]:
            st.warning("Affected earlier exports: " + ", ".join(report["affected_exports"]))

if partition == "test" and not get_state("test_unlocked", db_path):
    st.info(
        "Test collection is locked. Keep it isolated until the project freeze milestone; "
        "the annotation guide describes the explicit unlock record."
    )
    st.stop()
all_rows = queue(partition, path=db_path)
if reviewing and not required:
    st.success("No examples currently require manual review.")
    st.caption("Use Browse All to search, inspect or correct records.")
    st.stop()
if not all_rows:
    st.info("No records in this partition. Import candidates or choose another partition.")
    st.stop()
rows = required if reviewing else all_rows
if not reviewing:
    if mode in ("VALID", "GIBBERISH"):
        rows = [record for record in rows if record["human_label"] == mode]
    elif mode == "flagged":
        rows = [record for record in rows if record["flagged"]]
    elif mode == "unreviewed":
        rows = [record for record in rows if record["human_label"] is None]
    elif mode in {"human-reviewed", "provisional"}:
        rows = [record for record in rows if record["status"] == mode]
    elif mode == "manual-review":
        rows = queue(partition, "manual-review", path=db_path)
    if search:
        rows = [
            r for r in rows if search in r["input_text"].casefold() or search in r["id"].casefold()
        ]
# Review selection is restricted to the live required set. Browse selection remains
# independent of filters so correcting a label does not remove the open record.
state_key = f"cursor:{partition}:review" if reviewing else f"cursor:{partition}:all"
current_id = get_state(state_key, db_path)
selection_pool = rows if reviewing else all_rows
row = next((r for r in selection_pool if r["id"] == current_id), rows[0] if rows else all_rows[0])
index = next((i for i, record in enumerate(rows) if record["id"] == row["id"]), -1)
set_state(state_key, row["id"], db_path)
notes_key = f"notes:{row['id']}"
if notes_key not in st.session_state:
    st.session_state[notes_key] = row["notes"]


def persist(label=None, flag=None):
    try:
        fresh = next(r for r in queue(partition, path=db_path) if r["id"] == row["id"])
        if label is None and flag is None and st.session_state[notes_key] == fresh["notes"]:
            return True
        save_annotation(
            row["id"],
            label=label if label is not None else fresh["human_label"],
            flagged=fresh["flagged"] if flag is None else flag,
            notes=st.session_state[notes_key],
            path=db_path,
        )
        st.session_state["save_status"] = "Saved to SQLite"
        st.session_state["save_failed"] = False
        return True
    except Exception as error:
        st.session_state["save_status"] = f"Save failed; staying on this record: {error}"
        st.session_state["save_failed"] = True
        return False


def open_example(record_id):
    if persist():
        set_state(state_key, record_id, db_path)
        st.session_state.pop("save_status", None)


def advance_required():
    remaining = required_records()
    by_id = {r["id"] for r in remaining}
    # Wrap around unfinished records; an audit error can also add newly quarantined cases.
    following = rows[index + 1 :] + rows[:index]
    target = next((r for r in following if r["id"] in by_id), remaining[0] if remaining else None)
    if target:
        set_state(state_key, target["id"], db_path)


def complete_review(label):
    # One backend transaction saves the label/notes, clears the flag and updates the audit.
    # A failed write leaves the current record and queue untouched.
    if persist(label=label, flag=False):
        advance_required()


def flag_uncertain():
    if persist(flag=True):
        st.session_state["save_status"] = "Kept unresolved for later review."
        advance_required()


if not reviewing:
    st.caption(f"{len(rows)} matching examples")
    if rows:
        with st.container(height=220):
            for match in rows:
                preview = " ".join(match["input_text"].split())
                preview = preview[:77] + "…" if len(preview) > 80 else preview
                status = match["human_label"] or "Not labeled"
                if match["flagged"]:
                    status += " · flagged"
                elif match["status"] == "provisional":
                    status += " · provisional (answer hidden)"
                st.button(
                    f"{match['id']} · {status} · {preview}",
                    key=f"open:{match['id']}",
                    on_click=open_example,
                    args=(match["id"],),
                    use_container_width=True,
                )
    else:
        st.info("No matching examples. Change the search or queue filter.")
if index >= 0:
    location = "Review Queue" if reviewing else f"{mode} queue"
    st.caption(f"{row['id']} · Position {index + 1} of {len(rows)} in {location}")
else:
    st.caption(f"{row['id']} · selected example is outside the current search/filter")
    st.info("This example stays open after saving even when it no longer matches the filter.")
st.markdown(
    '<div dir="rtl" lang="ar" style="text-align:right;unicode-bidi:plaintext;'
    "white-space:pre-wrap;font-size:1.6rem;line-height:1.9;padding:1rem;"
    'border:1px solid #8886;border-radius:.5rem">' + html.escape(row["input_text"]) + "</div>",
    unsafe_allow_html=True,
)
st.write(
    f"Your label: **{row['human_label'] or 'Not labeled'}**"
    + (" · **Flagged for review**" if row["flagged"] else "")
)
if reviewing:
    reason_names = {
        "audit_sample": "Audit sample",
        "content_uncertain": "Needs review",
        "rule_not_admitted": "Manual labeling",
        "failed_audit": "Audit error review",
        "quality_issue": "Quality review",
    }
    st.caption(
        " · ".join(reason_names.get(reason, "Manual review") for reason in row["review_reasons"])
        or "Manual labeling"
    )
else:
    st.caption(f"Review status: {row['status']}")
    if row["review_reasons"]:
        st.caption("Review required: " + "; ".join(row["review_reasons"]))
left, right = st.columns(2)
for column, label, shortcut in ((left, "VALID", "1"), (right, "GIBBERISH", "2")):
    column.button(
        label,
        shortcut=shortcut,
        on_click=complete_review if reviewing else persist,
        kwargs={"label": label},
        use_container_width=True,
    )
st.text_area(
    "Notes (optional; saved on blur and before navigation)", key=notes_key, on_change=persist
)
if reviewing:
    st.button("Flag as uncertain", shortcut="F", on_click=flag_uncertain)
    st.caption("Uncertain examples stay unresolved and remain available in Review Queue.")
elif row["flagged"]:
    st.button(
        "Resolve review flag",
        disabled=row["human_label"] is None,
        on_click=persist,
        kwargs={"flag": False},
    )
else:
    st.button("Flag for Review", shortcut="F", on_click=persist, kwargs={"flag": True})
previous, following = st.columns(2)
previous.button(
    "Previous",
    shortcut="P",
    on_click=open_example,
    args=(rows[index - 1]["id"] if index > 0 else row["id"],),
    disabled=index <= 0,
)
following.button(
    "Next",
    shortcut="N",
    on_click=open_example,
    args=(
        rows[(index + 1) % len(rows)]["id"]
        if reviewing
        else rows[index + 1]["id"]
        if rows and index + 1 < len(rows)
        else row["id"],
    ),
    disabled=len(rows) <= 1 if reviewing else not rows or index == len(rows) - 1,
)
if not reviewing:
    unlabeled = [r for r in all_rows if r["human_label"] is None]
    if unlabeled:
        if index == len(rows) - 1:
            st.warning(
                f"End of this queue, but {len(unlabeled)} examples in this partition are still "
                "unlabeled. Queue position is not annotation progress."
            )
        current_position = next(i for i, r in enumerate(all_rows) if r["id"] == row["id"])
        after_cursor = [r for r in all_rows[current_position + 1 :] if r["human_label"] is None]
        st.button(
            "Open next unlabeled example",
            on_click=open_example,
            args=((after_cursor or unlabeled)[0]["id"],),
            help="Search the entire partition, wrapping to the beginning if needed.",
        )
st.caption(
    "Shortcuts: 1 VALID · 2 GIBBERISH · F flag · P previous · N next. "
    "Shortcuts are inactive while typing notes."
)
if st.session_state.get("save_status"):
    if st.session_state.get("save_failed"):
        st.error(st.session_state["save_status"])
    else:
        st.success(st.session_state["save_status"])
if not reviewing and row["human_label"] is not None:
    with st.expander("Source details (available after initial labeling)"):
        if st.checkbox("Show source and intended label"):
            st.json(provenance_after_review(row["id"], db_path))
