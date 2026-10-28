"""Run locally: .venv/bin/python -m streamlit run apps/annotate.py"""

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
    export_dataset,
    get_state,
    import_candidates,
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
st.caption("Single-reviewer pilot. Intended labels are hidden; your judgment is the annotation.")
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
    if "active_queue" not in st.session_state:
        st.session_state["active_queue"] = get_state("active_queue", db_path) or "all"
    partition = st.selectbox("Partition", ("train", "validation", "test"), key="active_partition")
    mode = st.selectbox(
        "Queue", ("all", "VALID", "GIBBERISH", "flagged", "unreviewed"), key="active_queue"
    )
    search = (
        st.text_input(
            "Search text or example ID",
            help="Match exact or partial Arabic text, or all or part of an example ID.",
        )
        .strip()
        .casefold()
    )
    set_state("active_partition", partition, db_path)
    set_state("active_queue", mode, db_path)
    st.caption(
        "Review the current training collection before further collection. "
        "Originals with transformed copies are archived and excluded."
    )
    with st.expander("Import candidates"):
        uploaded = st.file_uploader("Candidate JSONL", type="jsonl")
        if st.button("Import uploaded file", disabled=uploaded is None):
            try:
                rows = [
                    json.loads(line)
                    for line in uploaded.getvalue().decode("utf-8").splitlines()
                    if line.strip()
                ]
                st.success(f"Imported {import_candidates(rows, db_path)} candidates")
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
        st.caption(
            "Unlabeled and flagged records are excluded. Exports are local snapshots, "
            "not automatically cleared for public redistribution."
        )
        if st.button("Export partition"):
            try:
                folder = export_dataset(partition, version, ROOT / "data/exports", path=db_path)
                st.success(f"Saved {folder}")
            except (ValueError, OSError) as error:
                st.error(str(error))

stats = progress(db_path)
for column, (name, counts) in zip(st.columns(3), stats.items(), strict=True):
    with column:
        st.metric(name, f"{counts['reviewed']} / {counts['imported']}")
        st.caption(
            f"{counts['exportable']} exportable · {counts['flagged']} flagged · "
            f"{counts['unresolved']} unresolved"
        )

if partition == "test" and not get_state("test_unlocked", db_path):
    st.info(
        "Test collection is locked. Keep it isolated until the project freeze milestone; "
        "the annotation guide describes the explicit unlock record."
    )
    st.stop()
all_rows = queue(partition, path=db_path)
if not all_rows:
    st.info("No records in this partition. Import the pilot or choose another partition.")
    st.stop()
rows = all_rows
if mode in ("VALID", "GIBBERISH"):
    rows = [record for record in rows if record["human_label"] == mode]
elif mode == "flagged":
    rows = [record for record in rows if record["flagged"]]
elif mode == "unreviewed":
    rows = [record for record in rows if record["human_label"] is None]
if search:
    rows = [
        record
        for record in rows
        if search in record["input_text"].casefold() or search in record["id"].casefold()
    ]
# Selection is independent of filters so correcting a label never removes the open example.
state_key = f"cursor:{partition}:all"
current_id = get_state(state_key, db_path)
row = next(
    (record for record in all_rows if record["id"] == current_id), rows[0] if rows else all_rows[0]
)
index = next((i for i, record in enumerate(rows) if record["id"] == row["id"]), -1)
set_state(state_key, row["id"], db_path)
notes_key = f"notes:{row['id']}"
if notes_key not in st.session_state:
    st.session_state[notes_key] = row["notes"]


def persist(label=None, flag=None):
    try:
        fresh = next(r for r in queue(partition, path=db_path) if r["id"] == row["id"])
        if label is None and flag is None and st.session_state[notes_key] == fresh["notes"]:
            # Browsing should not create annotation revisions when nothing was edited.
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


st.caption(f"{len(rows)} matching examples")
if rows:
    with st.container(height=220):
        for match in rows:
            preview = " ".join(match["input_text"].split())
            preview = preview[:77] + "…" if len(preview) > 80 else preview
            status = match["human_label"] or "Not labeled"
            if match["flagged"]:
                status += " · flagged"
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
    st.caption(f"{row['id']} · Position {index + 1} of {len(rows)} in {mode} queue")
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
left, right = st.columns(2)
left.button(
    "VALID", shortcut="1", on_click=persist, kwargs={"label": "VALID"}, use_container_width=True
)
right.button(
    "GIBBERISH",
    shortcut="2",
    on_click=persist,
    kwargs={"label": "GIBBERISH"},
    use_container_width=True,
)
st.text_area(
    "Notes (optional; saved on blur and before navigation)", key=notes_key, on_change=persist
)
if row["flagged"]:
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
    args=(rows[index + 1]["id"] if rows and index + 1 < len(rows) else row["id"],),
    disabled=not rows or index == len(rows) - 1,
)
unlabeled = [record for record in all_rows if record["human_label"] is None]
if unlabeled:
    if index == len(rows) - 1:
        st.warning(
            f"End of this queue, but {len(unlabeled)} examples in this partition are still "
            "unlabeled. Queue position is not annotation progress."
        )
    # Imports can sort before the saved cursor. Wrap to the beginning rather than skip them.
    current_position = next(i for i, record in enumerate(all_rows) if record["id"] == row["id"])
    after_cursor = [
        record for record in all_rows[current_position + 1 :] if record["human_label"] is None
    ]
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
if row["human_label"] is not None:
    with st.expander("Source details (available after initial labeling)"):
        if st.checkbox("Show source and intended label"):
            st.json(provenance_after_review(row["id"], db_path))
