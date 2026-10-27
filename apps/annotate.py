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
    mode = st.selectbox("Queue", ("all", "unreviewed", "flagged"), key="active_queue")
    set_state("active_partition", partition, db_path)
    set_state("active_queue", mode, db_path)
    st.caption(
        "Targets: 1,000 train · 200 validation · 400 isolated test. Stop after the 100 pilot."
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
rows = queue(partition, mode, db_path)
if not rows:
    st.info("No records in this queue. Import the pilot, choose another queue, or resume later.")
    st.stop()
state_key = f"cursor:{partition}:{mode}"
current_id = get_state(state_key, db_path)
index = next((i for i, row in enumerate(rows) if row["id"] == current_id), 0)
row = rows[index]
set_state(state_key, row["id"], db_path)
notes_key = f"notes:{row['id']}"
if notes_key not in st.session_state:
    st.session_state[notes_key] = row["notes"]


def persist(label=None, flag=None):
    try:
        fresh = next(r for r in queue(partition, path=db_path) if r["id"] == row["id"])
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


def navigate(direction):
    if persist():
        next_row = rows[max(0, min(index + direction, len(rows) - 1))]
        set_state(state_key, next_row["id"], db_path)


st.caption(f"{row['id']} · {index + 1} of {len(rows)} in {mode} queue")
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
previous.button("Previous", shortcut="P", on_click=navigate, args=(-1,), disabled=index == 0)
following.button(
    "Next", shortcut="N", on_click=navigate, args=(1,), disabled=index == len(rows) - 1
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
