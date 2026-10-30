"""Deterministic held-out candidate construction; never assigns human labels."""

import hashlib
import random
from collections import Counter
from pathlib import Path

from jais_guard.data.annotations import connection, now, set_state
from jais_guard.data.test_workflow import sha, verify_lock, write_new

RUN = "test400-request-corruption-20261029-v2"
ARABIC = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
# Physical Arabic keyboard rows, not alphabetic proximity.
KEY_ROWS = ("ضصثقفغعهخحجد", "شسيبلاتنمكط", "ئءؤرلاىةوزظ")
CATEGORIES = {
    "meaningful_request": 80,
    "recoverable_corruption": 120,
    "severe_corruption": 90,
    "partial_unrecoverable": 50,
    "random_sequence": 40,
    "keyboard_repetition": 20,
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def nearby(char, rng):
    for row in KEY_ROWS:
        if char in row:
            i = row.index(char)
            return rng.choice(row[max(0, i - 1) : i] + row[i + 1 : i + 2] or ARABIC)
    return rng.choice(ARABIC)


def corrupt(text, seed, severity):
    """Return text plus actual edit trace. Severity is a proposal, never a label."""
    rng = random.Random(seed)
    chars = list(text)
    trace = []
    count = (
        max(1, len(text) // (18 - seed % 11))
        if severity == "recoverable_corruption"
        else int(len(text) * (0.42 if severity == "partial_unrecoverable" else 0.58))
    )
    mechanisms = ["delete", "adjacent_key", "transpose", "repeat", "join", "split"]
    if severity != "recoverable_corruption":
        mechanisms += ["delete", "delete", "drop_span", "drop_span"]
    for _ in range(count):
        if len(chars) < 4:
            break
        mechanism = rng.choice(mechanisms)
        index = rng.randrange(len(chars))
        before = "".join(chars)
        if mechanism == "delete":
            del chars[index]
        elif mechanism == "adjacent_key" and chars[index] != " ":
            chars[index] = nearby(chars[index], rng)
        elif mechanism == "transpose" and index + 1 < len(chars):
            chars[index], chars[index + 1] = chars[index + 1], chars[index]
        elif mechanism == "repeat":
            chars.insert(index, chars[index])
        elif mechanism == "join":
            spaces = [i for i, c in enumerate(chars) if c == " "]
            if spaces:
                del chars[rng.choice(spaces)]
        elif mechanism == "split":
            chars.insert(index, " ")
        elif mechanism == "drop_span":
            del chars[index : index + rng.randint(2, 5)]
        after = "".join(chars)
        if after != before:
            trace.append(
                {"operation": mechanism, "index_at_step": index, "before": before, "after": after}
            )
    return " ".join("".join(chars).split()), trace


def construct(parents):
    """340 independent request families + 60 independently seeded sequence families."""
    # 80 MSA + 30 per dialect, with 32/12 natural requests in those strata.
    groups = {
        d: [r for r in parents[:200] if r["dialect"] == d]
        for d in ("MSA", "Tunisian", "Egyptian", "Levantine", "Gulf")
    }
    groups["MSA"] += parents[200:240]
    valid = []
    used = set()
    for dialect, pool in groups.items():
        for i, parent in enumerate(pool[: 80 if dialect == "MSA" else 30]):
            category = (
                "meaningful_request"
                if i < (32 if dialect == "MSA" else 12)
                else "recoverable_corruption"
            )
            valid.append((parent, category))
            used.add(parent["text"])
    negatives = [p for p in parents if p["text"] not in used]
    assert len(valid) == 200 and len(negatives) == 140
    assignments = valid + [
        (p, "severe_corruption" if i < 90 else "partial_unrecoverable")
        for i, p in enumerate(negatives)
    ]
    assignments += [(None, "random_sequence")] * 40 + [(None, "keyboard_repetition")] * 20
    rows, sidecar = [], []
    # Unrelated real words, not a source of labels; recognizable accidental meaning is reviewable.
    word_bank = (
        "مسمار مظلة مساء دفتر تنور نافذة عشب طريق محبرة بستان "
        "ضريبة جناح منديل قافلة ركبة فنجان ضباب مرساة سلة حافلة "
        "حبل زيتون مئذنة ظرف موقد معطف تراب سباحة خاتم جزيرة "
        "ضرس رف ميزان وردة ملعقة طلاء قنديل عنب سقف حذاء "
        "قطن ممر فرشاة قارب ثلج قفل مخدة عدسة حصان علبة"
    ).split()
    for index, (parent, category) in enumerate(assignments):
        seed = 710000 + index
        rng = random.Random(seed)
        trace = []
        original = parent["text"] if parent else None
        if category == "meaningful_request":
            text = original
        elif parent:
            text, trace = corrupt(original, seed, category)
        elif category == "random_sequence":
            if index % 2:
                text = " ".join(rng.sample(word_bank, rng.randint(4, 10)))
            else:
                text = " ".join(
                    "".join(rng.choices(ARABIC, k=rng.randint(2, 9)))
                    for _ in range(rng.randint(3, 9))
                )
        else:
            row = KEY_ROWS[index % len(KEY_ROWS)]
            runs = []
            for _ in range(rng.randint(3, 8)):
                start = rng.randrange(len(row) - 2)
                fragment = row[start : start + rng.randint(2, 5)]
                runs.append(fragment * rng.randint(1, 3))
            text = " ".join(runs)
        family = "F-" + digest(RUN + str(index) + (original or "seeded-sequence"))[:20]
        identifier = "T-" + digest(family + text)[:16]
        intended = "VALID" if index < 200 else "GIBBERISH"
        row = {
            "id": identifier,
            "input_text": text,
            "partition": "test",
            "intended_label": intended,
            "intended_label_status": "provisional",
            "review_status": "pending-human-review",
            "source_method": "python",
            "subtype": category,
            "dialect": parent["dialect"] if intended == "VALID" else "not-assigned",
            "domain": "everyday-user-requests" if parent else "nonsemantic-input",
            "family_id": family,
            "provenance": {
                "source": RUN,
                "source_id": family,
                "license": "generated-content-local-review-only",
                "attribution": "Assistant-authored scenarios; Python transformations",
                "rights_evidence": (
                    "New authoring; no external corpus copied; public release unreviewed"
                ),
                "origin_type": "assistant-authored-scenario"
                if parent
                else "simulated-input-failure",
            },
            "generation": {
                "seed": seed,
                "recipe": "request-corruption-v2/" + category,
                "run_id": RUN,
                "parent_id": family if parent else None,
            },
        }
        rows.append(row)
        sidecar.append(
            {
                "id": identifier,
                "family_id": family,
                "original_text": original,
                "parent_dialect_hint": parent["dialect"] if parent else None,
                "parent_sha256": digest(original) if original else None,
                "seed": seed,
                "operations": trace,
                "mechanisms": sorted({t["operation"] for t in trace}),
                "proposal_category": category,
                "proposal_label": intended,
                "interpretation_note": "Provisional; review candidate without parent.",
            }
        )
    assert Counter(r["subtype"] for r in rows) == CATEGORIES
    assert len({r["family_id"] for r in rows}) == 400
    return rows, sidecar


def adopt(directory, previous, plan):
    """Append-only adoption: preserve old lock; inherit all selected model hashes."""
    directory, previous = Path(directory), Path(previous)
    if (directory / "selection_lock.json").exists():
        raise ValueError("Collection adoption already exists; never overwrite")
    lock, _ = verify_lock(previous)
    directory.mkdir(parents=True, exist_ok=True)
    write_new(directory / "plan.json", plan)
    lock = {
        **lock,
        "plan_sha256": sha(directory / "plan.json"),
        "parent_protocol": str(previous.resolve()),
        "parent_lock_sha256": sha(previous / "selection_lock.json"),
    }
    write_new(directory / "selection_lock.json", lock)
    write_new(
        directory / "adoption_receipt.json",
        {
            "created_at": now(),
            "parent_lock_sha256": lock["parent_lock_sha256"],
            "collection_protocol": "request-corruption-v2",
            "dataset": "test-400-v1",
            "consequences": (
                "Controlled corruption; original protocol retained; model selection unchanged"
            ),
            "parent_and_operations": "private/corruptions.jsonl; never displayed in reviewer UI",
            "historical_candidates": "Remain unchanged and excluded from this new collection",
            "human_labels": "None; all proposals pending blind human review",
        },
    )
    db = directory / "annotations.sqlite3"
    with connection(db):
        pass
    for key, value in {
        "test_unlocked": "collection-only",
        "active_partition": "test",
        "independent_test_workflow": "test-400-v1",
        "blind_sources": "1",
    }.items():
        set_state(key, value, db)
