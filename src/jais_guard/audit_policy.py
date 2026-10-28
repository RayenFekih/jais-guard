"""Pure v2 admission and workload projection: no database writes or model calls."""

import hashlib
import json
import math
import random
import re
import unicodedata

VERSION = "provisional-v2"
AYA_REVISION = "f9ea04583f02a8f86404ff6c58bf75fe637df8a2"
PUBLIC_RULES = {
    "google-research-datasets/tydiqa": ("da78f23f9119363459acbaf46bf89426ff26c259", "question"),
    "AmazonScience/massive": ("ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740", "localized_request"),
}
CONTEXT = re.compile(
    r"أكمل|اكمل|كمل|السابق|التالي|أعلاه|الفقرة|ترجم|لخص|المحدد|المذكور|هذه الجملة|هذا النص"
)
AYA_OPENING = re.compile(
    r"^(?:ما\b|ماهي|ماهو|ماذا|من\b|لمن|أين|اين|متى|كم\b|كيف|لماذا|هل\b|في أي|في اي|في ماذا|على أي|"
    r"شحال|شنو|كيفاش|وش\b|إيش|ايش|أي\b|اي\b|يعني|فين|ليه|"
    r"أخبرني|أخبرنى|حدثني|حدثنى|كلمني|كلمنى|اذكر|أذكر|اكتب|اعطني|أعطني|قارن|اعكس|"
    r"رشحلي|رشحلى|رشح|قولى|قل\b|أسرد|احكي|اشرح|فسر|قم بإنشاء|استخدم|ممكن تديني|ممكن تقولي|"
    r"معاك\b|معك\b|معكش\b|عندك\b|عندكم\b|محتاج\b|عايز\b|عاوز\b|بدي\b|نحب\b|طريقة عمل|كيفية\b|"
    r"تعرف\b|اختار\b|ممكن تقولى|كام\b)"
)
AYA_QUESTION = re.compile(
    r"\b(?:ما|ماذا|أين|اين|متى|كيف|كم|كام|لماذا|ليه|فين|أني|إيه|أيه|ازاى|إزاي|شنو|شحال|قديش|ليش)\b"
)
REPETITION = re.compile(r"([ثظضصقغخجطز])\1{4,9} ([ثظضصقغخجطز])\2{4,9}")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def admission(row):
    """Return (proposal, reason). A recognizable form is a heuristic, not human ground truth."""
    p, g = row["provenance"], row.get("generation", {})
    method, subtype, text = row["source_method"], row["subtype"], row["input_text"].strip()
    risk = "higher" if row.get("difficulty") == "hard" else "standard"
    if method == "public":
        # Comparison only: never rewrite the original candidate or its provenance.
        text = "".join(
            c
            for c in unicodedata.normalize("NFKC", text).replace("ـ", "")
            if not unicodedata.combining(c)
        )
        aya = p["source"] == "CohereLabs/aya_dataset"
        admitted = (p.get("revision"), subtype) == PUBLIC_RULES.get(p["source"])
        if aya:
            admitted = (
                p.get("revision") == AYA_REVISION
                and subtype == "human_instruction"
                and p.get("annotation_type") == "original-annotations"
            )
        if not admitted:
            return None, "rule_not_admitted"
        if (
            row["intended_label"] != "VALID"
            or CONTEXT.search(text)
            or not 8 <= len(text) <= 300
            or len(text.split()) < 2
            or (subtype == "question" and not text.endswith(("؟", "?")))
            or (
                aya
                and not (
                    AYA_OPENING.search(text)
                    or (text.endswith(("؟", "?")) and AYA_QUESTION.search(text))
                )
            )
        ):
            return None, "content_uncertain"
        label, origin = "VALID", "public-source-rule"
    elif method == "python":
        recipe = g.get("recipe")
        if subtype == "repeated_characters" and recipe == "two_repeated_character_blocks":
            match = REPETITION.fullmatch(text)
            safe = bool(match and match[1] != match[2])
            safe = safe and p["source"] == "JaisGuard deterministic Python recipes"
        elif subtype == "keyboard_noise" and recipe == "keyboard_noise":
            keyboard = "ضصثقفغعهخحج"
            compact = text.replace(" ", "")
            walks = {s[i : i + 8] for s in (keyboard, keyboard[::-1]) for i in range(4)}
            safe = (
                10 <= len(compact) <= 30
                and set(compact) <= set(keyboard)
                and any(walk in compact for walk in walks)
            )
        else:
            return None, "rule_not_admitted"
        if (
            not safe
            or g.get("parent_id")
            or row["intended_label"] != "GIBBERISH"
            or not isinstance(g.get("seed"), int)
            or isinstance(g.get("seed"), bool)
        ):
            return None, "content_uncertain"
        label, origin = "GIBBERISH", "python-rule"
    else:
        if not (g.get("instructions_file") or g.get("instructions")) or not isinstance(
            g.get("model_metadata"), dict
        ):
            return None, "content_uncertain"
        label, origin, risk = row["intended_label"], "codex-proposal", "higher"
    rule = json.dumps(
        [VERSION, method, p["source"], p.get("revision", "unknown"), subtype], ensure_ascii=False
    )
    return {"label": label, "origin": origin, "risk": risk, "rule_id": rule}, None


def boundary(row, batch):
    """Authoring chunks can share a run only with explicitly declared process-family metadata."""
    g, p = row.get("generation", {}), row["provenance"]
    for metadata, fields in (
        (g, ("run_id", "audit_family")),
        (p, ("collection_run", "input_family")),
    ):
        if any(
            k in metadata and (not isinstance(metadata[k], str) or not metadata[k].strip())
            for k in fields
        ):
            raise ValueError("Audit run and family metadata must be nonempty strings")
    if row["source_method"] == "public":
        return {
            "run": p.get("collection_run") or row.get("collection_batch") or batch,
            "family": p.get("input_family") or row["subtype"],
        }
    if g.get("run_id"):
        if not isinstance(g.get("audit_family"), str) or not g["audit_family"].strip():
            raise ValueError("A generation run_id requires an explicit audit_family")
        return {"run": g["run_id"], "family": g["audit_family"], "recipe": g.get("recipe")}
    return {
        "run": g.get("batch_id") or row.get("collection_batch") or batch,
        "family": g.get("recipe") or g.get("batch_id") or row["family_id"],
    }


def select_samples(members, risk, seed):
    rate, minimum = (0.5, 20) if risk == "higher" else (0.2, 10)
    target = min(len(members), max(minimum, math.ceil(rate * len(members))))
    strata = {}
    for row in sorted(members, key=lambda r: r["id"]):
        strata.setdefault(row["dialect"], []).append(row["id"])
    rng = random.Random(str(seed))
    dialects = sorted(strata)
    represented = dialects if target >= len(dialects) else rng.sample(dialects, target)
    allocation = {d: int(d in represented) for d in dialects}
    remaining = target - len(represented)
    if remaining:
        capacity = {d: len(strata[d]) - allocation[d] for d in dialects}
        quotas = {d: remaining * capacity[d] / sum(capacity.values()) for d in dialects}
        base = {d: math.floor(quotas[d]) for d in dialects}
        ties = {d: rng.random() for d in dialects}
        order = sorted(dialects, key=lambda d: (-(quotas[d] - base[d]), ties[d]))
        extras = set(order[: remaining - sum(base.values())])
        for d in dialects:
            allocation[d] += base[d] + int(d in extras)
    selected = {i for d in dialects for i in rng.sample(strata[d], allocation[d])}
    return sorted(selected), {
        d: {"population": len(ids), "sampled": sum(i in selected for i in ids)}
        for d, ids in sorted(strata.items())
    }


def project_workload(rows, batch, seed=0, *, disabled_rules=(), quality_flags=(), boundaries=None):
    """No admission, flag, audit or annotation is persisted by this function."""
    if not rows or any(r["partition"] != "train" for r in rows):
        raise ValueError("Provisional projections require a nonempty training batch")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate candidate IDs")
    groups, requests = {}, {}
    # A scheduling-policy version must not silently readmit a previously failed labeling rule.
    disabled_signatures = {tuple(json.loads(rule)[1:]) for rule in disabled_rules}
    for row in sorted(rows, key=lambda r: r["id"]):
        proposal, reason = admission(row)
        if proposal and tuple(json.loads(proposal["rule_id"])[1:]) in disabled_signatures:
            proposal, reason = None, "failed_audit"
        if proposal:
            process = boundaries[row["id"]] if boundaries is not None else boundary(row, batch)
            key = digest([VERSION, proposal["rule_id"], process])[:20]
            p = row["provenance"]
            group = groups.setdefault(
                key,
                dict(
                    id=key,
                    batch_id=batch,
                    source=p["source"],
                    source_revision=p.get("revision", "unknown"),
                    method=row["source_method"],
                    subtype=row["subtype"],
                    rule_id=proposal["rule_id"],
                    policy_version=VERSION,
                    boundary=process,
                    risk="standard",
                    members=[],
                ),
            )
            if proposal["risk"] == "higher":
                group["risk"] = "higher"
            group["members"].append(
                {
                    "id": row["id"],
                    "label": proposal["label"],
                    "origin": proposal["origin"],
                    "dialect": row["dialect"],
                }
            )
        if reason or row["id"] in quality_flags:
            requests[row["id"]] = "quality_issue" if row["id"] in quality_flags else reason
    sampled = set()
    for key, group in sorted(groups.items()):
        group["sampled_ids"], group["strata"] = select_samples(
            group["members"], group["risk"], f"{seed}:{key}"
        )
        group["population"] = len(group["members"])
        sampled.update(group["sampled_ids"])
    result = dict(
        policy_version=VERSION,
        batch_id=batch,
        seed=seed,
        candidates=len(rows),
        groups=[groups[k] for k in sorted(groups)],
        review_requests=requests,
        audit_samples=len(sampled),
        manual_only=len(set(requests) - sampled),
        required_reviews=len(sampled | set(requests)),
        unsampled_provisional=len(rows) - len(sampled | set(requests)),
        input_sha256=digest(sorted(rows, key=lambda r: r["id"])),
    )
    result["fingerprint"] = digest(result)
    return result
