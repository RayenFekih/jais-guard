"""Prepare/review/freeze test-400-v1. Never loads or evaluates any classifier."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from jais_guard.data.annotations import DEFAULT_DB
    from jais_guard.data.test_workflow import (
        check_candidates,
        freeze,
        import_collection,
        initialize,
        read_frozen_test,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "data/test400_v1")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser(
        "init", help="Lock six models/thresholds and initialize isolated review DB"
    )
    init.add_argument("--results", type=Path, default=ROOT / "results/fine_tuning")
    init.add_argument("--reference-db", type=Path, default=DEFAULT_DB)
    init.add_argument("--plan", type=Path, default=ROOT / "configs/test400_plan.json")
    for name in ("check", "import"):
        command = sub.add_parser(name)
        command.add_argument("candidates", type=Path)
        if name == "check":
            command.add_argument(
                "--allow-partial",
                action="store_true",
                help="Audit a 1-399-row draft without permitting import or freeze",
            )
    final = sub.add_parser("freeze", help="Export exactly 400 final human labels without overwrite")
    final.add_argument("--destination", type=Path, default=ROOT / "data/exports")
    final.add_argument("--confounding-note")
    verify = sub.add_parser("verify", help="Verify frozen export; never evaluate models")
    verify.add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        print(initialize(args.directory, args.results, args.reference_db, args.plan))
        print(
            "Protocol locked; no examples generated or imported. Mixed-script inputs stay separate."
        )
    elif args.command == "check":
        _, report = check_candidates(
            args.directory, args.candidates, allow_partial=args.allow_partial
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.allow_partial and not report["count_complete"]:
            print(
                "Partial draft only: import and freeze remain disabled "
                "until 400 candidates are ready."
            )
            if any(i["kind"] != "near_duplicate" or i["cross_partition"] for i in report["issues"]):
                raise SystemExit(1)
            return
        if (
            not report["ready_for_import"]
            or report["distribution"]["confounding_warnings"]
            or report["distribution"]["valid_dialect_target_shortfall"]
            or not report["distribution"]["proposed_class_target_met"]
        ):
            raise SystemExit(1)
    elif args.command == "import":
        print("Imported for blind review:", import_collection(args.directory, args.candidates))
    elif args.command == "freeze":
        print("Frozen:", freeze(args.directory, args.destination, args.confounding_note))
    else:
        rows = read_frozen_test(args.folder)
        print(f"PASS: {len(rows)} human-reviewed frozen examples. No model evaluation performed.")


if __name__ == "__main__":
    main()
