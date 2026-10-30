"""Check local Markdown link targets in project documentation without network calls."""

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main():
    failures = []
    count = 0
    for path in [
        ROOT / "README.md",
        ROOT / "PROJECT_CONTEXT.md",
        *sorted((ROOT / "docs").rglob("*.md")),
    ]:
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            target = target.strip().split(' "', 1)[0].strip("<>")
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            target = unquote(target.split("#", 1)[0])
            if not target:
                continue
            count += 1
            if not (path.parent / target).exists():
                failures.append(f"{path.relative_to(ROOT)}: {target}")
    if failures:
        raise SystemExit("Missing local link targets:\n" + "\n".join(failures))
    print(f"Verified {count} local documentation link targets (remote URLs not requested)")


if __name__ == "__main__":
    main()
