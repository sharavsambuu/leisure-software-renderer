#!/usr/bin/env python3
"""Step-7 item 3: emit the old->new namespace spelling mapping doc table.

Scans all root-visible definitions in canonical headers and writes a markdown
table `old spelling | owner-namespace spelling | owner` to stdout (or a file
via --out).
"""
import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
LIB = REPO / "cpp-folders" / "src" / "shs-renderer-lib"
INCLUDE = LIB / "include" / "shs"

DEF_PATTERNS = [
    r"^\s*(?:struct|class|union)\s+([A-Za-z_]\w*)\b",
    r"^\s*enum(?:\s+class)?(?:\s+\w+)?\s+([A-Za-z_]\w*)\b",
    r"^\s*using\s+([A-Za-z_]\w*)\s*=",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # token -> owners of defining headers (exact definition shapes only)
    owners = {}
    for header in sorted(INCLUDE.rglob("*.hpp")):
        rel = header.relative_to(INCLUDE)
        if rel.parts[0] in ("domains", "execution"):
            continue
        text = header.read_text()
        for p in (r"^\s*(?:struct|class|union)\s+([A-Za-z_]\w*)",
                  r"^\s*enum(?:\s+class)?(?:\s+\w+)?\s+([A-Za-z_]\w*)",
                  r"^\s*using\s+([A-Za-z_]\w*)\s*="):
            for match in re.finditer(p, text, re.M):
                owners.setdefault(match.group(1), set()).add(rel.parts[0])

    rows = []
    ambiguous = []
    for token, owners_ in sorted(owners.items()):
        if len(owners_) == 1:
            (owner,) = tuple(owners_)
            rows.append((owner, token))
        elif len(owners_) > 1:
            # Ambiguous spellings: the inline-namespace compat layer keeps the
            # old root spelling valid; consumers must pick the owner that
            # matches the entity they mean (see doc notes).
            ambiguous.append((token, sorted(owners_)))
    lines = [f"| `{token}` | `shs::{owner}::{token}` | {owner} |" for owner, token in rows]
    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(f"tokens: {len(owners)}, rows: {len(rows)}, ambiguous: {len(ambiguous)}")
    for token, owners_ in ambiguous:
        print(f"AMBIGUOUS {token}: {owners_}", file=__import__('sys').stderr)


if __name__ == "__main__":
    raise SystemExit(main())
