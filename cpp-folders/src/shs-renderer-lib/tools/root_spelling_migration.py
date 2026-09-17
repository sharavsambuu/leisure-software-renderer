#!/usr/bin/env python3
"""Step-7 item 3: migrate repo consumers from root `shs::X` spellings to
owner-namespace `shs::<owner>::X` spellings.

Resolution: a token `shs::X` is rewritten only when exactly one owner module
canonical header defines X (struct/class/enum/union/using-alias/free-function).
Ambiguous or unresolved tokens are reported and left untouched — the inline
namespace / using-declaration compatibility layer keeps them valid.

Usage:
  python3 tools/root_spelling_migration.py --check   # report only
  python3 tools/root_spelling_migration.py --write   # rewrite files
"""
import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
LIB = REPO / "cpp-folders" / "src" / "shs-renderer-lib"
INCLUDE = LIB / "include" / "shs"

OWNER_NS = {p.name for p in INCLUDE.iterdir() if p.is_dir()} if INCLUDE.exists() else set()

DEF_PATTERNS = [
    r"^\s*(?:struct|class|union)\s+{token}\b",
    r"^\s*enum(?:\s+class)?(?:\s+\w+)?\s+{token}\b",
    r"^\s*using\s+{token}\s*=",
    r"^\s{0,8}(?:inline\s+|constexpr\s+|static\s+)*[\w:<>,&*\s]+\b{token}\s*\(",
]

# Manually verified owners for tokens whose resolution is ambiguous because
# the heuristic function pattern also matches call sites in other modules.
OVERRIDES = {
    "Context": "app",                              # shs/app/context.hpp (rhi re-points the fwd decl)
    "ResourceRegistry": "resources",               # renderpath fwd decl was a call-site match
    "Scene": "scene",
    "cull_class_is_visible": "geometry",           # scene hit is a call
    "find_render_path_resource_by_semantic": "renderpath",  # rhi hit is a call
    "make_render_path_capability_set": "renderpath",  # app hit is a call
    "normalize_or": "geometry",
    "rasterize_mesh": "render",                    # app hit is a call
    "render_backend_type_name": "render",          # other hits are calls
    "technique_mode_bit": "render",
}


def resolve(tokens):
    mapping = {}
    ambiguous = {}
    unresolved = []
    for token in sorted(tokens):
        owners = set()
        for header in sorted(INCLUDE.rglob("*.hpp")):
            rel = header.relative_to(INCLUDE)
            if rel.parts[0] in ("domains", "execution"):
                continue
            text = header.read_text()
            if any(re.search(p.replace("{token}", re.escape(token)), text, re.M) for p in DEF_PATTERNS):
                owners.add(rel.parts[0])
        if len(owners) == 1:
            mapping[token] = owners.pop()
        elif len(owners) > 1:
            if token in OVERRIDES:
                mapping[token] = OVERRIDES[token]
            else:
                ambiguous[token] = sorted(owners)
        else:
            unresolved.append(token)
    return mapping, ambiguous, unresolved


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report only")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    # Tokens actually used by repo consumers.
    usage = {}
    for path in sorted((REPO / "cpp-folders").rglob("*")):
        if path.suffix not in (".cpp", ".hpp", ".h"):
            continue
        try:
            rel = path.relative_to(INCLUDE)
            continue  # canonical/forwarder headers are not consumers
        except ValueError:
            pass
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for token in set(re.findall(r"shs::([A-Za-z_]\w*)\b(?!::)", text)):
            usage.setdefault(token, set()).add(path)

    mapping, ambiguous, unresolved = resolve(usage)
    print(f"tokens used: {len(usage)}")
    print(f"mappable:    {len(mapping)}")
    print(f"ambiguous:   {len(ambiguous)} {sorted(ambiguous)[:12]}")
    print(f"unresolved:  {len(unresolved)} {sorted(unresolved)[:20]}")

    if not args.write:
        return 0

    files = sorted({p for paths in usage.values() for p in paths})
    changed_files = 0
    changed_hits = 0
    for path in files:
        text = path.read_text()
        out = text
        for token, owner in sorted(mapping.items()):
            out = re.sub(rf"shs::{token}\b(?!::)", f"shs::{owner}::{token}", out)
        if out != text:
            changed_hits += sum(len(re.findall(rf"shs::{t}\b(?!::)", text)) for t in mapping)
            changed_files += 1
            path.write_text(out)
    print(f"rewrote {changed_files} file(s) (hits approximated at rewrite time)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
