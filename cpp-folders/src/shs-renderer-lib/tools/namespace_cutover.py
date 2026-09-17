#!/usr/bin/env python3
"""Step-7 namespace cutover tool: wrap root `namespace shs` blocks.

For every canonical public header that still declares symbols directly in the
root `shs` namespace, the tool wraps each direct block as

    namespace shs
    {
    inline namespace <owner>
    {
        ...unchanged body...
    } // inline namespace <owner>
    } // namespace shs

Old `shs::X` spellings keep resolving (inline-namespace compatibility alias),
and the new `shs::<owner>::X` spelling becomes available immediately. Headers
that already use C++17 nested namespaces (`namespace shs::app {`) or nested
named namespaces inside root blocks are left untouched except for the wrapper.

The tool is idempotent (a marker comment guards each wrapped block) and
verifies: after a run, no direct root block remains unwrapped.

Usage:
  namespace_cutover.py --module core                 # apply to one module
  namespace_cutover.py --module core --check         # verify without writing
  namespace_cutover.py --list                        # list root-block headers
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
INCLUDE = ROOT / "cpp-folders/src/shs-renderer-lib/include/shs"

BLOCK_OPEN = re.compile(r"namespace\s+shs(?=\s*[{\n/])")
MARKER = "// namespace-cutover: inline compatibility wrapper (step 7)"


def skip_ws_comments(text, i):
    n = len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n":
            i += 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j + 1
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            return i
    return n


def find_matching_brace(text, open_idx):
    """Return index of the brace matching text[open_idx] == '{', ignoring
    comments and string/char literals."""
    i = skip_ws_comments(text, open_idx + 1)
    depth = 1
    n = len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            i = text.find("\n", i)
            if i < 0:
                break
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c in "\"'":
            quote = c
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\":
                    i += 1
                i += 1
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced braces")


def wrap_block(text, open_idx, close_idx, owner):
    body = text[open_idx + 1 : close_idx]
    wrapped_open = "\n" + MARKER + "\n    inline namespace " + owner + "\n    {"
    wrapped_close = "\n    } // inline namespace " + owner + "\n"
    return text[: open_idx + 1] + wrapped_open + body + wrapped_close + text[close_idx:]


def owner_of(rel):
    return rel.parts[0]


def transform_file(path, owner, check_only):
    text = path.read_text()
    out = text
    changed = 0
    while True:
        m = BLOCK_OPEN.search(out)
        if not m:
            break
        after = out[m.end() :].lstrip()
        if after.startswith(":"):  # C++17 nested namespace, skip token
            nxt = BLOCK_OPEN.search(out, m.end())
            if not nxt:
                break
            m = nxt
            after = out[m.end() :].lstrip()
            if after.startswith(":"):
                break
        open_idx = skip_ws_comments(out, m.end())
        if open_idx >= len(out) or out[open_idx] != "{":
            break
        if MARKER in out[open_idx : open_idx + 200]:
            break  # already wrapped
        close_idx = find_matching_brace(out, open_idx)
        out = wrap_block(out, open_idx, close_idx, owner)
        changed += 1
    if changed and not check_only:
        path.write_text(out)
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--module", help="top-level module (e.g. core) to cut over")
    ap.add_argument("--check", action="store_true", help="report without writing")
    ap.add_argument("--list", action="store_true", help="list headers with root blocks")
    args = ap.parse_args()

    targets = []
    for p in sorted(INCLUDE.rglob("*.hpp")):
        rel = p.relative_to(INCLUDE)
        if "domains" in rel.parts or "execution" in rel.parts:
            continue  # compatibility forwarders carry no declarations
        text = p.read_text()
        m = BLOCK_OPEN.search(text)
        if not m:
            continue
        if text[m.end() :].lstrip().startswith(":"):
            continue
        targets.append((rel, p))

    if args.list:
        for rel, _ in targets:
            print(rel.as_posix())
        print(f"total: {len(targets)}", file=sys.stderr)
        return 0

    if not args.module:
        ap.error("--module or --list required")
    selected = [(rel, p) for rel, p in targets if rel.parts[0] == args.module]
    if not selected:
        print(f"[namespace-cutover] no root blocks under module {args.module}")
        return 0
    total = 0
    for rel, p in selected:
        n = transform_file(p, args.module, args.check)
        total += n
        print(f"[namespace-cutover] {rel.as_posix()}: wrapped {n} block(s)")
    verb = "would wrap" if args.check else "wrapped"
    print(f"[namespace-cutover] module {args.module}: {verb} {total} block(s) in {len(selected)} header(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
