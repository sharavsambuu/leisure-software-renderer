#!/usr/bin/env python3
"""Doc-path / citation existence gate (ROP-4.3, owner ruling 2026-09-18).

Every doc citation must point at a file that exists. Three rules, all at
**zero baseline violations** when the gate landed (measured 2026-09-18):

  R1 code citations  — a repo-root-relative `docs/....md` path named in a C++
                       source/header must exist. This is the rule that would
                       have caught the archived Kleisli plan (cited at six live
                       sites in `include/shs` after it moved to
                       `docs/outdated/`) and the archived
                       `engine_domain_separation_migration.md` (two more).
  R2 markdown links  — a relative `](target)` link inside `docs/**/*.md` must
                       resolve against its own directory. Schemes (`https:`,
                       `file:`), absolute paths and pure `#anchors` are exempt:
                       review docs carry deliberately absolute `file://`
                       evidence links.
  R3 markdown prose  — a standalone `docs/....md` path in prose or backticks
                       must exist. A path embedded in a longer path
                       (`snake/docs/STATUS.md`) is NOT a repo-root citation and
                       is skipped, not failed.

Quoting a defect (R2/R3): a line carrying the marker `doc-paths: quoted` is
documentation *about* a broken path — an audit or close-out note quoting the old
spelling — not a citation of it, so its paths are skipped. The marker is
deliberately explicit and greppable, and it must be added per line; a gate that
forbade quoting a defect would force authors to hide their evidence, which is the
opposite of what this gate is for. See `docs/backlog/rop_hardening_todo.md` §0/§5
for uses.

Deliberately out of scope:

  - **Bare filenames** (`ARCHITECTURE.md`, `conventions.md`): legal. A
    demo-local doc resolves against its own project tree, not the repo root,
    so a bare name carries no path claim to verify.
  - **`docs/outdated/` and `docs/education/kdba_history/`**: read-only history
    (T2). A gate must never be the reason an archive gets edited, so archived
    files are exempt from the link rules (R2/R3). Citations *of* an archived
    path, made from live code or a live doc, are still checked — that is the
    direction the drift happened in.

Why the gate exists: the Kleisli vocabulary is law (Constitution II §8) but the
plan that published it was archived without repointing its live citations, and
no gate covered markdown paths — the others cover headers, includes, contracts,
gateways, KDBA boundaries, pure-value libraries and the backend seam.
See `docs/backlog/rop_hardening_todo.md` ROP-4.3.
"""

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # repository root

CODE_SUFFIXES = ('.hpp', '.h', '.cpp', '.cc')

# A repo-root-relative doc path. The lookbehind keeps a path embedded in a
# longer path (`.../snake/docs/STATUS.md`) from being read as a citation.
DOC_CITATION = re.compile(r'(?<![A-Za-z0-9_./-])docs/[A-Za-z0-9_./-]+\.md')
MD_LINK = re.compile(r'\]\(([^)\s]+)\)')
SCHEME = re.compile(r'^[A-Za-z][A-Za-z0-9+.-]*:')

# Read-only history (T2): never make the gate the reason an archive changes.
ARCHIVE_PARTS = ('outdated', 'kdba_history')

# Line-level exemption: the line documents a broken path instead of citing one.
QUOTED = 'doc-paths: quoted'


def is_archived(path):
    return any(part in ARCHIVE_PARTS for part in path.parts)


def read(path):
    return path.read_text(encoding='utf-8', errors='replace')


def code_files(root):
    src = root / 'cpp-folders' / 'src'
    if not src.is_dir():
        return
    for path in sorted(src.rglob('*')):
        if (path.is_file() and path.suffix in CODE_SUFFIXES
                and 'build' not in path.parts and '_deps' not in path.parts):
            yield path


def doc_files(root):
    docs = root / 'docs'
    if not docs.is_dir():
        return
    for path in sorted(docs.rglob('*.md')):
        if path.is_file():
            yield path


def code_citation_violations(root):
    """R1: a `docs/....md` path named anywhere in C++ must exist."""
    out = []
    for path in code_files(root):
        for number, line in enumerate(read(path).splitlines(), 1):
            for match in DOC_CITATION.finditer(line):
                if not (root / match.group(0)).exists():
                    out.append((path.relative_to(root), number, match.group(0)))
    return out


def link_violations(root):
    """R2: relative markdown links in live docs must resolve."""
    out = []
    for path in doc_files(root):
        if is_archived(path):
            continue
        for number, line in enumerate(read(path).splitlines(), 1):
            if QUOTED in line:
                continue
            for match in MD_LINK.finditer(line):
                target = match.group(1).split('#')[0]
                if not target or target.startswith('/') or SCHEME.match(target):
                    continue
                if not (path.parent / target).exists():
                    out.append((path.relative_to(root), number, target))
    return out


def prose_violations(root):
    """R3: standalone `docs/....md` paths in live docs must exist."""
    out = []
    for path in doc_files(root):
        if is_archived(path):
            continue
        for number, line in enumerate(read(path).splitlines(), 1):
            if QUOTED in line:
                continue
            for match in DOC_CITATION.finditer(line):
                if not (root / match.group(0)).exists():
                    out.append((path.relative_to(root), number, match.group(0)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path(os.environ.get('DOC_PATHS_ROOT', ROOT)),
                        help='repository root to scan (env: DOC_PATHS_ROOT)')
    args = parser.parse_args()
    root = args.root.resolve()

    rules = (
        ('R1 code citations', code_citation_violations),
        ('R2 markdown links', link_violations),
        ('R3 markdown prose', prose_violations),
    )

    failed = 0
    for name, check in rules:
        violations = check(root)
        if violations:
            failed = 1
            print(f'[doc-paths] FAIL: {name} — {len(violations)} citation(s) '
                  f'point at a file that does not exist')
            for path, number, target in violations:
                print(f'    {path}:{number}: {target}')
        else:
            print(f'[doc-paths] OK: {name}')

    if failed:
        print('[doc-paths] broken doc citations detected; repoint the citing '
              'site (archives are read-only history — never the archive)')
        return 1

    print('[doc-paths] all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
