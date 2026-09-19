#!/usr/bin/env python3
"""Claimed-primitive gate (ROP-1.3, owner ruling 2026-09-18).

A `core/` primitive must have **either a live consumer or a recorded
disposition**. The defect ROP-1 exists because neither held:
`shs/core/result.hpp` shipped, was never included by anything, was never
covered by a gate, and rotted from aspirational to law-contradicting while a
planning doc kept citing it as if it were the house railway.

Rules:

  R1 every direct `include/shs/core/*.hpp` primitive must be included by at
     least one other file in the library (`include/`, `src/`, `tests/`), or be
     listed in `tools/unclaimed_core_primitives.json` with a disposition;
  R2 the allow-list may not go stale — every listed header must still exist
     (a listed header that is gone means the record outlived its subject);
  R3 every allow-list entry must carry a non-empty `disposition` naming why it
     is permitted to be unclaimed (a pending proposal, or a tracked decision).

Allow-listed entries are still printed, as `KNOWN`, so the debt stays visible
in every CI run instead of silently passing. An entry is a *recorded decision*,
never an amnesty: the gate's purpose is to stop NEW unclaimed primitives.

Usage:
  check_claimed_primitives.py [--lib-root DIR]
Env override for the negative fixture: CLAIMED_PRIMITIVES_LIB_ROOT.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # .../shs-renderer-lib
ALLOWLIST = Path(__file__).parent / 'unclaimed_core_primitives.json'
CORE = 'core'
CODE_SUFFIXES = ('.hpp', '.h', '.cpp', '.cc')

INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"\n]+)[>"]', re.MULTILINE)
COMMENT = re.compile(r'/\*.*?\*/|//[^\n]*', re.DOTALL)


def includes(path):
    text = COMMENT.sub('', path.read_text(encoding='utf-8', errors='replace'))
    return INCLUDE.findall(text)


def primitives(lib_root):
    core = lib_root / 'include' / 'shs' / CORE
    if not core.is_dir():
        return []
    return sorted(p for p in core.glob('*.hpp') if p.is_file())


def consumers(lib_root):
    """Map `shs/core/<name>` -> list of files that include it."""
    found = {}
    for top in ('include', 'src', 'tests'):
        base = lib_root / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob('*')):
            if not path.is_file() or path.suffix not in CODE_SUFFIXES:
                continue
            if path.parent.name == CORE and path.parent.parent.name == 'shs':
                continue  # the primitive itself is not its own consumer
            for inc in includes(path):
                if inc.startswith(f'shs/{CORE}/'):
                    found.setdefault(inc, []).append(str(path.relative_to(lib_root)))
    return found


def load_allowlist(path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    return {entry['header']: entry for entry in data.get('entries', [])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lib-root', type=Path,
                        default=Path(os.environ.get('CLAIMED_PRIMITIVES_LIB_ROOT', ROOT)),
                        help='library root holding include/, src/, tests/ '
                             '(env: CLAIMED_PRIMITIVES_LIB_ROOT)')
    parser.add_argument('--allowlist', type=Path,
                        default=Path(os.environ.get('CLAIMED_PRIMITIVES_ALLOWLIST', ALLOWLIST)),
                        help='recorded-disposition file '
                             '(env: CLAIMED_PRIMITIVES_ALLOWLIST)')
    args = parser.parse_args()
    lib_root = args.lib_root.resolve()

    allowed = load_allowlist(args.allowlist)
    used = consumers(lib_root)
    failed = 0
    claimed = known = 0

    for header in primitives(lib_root):
        key = f'shs/{CORE}/{header.name}'
        if used.get(key):
            claimed += 1
            continue
        entry = allowed.get(key)
        if entry is None:
            print(f'[claimed-primitives] FAIL: R1 {key} is unclaimed — nothing '
                  f'includes it and no disposition is recorded. Give it a live '
                  f'consumer, a gate test, or an entry in '
                  f'tools/unclaimed_core_primitives.json')
            failed = 1
            continue
        disposition = (entry.get('disposition') or '').strip()
        if not disposition:
            print(f'[claimed-primitives] FAIL: R3 {key} is allow-listed with an '
                  f'empty disposition (a record must name why it is permitted)')
            failed = 1
            continue
        known += 1
        print(f'[claimed-primitives] KNOWN: {key} is unclaimed by decision — '
              f'{disposition}')

    # R2: the allow-list must not outlive its subjects.
    for key in sorted(allowed):
        if not (lib_root / 'include' / key).exists():
            print(f'[claimed-primitives] FAIL: R2 {key} is allow-listed but the '
                  f'header no longer exists — remove the stale entry')
            failed = 1

    if failed:
        print('[claimed-primitives] unclaimed primitives detected')
        return 1

    print(f'[claimed-primitives] all checks passed ({claimed} claimed, '
          f'{known} allow-listed with a recorded disposition)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
