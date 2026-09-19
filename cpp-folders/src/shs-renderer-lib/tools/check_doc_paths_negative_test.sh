#!/usr/bin/env bash
set -euo pipefail

# Negative fixture for the doc-path gate (ROP-4.3, owner ruling 2026-09-18):
#   - a C++ citation of a missing `docs/....md` path FAILs (R1);
#   - a broken relative markdown link FAILs (R2);
#   - a broken `docs/....md` path in prose FAILs (R3);
#   - a conformant tree PASSES, and the documented exemptions hold:
#     a bare filename, a path embedded in a longer path, a `file://` link and
#     a link broken *inside an archive* must all stay legal.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checker="${script_dir}/check_doc_paths.py"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

# --- fixture A: R1 — C++ cites a doc path that does not exist → FAIL ----------
mkdir -p "${tmp}/bad/cpp-folders/src/shs"
cat > "${tmp}/bad/cpp-folders/src/shs/demo.hpp" <<'EOF'
#pragma once
// Decision recorded in docs/backlog/gone_kleisli_plan.md.
namespace shs::demo { inline int value() { return 0; } }
EOF
if DOC_PATHS_ROOT="${tmp}/bad" "${checker}" >/dev/null 2>&1; then
  echo "[doc-paths-negative] FAIL: missing code citation passed R1"
  exit 1
fi

# --- fixture B: R2 — a broken relative markdown link → FAIL -------------------
mkdir -p "${tmp}/bad/docs"
cat > "${tmp}/bad/docs/page.md" <<'EOF'
See [the law](missing_law.md) for the rule.
EOF
if DOC_PATHS_ROOT="${tmp}/bad" "${checker}" >/dev/null 2>&1; then
  echo "[doc-paths-negative] FAIL: broken relative link passed R2"
  exit 1
fi

# --- fixture C: R3 — a broken doc path in prose → FAIL ------------------------
mkdir -p "${tmp}/bad3/docs"
cat > "${tmp}/bad3/docs/notes.md" <<'EOF'
The governing text is `docs/spec/does_not_exist.md` (prose citation).
EOF
if DOC_PATHS_ROOT="${tmp}/bad3" "${checker}" >/dev/null 2>&1; then
  echo "[doc-paths-negative] FAIL: broken prose path passed R3"
  exit 1
fi

# --- fixture C2: the `doc-paths: quoted` marker exempts a *documented* defect -
# (a gate that forbade quoting a broken path would force authors to hide
# evidence), but the exemption must be line-scoped, not file-scoped.
mkdir -p "${tmp}/c2/docs"
cat > "${tmp}/c2/docs/audit.md" <<'EOF'
The sweep repointed `docs/backlog/gone_plan.md` (quoted as evidence). <!-- doc-paths: quoted -->
EOF
if ! DOC_PATHS_ROOT="${tmp}/c2" "${checker}" >/dev/null 2>&1; then
  echo "[doc-paths-negative] FAIL: the 'doc-paths: quoted' marker did not exempt a documented defect"
  DOC_PATHS_ROOT="${tmp}/c2" "${checker}" || true
  exit 1
fi

mkdir -p "${tmp}/c2b/docs"
cat > "${tmp}/c2b/docs/audit.md" <<'EOF'
Repointed `docs/backlog/gone_plan.md`. <!-- doc-paths: quoted -->
Still live and still broken: `docs/backlog/also_gone.md`.
EOF
if DOC_PATHS_ROOT="${tmp}/c2b" "${checker}" >/dev/null 2>&1; then
  echo "[doc-paths-negative] FAIL: the 'doc-paths: quoted' marker leaked beyond its line"
  exit 1
fi

# --- fixture D: conformant tree must PASS ------------------------------------
mkdir -p "${tmp}/good/cpp-folders/src/shs" "${tmp}/good/docs/spec" \
         "${tmp}/good/docs/outdated"
cat > "${tmp}/good/docs/spec/value_oriented_programming.md" <<'EOF'
# Value-oriented programming
EOF
cat > "${tmp}/good/cpp-folders/src/shs/demo.hpp" <<'EOF'
#pragma once
// Law: docs/spec/value_oriented_programming.md §8.
namespace shs::demo { inline int value() { return 0; } }
EOF
cat > "${tmp}/good/docs/index.md" <<'EOF'
Law: [value-oriented programming](spec/value_oriented_programming.md) and
`docs/spec/value_oriented_programming.md` in prose.

Exemptions that must stay legal:
  - a bare filename: ARCHITECTURE.md (demo-local, no path claim)
  - a path embedded in a longer one: snake/docs/STATUS.md
  - an absolute evidence link: [review](file:///tmp/review.md)
EOF
cat > "${tmp}/good/docs/outdated/frozen.md" <<'EOF'
# Frozen history (T2) — a broken link here must NOT fail the gate
See [the old plan](long_gone.md).
EOF
if ! DOC_PATHS_ROOT="${tmp}/good" "${checker}" >/dev/null 2>&1; then
  echo "[doc-paths-negative] FAIL: conformant tree rejected by the doc-path gate"
  DOC_PATHS_ROOT="${tmp}/good" "${checker}" || true
  exit 1
fi

echo "[doc-paths-negative] all tests passed"
