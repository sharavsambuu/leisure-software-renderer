#!/usr/bin/env bash
# Migration step 5 — self-contained public headers gate.
#
# Every public header must compile standalone with exactly the exported
# package dependency set:
#   - the package include dir (include/),
#   - glm and any SDK include dirs present in this build (SDL3, assimp,
#     Vulkan, Jolt, xsimd, VMA),
# and with NO SHS_HAS_* feature defines and NO build-dir/stb include dirs.
# A failure means a header leaks a build-tree/generated/stb dependency or
# requires compile definitions to parse — a self-containment violation.
#
# Usage: header_self_containment_test.sh <flags-file> <source-root> [compiler]
set -uo pipefail

FLAGS_FILE="${1:?usage: header_self_containment_test.sh <flags-file> <source-root> [compiler]}"
SRC_ROOT="${2:?usage: header_self_containment_test.sh <flags-file> <source-root> [compiler]}"
CXX_BIN="${3:-${CXX:-g++}}"

INC_FLAGS_STR="$(tr '\n' ' ' < "$FLAGS_FILE")"
export INC_FLAGS_STR CXX_BIN

check_header() {
    local err
    err="$(mktemp /tmp/shs-selfcont.XXXXXX)"
    # shellcheck disable=SC2086
    if ! "$CXX_BIN" -std=c++23 -fsyntax-only -x c++ $INC_FLAGS_STR "$1" 2>"$err"; then
        echo "SELFFAIL: $1" >&2
        head -6 "$err" >&2
        rm -f "$err"
        return 1
    fi
    rm -f "$err"
    return 0
}
export -f check_header

TOTAL="$(find "$SRC_ROOT/include" -name '*.hpp' | wc -l)"
echo "header_self_containment: compiling $TOTAL public headers standalone"

find "$SRC_ROOT/include" -name '*.hpp' -print0 \
    | xargs -0 -P "$(nproc)" -n 1 bash -c 'check_header "$0"'
RC=$?

if [ "$RC" -ne 0 ]; then
    echo "FAIL: some public headers are not self-contained" >&2
    exit 1
fi
echo "PASS: all $TOTAL public headers compile standalone (no defines, no build-dir/stb includes)"
