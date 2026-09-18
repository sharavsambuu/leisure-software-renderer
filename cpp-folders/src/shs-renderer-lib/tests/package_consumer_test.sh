#!/usr/bin/env bash
# Migration step 5 — install/export/package-consumer test.
#
# Proves, against the *installed* package (not the build tree):
#   1. `cmake --install` produces a find_package()-able shs_renderer package
#      (Config + Version + exported targets + headers);
#   2. aggregate compatibility: consumers may link the `shs::renderer` /
#      `shs::renderer-values` aliases exported by Config.cmake;
#   3. source-tree/build-tree path leakage rejection: no absolute source or
#      build path appears anywhere in the installed package;
#   4. a minimal headless consumer (no SHS_HAS_* SDK usage in its own TU)
#      configures, builds, links and runs against the installed package.
#
# Usage: package_consumer_test.sh <build-dir> <source-dir> [toolchain] [cxx]
set -euo pipefail

BUILD_DIR="${1:?usage: package_consumer_test.sh <build-dir> <source-dir> [toolchain] [cxx]}"
SOURCE_DIR="${2:?usage: package_consumer_test.sh <build-dir> <source-dir> [toolchain] [cxx]}"
TOOLCHAIN_FILE="${3:-}"
CXX_BIN="${4:-${CMAKE_CXX_COMPILER:-}}"
# Args 5+ are the parent build's CMAKE_PREFIX_PATH entries (one per path).
shift 4 || true
# Join with ';' (CMake list separator) — the args arrive space-separated.
if [ "$#" -gt 0 ]; then
    _shs_ifs_save="$IFS"
    IFS=';'
    EXTRA_PREFIX_PATH="${*}"
    IFS="$_shs_ifs_save"
    unset _shs_ifs_save
else
    EXTRA_PREFIX_PATH=""
fi

SCRATCH="$(mktemp -d /tmp/shs-pkg-consumer.XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
PREFIX="$SCRATCH/prefix"

fail() { echo "FAIL: $*" >&2; exit 1; }

# 1. Install this build into a scratch prefix.
cmake --install "$BUILD_DIR" --prefix "$PREFIX" >/dev/null \
    || fail "cmake --install failed"
test -f "$PREFIX/lib/cmake/shs_renderer/shs_rendererConfig.cmake" \
    || fail "shs_rendererConfig.cmake not installed"
test -f "$PREFIX/lib/cmake/shs_renderer/shs_rendererConfigVersion.cmake" \
    || fail "shs_rendererConfigVersion.cmake not installed"
test -f "$PREFIX/lib/cmake/shs_renderer/shs_rendererTargets.cmake" \
    || fail "shs_rendererTargets.cmake not installed"
test -f "$PREFIX/include/shs/renderpath/renderpath.gateway.hpp" \
    || fail "public headers not installed"
test -e "$PREFIX/lib/libshs_renderer.a" -o -e "$PREFIX/lib/libshs_renderer.so" \
    || fail "compiled library not installed"

# 3. Source-tree/build-tree path leakage rejection. The installed package
#    must be relocatable: no absolute path from the source tree or build
#    tree may survive into headers or exported CMake files.
LEAKS="$(grep -RIl -e "$SOURCE_DIR" -e "$BUILD_DIR" "$PREFIX" || true)"
if [ -n "$LEAKS" ]; then
    echo "FAIL: source-tree/build-tree path leakage in installed package:" >&2
    echo "$LEAKS" >&2
    exit 1
fi

# 2+4. Minimal headless consumer: find_package + build + run.
mkdir -p "$SCRATCH/consumer"
cat > "$SCRATCH/consumer/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.20)
project(shs_pkg_consumer LANGUAGES CXX)

find_package(shs_renderer CONFIG REQUIRED)

add_executable(shs_pkg_consumer main.cpp)
# Aggregate compatibility seams: link via the aliases, not the raw names.
target_link_libraries(shs_pkg_consumer PRIVATE shs::renderer shs::renderer-values)
EOF
cat > "$SCRATCH/consumer/main.cpp" <<'EOF'
// Minimal headless package consumer: value-layer headers only, zero SDKs.
#include "shs/renderpath/renderpath.gateway.hpp"

namespace shs { int shs_renderer_compiled_target_anchor(); }

int main()
{
    if (shs::shs_renderer_compiled_target_anchor() != 0)
    {
        return 1;
    }
    return 0;
}
EOF

CONSUMER_ARGS=(-DCMAKE_PREFIX_PATH="$PREFIX")
if [ -n "$EXTRA_PREFIX_PATH" ]; then
    CONSUMER_ARGS+=(-DCMAKE_PREFIX_PATH="$PREFIX;$EXTRA_PREFIX_PATH")
fi
if [ -n "$TOOLCHAIN_FILE" ]; then
    CONSUMER_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE")
fi
if [ -n "$CXX_BIN" ]; then
    CONSUMER_ARGS+=(-DCMAKE_CXX_COMPILER="$CXX_BIN")
fi

cmake -S "$SCRATCH/consumer" -B "$SCRATCH/consumer/build" \
    "${CONSUMER_ARGS[@]}" > "$SCRATCH/consumer/configure.log" 2>&1 \
    || { cat "$SCRATCH/consumer/configure.log" >&2; fail "consumer configure failed"; }
cmake --build "$SCRATCH/consumer/build" > "$SCRATCH/consumer/build.log" 2>&1 \
    || { tail -40 "$SCRATCH/consumer/build.log" >&2; fail "consumer build failed"; }

"$SCRATCH/consumer/build/shs_pkg_consumer" \
    || fail "consumer binary failed to run"

echo "PASS: package consumer configured, built and ran from installed prefix"
