#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${1:-${ROOT_DIR}/build}"
OUT_DIR="${2:-${ROOT_DIR}/parity-captures}"
BUILD_DIR="$(cd -- "${BUILD_DIR}" && pwd)"

LEGACY_BIN="${BUILD_DIR}/src/hello-render-target/HelloPBRLightShafts"
PLUMB_BIN="${BUILD_DIR}/src/hello-plumbing/HelloPassPlumbing"
LEGACY_DIR="${BUILD_DIR}/src/hello-render-target"
PLUMB_DIR="${BUILD_DIR}/src/hello-plumbing"

echo "[phase7] build: ${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --target HelloPBRLightShafts HelloPassPlumbing -j4

if [[ ! -x "${LEGACY_BIN}" ]]; then
    echo "[phase7] missing binary: ${LEGACY_BIN}" >&2
    exit 1
fi
if [[ ! -x "${PLUMB_BIN}" ]]; then
    echo "[phase7] missing binary: ${PLUMB_BIN}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}/legacy" "${OUT_DIR}/plumbing"
OUT_DIR="$(cd -- "${OUT_DIR}" && pwd)"

for preset in 0 1 2; do
    echo "[phase7] capture legacy preset ${preset}"
    (
        cd -- "${LEGACY_DIR}"
        SDL_VIDEODRIVER=dummy SDL_RENDER_DRIVER=software "${LEGACY_BIN}" \
            --preset "${preset}" \
            --capture "${OUT_DIR}/legacy/view_${preset}.ppm" \
            --capture-after 8
    )

    echo "[phase7] capture plumbing preset ${preset}"
    (
        cd -- "${PLUMB_DIR}"
        SDL_VIDEODRIVER=dummy SDL_RENDER_DRIVER=software "${PLUMB_BIN}" \
            --preset "${preset}" \
            --capture "${OUT_DIR}/plumbing/view_${preset}.ppm" \
            --capture-after 8
    )
done

if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 \
        "${OUT_DIR}"/legacy/view_*.ppm \
        "${OUT_DIR}"/plumbing/view_*.ppm \
        > "${OUT_DIR}/SHA256SUMS.txt"
fi

echo "[phase7] captures written to: ${OUT_DIR}"
