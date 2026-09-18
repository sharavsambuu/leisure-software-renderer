#!/usr/bin/env bash
set -euo pipefail

# Backend seam symbol-hygiene gate (sdl3_cutover_runbook §7.5; governance
# todo G1.2, from the 2026-09-18 constitutions & laws review).
#
# Contract: the SDL2 windowing/texture anchor TU (src/platform_sdl2_anchor.cpp
# + everything it pulls in) resolves every SDL2/SDL2_image call through the
# dlopen Sdl2Api dispatch table. The compiled anchor object must therefore
# carry ZERO undefined `SDL_*` / `IMG_*` symbols. Reason: the flat ELF
# namespace binds undefined `SDL_Init`-shaped references to whatever same-name
# definition wins the link, and a co-linked static libSDL3 hijacks them (the
# dual-link failure the dlopen dispatch exists to prevent).
#
# The SDL3 anchor is NOT policed for undefined SDL_* symbols: normal SDL3 SDK
# linkage is legal there (the hijack hazard is the SDL2 dlopen seam only —
# runbook §7.5).
#
# Usage:
#   check_backend_seam_symbols.sh [--scan-root <dir>] [<object-file>...]
#   - positional args: object files to inspect;
#   - --scan-root <dir>: recursively collect `platform_sdl*anchor*.(o|obj)`.
# Honest failures (non-zero exit): `nm` unavailable, no SDL2 anchor object
# found among the inputs, or any undefined SDL_*/IMG_* symbol in the anchor.
# The test tree runs this AFTER the library is built (objects must exist).

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lib_root="$(cd "${script_dir}/.." && pwd)"

objects=()
scan_root=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scan-root)
            [[ $# -ge 2 ]] || { echo "[backend-seam-symbols] FAIL: --scan-root needs a directory argument"; exit 1; }
            scan_root="$2"; shift 2 ;;
        --scan-root=*) scan_root="${1#*=}"; shift ;;
        *) objects+=("$1"); shift ;;
    esac
done

if ! command -v nm >/dev/null 2>&1; then
    echo "[backend-seam-symbols] FAIL: nm not found on PATH — cannot verify the zero-undefined-SDL-symbol contract"
    exit 1
fi

if [[ -n "${scan_root}" ]]; then
    while IFS= read -r f; do objects+=("${f}"); done < <(
        find "${scan_root}" -type f \( -name 'platform_sdl*anchor*.o' -o -name 'platform_sdl*anchor*.obj' \) 2>/dev/null | sort)
fi

sdl2_objects=()
sdl3_objects=()
for obj in "${objects[@]}"; do
    case "$(basename "${obj}")" in
        platform_sdl2_anchor*) sdl2_objects+=("${obj}") ;;
        platform_sdl3_anchor*) sdl3_objects+=("${obj}") ;;
    esac
done

if [[ "${#sdl2_objects[@]}" -eq 0 ]]; then
    echo "[backend-seam-symbols] FAIL: no SDL2 anchor object found among inputs (${#objects[@]} objects scanned) — cannot verify the zero-undefined-SDL-symbol contract"
    exit 1
fi

failed=0
for obj in "${sdl2_objects[@]}"; do
    undef="$(nm -u "${obj}" 2>/dev/null | awk '{print $NF}' | grep -E '^(SDL_|IMG_)' | sort -u || true)"
    if [[ -n "${undef}" ]]; then
        echo "[backend-seam-symbols] FAIL: undefined SDL_*/IMG_* symbols in ${obj} — a direct SDK call sits in the dlopen-dispatch anchor TU (flat-namespace hijack hazard; sdl3_cutover_runbook §7.5)"
        echo "${undef}" | sed 's/^/    /'
        failed=1
    else
        echo "[backend-seam-symbols] OK: ${obj} — zero undefined SDL_*/IMG_* symbols (dlopen dispatch intact)"
    fi
done

for obj in "${sdl3_objects[@]}"; do
    echo "[backend-seam-symbols] INFO: SDL3 anchor ${obj} not policed — normal SDL3 linkage is legal (hijack hazard is SDL2-only)"
done

exit "${failed}"
