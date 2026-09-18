#!/usr/bin/env bash
set -euo pipefail

# Negative fixture for the backend seam symbol-hygiene gate (governance todo
# G1.2): proves the gate actually trips. A stub TU that merely *references*
# SDL_Init (extern "C" declaration — no SDK headers needed) must FAIL the
# gate; and the gate must fail honestly when no SDL2 anchor object is among
# its inputs (silent pass on un-verifiable input would be a false shield).
#
# Usage: check_backend_seam_symbols_negative_test.sh <cxx-compiler>

cxx="${1:?usage: check_backend_seam_symbols_negative_test.sh <cxx-compiler>}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gate="${script_dir}/check_backend_seam_symbols.sh"

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

cat > "${tmp}/sdl_ref_probe.cpp" <<'EOF'
extern "C" void SDL_Init();
void shs_seam_negative_probe() { SDL_Init(); }
EOF
"${cxx}" -std=c++20 -c -o "${tmp}/platform_sdl2_anchor_probe.o" "${tmp}/sdl_ref_probe.cpp"

if bash "${gate}" "${tmp}/platform_sdl2_anchor_probe.o" >/dev/null 2>&1; then
    echo "[backend-seam-symbols-negative] FAIL: gate passed an object with an undefined SDL_Init reference (must trip)"
    exit 1
fi
echo "[backend-seam-symbols-negative] OK: gate trips on an undefined SDL_* reference"

if bash "${gate}" "${tmp}/sdl_ref_probe.cpp" >/dev/null 2>&1; then
    echo "[backend-seam-symbols-negative] FAIL: gate passed with no SDL2 anchor object among inputs (must fail honestly)"
    exit 1
fi
echo "[backend-seam-symbols-negative] OK: gate fails honestly when no anchor object is provided"
