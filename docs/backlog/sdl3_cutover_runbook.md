# SDL3 cutover runbook (windowing dependency, 2026-09)

> **Status: EXECUTED and VERIFIED — full build green, CTest 43/43, both
> boundary gates green (2026-09-17).** Ruling context: SDL2→SDL3 now, no
> dual support (R4) — same clean-cutover precedent as the forwarder-tree and
> namespace retirements. SDL3 has been stable (3.2+) since 2025-01; the vcpkg
> port (`sdl3` 3.4.0, community `sdl3-image`) is mature.

## 1. Scope: complete site inventory (enumerated 2026-09-17)

The lib's entire SDL surface is **three header files and ~40 symbols** — the
migration is fully mechanical:

| # | Site | What changed |
|---|---|---|
| 1 | `include/shs/platform/sdl/sdl_runtime.hpp` | The windowed 2D adapter (renderer path) |
| 2 | `include/shs/resources/adapters/texture_loader_sdl.hpp` | SDL_image load path |
| 3 | `include/shs/rhi/vulkan/runtime/vk_backend.hpp` | Vulkan surface/window interop (3 call sites) |

Everything else was CMake + gate scripts + parked exps (see §3).

## 2. Mechanical rename table (the only non-obvious edits)

| SDL2 | SDL3 | Note |
|---|---|---|
| `<SDL2/SDL.h>` / `<SDL2/SDL_image.h>` / `<SDL2/SDL_vulkan.h>` | `<SDL3/SDL.h>` / `<SDL3/SDL_image.h>` / `<SDL3/SDL_vulkan.h>` | path rename |
| `SDL_Init(SDL_INIT_VIDEO \| SDL_INIT_TIMER) != 0` | `!SDL_Init(SDL_INIT_VIDEO)` | `SDL_INIT_TIMER` removed; `SDL_Init` returns bool |
| `SDL_CreateWindow(title, x, y, w, h, SDL_WINDOW_SHOWN)` | `SDL_CreateWindow(title, w, h, 0)` | x/y gone; windows shown by default, `SDL_WINDOW_SHOWN` retired |
| `SDL_QUIT` / `SDL_KEYDOWN` / `SDL_MOUSEMOTION` / `SDL_MOUSEBUTTONDOWN` / `SDL_MOUSEBUTTONUP` | `SDL_EVENT_QUIT` / `SDL_EVENT_KEY_DOWN` / `SDL_EVENT_MOUSE_MOTION` / `SDL_EVENT_MOUSE_BUTTON_DOWN` / `SDL_EVENT_MOUSE_BUTTON_UP` | event enum family |
| `e.key.keysym.sym` | `e.key.key` | keyboard event field |
| `SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_FOCUS_LOST` | `e.type == SDL_EVENT_WINDOW_FOCUS_LOST` | window events promoted to first-class events |
| `SDL_SetRelativeMouseMode(SDL_TRUE/SDL_FALSE)` | `SDL_SetWindowRelativeMouseMode(window, bool)` | per-window; `SDL_TRUE/SDL_FALSE` removed |
| `SDL_GetRelativeMouseMode() == SDL_TRUE` | `SDL_GetWindowRelativeMouseMode(window)` | same |
| `SDL_GetRelativeMouseMode() == SDL_TRUE` | `SDL_GetWindowRelativeMouseMode(window)` | same |
| `SDL_RenderCopy(...)` | `SDL_RenderTexture(...)` | renderer API |
| `SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED \\| SDL_RENDERER_PRESENTVSYNC)` | `SDL_CreateRenderer(win, nullptr)` + `SDL_SetRenderVSync(renderer, 1)` | flags param removed; accelerated is default, vsync is post-creation |
| `SDL_RENDERER_ACCELERATED` / `SDL_RENDERER_PRESENTVSYNC` | removed | see above |
| `SDL_BUTTON(X)` | `SDL_BUTTON_MASK(X)` | macro rename |
| `SDLK_l` / `SDLK_b` / `SDLK_m` | `SDLK_L` / `SDLK_B` / `SDLK_M` | SDL3 keycodes are uppercase-only |
| `const uint8_t* ks = SDL_GetKeyboardState(...)` | `const bool* ks = SDL_GetKeyboardState(...)` | return type |
| `IMG_Init(IMG_INIT_PNG \\| IMG_INIT_JPG)` / `IMG_Quit()` | **removed** | SDL3_image initializes formats on demand; the functions no longer exist |
| `SDL_FreeSurface(surf)` | `SDL_DestroySurface(surf)` | rename |
| `SDL_ConvertSurfaceFormat(surf, fmt, 0)` | `SDL_ConvertSurface(surf, fmt)` | signature |
| `SDL_GetRGBA(px, &format, ...)` | `SDL_GetPixelFormatDetails(format)` + `SDL_GetRGBA(px, details, nullptr, ...)` | pixel-format details are now separate |
| `SDL_Vulkan_GetInstanceExtensions(win, &n, buf)` | `SDL_Vulkan_GetInstanceExtensions(&n)` → `const char* const*` | no window arg, no out-array variant |
| `SDL_Vulkan_CreateSurface(win, inst, &surf)` | `SDL_Vulkan_CreateSurface(win, inst, nullptr, &surf)` | props struct (nullable) added; still returns bool |
| `SDL_Vulkan_GetDrawableSize(win, &w, &h)` | `SDL_GetWindowSizeInPixels(win, &w, &h)` | renamed, moved out of the vulkan header |

CMake renames: `find_package(SDL2/SDL2_image)` → `SDL3/SDL3_image`;
`SDL2::SDL2` → `SDL3::SDL3` (or `-static`); `SDL2_image::Main` alias →
`SDL3_image::Main`; option `SHS_RENDERER_WITH_SDL2` → `SHS_RENDERER_WITH_SDL3`;
define `SHS_HAS_SDL2` → `SHS_HAS_SDL3`; probe var `SHS_SDL2_HAS_VULKAN` →
`SHS_SDL3_HAS_VULKAN`. The SDL2-era probe read `SDL_VIDEO_VULKAN` out of
SDL's private config header; SDL3 ships **no** public config header, so the
probe is now the presence of `SDL_vulkan.h` in the installed include dir.
Also: `SDL3_image::SDL3_image` can be an ALIAS of the `-static` target, so
the `SDL3_image::Main` alias ladder resolves `ALIASED_TARGET` first (CMake
forbids alias-of-alias).

## 3. Sweep coverage

- **Active build (lib + exps-rendering-adventures):** the three lib files +
  lib CMake + `cmake/shs_rendererConfig.cmake.in`. Adventures contain **zero**
  SDL references (headless PNG path is SDL-free by design) — nothing to move.
- **Gate scripts:** `check_include_graph.py` (`SDL2/` → `SDL3/` in
  `SDK_INC_RE` + negative fixture), `check_kdba_boundaries.sh` (pio gate
  `<SDL2/` → `<SDL3/`; `SDL_[A-Z]` token pattern is version-agnostic).
- **Parked trees (exps-other / exps-software-renderer / exps-gpu-renderer):**
  mechanical `SDL2` → `SDL3` rename across sources and CMakeLists (includes,
  target names, `SHS_SDL3_HAS_VULKAN`, `SHS_HAS_SDL3`, `SDL_image.h` include
  path) so un-parking works; these trees are commented out of the root build
  and were NOT build-verified — they still use SDL2-era call shapes that
  SDL3 removed (`IMG_Init`, 5-arg `SDL_CreateWindow`, renderer-flags
  `SDL_CreateRenderer`, `SDL_FreeSurface`, lowercase `SDLK_*`, …) and must
  get the full §2 rename pass at un-park time.

## 4. Environment notes (2026-09-17, this machine)

- `/opt/vcpkg` is root-owned → SDL3 installed into a **user-local vcpkg**
  (`~/vcpkg`, toolkit 2026-07-27, SDL3 3.4.0 + SDL3_image 3.4.4) with explicit
  `--triplet x64-linux`; the main toolchain stays `/opt/vcpkg` and configure
  adds `-DCMAKE_PREFIX_PATH=$HOME/vcpkg/installed/x64-linux`.
- **`sdl3[vulkan]` feature is mandatory** — it is NOT a default feature;
  without it the port builds without `SDL_Vulkan_*` support.
- **`sdl3-image` has NO default features** — request `sdl3-image[png,jpeg]`
  explicitly. The SDL2-era feature name `libjpeg-turbo` does not exist on the
  sdl3-image port (it uses plain `jpeg`). Without `png`, `IMG_INIT_PNG`
  would be missing (moot: SDL3_image removed `IMG_Init` entirely).
- System `libsdl2-dev` packages may remain installed; package names
  (`SDL3` vs `SDL2`) do not collide.

## 5. Verification (the parity gate) — PASSED 2026-09-17

1. Full reconfigure + build of `cpp-folders` (active trees only) — zero
   errors.
2. Full `ctest`: **43/43 passed** (VOP boundary, header migration,
   self-containment, package consumer, inventory, sw/vk parity — the
   adventures headless PNG parity tests included).
3. Adventures PNG parity: the headless demos never linked SDL; their parity
   tests pass unchanged, proving the lib cutover leaked nothing into the
   headless path.
4. Gate scripts green: `check_include_graph.py` (R2 allowlist now `SDL3/`),
   `check_kdba_boundaries.sh` (pio gate `<SDL3/`).

Test-infrastructure fixes made in the same pass (needed by the dual-vcpkg
setup): `package_consumer_test.sh` now accepts the parent build's
`CMAKE_PREFIX_PATH` entries (args 5+, joined with `;`) so the installed
package's `find_dependency(SDL3)` resolves; the header-inventory JSON was
regenerated for the renamed includes.

## 6. Follow-ups

- Tier2-interactive demo (windowed, `--window` opt-in) now lands directly on
  the SDL3 seam; `SdlRuntime` remains the only SDL-including adapter
  (enforcement-gated by `check_include_graph.py` R2).
- `vk_backend.hpp` still includes SDL headers directly (acceptable for the
  parked monolith; the P3 pod decomposition should move surface creation
  behind `IPlatformRuntime` — same seam as `external_engine_seams.md`).
- hello-3d-demos SDL2_mixer → SDL3_mixer is parked with the trees.
