# SDL3 cutover runbook (windowing dependency, 2026-09)

> **Status: EXECUTED and VERIFIED — full build green, CTest 43/43, both
> boundary gates green (2026-09-17).** Ruling context: SDL2→SDL3 now, no
> dual support (R4) — same clean-cutover precedent as the forwarder-tree and
> namespace retirements. SDL3 has been stable (3.2+) since 2025-01; the vcpkg
> port (`sdl3` 3.4.0, community `sdl3-image`) is mature.
>
> **SUPERSEDED 2026-09-17 (same day, owner ruling): the SDL3-only stance is
> reversed — see §6.** The windowing seam is now **platform-agnostic**: SDL2
> AND SDL3 are both supported backends behind `IPlatformRuntime`
> (`window_backend.hpp`), with SFML/GLFW pluggable later. Everything in
> §1–§5 below remains the accurate record of the SDL3 cutover itself.

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

## 7. Addendum — platform-agnostic seam, SDL2 + SDL3 dual support (2026-09-17)

**Owner ruling (supersedes the §0 "no dual support" R4 ruling, same day):**
the windowing dependency becomes a **platform-agnostic seam with SDL2 and
SDL3 both supported**; SFML/GLFW plug into the same seam later. The SDL3
cutover work itself (§1–§5) is untouched — this addendum re-adds SDL2 as a
second backend and, in doing so, executes the deferred vk_backend seam fix
(the old §6 follow-up bullet is now DONE).

### 7.1 The seam

| Piece | File | Role |
|---|---|---|
| Vulkan window interop | `shs/platform/platform_runtime.hpp` | `IVulkanWindowInterop` (instance extensions, surface creation, drawable pixel size) + `IPlatformRuntime::window_vulkan_interop()` (defaulted `nullptr`) — **`vk_backend.hpp` includes zero windowing SDKs now** |
| Backend selection | `shs/platform/window_backend.hpp` (new) | `WindowBackend {Auto, Sdl3, Sdl2}`, `WindowBackendCreateResult` (runtime + actual backend + honest `note`), `create_platform_runtime()` — zero SDK includes, runtime dispatch |
| SDL3 adapter | `shs/platform/sdl/sdl3_runtime.hpp` (canonical; `sdl_runtime.hpp` is now a compat forwarder aliasing `SdlRuntime = Sdl3Runtime`) | `Sdl3Runtime` + `Sdl3VulkanInterop` |
| SDL2 adapter | `shs/platform/sdl/sdl2_runtime.hpp` (new) | `Sdl2Runtime` + `Sdl2VulkanInterop`, SDL2 call shapes (§2 table, right-to-left) |
| Texture dispatch | `shs/resources/adapters/texture_loader.hpp` (new) | `load_texture2d_image(path, flip_y, ImageSource)` / `import_texture_image(...)` — Auto tries SDL3_image then SDL2_image; SDL-specific entry points (`load_texture2d_sdl_image` / `_sdl2_image`) stay valid |
| Anchor TUs | `src/platform_sdl3_anchor.cpp` / `src/platform_sdl2_anchor.cpp` | compiled unconditionally; each TU includes exactly one SDL SDK under `SHS_HAS_SDL3=1` / `SHS_HAS_SDL2=1` |

**Iron rule: SDL2 and SDL3 headers must never share a translation unit.**
That is why dispatch is extern-factory based (anchors), not header-level
`#if` over both SDKs. `window_backend.hpp` and `texture_loader.hpp` carry no
SDK includes at all.

### 7.2 `vk_backend.hpp` — surface creation moved behind the seam

`InitDesc.window` (`SDL_Window*`) → `InitDesc.window_interop`
(`platform::IVulkanWindowInterop*`, obtained from
`runtime.window_vulkan_interop()`). All four SDL call sites
(`SDL_Vulkan_CreateSurface`, `SDL_Vulkan_GetInstanceExtensions`,
2× `SDL_GetWindowSizeInPixels`) now go through the interop. `<SDL3/SDL.h>`
and `<SDL3/SDL_vulkan.h>` are gone from the RHI.

**Break:** callers pass `desc.window_interop = runtime.window_vulkan_interop();`
instead of `desc.window = win;`. Parked trees set the old field — fix at
un-park time (mechanical, one line per demo).

### 7.3 Build plumbing

- `SHS_RENDERER_WITH_SDL3` (ON) — SDL3 REQUIRED when on (unchanged).
- `SHS_RENDERER_WITH_SDL2` (ON, new) — best-effort QUIET: CMake config first,
  pkg-config fallback; **missing SDL2 degrades honestly to an SDL3-only
  build** (STATUS note, no FATAL). Define `SHS_HAS_SDL2=1` + link when found;
  `SHS_SDL2_HAS_VULKAN` probed like the SDL3 one (parent-scope exported).
- CMake ≥ 3.29 `INTERFACE_SDL_VERSION` conflict detection rejects a dual
  SDL2+SDL3 link — cleared explicitly (in-tree CMakeLists + installed
  `shs_rendererConfig.cmake.in`) because the dual link is intentional and
  SDK-isolated per TU.
- Package: `SHS_RENDERER_PKG_WITH_SDL2` re-discovered via
  `find_dependency(SDL2 CONFIG)` + `find_dependency(SDL2_image CONFIG)`
  (pkg-config-only discovery is a documented consumer-side limitation).

### 7.4 Verification (2026-09-17)

- Full build: zero errors, zero warnings, both `SHS_HAS_SDL2=1` and
  `SHS_HAS_SDL3=1` active on this machine (SDL2 2.30 via vcpkg config,
  SDL3 3.4.0).
- **CTest 43/43** including `shs_renderer_header_self_containment_test`
  (both new adapter headers compile standalone, no defines),
  `shs_renderer_package_consumer_test` (dual-SDL installed package consumed
  by a fresh consumer), header-inventory, include-graph and kdba gates
  (both extended for `SDL2/` / `SDL2_image/` include paths).
- CMake 3.29 generate-time conflict resolved by clearing
  `INTERFACE_SDL_VERSION` on the imported SDL targets (in-tree + consumer
  config) — documented, not a workaround of the isolation rule.
- **Live dispatch smoke** (manual, /tmp): a consumer TU calling
  `create_platform_runtime(..., WindowBackend::Sdl3)` +
  `load_texture2d_image()` linked against `libshs_renderer.a` with both SDKs
  present created a real SDL3 window/renderer/texture at runtime and the
  honest-failure texture path returned empty — seam verified end-to-end.
- **Known limitation (documented, honest):** SDL2 and SDL3 *static* archives
  collide on dynamic-API symbols (`SDL_AddEventWatch`, ...) when both are
  linked into ONE binary — inherent to both SDKs exporting identical C
  symbols, not to this seam. Shared SDL2 (or shared SDL3) plus the other
  static is fine, as is a single-SDK build. The dispatch dispatches at
  runtime, so binaries that only ever create one backend's runtime are
  unaffected as long as the unused SDK's anchor objects are not pulled
  (archive member granularity handles this when only the created backend's
  factory is referenced via Auto+success paths... a dual-static binary that
  references BOTH factories must link at least one SDK shared).
- Headless adventures PNG parity unchanged (windowing seam is not reachable
  from the SDL-free path).

### 7.5 Dual-link resolution — system-shared-first + dlopen dispatch (2026-09-17, later session)

Resolution of the §7.4 "known limitation", plus the build-composition and
packaging follow-through for the demo `--window` front-end:

- **Link composition (system-shared-first):** with both SDKs discovered, the
  demos link SHARED system `libSDL2-2.0.so` + shared `libSDL2_image` and
  STATIC SDL3/SDL3_image (vcpkg). Build is clean: zero multiple-definition
  collisions. The `INTERFACE_SDL_VERSION` conflict detection stays cleared.
- **Export-interface rule (enforced by tests):** the `SDL2::SDL2` /
  `SDL2_image` links on `shs_renderer` must be `$<BUILD_INTERFACE:>`-wrapped,
  or the installed `shs_rendererTargets.cmake` references system paths and
  breaks package consumers (observed at Targets line 61).
  `package_consumer_test` + `header_inventory_check` guard this.
- **Dual-link hijack (root cause found with gdb):** even with shared libSDL2
  on the link line, the flat ELF namespace + link order made the STATIC
  `libSDL3.a` satisfy the SDL2 backend's `SDL_*` references — the demo
  executable's `SDL_Init` bound to SDL3's `SDL_dynapi_procs.h`, so SDL2's
  5-arg `SDL_CreateWindow` call was decoded with SDL3's 3-arg ABI and window
  creation failed (honest nullptr). Same-name C symbols cannot be bound to
  two different libraries in one binary; no link order fixes both backends.
- **Fix: dlopen dispatch (hijack shield).** `sdl2_runtime.hpp` now loads
  `libSDL2-2.0.so.0` via `dlopen(RTLD_LOCAL)` and dispatches every SDL2 and
  SDL2_image call through an `Sdl2Api` function-pointer table
  (`shs_sdl2_api()`; SDL2_image is optional, IMG_* null-guarded). The SDL2
  backend TU owns **zero** undefined `SDL_*`/`IMG_*` symbols — verify with
  `nm -u .../platform_sdl2_anchor.cpp.o` (must list none). The texture
  adapter `texture_loader_sdl2.hpp` routes through the same table. Linking
  libSDL2 into the lib remains legal (keeps the soname resolvable); nothing
  references it statically. Result: `--backend=sdl2` works in dual-linked
  demo binaries. dlopen failure → honest "backend not available" contract
  unchanged. Non-dlopen platforms (Windows) report honest unavailability.
- **Wayland note:** SDL2 2.30 on this Wayland session selects the x11 video
  driver by default and window creation succeeds — no `SDL_VIDEODRIVER` pin
  required. A broken driver env (`SDL_VIDEODRIVER=nosuchdriver`) exercises
  the honest-failure path: `windowed mode unavailable: ...` on stderr, demo
  PNG already written, clean exit.
- **Verification (2026-09-17, this session):** full build 0 errors / 0
  warnings; anchor `nm -u` free of SDL/IMG refs; **CTest 43/43**; kdba
  boundary + include-graph + header-inventory gates green. Live smokes:
  headless PNG parity; `--window` auto → SDL3 live-present (exit 124 by
  timeout design); `--window --backend=sdl2` → `requested SDL2 runtime
  created` (exit 124); `_vk` twin auto → SDL3; in-process probe pushed F12 /
  Esc / QUIT through the dlopen-dispatched `pump_input` →
  `save_screenshot=1`, `quit=1` (F12→`<stem>_export_1.png` inside a live
  demo window not machine-tested — no key injector on this box — but the
  mapping is probe-verified and the export path is shared with SDL3).
- **Zero-undef contract is now machine-enforced (2026-09-18, governance todo
  G1.2):** the manual `nm -u` check above is codified as
  `tools/check_backend_seam_symbols.sh` — CTest
  `shs_renderer_backend_seam_symbols_check` fails if any
  `platform_sdl2_anchor*` object carries undefined `SDL_*`/`IMG_*` symbols
  (the flat-namespace hijack class above). The SDL3 anchor is reported, not
  policed (normal SDK linkage is legal there; the hazard is the SDL2 dlopen
  seam only). A negative fixture
  (`shs_renderer_backend_seam_symbols_negative_test`) proves the gate trips
  on a stub TU that references `SDL_Init` and fails honestly when no anchor
  object is found. House rule going forward: any new SDL2/IMG call must enter
  through the `Sdl2Api` dispatch table — direct calls will turn the gate red.
