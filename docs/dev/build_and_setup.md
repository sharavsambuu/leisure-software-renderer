# Build & Platform Setup Guide

Canonical, per-platform setup and build instructions for the converged single-library
tree (`shs-renderer-lib`). This document ingests the platform setup sections that used
to live in the root `README.md`, corrects them to the post-convergence layout, and adds
the Slang (`slangc`) toolchain installation.

> Agent-side build/test workflow, shell traps through the Windows→WSL bridge, and
> CMake/C++ pitfalls: `docs/dev/cpp_compilation_workflow.md` (read it before running
> anything as an agent).

## Tree layout (post-convergence, 2026-09)

```
cpp-folders/
  CMakeLists.txt          # single root project; one add_subdirectory
  src/
    shs-renderer-lib/     # THE library (formerly shs-core-lib + absorbed shs-gpu-lib)
                          #   targets: shs_renderer / shs::renderer,
                          #            shs_renderer_values / shs::renderer-values
    exps-software-renderer/   # software-path demos (parked)
    exps-gpu-renderer/        # Vulkan-path experiments (parked)
    exps-other/               # parallelization / misc experiments (parked)
    assets/
  build/                  # build dir
```

Capability gates (configure-time):

- `SHS_HAS_VULKAN=1` — defined when Vulkan is found (`find_package(Vulkan)`, with an
  Apple `VULKAN_SDK` fallback). Without it, `shs_renderer` builds with **zero** edge
  sources; a Vulkan backend request falls back to `SoftwareRenderBackend` via
  `create_render_backend()` (honest `RenderBackendCreateResult`: `requested` / `active` / `note`).
- `SHS_HAS_VMA=1` — Vulkan Memory Allocator linked when available.
- `SHS_HAS_SLANG` — planned Slang gate (see the Slang section below).

Vulkan auto-detection is the default: **no manual CMake setting is required** on a
normal machine. To force a GPU-free configure (software-only), see
[GPU-free configure](#gpu-free-configure-software-only).

## Common prerequisites

All platforms need: CMake ≥ 3.20, a C++20 compiler, vcpkg, and the vcpkg packages
below. Vulkan SDK and Slang are per-platform.

### vcpkg packages (required at configure time)

`find_package(... REQUIRED)` — missing any of these aborts CMake:

```bash
vcpkg install "sdl2[vulkan]"               # window/input backend for every demo
vcpkg install "sdl2-image[libjpeg-turbo]"  # image loading: REQUIRED for hello-render-target demos
                                           # (HelloShadowMapping, HelloWater, HelloIblSkybox*, ...)
                                           # NOTE: installs as SDL2_image::SDL2_image-static on x64-linux
vcpkg install glm                          # math library used everywhere
vcpkg install assimp                       # model loading (hello-3d-primitives + hello-render-target)
vcpkg install vulkan-memory-allocator      # REQUIRED at configure time even for CPU-only work
```

Optional / feature-gated:

```bash
vcpkg install joltphysics    # physics experiments (exps-gpu-renderer/exp-plumbing)
vcpkg install lua            # scripting experiments (configure tolerates absence)
```

### Slang (`slangc`) — shader compiler toolchain

Slang is the chosen shader pipeline going forward (Slang → SPIR-V for the Vulkan edge,
with reflection feeding the pod driver). Key property: **`slangc` is a standalone
compiler — it needs no GPU, no Vulkan SDK, and no display**, so shader artifacts build
on GPU-free machines just like everything else.

As of 2026-09 there is **no vcpkg `slang` port**, so install the official prebuilt
release directly (pinned here: **v2026.17.1**). Long-term, CMake will auto-discover
`slangc` via `find_program` and gate on `SHS_HAS_SLANG`; a manual install is the
supported path today.

**Linux x86_64:**

```bash
mkdir -p ~/slang && cd ~/slang
wget https://github.com/shader-slang/slang/releases/download/v2026.17.1/slang-2026.17.1-linux-x86_64.tar.gz
tar -xzf slang-2026.17.1-linux-x86_64.tar.gz
echo 'export PATH="$HOME/slang/slang/bin:$PATH"' >> ~/.bashrc   # dir containing slangc
source ~/.bashrc
slangc -h    # verify: prints usage
```

(Linux aarch64: use `slang-2026.17.1-linux-aarch64.tar.gz`. Older glibc hosts can use
the `-glibc-2.27` / `-glibc-2.28` variants.)

**macOS (Apple Silicon):**

```bash
mkdir -p ~/slang && cd ~/slang
curl -LO https://github.com/shader-slang/slang/releases/download/v2026.17.1/slang-2026.17.1-macos-aarch64.tar.gz
tar -xzf slang-2026.17.1-macos-aarch64.tar.gz
echo 'export PATH="$HOME/slang/slang/bin:$PATH"' >> ~/.zshrc
xattr -d com.apple.quarantine ~/slang/slang/bin/slangc 2>/dev/null  # if Gatekeeper complains
slangc -h
```

(Intel Macs: `slang-2026.17.1-macos-x86_64.tar.gz`.)

**Windows 11 (x86_64):**

Download `slang-2026.17.1-windows-x86_64.zip` from the release page, extract to e.g.
`C:\slang`, then add `C:\slang\bin` to `PATH` (system properties → environment
variables). Verify with `slangc -h` in a new shell.

(aarch64 Windows: `slang-2026.17.1-windows-aarch64.zip`.)

Release page: <https://github.com/shader-slang/slang/releases>

## On Ubuntu 24.04

```bash
sudo apt install automake m4 libtool cmake build-essential autoconf autoconf-archive \
  automake libtool-bin python3.12-venv python3.13-venv
```

**LunarG Vulkan SDK:**

```bash
mv ~/Downloads/vulkansdk-linux-x86_64-1.4.341.1.tar.xz ~/vulkan
cd ~/vulkan && tar -xvf vulkansdk-linux-x86_64-1.4.341.1.tar.xz
# add to bottom of ~/.bashrc:
source ~/vulkan/1.4.341.1/setup-env.sh
```

**vcpkg:** <https://lindevs.com/install-vcpkg-on-ubuntu>

```bash
export VCPKG_ROOT="/opt/vcpkg"
```

Then install the [vcpkg packages](#vcpkg-packages-required-at-configure-time) and
[Slang](#slang-slangc--shader-compiler-toolchain) as above.

**Configure + build:**

```bash
cd cpp-folders && mkdir -p build && cd build
export VCPKG_ROOT="/opt/vcpkg"
cmake .. -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
make -j20
ctest --output-on-failure                       # 5/5 expected (VOP boundary + tests)
```

Demos are currently **parked** (they predate the `create_render_backend()` contract —
see the note at the bottom). A software-path demo still runs the classic way:

```bash
cd src/exps-software-renderer/hello-pixel-primitives && ./HelloPixel
```

## On macOS

MoltenVK / Vulkan guidance:

- <https://github.com/MrVideo/ARMMoltenVKGuide>
- <https://vulkan-tutorial.com/Development_environment#page_MacOS>
- <https://vulkan.lunarg.com/sdk/home#mac>

vcpkg via brew, packages with `:arm64-osx` triples:

```bash
brew install vcpkg
git clone https://github.com/microsoft/vcpkg.git "$HOME/vcpkg"
export VCPKG_ROOT="$HOME/vcpkg"

vcpkg install "sdl2[vulkan]:arm64-osx" --recurse
vcpkg install "sdl2-image[libjpeg-turbo]:arm64-osx"
vcpkg install "glm:arm64-osx"
vcpkg install "assimp:arm64-osx"
vcpkg install "joltphysics:arm64-osx"
vcpkg install "vulkan-memory-allocator:arm64-osx"
vcpkg install "lua:arm64-osx"
```

Install the macOS Slang release (see the
[Slang section](#slang-slangc--shader-compiler-toolchain)).

CMake Vulkan detection: Linux/Windows use normal `find_package(Vulkan)`; macOS tries
normal detection first, then falls back to the `VULKAN_SDK` path if needed.

**Configure + build:**

```bash
cd cpp-folders && mkdir -p build && cd build
export VCPKG_ROOT="$HOME/vcpkg"
export VK_ICD_FILENAMES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json"
export VK_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d"
cmake .. -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
make -j20
```

## On Windows 11

- Set the vcpkg toolchain variable: system properties → environment variables →
  system variables → New:
  `CMAKE_TOOLCHAIN_FILE = C:\src\misc\vcpkg\scripts\buildsystems\vcpkg.cmake`
  (adjust to your install location).

```bat
vcpkg install sdl2[vulkan] --recurse
vcpkg install sdl2-image
vcpkg install sdl2-image:x64-windows-static
vcpkg install --recurse sdl2-image[libjpeg-turbo]
vcpkg install libjpeg-turbo
vcpkg install glm
vcpkg install assimp
vcpkg install joltphysics
vcpkg install vulkan-memory-allocator:x64-windows
vcpkg install lua:x64-windows
```

Install the Windows Slang release (see the
[Slang section](#slang-slangc--shader-compiler-toolchain)).

Use CMake-GUI with **Visual Studio 17 2022**.

## GPU-free configure (software-only)

Validated escape hatch for machines without any Vulkan/SDK:

```bash
cmake .. -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
         -DCMAKE_DISABLE_FIND_PACKAGE_Vulkan=TRUE
```

Result: `Vulkan_FOUND` unset → `SHS_HAS_VULKAN` not defined → the Vulkan runtime edge
(monolith `shs/rhi` + `shs/pipeline/vk_*` + VMA) is **not compiled at all**; the build
stays green, ctest passes. The pure pod driver (`shs/execution/rhi/drivers/vulkan/`)
still builds — it is software by construction.

## Demo restart note (parked demos)

Existing demos hard-fail on concrete Vulkan types. New demos must:

1. Call `create_render_backend()` (Vulkan auto-detected, no manual CMake setting).
2. Branch on `RenderBackendCreateResult.active` / `BackendCapabilities` — never
   hard-cast to concrete Vulkan types. On a no-Vulkan build the request honestly
   resolves to `SoftwareRenderBackend` with an explanatory `note`.

## Related docs

- Agent build/test workflow + pitfalls: `docs/dev/cpp_compilation_workflow.md`
- Agent environment rules: `docs/dev/agent_environment.md`
- Constitutions: `docs/spec/conventions.md`, `docs/spec/value_oriented_programming.md`
- Root README (high-level about, results gallery, references)