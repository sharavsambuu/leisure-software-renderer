# C++ Compilation & Agent Workflow (leisure-software-renderer)

This document is written for **AI coding agents** working in this repo. It defines the exact, validated build/test workflow and the corner cases that are easy to get wrong when an agent drives compilation from a Windows shell into WSL Ubuntu 24.04. Read `docs/dev/agent_environment.md` first — it explains why commands behave the way they do here.

## The agent session loop (memorize this; everything below is detail)

For ANY code task in this repo, run this loop — it is the shortest path to a verified result:

1. **Read context first** (file tools, no shell): the target demo's `STATUS.md` if it exists
   (e.g. `cpp-folders/src/hello-3d-demos/snake/docs/STATUS.md` — canonical per-demo state written by
   previous agents), plus the files you will edit. Per-demo STATUS.md files override stale
   IMPLEMENTATION_PLAN.md files.
2. **Edit** with `write_to_file`/`replace_in_file` only.
3. **Reconfigure if any CMakeLists.txt changed**: `cmake .` inside the build dir (template below).
4. **Build ONE target** for fast iteration; **full build** before declaring done.
5. **Verify from the log**: `EXIT=0`, `grep -cE 'error:'` = 0, and for warning hygiene
   `grep 'warning:' log | grep -v shs_renderer.hpp | wc -l` = 0 for files you touched.
6. **Stop.** Do not run redundant verification passes.

If a build fails: read the FIRST error in the log (later ones are usually cascades), check the
pitfalls list below, fix, rebuild. Never guess-fix more than one thing between builds.

## The one rule (repeated, because it is the whole problem)

> **An agent's shell command runs in Windows `cmd.exe`, whose cwd is a broken UNC path.**
> Commands with no explicit filesystem path silently run against Windows root (`C:\Windows`) and return wrong results — they do NOT error out. To touch the project you must pass an **explicit WSL path** (inside a `wsl` bash wrapper) or use the file tools directly.

## Project layout (validated)

- Root build dir: `cpp-folders/`
  - Top-level `CMakeLists.txt`: C++20, `include(CTest)`, fetches stb / xsimd / JoltPhysics via FetchContent, finds VulkanMemoryAllocator + Lua 5.5 via vcpkg.
  - Per-target dirs under `cpp-folders/src/` (each has its own `CMakeLists.txt`):
    - `hello-shs-renderer`, `hello-pixel-primitives`, `hello-shaders`, `hello-parallelization`, `hello-3d-primitives`, `hello-3d-demos`, `hello-render-target`, `hello-other-exps` (demos)
    - `shs-renderer-lib` (the library + the VOP tests), `exp-plumbing`, `exp-rendering-techniques`
  - Build artifacts land in `cpp-folders/build/` (already present; contains a Makefile, CMakeCache.txt, Testing/, _deps/).

## The build/test workflow (validated)

All of this runs from **bash**, not from Windows `cmd.exe`. An agent must wrap every command so bash resolves WSL paths natively:

```
wsl -e bash -lc "cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders && <linux command>"
```

### 1. Configure (first time, or after CMake changes)

```bash
cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders && mkdir -p build && cd build
export VCPKG_ROOT="/opt/vcpkg"
cmake .. -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
```

- `mkdir -p` is idempotent — safe to re-run.
- The toolchain file wires vcpkg into CMake so the fetched deps resolve from `/opt/vcpkg`.

### 2. Build (parallel)

```bash
cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build && make -j20
```

- `make` needs a real Linux build dir; it cannot run against Windows root. Always wrap in the `wsl` bash form with an explicit path.
- Re-running is safe (incremental). If you changed CMake, re-run step 1 first.

### 3. Test (VOP boundary + deterministic core)

```bash
cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build && ctest -R "shs_renderer_vop_(boundary_check|tests)" --output-on-failure
```

- Registered tests live in `src/shs-renderer-lib/CMakeLists.txt`: `shs_renderer_vop_boundary_check` (a custom target) and `shs_renderer_vop_tests` (an executable). The `-R "..."` regex selects exactly those.
- `--output-on-failure` prints failing test output; without it ctest is quiet on success.

### 4. Run a demo binary (optional, after build)

```bash
cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build && cd src/hello-pixel-primitives && ./HelloPixel
```

- Demos are the `hello-*` targets; each produces an executable under its own subdir of `build/src/`.

## Nested sub-CMake project structure (validated)

The `cpp-folders/src/hello-3d-demos/` aggregator is a **multi-root CMake layout**: one top-level
aggregator plus one self-contained `CMakeLists.txt` per demo pod (`fps/`, `plane/`, `tetris/`,
`snake/`). Shared dependencies are defined once in the parent and exposed to every subdir; each
demo owns its own internal include paths.

### Responsibilities split (do NOT duplicate)

- **Parent aggregator** (`hello-3d-demos/CMakeLists.txt`) — defines shared targets/aliases and adds
  only the **pod roots** (`fps`, `plane`, `tetris`, `snake`) to the global include path, plus the
  shared renderer library + fetched deps (stb/xsimd/Jolt) + vcpkg-provided packages. It must NOT list
  any demo's internal subdirs — those belong in each demo's own file.
- **Per-demo** (`hello-3d-demos/<demo>/CMakeLists.txt`) — defines its executable, compile options,
  link libraries, platform flags, and the **internal include dirs** for its own pod (e.g. `domains/`,
  `edges/`, `config/`). It must NOT redefine shared targets (`find_package` + aliases) or re-add the
  parent's demo roots — that causes "target already exists" / duplicate-include errors.

### How internal includes resolve

Each source header uses **bare** cross-subdir includes (e.g. `"snake.contract.hpp"`,
`"../config/levels/snake_level_01.hpp"`, `"shs_renderer.hpp"`). The demo's own `CMakeLists.txt` adds
the specific subdirs that are actually referenced, using `${CMAKE_CURRENT_LIST_DIR}` so the paths stay
correct regardless of where CMake is invoked from:

```cmake
include_directories(${CMAKE_CURRENT_LIST_DIR}/config)
include_directories(${CMAKE_CURRENT_LIST_DIR}/config/levels)
include_directories(${CMAKE_CURRENT_LIST_DIR}/domains/matrix)
```

- Only list dirs that a source header actually `#include`s. Verify with:
  ```bash
  wsl -e bash -lc "cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/hello-3d-demos && grep -rn '#include' --include='*.hpp' --include='*.cpp' . | grep '<demo-name>'"
  ```
- Subdirs referenced by **no** source header (e.g. `snake/domains/environment`, `snake/domains/progression`
  as an include dir, `snake/edges/rasterizer`) are intentionally omitted — dead dirs on the include path
  only mask future mistakes. (`progression` headers are reached by relative path from `main`, not bare includes.)
- The shared renderer (`shs_renderer.hpp`) is inherited from the parent's global include path; do not
  re-add it in the demo file (it would duplicate the include).

### Reconfiguring after a CMake change

If you edited any `CMakeLists.txt` (parent or per-demo), re-run step 1 (`cmake ..`) before building,
or the new targets/include paths will not appear. This is incremental and safe to repeat.

## The `build_vcpkg` build dir (validated, preferred for demo work)

`cpp-folders/build_vcpkg/` is an **already-configured** vcpkg build tree (Makefile generator,
toolchain wired, `_deps/` fetched). Prefer it over creating a fresh `build/` — configure is slow
(~15 s) and unnecessary when the cache is valid:

```bash
# Reconfigure after any CMakeLists.txt edit (run INSIDE the build dir — note `cmake .`, not `cmake ..`):
wsl.exe -d Ubuntu-24.04 --cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_vcpkg -- bash -c "cmake . > /tmp/cfg.log 2>&1; echo EXIT=$?; tail -3 /tmp/cfg.log"

# Build ONE target (much faster than `make -j20` for everything):
wsl.exe -d Ubuntu-24.04 --cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_vcpkg -- bash -c "cmake --build . --target Hello3DSnake -j$(nproc) > /tmp/b.log 2>&1; echo EXIT=$?; grep -cE 'error:' /tmp/b.log; tail -2 /tmp/b.log"
```

- `cmake .` inside an existing configured dir re-runs configure with cached values — no toolchain flag needed.
- `--target <Name>` builds just that target and its deps. Demo target names: `HelloPixel`, `Hello3DSnake`, etc.
- Log to `/tmp/*.log`, then `grep -E 'error:'` + `tail` it. Full g++ output is huge and gets truncated through the Windows bridge.
- Binary output path: `build_vcpkg/src/hello-3d-demos/<demo>/<TargetName>` (e.g. `.../snake/Hello3DSnake`).

## Shell-quoting traps through the cmd→wsl bridge (validated)

The agent's command passes through **two** shells (`cmd.exe` → `bash -c "..."`). Nested quoting breaks silently:

- **Never nest double quotes** inside the outer `bash -c "..."`. Use single quotes inside, or none:
  ```bash
  # BAD  (inner "..." terminates the outer string → mangled command):
  wsl.exe ... -- bash -c "echo "exit: $?""
  # GOOD:
  wsl.exe ... -- bash -c "echo EXIT_CODE=$?"
  ```
- **Never embed literal newlines** in the inline command (e.g. inside `tr ' ' '
'`) — cmd eats them and bash reports `unexpected EOF while looking for matching quote`.
- **Avoid `$()`/parens-heavy one-liners** (`python3 -c "..."`) inline — they trip cmd's parser. For anything multi-step or quote-heavy, **write a small script file with `write_to_file` and run it**: `wsl.exe -d Ubuntu-24.04 --cd <repo> -- bash script.sh`. Delete it afterwards.
- `$(nproc)` and `$?` survive fine because cmd does not expand `$`.
- Redirect verbose output to a file and inspect with `grep`/`tail` instead of streaming it.

## C++ pitfalls in this repo (validated by real compile failures)

These cost real debugging time — check them before building:

1. **`std::array<glm::ivec2, N>` needs DOUBLE braces.** Single-brace init (`= { {1,2}, {3,4} }`)
   fails on GCC with `too many initializers` (brace elision breaks against glm::vec's ctor set).
   Always write `= {{ {1,2}, {3,4} }}`. Verified minimal repro in `/tmp` before fixing.
2. **Same-name headers shadow via same-dir resolution.** A bare `#include "snake.contract.hpp"` from
   `domains/spatial_fx/` resolves to a file of that name **in the same directory first**, regardless of
   `-I` order. snake had two `snake.contract.hpp` (matrix + spatial_fx); the spatial_fx plan silently
   got the wrong one. Fix: give contracts unique names per pod (tetris convention: root-level shared
   vocab + `spatial_fx.contract.hpp`), then rely on `-I` dirs for cross-pod bare includes.
3. **Sibling namespaces are NOT searched by unqualified lookup.** Code in `namespace snake::spatial_fx`
   cannot see `snake::matrix::SnakeSnapshot` as plain `SnakeSnapshot` even if the header is included.
   Add explicit `using snake::matrix::SnakeSnapshot;` declarations (or fully qualify).
4. **Strict warning flags are on** (`-Wall -Wextra -Wconversion -Wsign-conversion -Wshadow …`).
   Recurring fixes: loop indices over `.size()` must be `size_t`; `static_cast<uint8_t>` for
   `shs::Color` channel expressions; `static_cast<float>` on SDL `Uint32` math;
   `begin() + static_cast<std::ptrdiff_t>(i)` for iterator arithmetic; `(void)param` for unused params.
5. **SDL API names**: there is no `SDL_QuitAudio()` — use `SDL_QuitSubSystem(SDL_INIT_AUDIO)`.
6. **Pre-existing warnings in `shs_renderer.hpp`** (-Wshadow in ZBuffer/Viewer ctors, some
   -Wsign-conversion) come from the SHARED renderer and appear in every demo. Don't chase them from a
   demo task; they need a dedicated renderer cleanup. Filter them out when counting your own warnings:
   `grep 'warning:' build.log | grep -v shs_renderer.hpp | wc -l`
7. **CMake ALIAS namespace is shared across subsequently-processed sibling directories** (validated
   2026-08-22 with an isolated two-subdir probe). A non-global alias created in subdir A IS visible to
   sibling B added later — so a fallback/no-op alias can POISON the name for later siblings.
   Symptom seen: full `make -j20` failed linking HelloShadowMapping / HelloWater /
   HelloShadowMappingSoft / HelloIblSkybox / HelloIblSkyboxOpt with undefined `IMG_Init/IMG_Load/
   IMG_Quit`, while configure printed "SDL2_image not found" even though vcpkg HAD sdl2-image installed.
   Root cause chain: vcpkg's static x64-linux SDL2_image defines ONLY the qualified target
   `SDL2_image::SDL2_image-static` (verify with a probe project: `if(TARGET ...) message(...)` for each
   candidate name). The hello-3d-demos aggregator's checks missed that name → fell into its no-op
   INTERFACE fallback → hello-render-target (processed later) saw `SDL2_image::Main` already defined,
   skipped creating its real alias → linked without libSDL2_image.a. Fix: check all real shapes
   (`SDL2_image::SDL2_image`, `SDL2_image::SDL2_image-static`, legacy unqualified `SDL2_image-static`)
   before any no-op fallback. Diagnosis pattern that cracked it: read the generated
   `build/<dir>/CMakeFiles/<Target>.dir/link.txt` to see EXACTLY which libs CMake put on the link line.
8. **SDL pixel-format byte order (for any screenshot/pixel analysis)**: `SDL_PIXELFORMAT_RGBA32`
   resolves to ARGB8888 packing on little-endian → memory bytes are **B,G,R,A**, not R,G,B,A. A pixel
   reader that assumes RGBA silently swaps R and B (symptom: color classifiers match nothing / wrong
   things). Read pixels as `(byte[2], byte[1], byte[0])`. Validated while building the snake screenshot
   analyzer (`cpp-folders/_diag_snake/analyze_frame.py`).
9. **Coordinate-system vs camera mismatch (Constitution I)**: if a 3D scene renders edge-on or
   mirrored, check the geometry plane against `docs/spec/conventions.md` (+Y up, +Z forward) BEFORE
   touching the camera. Board/floor geometry belongs in the XZ plane; map grid (x,y) → world
   `(x, 0, -y)` (proper rotation, det=+1). NEVER `(x, 0, y)` — that reflection flips triangle winding
   (backface culling + mirrored lighting). Also check projection aspect = canvas_w/canvas_h (a
   hardcoded 1.0 squeezes the image; see tetris.plan.hpp for the convention).

## Agent workflow (step by step)

1. **Read before you run.** For any non-trivial change, first `read_file`/`search_files` the target and its neighbors. Do not guess a shell command to "discover" state — it burns tokens and returns wrong results here.
2. **Prefer file tools for all file access** (`read_file`, `write_to_file`, `search_files`) with workspace-relative paths (e.g., `cpp-folders/src/shs-renderer-lib/CMakeLists.txt`). No shell needed, no wasted tokens.
3. **Reserve `execute_command` for non-filesystem tasks** — pure diagnostics that don't touch the project path (`echo`, `sleep`, arithmetic).
4. **When you must run a filesystem command**, wrap it in `wsl -e bash -lc "cd <WSL path> && ..."` so bash resolves WSL paths natively (see above). Do **not** use bare relative commands expecting the project cwd — they silently hit Windows root.
5. **One explicit path per command.** Never chain multiple filesystem operations expecting shared cwd; each needs its own explicit WSL path (or use file tools).
6. **Small, verifiable steps.** Prefer a command whose success you can confirm from its own output (e.g., `... && pwd` first) before chaining expensive work onto an unverified cwd.
7. **Prefer idempotent commands** (`mkdir -p`, incremental `make`). Re-running a build is safe; re-running a destructive command is not — check state first.
8. **When output looks wrong, suspect the cwd before the tool.** In this environment a "wrong" result almost always means Windows-root cwd, not a broken tool.

## Corner cases (edge behaviors that are easy to get wrong)

- **`ls`, `cat`, `git status` with no explicit path** do NOT error — they silently list/operate on Windows root (`C:\Windows`). The absence of an error is the trap; there is no failure signal.
- **A bare `wsl -e bash -lc "pwd"`** (no explicit WSL path) lands in `/mnt/c/Windows`, not the project. It succeeds but points at the wrong tree. Always pass an explicit `/home/...` path.
- **`cd /home/... && pwd` run directly** (not wrapped in `wsl`) fails with "The system cannot find the path specified." — cmd.exe cannot resolve WSL paths; only a bash wrapper can.
- **Explicit UNC args DO work** (`dir \\wsl$\Ubuntu-24.04\...`, `git -C "\\wsl$\..." status`). The launcher's broken cwd does not prevent an explicit UNC argument from resolving — the two are independent.
- **`make`/`cmake`/`ctest` need a real Linux build dir.** They cannot run against Windows root; they must be wrapped in `wsl -e bash -lc "cd <WSL build dir> && ..."`. A bare `make` from cmd.exe will not find the Makefile and is useless.
- **The banner + a successful exit code can coexist.** Exit code 0 with the UNC banner printed does NOT mean the command ran in the project — it may have run against Windows root. Verify output content, not just exit status.
- **`cmake ..` from `build/`** reconfigures; if you changed a top-level or per-target `CMakeLists.txt`, reconfigure before building or the new targets won't appear.
- **vcpkg toolchain is required at configure time.** Forgetting `-DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake` makes CMake fail to find VulkanMemoryAllocator / Lua 5.5 and the fetched deps.

## Why this happens (root cause, for context)

The `execute_command` tool launches Windows `cmd.exe` and forces it to start with the WSL project directory as its current working directory, expressed as a UNC path:

```
CMD.EXE was started with the above path as the current directory.
UNC paths are not supported. Defaulting to Windows directory.
The system cannot find the path specified.
```

That banner appears on **every** command and is harmless — it only describes the launcher's cwd, not your arguments. The real trap is what happens next: bare filesystem commands silently hit Windows root (no error), explicit WSL paths fail ("cannot find the path"), but explicit UNC args DO resolve. Wrapping in `wsl -e bash` with an explicit `/home/...` path is the only reliable way to run Linux tools from the Windows shell.

## Related docs

- Agent environment rules & command templates: `docs/dev/agent_environment.md` (read FIRST)
- Project constitutions (architecture): `docs/spec/conventions.md`, `docs/spec/value_oriented_programming.md`, `docs/spec/dod_ecs_architecture.md`
- VOP roadmap: `docs/roadmap/value_oriented_programming_first_class_roadmap.md`
- Per-demo canonical state: `cpp-folders/src/hello-3d-demos/<demo>/docs/STATUS.md` where present (e.g. `snake/docs/STATUS.md`)
- README top-level build instructions (incl. vcpkg package list) for Ubuntu / macOS / Windows.
