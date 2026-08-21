# C++ Compilation & Agent Workflow (leisure-software-renderer)

This document is written for **AI coding agents** working in this repo. It defines the exact, validated build/test workflow and the corner cases that are easy to get wrong when an agent drives compilation from a Windows shell into WSL Ubuntu 24.04. Read `docs/dev/agent_environment.md` first — it explains why commands behave the way they do here.

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

- Project constitutions (architecture): `docs/spec/conventions.md`, `docs/spec/value_oriented_programming.md`, `docs/spec/dod_ecs_architecture.md`
- VOP roadmap: `docs/roadmap/value_oriented_programming_first_class_roadmap.md`
- Agent environment notes: `docs/dev/agent_environment.md`
- README top-level build instructions for Ubuntu / macOS / Windows.
