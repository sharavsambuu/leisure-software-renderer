# Agent Environment Notes (WSL Ubuntu 24.04 + Windows 11)

This document is written for **AI coding agents** working in this repo, not humans. It exists to prevent wasted tool calls and tokens from environment quirks that are easy to miss on first contact.

## TL;DR — the one rule

> **`execute_command` launches Windows `cmd.exe`, whose cwd is a broken UNC path.**
> Every command prints a harmless "UNC paths are not supported" banner, but more importantly: **commands with no explicit filesystem path silently run against Windows root (`C:\Windows`) and return wrong results — they do NOT error out.** To touch the project you must pass an **explicit path** as an argument. For reading/writing/searching files, prefer the dedicated file tools (`read_file`, `write_to_file`, `search_files`) — no shell needed.

## Environment layout

- OS: WSL Ubuntu 24.04 (default shell is `bash`).
- Host: Windows 11. The project lives on the WSL filesystem under `/home/sharavsambuu/src/dev/leisure-software-renderer`.
- From Windows, that maps to `\\wsl$\Ubuntu-24.04\home\sharavsambuu\src\dev\leisure-software-renderer` (UNC path).

## The execute_command launcher (root cause)

The `execute_command` tool launches **Windows `cmd.exe`** and forces it to start with the WSL project directory as its current working directory, expressed as a UNC path. Windows `cmd.exe` does not understand that cwd:

```
CMD.EXE was started with the above path as the current directory.
UNC paths are not supported. Defaulting to Windows directory.
The system cannot find the path specified.
```

That banner appears on **every** command and is harmless — it only describes the launcher's cwd, not your arguments. The real trap is what happens next, which depends on whether you pass an explicit path:

| What you run | Actual result (validated) |
| :--- | :--- |
| Pure commands (`echo`, `sleep`, arithmetic, env inspection that ignores cwd) | ✅ Works — output is correct |
| Filesystem command with **no** explicit path (`ls`, `cat`, `git status`, `cmake ..`) | ⚠️ **Succeeds but silently runs against Windows root `C:\Windows`** and returns wrong results. It does NOT error out. This is the dangerous case for small agents: no failure signal, just a bogus listing of System32/Temp/Microsoft.NET. |
| Explicit WSL path (`cd /home/sharavsambuu/... && pwd`) | ❌ Fails with "The system cannot find the path specified." — `cmd.exe` cannot resolve `/home/...`. |
| Explicit UNC path as an argument (`dir \\wsl$\Ubuntu-24.04\...`, `git -C "\\wsl$\..." status`) | ✅ **Works** — explicit UNC paths DO resolve when passed as arguments, even though the launcher's cwd is broken. |

This is an environment limitation, not project corruption. The project files are intact and readable via the file tools.

## What works vs what doesn't (via execute_command)

| Operation | Status |
| :--- | :--- |
| Read/write/search files with workspace-relative paths (`@workspace:docs/spec/conventions.md`) | ✅ Works — use these for all file access |
| `echo`, `sleep`, pure arithmetic, env inspection that ignores cwd | ✅ Works |
| Filesystem command **without** an explicit path (`ls`, `cat`, `git status`, `cmake ..`) | ⚠️ Silently runs against Windows root — wrong results, no error. Do not rely on it. |
| Explicit WSL `/home/...` paths in a command | ❌ Fails ("cannot find the path") |
| Explicit UNC `\\wsl$\Ubuntu-24.04\...` paths as arguments | ✅ Works — this is the reliable way to touch the project from cmd.exe |

## The WSL intermediary (validated, preferred for build/test)

When a command genuinely needs the Linux filesystem (cmake, make, ctest, git), wrap it in `wsl` so bash resolves WSL paths natively. This is **the** reliable way to run Linux tools from the Windows shell:

```
wsl -e bash -lc "cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders && <linux command>"
```

Validated behavior (all confirmed via `execute_command`):

| Command | Result |
| :--- | :--- |
| `wsl -e bash -lc "echo from-wsl && pwd"` | ✅ Works — but note the bare cwd lands in `/mnt/c/Windows` (the broken UNC path translated to a Windows drive). Always pass an explicit WSL path. |
| `wsl -e bash -lc "cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders && pwd && ls"` | ✅ Works — resolves the WSL project natively and lands in the project (lists `CMakeLists.txt`, `build`, `src`). |

**Rule:** never rely on a bare `wsl` command inheriting the project cwd. Always pass an explicit `/home/...` path inside the quoted bash string so it resolves to the WSL filesystem, not Windows root.

## Recommended agent workflow

1. **Prefer file tools for everything about files.** Use `read_file`/`write_to_file`/`search_files` with workspace-relative paths (e.g., `docs/spec/conventions.md`). No shell needed, no wasted tokens.
2. **Reserve `execute_command` for non-filesystem tasks** — pure diagnostics that don't touch the project path (`echo`, `sleep`, arithmetic).
3. **If you must run a filesystem command**, wrap it in `wsl -e bash -lc "cd <WSL path> && ..."` so bash resolves WSL paths natively (see above). Do **not** use bare relative commands expecting the project cwd — they silently hit Windows root.
4. **Never retry a failing filesystem command with a different WSL path.** The failure is the launcher's cwd, not your argument. Switch strategy (use file tools) instead of rephrasing.

## General tips (token efficiency & reliability)

- **Read before you run.** For any non-trivial change, first `read_file`/`search_files` the target and its neighbors. Guessing at a shell command to "discover" state burns tokens and produces wrong results in this environment.
- **One explicit path per command.** Never chain multiple filesystem operations expecting shared cwd; each needs its own explicit WSL path (or use file tools).
- **Treat the UNC banner as noise.** It appears on every `execute_command`. Do not spend tokens re-reading or reacting to it — only act on the actual output.
- **Small, verifiable steps.** Prefer a command whose success you can confirm from its own output (e.g., `... && pwd` first) before chaining expensive work onto an unverified cwd.
- **Prefer idempotent commands** (`mkdir -p`, `cmake --fresh` only when needed). Re-running a build is safe; re-running a destructive command is not — check state first.
- **When output looks wrong, suspect the cwd before the tool.** In this environment a "wrong" result almost always means Windows-root cwd, not a broken tool.

## Corner cases (edge behaviors that are easy to get wrong)

- **`ls`, `cat`, `git status` with no explicit path** do NOT error — they silently list/operate on Windows root (`C:\Windows`). The absence of an error is the trap; there is no failure signal.
- **A bare `wsl -e bash -lc "pwd"`** (no explicit WSL path) lands in `/mnt/c/Windows`, not the project. It succeeds but points at the wrong tree. Always pass an explicit `/home/...` path.
- **`cd /home/... && pwd` run directly** (not wrapped in `wsl`) fails with "The system cannot find the path specified." — cmd.exe cannot resolve WSL paths; only a bash wrapper can.
- **Explicit UNC args DO work** (`dir \\wsl$\Ubuntu-24.04\...`, `git -C "\\wsl$\..." status`). The launcher's broken cwd does not prevent an explicit UNC argument from resolving — the two are independent.
- **`make`/`cmake`/`ctest` need a real Linux build dir.** They cannot run against Windows root; they must be wrapped in `wsl -e bash -lc "cd <WSL build dir> && ..."`. A bare `make` from cmd.exe will not find the Makefile and is useless.
- **The banner + a successful exit code can coexist.** Exit code 0 with the UNC banner printed does NOT mean the command ran in the project — it may have run against Windows root. Verify output content, not just exit status.

## Build / test commands (for humans and agents that can run bash)

Run these from a real `bash` terminal (VS Code integrated terminal or Git Bash), **not** via `execute_command`:

```bash
cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders && mkdir -p build && cd build
export VCPKG_ROOT="/opt/vcpkg"
cmake .. -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
make -j20

# Run the VOP boundary + deterministic-core tests:
ctest -R "shs_renderer_vop_(boundary_check|tests)" --output-on-failure
```

## Related docs

- Project constitutions (architecture): `docs/spec/conventions.md`, `docs/spec/value_oriented_programming.md`, `docs/spec/dod_ecs_architecture.md`
- VOP roadmap: `docs/roadmap/value_oriented_programming_first_class_roadmap.md`
- C++ compilation & agent workflow for this project: `docs/dev/cpp_compilation_workflow.md`
- README top-level build instructions for Ubuntu / macOS / Windows.
