# Agent Environment Notes (WSL Ubuntu 24.04 + Windows 11)

Written for **AI coding agents**, not humans. Goal: complete the task with the fewest tool calls and tokens. Read this first — it is shorter than the constitutions and overrides them on process questions only.

## START HERE — follow these exactly, do not guess
- **[HARD RULE] Never use `execute_command` or any shell to read/list/search files.** Use the file tools instead: `read_file`, `write_to_file`, `search_files` with workspace-relative paths (e.g., `docs/spec/conventions.md`). This single rule removes most wasteful CLI text here.
- **READ-ONLY TASKS → file tools only, no shell.** If the task is "recall X", "read Y", or "survey Z", do it purely with `read_file`/`search_files`. Do NOT run `ls`, `find`, `dir`, PowerShell, or any listing to discover files. The file tools already return full contents; a shell listing is pure waste and can be wrong here (see Anti-recurrence notes).
- **For anything that must run on Linux** (`cmake`, `make`, `ctest`, `git`), wrap it in:

  ```bash
  wsl -e bash -lc "cd /home/sharavsambuu/src/dev/leisure-software-renderer && <linux command>"
  ```

  Always pass an explicit WSL path inside the string. Never rely on inherited cwd (a bare `wsl` lands in `/mnt/c/Windows`).
- **Treat the "UNC paths are not supported" banner as noise.** It prints on every `execute_command`. Ignore it; act only on real output.

## Hard rules (do not violate)
1. **File access → file tools only.** Never use `ls`, `cat`, `find`, or `git status` to discover files, list dirs, or read content. The file tools already return full file contents — a shell listing is pure waste and can be wrong here.
2. **`execute_command` ONLY for pure non-filesystem ops** that ignore the project path: `echo`, `sleep`, arithmetic, env inspection. If it touches files/paths, use file tools or an explicit WSL path instead.
3. **Never retry a failing filesystem command with a different WSL path.** The failure is the launcher's broken cwd, not your argument. Switch strategy (use file tools) rather than rephrasing — each failed shell call burns tokens for nothing.

## Anti-recurrence notes (from a real slip-up)
- **A "recall / read-only" task must never spawn CLI listing commands.** The instinct to `ls`/`find` the tree before reading is exactly what wastes tokens here. Prefer `search_files` over the whole workspace when you want breadth without opening every file. If you already opened a doc, do not re-list or re-read it — you have its contents in context.
- **Failure signatures that mean "wrong cwd", NOT "your command was wrong":**
  - `"UNC paths are not supported"` + `CMD.EXE was started with the above path as the current directory.` → launcher banner, ignore.
  - `"The system cannot find the file specified."` / `"No such file or directory"` for a **relative** path like `docs/...` or `cpp-folders/...` → ran against Windows root (`C:\Windows`). Do NOT retry with another relative path; switch to file tools immediately.
  - A plausible-but-wrong listing (e.g., empty output, or content from the wrong tree) → it silently executed in `C:\Windows`. Suspect cwd, not the tool. Fall back to file tools instead of rephrasing.
- **The `@workspace:` prefix is NOT a filesystem path for cmd.exe.** Prefixing an argument with `@workspace:` (e.g., `ls @workspace:docs`) makes Windows CMD try to resolve it as a literal filename and fail — this tool's workspace alias only works through the file tools, not inside a shell command. To touch files from a shell, use plain relative paths wrapped in `wsl -e bash -lc "cd /home/... && ..."`, or explicit UNC args (`\\wsl$\Ubuntu-24.04\...\path`).
- **Do not launch broad recursive scans (PowerShell `Get-ChildItem -Recurse`, `find .`) from the Windows shell.** They wander into unrelated trees, hit permission-denied noise, and can hang or spawn runaway processes. Scope any filesystem op to a specific doc via file tools instead. If a scan is genuinely needed, keep it shallow and scoped, and never let it run unattended in the background.
- **STOP early on read-only tasks.** You are done when success criteria are met by real output from the file tools or a wrapped Linux command. Do not add extra verification passes (re-listing trees, re-reading already-read files, redundant checks).

## The one environment fact to internalize
`execute_command` launches Windows `cmd.exe` whose cwd is a broken UNC path. A bare filesystem command **silently runs against `C:\Windows` and returns wrong results without erroring.** So: if a shell command gives a plausible-but-wrong listing, it ran in the wrong place — do not retry with another path; switch to file tools.

## Decision table (validated behavior)

| Operation | Result | What to do |
| :--- | :--- | :--- |
| `read_file` / `write_to_file` / `search_files` with a workspace-relative path (`@workspace:...`, `docs/...`) | ✅ Works — returns full file content | **Always prefer this for any file access.** No shell needed. |
| Pure non-filesystem op via `execute_command` (`echo`, `sleep`, arithmetic, env inspection that ignores cwd) | ✅ Works | Allowed only when no file tool fits. Otherwise use file tools. |
| Filesystem command with **no** explicit path (`ls`, `cat`, `find`, `git status`, `cmake ..`) | ⚠️ Silently runs against `C:\Windows` — wrong results, no error | Never rely on it. Use file tools or an explicit WSL path. |
| Explicit WSL `/home/...` path run **directly** (not wrapped in `wsl`) | ❌ "The system cannot find the path specified." | Don't do this; wrap in `wsl -e bash -lc "cd /home/... && ..."` instead. |
| Explicit UNC `\\wsl$\Ubuntu-24.04\...` as an **argument** (`dir`, `git -C "\\wsl$..." status`) | ✅ Works — reliable way to touch the project from cmd.exe | Valid, but file tools are simpler for reads/writes/searches. |
| Linux tool wrapped: `wsl -e bash -lc "cd /home/... && <cmd>"` with explicit path | ✅ Works (`cmake`, `make`, `ctest`, `git`) | The only correct way to run Linux build/test from the Windows shell. |
| Bare `wsl -e bash -lc "pwd"` (no explicit WSL path) | ⚠️ Lands in `/mnt/c/Windows` — succeeds but wrong tree | Always pass an explicit `/home/...` path inside the string. |

## Current state anchor (do not re-probe this)
- The build dir `cpp-folders/build` already exists and is configured; VOP tests are registered. **Do not re-run cmake** unless you changed a top-level or per-target `CMakeLists.txt`.
- If a command's output looks wrong, suspect the cwd (Windows root), not a broken tool — then fall back to file tools instead of retrying with another path.

## STOP conditions
You are done when the task is complete and its success criteria are met by real output from the file tools or a wrapped Linux command. Do **not** add extra verification passes beyond what the task requires (e.g., don't re-list the tree, re-read files you already read, or run redundant checks).

## When you must run Linux — the only legitimate shell use
Wrap every filesystem-touching command in `wsl` with an explicit path so bash resolves WSL paths natively:

```bash
wsl -e bash -lc "cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders && <linux command>"
```

- **One explicit path per command.** Never chain multiple filesystem operations expecting a shared cwd; each needs its own explicit WSL path (or use file tools).
- **Prefer idempotent commands** (`mkdir -p`, incremental `make`). Re-running a build is safe; re-running a destructive command is not — check state first.

## Build / test commands (run from bash, NOT via execute_command)
```bash
wsl -e bash -lc "cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders && mkdir -p build && cd build"
export VCPKG_ROOT="/opt/vcpkg"
cmake .. -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
make -j20
ctest -R "shs_renderer_vop_(boundary_check|tests)" --output-on-failure   # run only when the task needs it
```

## Related docs
- Project constitutions (architecture): `docs/spec/conventions.md`, `docs/spec/value_oriented_programming.md`, `docs/spec/dod_ecs_architecture.md`
- VOP roadmap: `docs/roadmap/value_oriented_programming_first_class_roadmap.md`
- C++ compilation & agent workflow for this project: `docs/dev/cpp_compilation_workflow.md`
- README top-level build instructions for Ubuntu / macOS / Windows.
