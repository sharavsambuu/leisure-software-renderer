# Agent Environment Notes (WSL Ubuntu 24.04 + Windows 11)

Written for **AI coding agents**, not humans. Goal: complete the task with the fewest tool calls and tokens. Read this first — it is shorter than the constitutions and overrides them on process questions only.

## START HERE — follow these exactly, do not guess
- **[HARD RULE] Never use `execute_command` or any shell to read/list/search files.** Use the file tools instead: `read_file`, `write_to_file`, `search_files` with workspace-relative paths (e.g., `docs/spec/conventions.md`). This single rule removes most wasteful CLI text here.
- **READ-ONLY TASKS → file tools only, no shell.** If the task is "recall X", "read Y", or "survey Z", do it purely with `read_file`/`search_files`. Do NOT run `ls`, `find`, `dir`, PowerShell, or any listing to discover files. The file tools already return full contents; a shell listing is pure waste and can be wrong here (see Anti-recurrence notes).
- **For anything that must run on Linux** (`cmake`, `make`, `ctest`, `git`), wrap it in ONE of these two
  validated forms:

  ```bash
  # PREFERRED (--cd sets the Linux cwd natively; no cd-chaining needed):
  wsl.exe -d Ubuntu-24.04 --cd /home/sharavsambuu/src/dev/leisure-software-renderer -- bash -c "<linux command>"

  # Equivalent classic form:
  wsl -e bash -lc "cd /home/sharavsambuu/src/dev/leisure-software-renderer && <linux command>"
  ```

  Always pass an explicit WSL path. Never rely on inherited cwd (a bare `wsl` lands in `/mnt/c/Windows`).
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
- TWO configured build dirs exist: `cpp-folders/build` (classic) and `cpp-folders/build_vcpkg`
  (preferred for demo work). Both are configured with the vcpkg toolchain; VOP tests are registered.
  **Do not re-run cmake** unless you changed any `CMakeLists.txt` (then `cmake .` inside the build dir).
- If a command's output looks wrong, suspect the cwd (Windows root), not a broken tool — then fall back to file tools instead of retrying with another path.

## Command templates (copy-paste safe, all validated)

Use these EXACT shapes. Do not improvise variants — nested quotes, embedded newlines, and `$()`
substitution break silently through the cmd→wsl bridge (details in
`docs/dev/cpp_compilation_workflow.md` § Shell-quoting traps).

```bash
# Configure/reconfigure after editing ANY CMakeLists.txt (run INSIDE the build dir):
wsl.exe -d Ubuntu-24.04 --cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_vcpkg -- bash -c "cmake . > /tmp/cfg.log 2>&1; echo EXIT=$?; tail -3 /tmp/cfg.log"

# Build ONE target (fast feedback loop while iterating on code):
wsl.exe -d Ubuntu-24.04 --cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_vcpkg -- bash -c "cmake --build . --target Hello3DSnake -j$(nproc) > /tmp/b.log 2>&1; echo EXIT=$?; grep -cE 'error:' /tmp/b.log; tail -2 /tmp/b.log"

# Full build (all targets; use when CMake structure changed or before declaring done):
wsl.exe -d Ubuntu-24.04 --cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_vcpkg -- bash -c "cmake --build . -j$(nproc) > /tmp/full.log 2>&1; echo EXIT=$?; grep -cE 'error:' /tmp/full.log; tail -3 /tmp/full.log"

# VOP validation tests:
wsl.exe -d Ubuntu-24.04 --cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_vcpkg -- bash -c "ctest -R 'shs_renderer_vop_(boundary_check|tests)' --output-on-failure"

# git state (read-only inspection is fine via shell when wrapped):
wsl.exe -d Ubuntu-24.04 --cd /home/sharavsambuu/src/dev/leisure-software-renderer -- bash -c "git status --short | head -20"
```

Rules encoded above:
- **Always redirect verbose output to `/tmp/*.log`, then `grep`/`tail` it.** Full compiler output streamed
  through the bridge gets truncated; the log+grep pattern never loses data.
- **Count errors with `grep -cE 'error:'`** and check `EXIT=$?` — but remember exit 0 + plausible-looking
  output can still mean wrong-cwd execution; verify content mentions real project paths.
- **Multi-step or quote-heavy diagnostics → write a script file** (`write_to_file`), run it
  (`-- ... -- bash path/to/script.sh`), delete it afterwards. Never inline them.

## STOP conditions
You are done when the task is complete and its success criteria are met by real output from the file tools or a wrapped Linux command. Do **not** add extra verification passes beyond what the task requires (e.g., don't re-list the tree, re-read files you already read, or run redundant checks).

## When you must run Linux — the only legitimate shell use
Wrap every filesystem-touching command in `wsl` with an explicit path so bash resolves WSL paths natively:

```bash
wsl -e bash -lc "cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders && <linux command>"
```

- **One explicit path per command.** Never chain multiple filesystem operations expecting a shared cwd; each needs its own explicit WSL path (or use file tools).
- **Prefer idempotent commands** (`mkdir -p`, incremental `make`). Re-running a build is safe; re-running a destructive command is not — check state first.

## Related docs
- Project constitutions (architecture): `docs/spec/conventions.md`, `docs/spec/value_oriented_programming.md`, `docs/spec/dod_ecs_architecture.md`
- VOP roadmap: `docs/roadmap/value_oriented_programming_first_class_roadmap.md`
- C++ compilation & agent workflow for this project: `docs/dev/cpp_compilation_workflow.md`
- README top-level build instructions for Ubuntu / macOS / Windows.
