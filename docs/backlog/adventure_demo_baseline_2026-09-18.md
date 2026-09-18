# AD0 — Fresh reproducible baseline (closed 2026-09-18)

Owner: `adventure_demo_conformance_backlog.md` AD0. Evidence recorded per its
acceptance rule ("reproducible commands and fresh results attached to
close-out notes; library test counts are not substituted for demo coverage").

## Environment (recorded 2026-09-18)

- Repo: `/home/sharavsambuu/src/dev/leisure-software-renderer`, commit `03467a3`
- Build dir: `cpp-folders/build`, `CMAKE_BUILD_TYPE=Debug`
- Compiler: GCC 13.3.0 (`/usr/bin/c++`, Ubuntu 13.3.0-6ubuntu2~24.04.1), C++23
- Toolchain: vcpkg `/opt/vcpkg/scripts/buildsystems/vcpkg.cmake`,
  prefix `/home/sharavsambuu/vcpkg/installed/x64-linux`; SDL2+SDL3+Assimp ON
- Vulkan: API 1.4.318, device `llvmpipe (LLVM 20.1.2, 256 bits)`, driver
  `llvmpipe` (Mesa 25.2.8 user-space) — headless lavapipe-class device
- Slang: `slangc` from Vulkan SDK 1.4.341.1
  (`/home/sharavsambuu/vulkan/1.4.341.1/x86_64/bin/slangc`)

## Reproduce

```sh
cd cpp-folders/build
cmake --build . --target t0_tri_barycentric_sw t0_projection_sw \
  t0_depth_blend_sw t0_texture_sampling_sw t0_stencil_sw \
  t1_normal_mapping_sw t0_tri_barycentric_vk t0_projection_vk \
  t0_depth_blend_vk t0_texture_sampling_vk t0_stencil_vk t1_normal_mapping_vk
python3 cpp-folders/src/exps-rendering-adventures/tier0-rasterization-foundations/tools/t0_parity_suite.py
```

## Fresh six-pair parity (all binaries rebuilt, suite regenerates PNGs)

| Pair | differ | % | within-1 | max | Envelope | Result |
|---|---|---|---|---|---|---|
| 01 barycentric | 4 | 0.00% | 4 | 1 | near_exact (≤8, 1 LSB) | PASS |
| 02 projection | 37884 | 12.33% | 28419 | 217 | tol13 (≥70% within 1) | PASS |
| 03 depth_blend | 42193 | 13.73% | 42193 | 1 | tol1 | PASS |
| 04 texture_sampling | 6751 | 2.20% | 6751 | 1 | tol1 | PASS |
| 05 stencil | 0 | 0.00% | 0 | 0 | exact | PASS |
| 08 normal_mapping | 80744 | 26.28% | 79400 | 2 | tol27 | PASS |

All six Vulkan twins were **available and exercised** on this device — no
backend recorded as unavailable; none substituted by library test counts.

## Envelope re-pins (investigated, not blind — S7 discipline)

The suite breached on first fresh run: 01 (4 px @ 1 LSB vs EXACT) and
02 (12.33% vs the 12.0% cap). Both re-pins are backed by a decisive
cross-check: a `git worktree` at **`983925c`** (pre-R1, before the
rasterizer hot-path series *and* this session's K-G1/G2/G3 work) with the
identical toolchain rebuilds both pairs and reproduces the **exact same
numbers** (01: `differ=4 within1=4 max=1`; 02: `differ=37884 (12.33%)
within1=28419 max=217`). Conclusion: **pre-existing device drift** from the
Mesa 25.2.8 / LLVM 20.1.2 llvmpipe set vs the 2026-09-16 pin — not a defect
in the demos and not a regression from R1–R3 or the G-track slices (whose
own golden/parity gates stayed byte-identical). Re-pins:

- 01 `exact` → `near_exact` (differ ≤ 8, all within 1 LSB, max ≤ 1)
- 02 `tol12` → `tol13` (pct cap 12.0 → 13.0; within-1 ratio and max drift
  caps unchanged)

Both re-pins carry their dated rationale in `t0_parity_suite.py`. Shrink-only
rule applies from here: a future device set that beats these numbers may
tighten them; nothing may loosen them without a fresh investigation note.

## Defect-vs-regression separation (AD0 bullet 3)

- Existing defects found: the two device-drift items above (now pinned).
- Refactor regressions found: **none** (pre-R1 cross-check, above).
- CMake/CI wiring inspected before claiming anything is "manual": demos
  were **not** CTest-registered at baseline (no `add_test` in either tier's
  CMakeLists); the parity suite existed only as a hand-run script with
  hard-coded `/home/...` build paths. Registration + de-machining is AD1's
  scope and follows in the same run.
