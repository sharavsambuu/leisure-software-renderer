# Shader Identity Manifest — Evidence (2026-09-18)

> Status: **landed, verified**. Owner: [`slang_utilization_plan.md`](../roadmap/slang_utilization_plan.md)
> §3 P1.5 ("shader manifest value desc — dual-realization design"). Companion
> artifacts: [`engine_header_inventory.json`](engine_header_inventory.json)
> (regenerated, Rule 15), [`remaining_todos_2026-09-18.md`](remaining_todos_2026-09-18.md)
> §A0. Provenance: this is the first slice aimed at the *shader* half of the
> backend-agnostic vision, taken immediately after
> [`open_pass_id_registry_evidence_2026-09-18.md`](open_pass_id_registry_evidence_2026-09-18.md).

## 1. What problem this closes

Before this slice, shader identity was **not data**. The same authored shader
existed as two hand-written artifacts joined only by convention and test
discipline:

- `tests/shaders/offscreen_pipeline.slang` — `vs_main`, `vs_uploaded`, `fs_main`.
- `include/shs/rhi/software/sw_offscreen.hpp` — a CPU program plus two string
  literals (`vertex_entry = "vs_main"`, `fragment_entry = "fs_main"`) compared by
  hand inside `prepare_offscreen()`.

Nothing *named* the shader. Nothing recorded that these two artifacts are the
same shader, that one more exists than the other (`vs_uploaded` has no CPU
counterpart), or that no OpenGL realization exists at all. Adding a second GPU
backend would have cost a second hand-written copy — the same class of lock-in
that the open `PassId` work removed for passes.

## 2. What landed

| Artifact | Lines | Role |
| :--- | ---: | :--- |
| `include/shs/render/shader/shader_identity.hpp` | 357 | `ShaderId`, canonical entry-name constants, `ShaderEntryPoints`, realization masks, `ShaderDesc`, `ShaderBinding`, closed `ShaderIdentityError`, `ShaderManifest` (register/has/get/resolve) |
| `include/shs/render/shader/builtin_shader_manifest.hpp` | 82 | Value-tier builtin census (6 identities), each honestly software-only |
| `tests/shader_identity_tests.cpp` | 278 | Gate `shs_renderer_shader_identity_tests` — 83 executed assertions, GPU-free |
| `include/shs/rhi/software/sw_offscreen.hpp` | +64/−11 | Real consumer: entry-name law now resolves through the manifest; entry constants single-sourced |
| `include/shs/app/backend/backend_factory.hpp` | 1 line | OpenGL selection note made truthful (see §6) |

The manifest is **caller-owned, copyable and comparable** — deliberately the
same shape as `PassIdRegistry`, and deliberately *not* a global: the determinism
gates forbid ambient registries, and a global manifest would decide shader
resolution invisibly.

## 3. The law, as data

One `ShaderId` names one authored shader. `resolve(id, backend, declared_entries)`
returns either a binding or a *named* refusal:

| Situation | Result |
| :--- | :--- |
| Registered, backend realized, entries agree | `ShaderBinding`: software → `program` + `has_program`; GPU → `module` + `entries`, **no** program |
| Backend bit absent (OpenGL today; Vulkan for the value builtins) | `BackendNotRealized` — never a fallback to another backend's realization |
| Declared entry points differ from the registered ones | `EntryPointMismatch` |
| Never registered / `Unknown` / `Count` sentinel | `UnknownShader` — a hard miss, never an alias |
| Registration whose name contradicts the id, or duplicates it | `NameMismatch` / `AlreadyRegistered` |
| Software realization declared without a factory | `MissingCppImpl` |
| GPU realization declared without a module or without entries | `MissingModule` / `MissingEntryPoints` |
| No realization declared at all | `NoRealization` |

Registration is **verified pairing**: the id and the name must agree with the
descriptor registered under them. The builtin table takes both names from
`shader_id_builtin_name(id)`, so a builtin's two sides cannot drift apart.

The offscreen identity is the one with two realizations today, and the manifest
records the asymmetry that previously lived only in prose: `vs_main`/`fs_main`
on both sides, `vs_uploaded` **GPU-only**.

## 4. Real consumer, not a synthetic exercise

`SoftwareOffscreenExecution::prepare_offscreen()` previously compared two string
literals. It now resolves the descriptor's declared entry points through the
caller-owned manifest and refuses on any non-resolution. Behaviour is preserved
(only the two registered entry names prepare; everything else returns 0), with
one deliberate hardening: a null `entry` pointer is now a *refusal* instead of
undefined behaviour.

The execution path deliberately still calls the concrete
`flat_triangle_program()`. Type erasure is confined to the identity seam so the
software rasterizer keeps inlining the per-pixel fragment call (R1,
renderer-lib review 2026-09-18). The erased form is built *from* the concrete
factory by `make_erased_program`, so the two cannot restate divergent shader
bodies — and the gate checks their outputs agree.

## 5. Verification

- Gate `shs_renderer_shader_identity_tests`: **83 assertions**, all pass,
  GPU-free (no device, no `Context`, no SPIR-V execution). It covers vocabulary
  totality, every refusal mode, caller-owned equality/copy semantics, known
  answers *through* the resolved program (the authored flat colour and the
  authored position pass-through), erased-vs-concrete agreement, and the real
  consumer's accept/refuse behaviour.
- The gate reads the authored `offscreen_pipeline.slang` and asserts the
  registered module stem names a real file containing the declared entry points
  and `vs_uploaded`. The Vulkan half of the identity is therefore checked
  against the source of truth that is actually compiled, not against a second
  literal — the cheap half of the plan's P2 reflection idea.
- Full CTest **73/73** (was 72); `check_kdba_boundaries.sh` all green, contract
  placement now spanning **228** headers (was 226); include graph acyclic;
  header self-containment green; `engine_header_inventory.json` regenerated
  (Rule 15). Zero compiler warnings.
- **Three mutation probes**, each failing the gate and each reverted
  byte-identically (`cmp` verified):
  1. `kShaderEntryFsMain` renamed → `the declared fragment entry is absent from
     the authored module` (the source-of-truth check has teeth).
  2. Offscreen identity greedily granted the OpenGL bit → `OpenGL resolved a
     realization it does not have`.
  3. Authored flat colour changed on the CPU side → `the resolved offscreen
     fragment is not the authored flat color`.

## 6. Deliberate non-claims (residuals)

1. **No consumer/open shader ids.** `ShaderId` is a closed builtin vocabulary
   plus a `Count` sentinel. Consumer-owned shader ids are the rule-of-two
   follow-up and must reuse the `PassIdRegistry` shape, not invent a second one.
2. **The Vulkan binding is descriptive truth, not yet loader input.** The
   manifest states which module and entries realize an identity; the Vulkan edge
   still loads `.spv` paths itself. Making the manifest the loader's input is P2
   work, not this slice.
3. **The value builtins have no GPU modules.** Blinn-Phong, PBR-MR, lit and the
   three debug views are registered software-only. That is not a shortcut: it
   makes the dual-realization gap a *census* instead of an invisible debt. Their
   GPU counterparts arrive with the P3 migration.
4. **The frozen GLSL-era modules are not registered** — registering them would
   claim realizations that are not on the build path.
5. **OpenGL is still a stub backend class.** This slice makes the refusal
   explicit and machine-checked in the identity layer, and the selection message
   truthful; it does not build the OpenGL seat.
6. **The authored-source check verifies names, not semantics or layouts.** It
   proves the registered entry points exist in the compiled source; it does not
   verify descriptor/push-constant layout agreement (P2).

## 7. Rejected alternative

Storing resolved `ShaderProgram` values, or a global manifest, was rejected:
erased programs in a shared table would move the hot path onto `std::function`
calls and contradict R1, and a global registry is ambient state the determinism
gates exist to prevent. Descriptors hold a *factory pointer* only; programs are
constructed per resolution.

## 8. Follow-ups

1. Consumer/open shader ids (rule-of-two over `PassIdRegistry`).
2. P2 reflection: validate registered entry points and struct layouts against
   `slangc` reflection output, replacing the text check with a layout check.
3. P2.5/P3: register migrated pass modules as they leave GLSL, flipping
   software-only identities into dual-realization ones so the census shrinks
   visibly.
4. OpenGL seat: realize it (Slang→GLSL is now the natural route) or stop
   offering it as selectable — an owner ruling.
