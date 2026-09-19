# Open shader-ID registry — evidence (2026-09-18)

> Status: **CLOSED 2026-09-18.** Owner: Constitution I §7 *No User Lock-In
> (Pluggability)* (`docs/spec/conventions.md` §7.1) via the formal contract
> [`docs/arch/render_path_architecture.md`](../arch/render_path_architecture.md)
> §4 **graduation requirement 6** ("Open shader-identity registry"). Companion
> artifacts: [`engine_header_inventory.json`](engine_header_inventory.json)
> (regenerated, Rule 15), [`remaining_todos_2026-09-18.md`](remaining_todos_2026-09-18.md)
> §A0. Provenance: this closes the residual P1.5 stated in its own words —
> "no consumer/open shader ids yet" — and is the third instance of the
> `PassIdRegistry` shape, taken immediately after
> [`open_pass_id_registry_evidence_2026-09-18.md`](open_pass_id_registry_evidence_2026-09-18.md)
> and [`shader_identity_manifest_evidence_2026-09-18.md`](shader_identity_manifest_evidence_2026-09-18.md).

## 1. What problem this closes

P1.5 made shader identity *data*: one `ShaderId` names one authored shader, each
backend resolves its own realization, and a backend with no realization refuses
loudly. But `ShaderId` was still a **closed enum with a `Count` sentinel** — the
exact shape `PassId` shed on 2026-09-18. Consequences:

- A consumer/demo could not name its own shader. The only route was editing the
  core enum (`ShaderId::MyThing = 9, Count = 10`) and the core census
  (`builtin_shader_manifest.hpp`) — a core edit for a consumer-owned identity.
- `ShaderManifest::register_shader` was bounded by `slots_[id]` over a fixed
  `std::array<Slot, Count>`, so "registered" could only ever mean "one of the
  eight builtin slots".
- The law was therefore asymmetric: passes were first-class for consumers,
  shaders were not, and §7.1 rule 1 names the identity vocabulary as additive.

## 2. What landed

| Artifact | Lines | Role |
| :--- | ---: | :--- |
| `include/shs/core/open_id_hash.hpp` | **58 (new)** | The shared content-addressing law — `core::open_id_offset(name, range_capacity)`. Hoisted here so the pass and shader registries are one mechanism, not two copies |
| `include/shs/render/shader/shader_id.hpp` | **170 (new)** | `ShaderId` + range law (`kShaderIdBuiltinMin/Max`, `kShaderIdOpenBase/Max`, `kShaderIdReserved`, `kShaderIdUnknown`), vocabulary pins (`static_assert`), `shader_id_is_builtin/_is_open/_in_valid_range`, `shader_id_name_or_null`, `parse_shader_id`, `kShaderIdOpenSpelling`. Mirrors `renderpath/planning/pass_id.hpp` |
| `include/shs/render/shader/shader_id_registry.hpp` | **164 (new)** | `ShaderIdRegistry` — `intern` / `try_name` / `contains` / `is_open` / `registered` / `open_count` / `clear` / `==`, content-addressed via the shared law. Mirrors `renderpath/execution/pass_id_registry.hpp` |
| `include/shs/render/shader/shader_identity.hpp` | 357 → **489** | Umbrella header: re-exports the two new ones, so **no consumer include changed**. `ShaderManifest` now OWNS a `ShaderIdRegistry` and gains open-slot storage, `intern_shader`, `register_named_shader`, `shader_name`, `open_count`, `builtin_capacity`; `has`/`get`/`resolve`/`size`/`clear`/`==` cover both halves. Two new named refusals |
| `include/shs/renderpath/execution/pass_id_registry.hpp` | +1 include, −12 | `open_offset` now **delegates** to the shared law (behaviour-identical: the same FNV-1a + mix + fold over the same capacity) |
| `tests/shader_id_open_registry_tests.cpp` | **638 (new)** | Gate `shs_renderer_shader_id_open_tests` — 14 named cases, GPU-free, value-tier only |
| `tests/shader_identity_tests.cpp` | 278 → **312** | P1.5 parity guards: the closed-vocabulary reading is unchanged for every value it could hold; builtin manifests mint nothing |

The manifest is still **caller-owned, copyable and comparable** — deliberately
*not* a global. No ambient registry decides shader resolution.

## 3. The shape, as data

The third instance is the *same* shape, not a lookalike:

| Concern | `PassId` | `ShaderId` |
| :--- | :--- | :--- |
| Law header | `renderpath/planning/pass_id.hpp` | `render/shader/shader_id.hpp` |
| Registry header | `renderpath/execution/pass_id_registry.hpp` | `render/shader/shader_id_registry.hpp` |
| Null / builtin / open / reserved | `0` / `1…1023` / `1024…65534` / `65535` | identical |
| Offset law | `PassIdRegistry::open_offset` | `ShaderIdRegistry::open_offset` — **both call `core::open_id_offset`** |
| Builtin names never mint | `intern("taa")` → `PassId::TAA` | `intern("blinn_phong")` → `ShaderId::BlinnPhong` |
| Open id derived from the name | content-addressed | content-addressed |
| Foreign open id | hard miss via `try_name` | hard miss via `try_name` |
| Owner of the registry | `PassFactoryRegistry` owns a `PassIdRegistry` | `ShaderManifest` owns a `ShaderIdRegistry` |

**The rule of two was discharged by hoisting, not by copying.** Rather than
writing the FNV-1a/mix/fold arithmetic a second time, the mechanism moved to
`shs/core/open_id_hash.hpp` and `PassIdRegistry::open_offset` became a thin
forwarder — so "the same name yields the same id in every process" is now one
law with two users, and a divergence is impossible rather than merely unlikely.
The gate asserts it directly (`shared_offset_law`:
`ShaderIdRegistry::open_offset(n) == PassIdRegistry::open_offset(n)` for real
consumer names; both capacities equal), and probe 1 below proves that breaking
the one law fails **both** namespaces' gates.

## 4. The law, as data — what an open identity can and cannot do

| Situation | Result |
| :--- | :--- |
| Consumer mints a name and registers a software descriptor | `ShaderId` in the open range; resolves to `program` + `has_program` |
| ... and asks for Vulkan / OpenGL | `BackendNotRealized` — **never** a fallback to the software program |
| ... with a Vulkan realization (module + entries) | resolves to `module` + `entries`, `has_program == false` |
| Declared entry points differ from the registered ones | `EntryPointMismatch` |
| An open id minted by a **different** manifest is registered here | `UnregisteredOpenId` — a foreign id can never fill a slot |
| Two distinct names hashing to one slot | second `intern` → `nullopt`; `register_named_shader` → `NameCollision` (nothing overwritten) |
| A builtin name is interned | resolves to the builtin id — a consumer can never shadow a core shader |
| Empty name, or the reserved `"open_shader"` spelling | `nullopt` — never a registration key |
| `Unknown`, `Count`, the reserved gap, the reserved top slot | `UnknownShader` (unchanged from P1.5) |

Two refusals are **new** and named, because opening the id space created two
genuinely new failure modes: `UnregisteredOpenId` (an open id this manifest
never minted) and `NameCollision` (the one real failure mode of content
addressing). Neither is conflated with an existing error.

## 5. Verification

| Check | Result |
| :--- | :--- |
| New gate `shs_renderer_shader_id_open_tests` (14 named cases) | **14/14 pass** — range law + pins, builtin names never mint, content-addressed ids + order independence + copy/value equality, determinism across instances, **shared offset law vs `PassIdRegistry`**, null/reserved rejection, capacity arithmetic, loud collision (brute-forced real collision), foreign open id hard miss, consumer identity with zero core edits, the realization law on open ids, descriptor invariants on open ids, manifest equality/copy/clear, builtin paths unchanged |
| P1.5 gate `shs_renderer_shader_identity_tests` (parity guards added) | pass — the closed-vocabulary reading is unchanged for every value it could hold (`shader_id_is_registerable`, `shader_id_is_builtin`), every builtin manifest reports `open_count() == 0` and mints nothing, `builtin_capacity()` is unchanged |
| Pass gate `shs_renderer_pass_id_open_tests` | pass — proves the `open_offset` delegation is behaviour-identical |
| Full CTest | **74/74** (was 73); **74/74** again with `SHS_CONTRACTS_ENFORCED=1` |
| Build | 0 errors, 0 warnings (`-Wall -Wextra -Wpedantic`) |
| `check_kdba_boundaries.sh` (incl. gate 8 contract placement) | all checks passed, **232** headers scanned (was 230) |
| `check_include_graph.py` | acyclic; SDK placement + value-tier purity hold (the new `core` → `render`/`renderpath` edges are all value-tier) |
| `check_pure_value_libraries.sh` | pass (the new header sits in `core`, a classified pure library, and adds no stateful machinery) |
| `inventory_headers.py --write` | regenerated in this slice (Rule 15), `header_count` 230 → **233**; regeneration verified **idempotent** (identical md5) |

### Mutation probes (each must fail the gate, then be reverted byte-identically)

| Probe | Result |
| :--- | :--- |
| Collapse the **shared** law (`open_id_offset` → constant 0) | **fails both** `shs_renderer_shader_id_open_tests` **and** `shs_renderer_pass_id_open_tests` (and the P1.5 gate) — the one law genuinely binds two namespaces |
| Silent aliasing in `desc_of` (an unresolvable open id falls back to the first registered slot) | gate **exit 1** on `foreign_open_id_is_a_hard_miss`, `manifest_equality_and_copy`, `builtin_paths_unchanged` |
| Equality forgets the minted names (`ids_ != other.ids_` removed) | gate **exit 1** on `manifest_equality_and_copy` |
| An absent realization bit falls back instead of refusing | gate **exit 1** on `open_id_realization_law` and on the P1.5 gate |

After each probe the source was restored from a byte-identical backup (`cmp`
verified) and the gates returned to exit 0. No `PROVE-FAIL` residue remains.

**Harness honesty note:** the first probe run reported "restored: gates STILL
FAIL" because the harness took its backup *after* mutating, so `cmp` agreed on a
mutated pair and left the mutation in the tree; a second run then hit a
make-timestamp race (restore landing in the same second as the object build, so
a stale mutated binary was re-run). Both faults were in the **harness**, not the
gates — the four results above are from the corrected harness (backup before
mutation; forced distinct mtime on restore), and the tree was verified clean
(`0` `PROVE-FAIL` hits, all three gates green) afterwards.

## 6. DoD (met)

1. A consumer/demo-owned shader identity is minted, registered, resolved and
   named through a typed id **with no core edit** — no `ShaderId` enum change,
   no `builtin_shader_manifest.hpp` change (`consumer_shader_no_core_edit`).
2. Every builtin path keeps its exact previous behavior:
   `shader_id_is_registerable` and `shader_id_is_builtin` are unchanged for
   every value the old closed vocabulary could hold; the builtin manifests mint
   nothing and report `open_count() == 0`; the P1.5 realization/refusal law is
   intact (`builtin_paths_unchanged`, plus the P1.5 gate's own parity guards).
3. Failure modes are loud, named and total: collision refused with nothing
   overwritten, a foreign id a hard miss, verified `(id, name)` pairing, null /
   `Count` / gap / reserved ids never resolvable, and the descriptor invariants
   enforced by one shared validation path for both halves.
4. The rule of two is discharged by **one** mechanism: the offset law lives in
   `shs/core/open_id_hash.hpp` and both registries call it, asserted by
   `shared_offset_law` and prove-failed across both namespaces by probe 1.
5. Reproducibility: ids are order-independent and process-stable, so anything
   that names a consumer shader stays valid.

## 7. Not claimed / residual (honest limits)

1. **16-bit open range.** `ShaderId` stays `uint16_t`, so there are 64,511 open
   slots, one shader per name-hash slot. Widening the id type is a separate,
   gated decision; nothing here changes `ShaderId`'s type or size.
2. **Collision residual, precisely.** Two *distinct* names hashing to one slot
   cannot both be registered in one manifest (the second `intern` returns
   `nullopt`, or `register_named_shader` reports `NameCollision`). If such a
   pair exists and a *saved* identity built against one is replayed against a
   manifest holding the other, the shared id resolves to the registered name —
   which is why the verified `(id, name)` pairing exists and why resolution keys
   on the name. A real colliding pair is brute-forced in the gate, so this path
   is tested, not theoretical.
3. **A refused registration leaves its name minted.** `register_named_shader`
   mints first and can then refuse the descriptor (`NoRealization`,
   `MissingCppImpl`, ...), leaving the name in the registry because
   `ShaderIdRegistry` offers no `remove` — the same asymmetry `PassIdRegistry`
   has, and `intern_pass_id` + a failed `register_factory` behaves identically.
   `size()` / `open_count()` count **registered** identities, so the observable
   surface is honest; the gate pins this exact behavior rather than leaving it
   implicit.
4. **`ShaderManifest::operator==` is no longer `constexpr`.** The open half
   lives in `std::deque` / `std::string`, so no invocation could be
   constant-evaluated; claiming `constexpr` would have been ill-formed
   no-diagnostic-required. Equality now also compares the minted names (a
   minted-but-unregistered identity is observable through `shader_name`).
5. **P1.5's remaining residuals are untouched by this slice**, and this slice
   does not pretend otherwise: the Vulkan binding is still descriptive truth
   rather than loader input (P2); the value builtins still have no GPU modules
   (the dual-realization census is unchanged); the authored-source check still
   verifies names, not layouts (P2 reflection); OpenGL is still a stub seat.
   Opening the id space does not advance any of those.
6. **No consumer actually uses an open shader id yet.** The capability is proven
   by the gate, not by a shipping consumer — the demo/migration work that would
   exercise it is P2.5/P3 (pass modules leaving GLSL), which is exactly where the
   census is meant to shrink.

## 8. Follow-ups

1. **P2.5/P3** — register migrated pass modules as they leave GLSL, flipping
   software-only identities into dual-realization ones; the open range is now
   available for the consumer-owned identities those migrations will need.
2. **Grad req 2 (technique/light preset registries)** — the remaining closed
   enums in §4. They should reuse the same shape; the shared offset law is now in
   place for the third and fourth instances to call rather than copy.
3. **P2 reflection** — validate registered entry points and struct layouts
   against `slangc` reflection output, replacing the text check with a layout
   check.
4. **OpenGL seat** — realize it or stop offering it as selectable (an owner
   ruling, unchanged by this slice).
