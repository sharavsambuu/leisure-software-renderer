# Open pass-semantic registry — evidence (2026-09-18)

**Row:** RP-4 — open `PassSemantic` (backlog tracker
`docs/backlog/remaining_todos_2026-09-18.md`).
**Graduation requirement:** arch §4 req 7
(`docs/arch/render_path_architecture.md`) — added 2026-09-18 by the audit that
found the requirement list had never scheduled semantic or technique vocabulary,
and shipped the same day.
**Status:** ✅ done. Full suite **75/75** (was 74/74).

## 1. What problem this closes

`PassSemantic` (`planning/pass_contract.hpp`) was a **closed 16-value enum**
(`Unknown = 0` … `HistoryMotion = 15`) with no `Custom` member and no open
registered range. A consumer could therefore **not name a G-buffer channel the
builtins did not anticipate**. That is precisely what the *"arbitrary G-buffer
layouts"* goal needs, and it was a Lock-In residual in the literal sense of
Constitution I §7 — the arrangement machinery around the vocabulary
(`RenderPathCompiler`, resource planning, barrier planning) was already built
and dynamic, while the vocabulary itself stayed closed, so a custom layout could
be orchestrated but not *named*.

Same shape as reqs 1 and 6, so this is the third application of a mechanism
that already exists rather than a new mechanism:

| Namespace | Builtin | Open range | Registry | Law |
| :--- | :--- | :--- | :--- | :--- |
| Pass ids (req 1) | `PassId` (16) | 1024…65534 | `PassIdRegistry` | `core::open_id_offset` |
| Shader ids (req 6) | `ShaderId` | open | `ShaderIdRegistry` | `core::open_id_offset` |
| **Semantics (req 7)** | **`PassSemantic` (15 + `Unknown`)** | **1024…65534** | **`PassSemanticRegistry`** | **`core::open_id_offset`** |

No fourth copy of the hash law was added; `PassSemanticRegistry::open_offset`
delegates to the shared header exactly as the other two do.

## 2. What landed

- **Range law + vocabulary pins** in `pass_contract.hpp`, immediately after the
  enum (`kPassSemanticUnknown/BuiltinMin/BuiltinMax/OpenBase/OpenMax/Reserved`,
  `pass_semantic_is_builtin/_is_open/_in_valid_range`,
  `kPassSemanticOpenSpelling`), including two `static_assert` pins: `Unknown`
  stays the null semantic, and the builtin enum can never silently grow into the
  open range. (A silent overflow here would be load-bearing, because a resource
  spec keys off the semantic.)
- **`PassSemanticRegistry`** — new header
  `include/shs/renderpath/planning/semantic_registry.hpp` (179 lines), same
  `intern` / `try_name` / `contains` / `is_open` / `registered` / `open_count` /
  `clear` / order-independent `operator==` surface as `PassIdRegistry`.
- **All four switch surfaces answered** (the full extent of the change surface,
  verified by grepping every `case PassSemantic::` in `include/`, `src/` and
  `tests/` — exactly two files, four switches):
  - `pass_semantic_name` — open ids return the open spelling via an explicit
    `default:`.
  - `default_pass_semantic_descriptor` — open ids take the neutral defaults
    (Screen / Linear / Transient / sampled / not storage) by an **explicit early
    return**.
  - `render_path_resource_id_for_semantic` — open ids return the open spelling,
    plus a new **registry-aware overload** that resolves the registered name.
  - `make_default_resource_spec_for_semantic` — needed **no change**: its
    existing `default: break` already gave open semantics the neutral
    Full/Texture2D/transient default rather than either builtin carve-out
    (Absolute-2048 shadow map, Tile storage-buffer light culling). A
    registry-aware overload was added beside it.
- **`pass_semantic_name_or_null`** and **`parse_pass_semantic`** as the
  builtin-only counterparts of `pass_id_name_or_null` / `parse_pass_id`.
  `parse_pass_semantic` deliberately does **not** parse open names: their ids are
  minted from the name by the registry, and a second parse-back mapping would be
  a drifting duplicate of it.
- **Gate:** `tests/semantic_id_open_registry_tests.cpp` (455 lines), 12
  sub-checks, registered as `shs_renderer_semantic_id_open_tests` (test #17),
  value-tier and GPU-free (`shs::renderer-values` only).
- **Header inventory regenerated** (`inventory_headers.py --write`): 233 → 234
  headers, including the new dependency edges from
  `render_path_resource_plan.hpp`. Re-run verified **idempotent** (md5
  `9caf6e7a…` stable across two consecutive runs).

  **Gotcha, recorded because it cost a cycle:** the inventory maps both each
  header's dependencies *and* its **consumers**, and it enumerates consumers
  from **git-tracked** files. A first regeneration taken while the new gate test
  was still untracked therefore omitted its consumer edges — and the check
  *passed* in that state, i.e. it was green for the wrong reason. Once the test
  was staged the check went stale again. **Regenerate the inventory *after*
  staging any new source or header file**, not before.

## 3. The one non-obvious design decision

Semantics open *cleanly* — `TechniquePassContract::semantics` is a
`std::vector<PassSemanticRef>`, so unlike the technique-mode bitmask there is no
mask-representation question and no 32-value ceiling. That is why RP-4 was the
row to run first. But one real hazard showed up, and it is worse here than in
the pass-id namespace:

> A resource spec's `id` is **derived from the semantic**. Without a registry,
> two *distinct* open semantics would both fall back to
> `kPassSemanticOpenSpelling` and **silently merge into one planned resource**.

In the pass-id namespace the analogous fallback is only a display spelling; here
it can collapse two channels into one allocation. So the bare
`render_path_resource_id_for_semantic(PassSemantic)` is documented as
*non-distinguishing*, and a registry-aware overload
`render_path_resource_id_for_semantic(PassSemantic, const PassSemanticRegistry&)`
is provided as the recommended path (plus the spec-builder equivalent). The gate
asserts **both halves**: distinct names yield distinct ids through the registry,
and the bare spelling's aliasing behaviour is asserted rather than hidden, so
nobody can "fix" it to something silent later.

Second decision: the descriptor's early return is guarded on
`pass_semantic_is_open(semantic)` and deliberately **not** on
`!pass_semantic_is_builtin(semantic)`. The wider guard would have been a
one-token difference and would have silently changed `Unknown`'s descriptor
(`None` / `Unknown` / unsampled → Screen / Linear / sampled), breaking AD0
parity for the null semantic. `test_builtin_paths_unchanged` pins this
explicitly.

## 4. Verification

| Check | Result |
| :--- | :--- |
| `shs_renderer_semantic_id_open_tests` (new, GPU-free, value-tier) | **12/12 pass** — range law + vocabulary pins, builtin names never mint, content-addressed + idempotent ids, order-independence + value equality across instances, null/reserved rejection, capacity arithmetic, loud collision on a brute-forced *real* collision, foreign-id hard miss, neutral descriptor for open semantics, contract acceptance with and without overrides, registry-aware resource-id distinguishing vs the bare-spelling alias, and AD0 parity for all 15 builtins incl. `Unknown` and both resource carve-outs |
| Full CTest (`cpp-folders/build`) | **75/75** (was 74/74) |
| Build | 0 errors, 0 warnings (`-Wall -Wextra -Wpedantic`) |
| `check_kdba_boundaries.sh` (incl. gate 8 contract placement) | all checks passed, **233** headers scanned |
| `shs_renderer_include_graph_tests` + `_gate` | pass (acyclic; no cycle introduced by the new header) |
| `shs_renderer_header_self_containment_test` | pass (scans the new header) |
| `shs_renderer_header_inventory_check` | pass after `inventory_headers.py --write` (233 → 234 headers) |

### Mutation probes (each must fail the gate, then be reverted byte-identically)

| Probe | Result |
| :--- | :--- |
| Widen `pass_semantic_is_builtin` into the open range (`kPassSemanticBuiltinMax` → `kPassSemanticOpenMax`) | gate **exit 1**, 8 sub-checks failed (`range_law`, `content_addressed_ids`, `determinism_across_instances`, `null_and_reserved_rejected`, `collision_is_loud`, `foreign_open_id_is_a_hard_miss`, `resource_id_is_distinguishing`) |
| Collapse content addressing (`open_offset` → `core::open_id_offset(name, 1u)`, i.e. constant offset 0) | gate **exit 1** on 5 sub-checks (`content_addressed_ids`, `determinism_across_instances`, `foreign_open_id_is_a_hard_miss`, `resource_id_is_distinguishing`) |
| Reintroduce silent aliasing in `try_name` (unresolved id falls back to the first registered name) | gate **exit 1** on `foreign_open_id_is_a_hard_miss` |

After each probe both mutated sources were restored from byte-identical backups
(`diff -q` verified) and the gate returned to exit 0.

## 5. DoD (met)

1. A technique/consumer-owned channel is **named, registered, declared in a
   `TechniquePassContract`, and planned into a resource with a distinguishing
   id** — with no core edit of any kind
   (`contract_accepts_open_semantic`, `resource_id_is_distinguishing`).
2. Every builtin path keeps its exact previous behaviour, including `Unknown`,
   which the open-range guard must not catch, and both builtin resource
   carve-outs (`builtin_paths_unchanged`).
3. Failure modes are loud and total: collision refused with nothing overwritten,
   foreign id a hard miss, null/reserved ids never resolvable, and the
   alias-prone bare spelling asserted rather than left silent.
4. Reproducibility: ids are order-independent and process-stable, so a saved
   recipe, replay log or barrier table naming a consumer channel stays valid.

## 6. Not claimed / residual (honest limits)

- **`PassSemanticEncoding` was in the row's title and was NOT opened.** It is
  the *attachment-packing* axis (which channels occupy which target
  format/channel) — a **new abstraction**, not a closed enum to extend, since
  nothing today states that mapping. Opening both at once would have conflated
  "name a channel" with "decide its physical format". It remains deliberately
  unscheduled; see arch §4's "Blind spot" note. **This is the real ceiling on
  the G-buffer goal** — arbitrary *layouts* are still not expressible, only
  arbitrary channel *names*.
- **No out-of-library consumer.** The gate exercises the consumer *role* in
  library tests, exactly where reqs 1 and 6 did. There is no demo-level proof,
  because the render-path library currently has **no consumer outside its own
  tests**: the active `exps-rendering-adventures` tree links `shs::renderer` only
  for a platform seam and includes no `shs/renderpath` header, and the three
  `exps` trees that might have contained one are parked behind a whole-tree
  alias/include-path migration (`cpp-folders/CMakeLists.txt:103-107`, `644be48`).
  RP-0 was timeboxed to un-park one and closed as *not started*; see its row.
- **16-bit open range.** `PassSemantic` is `uint16_t` (64,511 consumer slots, one
  semantic per name-hash slot). Widening the type is a separate, gated decision;
  nothing here changes the enum's type or size.
- **Collision residual, precisely.** Two distinct names hashing to the same slot
  cannot both register in one registry (the second `intern` returns `nullopt`).
  If a plan built against one is run against a registry holding the other, the
  shared id resolves to *that* registry's name — which is exactly why the
  **registry-aware** resource id exists and is the recommended path. A real
  colliding pair is constructed in the gate so this path is tested, not
  theoretical.
- **The two debug switches in the parked exps tree were not touched.**
  `exp-plumbing/hello_rendering_paths.cpp` and
  `demo_forward_classic_renderpath.cpp` switch over `PassSemantic` but are **not
  built** (commented out at `cpp-folders/CMakeLists.txt:120`). They only print
  names, and `pass_semantic_name` is total for open ids, so they stay
  correct-or-honest on re-enable. *Untouched* is the honest word: they were not
  verified.
- **Not an ABI/API break**, but the header inventory is regenerated and the new
  header is covered by the self-containment and include-graph gates.

## 7. Follow-ups

- **RP-5** (technique vocabulary) still carries the series' one genuine pending
  decision — the `uint32_t` `TechniqueMode` bitmask representation — and stays
  blocked until that ruling lands.
- **RP-8** (composition-tier substrate policy) is independent of RP-4…RP-7 and
  can be pulled forward.
- The **attachment-packing schema** is the piece that would let *arbitrary
  G-buffer layouts* actually be authored; it needs its own design, not a row.
