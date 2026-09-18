# Open pass-ID registry — evidence (2026-09-18)

> Status: **CLOSED 2026-09-18.** Owner: Constitution I §7 *No User Lock-In
> (Pluggability)* (`docs/spec/conventions.md` §7.1) via the formal contract
> [`docs/arch/render_path_architecture.md`](../arch/render_path_architecture.md)
> §4 **graduation requirement 1** ("Open pass IDs"). Roadmap: P3 box 6, first
> half (`domain_pod_engine_rollout_roadmap.md`), consolidated tracker A0.
> Independently flagged by the 2026-09-18 Antigravity library review
> (`docs/review/2026-09-18_antigravity_shs_renderer_lib_review.md` §3.1,
> "Closed `PassId` Enumeration").

## What was required

§4 req 1, verbatim intent: `PassId` is a closed 16-value enum, so
consumer/demo-owned passes could not be first-class — the demo-authoring story
(§3) was blocked. Required: a builtin range + an open registered range (or
stable-string-hash contract keys) so a consumer pass needs **no core edit**.

## What shipped

**Range law** — `shs/renderpath/planning/pass_id.hpp`:

- `kPassIdBuiltinMin/Max` = 1…1023, `kPassIdOpenBase/Max` = 1024…65534,
  `kPassIdReserved` = 65535, `kPassIdUnknown` = 0.
- `pass_id_is_builtin` / `pass_id_is_open` / `pass_id_in_valid_range`.
- Vocabulary pins (`static_assert`): the builtin enum can never grow into the
  open range, and `PassId::Unknown` stays the null id.
- `pass_id_name_or_null`, and an honest `kPassIdOpenSpelling` for an open id
  whose name lives in a registry (never used as a registration key).
- **`pass_id_is_standard` is now behavior-identical to
  `pass_id_is_builtin`** for every value the enum can hold, so all pre-existing
  call sites keep their exact semantics (parity guard in the gate).

**Registry** — new `shs/renderpath/execution/pass_id_registry.hpp`:

- `PassIdRegistry` — explicit, caller-owned, copyable, snapshot-comparable. No
  ambient global, so a plan and its replay stay deterministic.
- `intern(name)` is **total**: builtin name → builtin id (a consumer can never
  shadow a core pass); repeat → same id (idempotent); new → registered;
  empty/`"unknown"`/collision → `nullopt`. No throw, no partial state.
- **Ids are content-addressed** (`open_offset` = mixed FNV-1a, folded into the
  range): an id is a pure function of the *name*. Consequences: stable across
  processes/translation units, order-independent (registration order can never
  leak into a plan), and `try_name(foreign_id)` can only return the name that id
  was derived from or `nullopt` — never a different pass.
- `try_name` / `contains` / `is_open` / `registered` / `open_count` / `clear`.

**Registration + planning integration**:

- `PassFactoryRegistry` owns a `PassIdRegistry`: `intern_pass_id`,
  `pass_id_registered_name`, `pass_ids()`; every typed overload
  (`register_factory`/`has`/`create`/`register_descriptor`/
  `try_get_descriptor`/`try_get_contract_hint`/`supports_backend_hint`/
  `supports_technique_mode_hint`) resolves its key through the registry, so an
  open id works everywhere a builtin id does, and an unresolvable id is a hard
  miss — the key is never guessed.
- **Verified `(id, name)` pair**: `register_factory(id, expected_name, factory)`
  and `has(id, expected_name)` refuse a mismatched pairing. This is what makes a
  foreign or colliding id safe.
- `make_render_path_pass_entry(name, pass_id, required)` — the consumer entry
  overload (the `PassId`-only overload can only spell builtin names).
- `render_path_plan_has_pass(plan, PassId)` now answers for open ids, and a new
  registry overload resolves a string-keyed plan entry by name.
- `RenderPathCompiler` resolves, canonicalises and validates entries with open
  ids (`canonical_id` = builtin spelling | registry name | entry text), warns on
  an exact-name mismatch for open ids, and keeps every builtin path identical.

## Why not mint-order ids (rejected alternative)

The first cut minted `PassId = base + registration_index`. The gate caught the
hole immediately: id `base+0` means "whatever was registered first *here*", so a
plan or registry other than the minting one silently resolves that id to the
**wrong pass**. Content addressing makes cross-registry confusion degrade to a
miss instead of a wrong execution. Recorded here because the rejected design
looks simpler and will be proposed again.

## Verification

| Check | Result |
| :--- | :--- |
| `shs_renderer_pass_id_open_tests` (new, GPU-free, header-only) | **12/12 pass** — range law + pins, builtin names never register, content-addressed ids, order independence + value equality, null/reserved rejection, capacity arithmetic, loud collision (brute-forced real collision), foreign id hard miss, typed registry on open ids, consumer pass plans + queries with zero core edits, string-keyed plan by name, builtin paths unchanged (incl. rejection paths) |
| Full CTest (`cpp-folders/build`) | **72/72** (was 71/71) |
| Build | 0 errors, 0 warnings (`-Wall -Wextra -Wpedantic`) |
| `check_kdba_boundaries.sh` (incl. gate 8 contract placement) | all checks passed, **226** headers scanned |
| `check_include_graph.py` | acyclic; SDK placement + value-tier purity hold |
| `header_self_containment_test` | pass (scans the new header) |
| `inventory_headers.py --write` | regenerated in this slice (Rule 15), 45 insertions |

### Mutation probes (each must fail the gate, then be reverted)

| Probe | Result |
| :--- | :--- |
| Widen `pass_id_is_builtin` into the open range | gate **exit 1**, 9 assertions failed (`range_law`, classification, registry typed paths) |
| Collapse content addressing (`open_offset` → constant) | gate **exit 1** on `content_addressed_ids`, `determinism_across_instances`, `foreign_open_id_is_hard_miss` |
| Reintroduce silent aliasing in `try_name` (fall back to the first registered name) | gate **exit 1** on `content_addressed_ids`, `foreign_open_id_is_hard_miss` |

After each probe the source was restored from a byte-identical backup and the
gate returned to exit 0 (`diff -q` verified both mutated files).

## DoD (met)

1. A consumer/demo-owned pass is registered, planned, queried and created
   through a typed id **with no core edit** — no `PassId` enum change, no planner
   fork (`consumer_pass_plans_no_core_edit`).
2. Every builtin path keeps its exact previous behavior; `pass_id_is_standard`
   is behavior-identical for every value the enum can hold, and the compiler's
   rejection paths (unregistered required pass, keyless required entry, text/typed
   mismatch warning) are unchanged (`builtin_paths_unchanged`).
3. Failure modes are loud and total: collision refused, foreign id a hard miss,
   verified `(id, name)` pairing, null/reserved ids never resolvable.
4. Reproducibility: ids are order-independent and process-stable, so plans,
   replay logs and barrier tables stay valid.

## Not claimed / residual (honest limits)

- **16-bit open range.** `PassId` is `uint16_t`, so there are 64,511 consumer
  slots, one pass per name-hash slot. Widening the id type is a separate, gated
  decision; nothing in this slice changes `PassId`'s type or size.
- **Collision residual, precisely.** Two *distinct* consumer names that hash to
  the same slot cannot both be registered in one registry (the second `intern`
  returns `nullopt`). If such a pair exists and a plan built against one is run
  against a registry holding the other, the shared id resolves to the registered
  name — which is why the verified `(id, name)` registration/query pair exists
  and why execution keys on the name. A real colliding pair is constructed in
  the gate so this path is tested, not theoretical.
- **Grad req 2 (light/technique registries) is not in this slice.** The box was
  split; the second half is tracked as an open roadmap item and tracker A0 entry.
- **Not an ABI/API break**, but the header inventory and the two touched planning
  headers are regenerated/covered as required.
- The Antigravity review's suggestion of "open 64-bit string-hash IDs" is
  satisfied in *mechanism* (stable name-hash ids + a dynamic registry), not in
  width; see the 16-bit residual above.

