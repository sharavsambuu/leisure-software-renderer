# Domain Value Object Error Flow — shs-renderer-lib

> Status: living catalog (2026-09-16, KDBA hardening). The failure-rail mirror
> of `EVENT_FLOW.md`: every closed error enum riding an `std::expected<T, E>`
> channel or a rejection fact is listed here, per pod. **Drift law:**
> `check_kdba_boundaries.sh` FAILs if an error enum declared in
> `domains/*/*.event.hpp` or `domains/*/*.contract.hpp` is missing from this
> file. Rule 11 requires one error-enum family per bounded context; Rule 12
> requires the `.or_else()` compensator to consume the emitted rejection fact.

## How to read this

- **Channel** = where the error rides: the per-command `expected` error rail
  inside a gateway's Kleisli arrows (house shape per
  `docs/backlog/kdba_kleisli_migration_plan.md`: the batch rim is a `Step`
  summary — infallible wherever every real failure is already a materialized
  rejection fact), or a rejection *fact* (e.g. `PathSwapRejectedEvent`)
  materialized on the failure rail while persistent state stays pristine.
- **Closed vocabulary only**: error payloads are `enum class X : uint8_t` —
  no `std::string` diagnostics inside pod/intent vocabulary (§8). Error enums
  are named `*Error`, `*Rejection`, or `*Reason` so the drift gate can find
  them; new error vocabularies must follow this naming law.
- **Fact, not command** (Rule 8.1): a rejection reason describes what
  happened; it never carries downstream instructions.

## renderpath — `shs::renderpath::PathSwapRejectionReason`

Declared in `renderpath.event.hpp` beside the events it explains (the
compile/reject pair shares one vocabulary).

| Value | Meaning |
|---|---|
| `CompileInvalid` | The candidate recipe failed compilation (fallback reason). |
| `EmptyPassChain` | The pass chain was empty. |
| `BackendUnavailable` | No driver for the requested backend target. |
| `MissingRequiredPass` | A pass required by the technique was absent. |
| `DepthUnsupported` | Depth prepass unsupported by the capability set. |
| `OcclusionUnsupported` | Occlusion culling unsupported by the capability set. |

Rides: the `expected` error channel of the compile resolve chain
(`renderpath.gateway.hpp` `detail::compile_render_path_plan`) and the
`PathSwapRejectedEvent` fact (previous plan kept — the keep-previous-plan
invariant). The compiler's native `RenderPathCompileRejection` enum maps to
this pod vocabulary via `map_rejection`; plan error strings are diagnostics
only, never the error channel.

## Silent error rails (no error channel today)

- `frame`, `geometry`, `lighting`, `camera`, `resources`, `sky`, `scene`,
  and `gfx`: monostate identity gateways; no domain failures or mutations.
  Their plain Step values introduce no invented `None` error enum.
- `input`: intents emit their corresponding facts; no domain error rail.
- `logic`: invalid start/force ids and unstarted commands are materialized as
  typed rejection facts (see `EVENT_FLOW.md`), not an error enum. No-rule and
  same-state outcomes emit observation facts, not failures. The batch Step
  counts outcomes; ticks still advance time when the state id is unchanged.

A new fallible transition with a closed error enum adds its table here in the
same commit. Absence of an enum does not mean absence of rejection facts.

## Run C transition and compensation audit (2026-09-17)

Scope: domain rejection paths, not allocation exceptions or host rollback.
Batches commit commands sequentially; a later rejection does not undo earlier
accepted commands. No production cross-pod saga ships in these gateways.

| Pod | Failure classification and Rule 12 disposition |
|---|---|
| camera | Infallible monostate identity; no mutations or multi-step flows. |
| frame | Infallible monostate identity; no mutations or multi-step flows. |
| geometry | Infallible monostate identity; no mutations or multi-step flows. |
| gfx | Infallible monostate identity; no mutations or multi-step flows. Registry allocation is outside this gateway. |
| lighting | Infallible monostate identity; no mutations or multi-step flows. |
| resources | Infallible monostate identity; no mutations or multi-step flows. Resource IO/registry failures are outside this gateway. |
| scene | Infallible monostate identity; no mutations or multi-step flows. |
| sky | Infallible monostate identity; no mutations or multi-step flows. |
| input | Infallible intent handlers emit translation, applied rotation (after pitch clamp), toggle, or quit facts. No fallible multi-step flow or compensator. Quit is idempotent; movement and toggles are not claimed idempotent. |
| logic | Start/force validation and unstarted commands reject before mutation with typed facts. No-rule/same-state outcomes are observations; a started tick advances elapsed time. Valid transitions emit exit/enter facts; no fallible stage follows mutation on the domain rail. No multi-step saga; these facts alone do not restore previous elapsed time. |
| renderpath | Preset/technique/view/shadow candidates compile before installation. The six reasons above are the only closed domain error family. Rejection preserves recipe, plan, and generation and appends only `PathSwapRejectedEvent`; accepted swaps append `PathCompiledEvent` and, for setters, the corresponding change fact. Runtime toggle setters are infallible and idempotent. No compensation is needed for rejected candidates because none were installed. Successful installs are not generally reversible from these facts alone; host snapshots remain P6.3. |

Evidence: identity Step tests across all eight identity pods; input Step/fact
and logic rejection/same-state tests; renderpath
`rejected_view_culling_has_no_changed_fact` and
`rejected_swap_facts_and_recovery` (state, generation, prefix preservation,
replay, empty batch, accepted retry); saga `rollback_full` restores both stock
and the wallet debited before the later failure using reverse emitted facts.
The renderpath regression was observed red before moving change facts into
accepted branches. Allocation failure while copying state/appending events is
not a modeled domain rejection and no strong exception/rollback guarantee is
asserted here. New fallible multi-stage flows require their own compensation
facts/tests before shipment; do not interpret this audit as universal undo.

## Saga compensators (Rule 12)

Every rejection fact a compensator may need to undo exists as a row here or
in `EVENT_FLOW.md` *before* the saga ships; the compensator consumes the
catalog, never ad hoc done-flags or phantom POD flags. A payment/debit-style
mutation whose compensator restores only flagged stages while a prior debit
leaks (the wallet-leak shape) is non-conforming — rejected at review even
when the error channel carries context.