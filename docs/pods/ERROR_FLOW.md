# Domain Pod Error Flow — shs-renderer-lib

> Status: living catalog (2026-09-16, KDBA hardening). The failure-rail mirror
> of `EVENT_FLOW.md`: every closed error enum riding an `std::expected<T, E>`
> channel or a rejection fact is listed here, per pod. **Drift law:**
> `check_vop_boundaries.sh` FAILs if an error enum declared in
> `domains/*/*.event.hpp` or `domains/*/*.contract.hpp` is missing from this
> file. Rule 11 requires one error-enum family per bounded context; Rule 12
> requires the `.or_else()` compensator to consume the emitted rejection fact.

## How to read this

- **Channel** = where the error rides: the `expected` error channel of a
  Kleisli arrow (house signature: `(State, span<const Action>, dt) ->
  expected<Step{NextState, Events}, DomainError>`), or a rejection *fact*
  (e.g. `PathSwapRejectedEvent`) materialized on the failure rail while
  persistent state stays pristine.
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
(`renderpath.reducer.hpp` `detail::compile_render_path_plan`) and the
`PathSwapRejectedEvent` fact (previous plan kept — the keep-previous-plan
invariant). The compiler's native `RenderPathCompileRejection` enum maps to
this pod vocabulary via `map_rejection`; plan error strings are diagnostics
only, never the error channel.

## Silent error rails (no error channel today)

Pods whose reducers are identity transitions or total functions carry no
error enum: `input`, `logic`, `frame`, `geometry`, `lighting`, `camera`,
`resources`, `sky`, `scene`, `gfx`. The first real fallible transition in
any of them adds a table above in the same commit.

## Saga compensators (Rule 12)

Every rejection fact a compensator may need to undo exists as a row here or
in `EVENT_FLOW.md` *before* the saga ships; the compensator consumes the
catalog, never ad hoc done-flags or phantom POD flags. A payment/debit-style
mutation whose compensator restores only flagged stages while a prior debit
leaks (the wallet-leak shape) is non-conforming — rejected at review even
when the error channel carries context.