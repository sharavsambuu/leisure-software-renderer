# Domain Pod Event Flow — shs-renderer-lib

> Status: living catalog (2026-09-16, R5b P4.5). Every discrete event value
> emitted by a pod gateway is listed here, per pod, with its meaning and
> producer. **Drift law:** `check_kdba_boundaries.sh` FAILs if any `*Event`
> struct in `domains/*/*.event.hpp` is missing from this file — the catalog
> and the code cannot diverge. Name tables in code (`renderpath_event_name`,
> `input_event_name`, `fsm_event_name_traffic`) are the P6 overlay's label
> source; this file is the human mirror of those tables.

## How to read this

- **Producer** = the pod's Kleisli gateway (house shape per
  `docs/outdated/kdba_kleisli_migration_plan.md` (archived 2026-09-17): batch rim
  `(State, span<Command>, Context, arena) -> Step` summary — infallible
  wherever every real failure is already a materialized rejection fact;
  per-command `expected` rails live inside the arrows; failure keeps state +
  materializes a rejection fact).
- **Fact, not command** (Constitution Rule 8.1): each entry is a raw
  state-transition fact. Downstream gateways/edges interpret; the event
  itself never carries downstream instructions.
- Pods with `std::monostate` vocabularies (frame, geometry, lighting,
  camera, resources, sky, scene, gfx) emit nothing by construction and are
  listed once at the bottom so their silence is explicit, not assumed.

## renderpath — `shs::renderpath::RenderPathEvent`

| Event | Emitted when |
|---|---|
| `PathCompiledEvent` | A recipe compiled valid and became the active plan (carries technique/mode/pass count). |
| `PathSwapRejectedEvent` | A compile failed; the pod kept the previous plan (carries native `PathSwapRejectionReason`). |
| `TechniqueSwitchedEvent` | The rendering technique changed (previous → current), only after an accepted swap. Rejected technique/view/shadow swaps emit only `PathSwapRejectedEvent`, never a switched/changed fact (Run C K3.3 regression). |
| `ViewCullingModeChangedEvent` | The view-chain culling mode changed (previous → current). K4.2 split (Run A 2026-09-17): was the positional-bool `CullingModeChangedEvent`. |
| `ShadowCullingModeChangedEvent` | The shadow-chain culling mode changed (previous → current). K4.2 split (Run A 2026-09-17). |
| `RuntimeToggledEvent` | A runtime flag flipped (flag id + post-toggle value). |
| `TechniqueUnchangedEvent` | A technique command requested the already-active technique — no-op fact (K3.2 zero-signal-loss, Run A). |
| `ViewCullingUnchangedEvent` | A view-culling command requested the already-active mode — no-op fact (K3.2, Run A). |
| `ShadowCullingUnchangedEvent` | A shadow-culling command requested the already-active mode — no-op fact (K3.2, Run A). |

## input — `shs::input::InputEvent`

| Event | Emitted when |
|---|---|
| `CameraTranslatedEvent` | A MoveLocal action applied (carries the applied world-space delta). |
| `CameraRotatedEvent` | A Look action applied (carries applied yaw/pitch deltas, post-clamp). |
| `RuntimeFlagToggledEvent` | Light-shafts/bot flag flipped (flag id + post-toggle value). |
| `QuitRequestedEvent` | A Quit action applied (quit_requested now true). |

## logic — `shs::logic::FsmEvent<TStateId>`

| Event | Emitted when |
|---|---|
| `FsmStarted` | The machine began at the initial state (rejected ids emit `FsmStartRejected` instead). |
| `FsmStateEntered` | A state became current (also right after `FsmStarted`). |
| `FsmStateExited` | A state stopped being current (always immediately before its `FsmStateEntered`). |
| `FsmTransitionRejected` | A Force targeted an unknown state (current unchanged). |
| `FsmStartRejected` | A Start targeted an unknown state (machine stays unstarted). |
| `FsmSignalRejected` | A Signal arrived while the machine is not started (K3.2 fact, Run B). |
| `FsmSignalNoRule` | A Signal matched no rule from the current state (K3.2 fact, Run B). |
| `FsmTickUnstarted` | A Tick arrived while the machine is not started (K3.2 fact, Run B). |
| `FsmTickNoRule` | A Tick fired no time-gated rule (K3.2 fact, Run B). |
| `FsmForceUnstarted` | A Force arrived while the machine is not started (K3.2 fact, Run B; a silent drop the audit missed, closed by zero-signal-loss). |
| `FsmSignalUnchanged` | A signal matched a same-state rule (carries signal id); no exit/enter or time reset. |
| `FsmForceUnchanged` | Force requested the current state (carries target); no exit/enter or time reset. |
| `FsmTickUnchanged` | A tick matched a same-state rule; state id is unchanged, but elapsed time still advances. |

## Silent pods (monostate event vocabulary — emit nothing)

`frame` (`FrameEvent`), `geometry` (`GeometryEvent`), `lighting`
(`LightingEvent`), `camera` (`CameraEvent`), `resources`
(`ResourcesEvent`), `sky` (`SkyEvent`), `scene` (`SceneEvent`), `gfx`
(`GfxEvent`). Their gateways are identity transitions; the first real
transition in any of them adds a row to a table above in the same commit.
## Saga fact requirements (Rule 12)

Every mutation a compensator may need to undo must exist here as a fact row
*before* the saga ships: the compensator consumes this catalog, never ad hoc
done-flags. A payment/debit-style mutation without a corresponding fact row
is non-conforming — the wallet-leak shape (restoring only flagged stages
while a prior debit leaks) is rejected at review even when the error channel
carries the context.

## Persistence and evolution policy (2026-09-17)

Under Constitution II §11.1, published wire schemas preserve field meanings
and stable type identifiers within each version. Do not silently repurpose a
shipped event or encode its `std::variant` index as a persistent identifier.
Changes require a new schema/type version and either tested migration or an
explicit unsupported-version rejection. This is compatibility of published
formats, not a requirement to keep all historical C++ variants forever.

Persistent payloads encode values and stable logical IDs, never raw pointers,
spans, allocator addresses or object memory. An in-memory handle requires a
specified reconstruction/remapping policy. Snapshot/command codecs and old-log
fixtures remain P6.2 work; this catalog does not establish serializability.
Facts are observations, not necessarily a complete state reconstruction log.
Snapshot + commands + recorded external inputs is the default replay contract.
Retention, checkpoint cadence, ordering and unknown-version handling must be
specified before persistence ships (S5 in the active conformance backlog).

## Failure-rail mirror

Closed error enums (the `expected` error channel and rejection facts) are
cataloged in `docs/pods/ERROR_FLOW.md` under the same drift gate —
rejections are facts too.
