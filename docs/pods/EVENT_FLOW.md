# Domain Pod Event Flow — shs-renderer-lib

> Status: living catalog (2026-09-16, R5b P4.5). Every discrete event value
> emitted by a pod gateway is listed here, per pod, with its meaning and
> producer. **Drift law:** `check_kdba_boundaries.sh` FAILs if any `*Event`
> struct in `domains/*/*.event.hpp` is missing from this file — the catalog
> and the code cannot diverge. Name tables in code (`renderpath_event_name`,
> `input_event_name`, `fsm_event_name_traffic`) are the P6 overlay's label
> source; this file is the human mirror of those tables.

## How to read this

- **Producer** = the Kleisli pipeline that emits the event (house signature:
  `(State, span<const Command>, Context) -> expected<Step{NextState, Events}, DomainError>`; failure keeps state + materializes a rejection fact).
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
| `TechniqueSwitchedEvent` | The rendering technique changed (previous → current). |
| `CullingModeChangedEvent` | View or shadow culling mode changed (previous → current). |
| `RuntimeToggledEvent` | A runtime flag flipped (flag id + post-toggle value). |

## input — `shs::input::InputEvent`

| Event | Emitted when |
|---|---|
| `CameraTranslatedEvent` | A MoveLocal action applied (carries the applied world-space delta). |
| `CameraRotatedEvent` | A Look action applied (carries applied yaw/pitch deltas, post-clamp). |
| `RuntimeFlagToggledEvent` | Light-shafts/bot flag flipped (flag id + post-toggle value). |
| `QuitRequestedEvent` | A Quit action applied (quit_requested now true). |

## logic — `shs::FsmEvent<TStateId>`

| Event | Emitted when |
|---|---|
| `FsmStarted` | The machine began at the initial state (rejected ids emit `FsmStartRejected` instead). |
| `FsmStateEntered` | A state became current (also right after `FsmStarted`). |
| `FsmStateExited` | A state stopped being current (always immediately before its `FsmStateEntered`). |
| `FsmTransitionRejected` | A Force targeted an unknown state (current unchanged). |
| `FsmStartRejected` | A Start targeted an unknown state (machine stays unstarted). |

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

## Failure-rail mirror

Closed error enums (the `expected` error channel and rejection facts) are
cataloged in `docs/pods/ERROR_FLOW.md` under the same drift gate —
rejections are facts too.
