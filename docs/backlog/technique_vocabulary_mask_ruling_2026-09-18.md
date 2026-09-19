# Render-technique vocabulary — mask-representation ruling (2026-09-18)

**Row:** RP-5 — unify, then open, the render-technique vocabulary (backlog tracker
`docs/backlog/remaining_todos_2026-09-18.md`).
**Graduation requirement:** arch §4 req 8 (`docs/arch/render_path_architecture.md`).
**Status:** ✅ the blocked decision is **made**; RP-5 itself remains **not started**.
**Provenance:** taken 2026-09-18 under delegated authority ("go ahead on those two
things"), so it is **binding for RP-5's implementation unless the owner overturns
it** — flagged here rather than written as an owner ruling, because no owner
sign-off was explicitly given for this specific decision. All three options were
weighed on evidence from the tree, below.

## Decision

**Option "cap at 32": the open `TechniqueMode` range is bounded by the existing
32-bit mask representation.** No mask widening, no side set. Concretely: builtins
`0..4` (pinned), reserved `5`, open base `6`, open max `31` ⇒ **26
consumer-authorable technique modes**, with the ceiling enforced by a
`static_assert` beside the enum and by the registry's capacity.

## Why the other two options lost

**(a) Widen the mask to 64-bit.** Mechanically cheap — and that was the trap.
`technique_mode_bit` is `1u << value`; `supported_modes_mask`
(`TechniquePassContract`) and `active_modes_mask` (`FrameParams::TechniqueParams`)
are `uint32_t`. Nothing serializes either mask: the only externalization anywhere
in the tree is a hex string in a `PluggablePipeline` diagnostic (`mask=0x...`), so
widening would migrate no data and no wire format. But it *would* change the
declared type of a field of `FrameParams`, which is the frame pod's
replay-snapshotted state (`pod_test_kit::empty_log_is_stable<FrameParams, ...>` /
`replay_is_deterministic<FrameParams, ...>`; `TechniqueParams` compares by
defaulted `operator==`). Buying 32 more slots by re-typing the frame state for a
vocabulary with 5 builtins, whose named consumer motivation (mobile shading lanes)
is single-digit, is the wrong trade — and unlike the range law it is not cheaply
reversible.

**(b) `uint32_t` + a side set of open ids.** Not disqualified by the pod kit — the
kit requires only value semantics (`operator==`) and copyability, and an
`unordered_set` satisfies both — so this option had to be argued down on cost, not
convention. It loses on two counts. First, "does this pass support this mode?"
becomes a **two-mechanism query** (bit test *or* set membership), and every
participation check in the tree would have to remember to ask twice:
`technique_mode_in_mask` is called from `render_pass.hpp` (`supports_technique_mode`),
`pass_registry.hpp` (descriptor hint) and `pluggable_pipeline.hpp` (both the active
mask and the contract mask). A check that asks once is silently *partial* — the
exact silent-load-bearing failure this series keeps closing. Second, the existing
32-bit masks do not go away in this option (they still carry the builtin half), so
the representation ends up genuinely hybrid: two encodings for one relation, plus a
container in the per-frame value struct.

## Why cap-at-32 is not a compromise

The mask *is* the representation. `1u << v` for an unsigned 32-bit type is
well-defined for `v ∈ [0,31]` and undefined at 32 — so **any** open range ≤ 31
needs no representation change at all, and any range ≥ 32 needs a type change
everywhere. Capping the range to what the representation can express is the
smallest change that makes the goal true: a consumer registers a technique mode by
name, with zero core edits.

## The escape hatch is explicit

The ceiling lives in exactly one place (`kTechniqueModeOpenMax`). If a consumer
ever needs more than 26 open modes the move is a mechanical widening to `uint64_t`
with `kTechniqueModeOpenMax = 63` — and it will be *visible* when the need is real,
because a registration past `OpenMax` must be a named refusal
(`OpenRangeExhausted`), never a truncation and never an aliasing wrap.

## Scope correction found while ruling

The backlog row and arch §4 req 8 say "two closed enums describe one axis". There
are **three**:

| enum | values | spelling | header |
| :--- | :--- | :--- | :--- |
| `TechniqueMode` | 5 | `Forward`, `ForwardPlus`, `Deferred`, `TiledDeferred`, `ClusteredForward` | `render/frame/technique_mode.hpp` |
| `RenderPathPreset` | 5 | **identical, value for value** | `planning/render_path_presets.hpp` |
| `RenderPathRenderingTechnique` | 3 | `ForwardLit`, `ForwardPlus`, `Deferred` | `planning/render_path_recipe.hpp` |

They are not equally redundant, and the difference decides the retirement order:

- **`RenderPathRenderingTechnique` is the one with the defect.** `technique_mode_for()`
  maps exactly its three values and defaults everything else to `Forward`, which is
  why `TiledDeferred` and `ClusteredForward` have no authoring path from a recipe.
  Its relation to `TechniqueMode` must be made total before the range opens, because
  a partial relation plus an open range is a *worse* trap than either alone: an
  open mode would silently resolve to `Forward`.
- **`RenderPathPreset` ⇄ `TechniqueMode` is an exact 1:1** — `render_path_preset_mode`
  / `render_path_preset_for_mode` are total switches with no default-worthy value and
  no residual (no preset maps to a non-mode, no two presets share a mode). So as
  things stand it carries no information beyond the mode. Caveat worth stating rather
  than assuming: a path preset is arguably a *selection key* (which recipe) rather
  than a *resolved technique* (what the recipe does), so the honest move is for RP-5
  to decide explicitly whether the preset is a redundant spelling to retire or a
  distinct concept to keep — on the evidence that today the mapping is
  information-free, not on the assumption that the names match.

## Shape the implementation must take

Mirroring the pass-semantic / pass-id / shader-id law exactly, so an author who
understands one namespace understands all four:

```cpp
inline constexpr uint8_t kTechniqueModeBuiltinMin = 0u;    // Forward
inline constexpr uint8_t kTechniqueModeBuiltinMax = 4u;    // ClusteredForward
inline constexpr uint8_t kTechniqueModeReserved   = 5u;    // never assignable
inline constexpr uint8_t kTechniqueModeOpenBase   = 6u;
inline constexpr uint8_t kTechniqueModeOpenMax    = 31u;   // the mask's last bit
// capacity = kTechniqueModeOpenMax - kTechniqueModeOpenBase + 1 = 26
```

- The **reserved slot** exists so that adding a 6th builtin later cannot silently
  alias an id already minted from a name by a consumer; the top of the builtin range
  is pinned by `static_assert` so growth is a compile error, not a collision.
- `technique_mode_bit` keeps its exact signature (`1u << static_cast<uint32_t>(m)`):
  with the range law above every reachable id is ≤ 31, so the shift is well-defined
  *by construction* rather than by luck, and no call site changes.
- Open ids are carried as `static_cast<TechniqueMode>(id)` — the type is the carrier,
  exactly as it already is for `PassSemantic` — so the existing closed switch
  surfaces take an open branch guarded by `technique_mode_is_open`, never by
  `!is_builtin` (the wider guard would silently adopt the reserved slot).
- A registration past `OpenMax` must be a **named refusal** (`OpenRangeExhausted`),
  never a truncation and never an aliasing wrap. That is what makes the ceiling
  visible if the escape hatch is ever needed.

## What this ruling does NOT decide

- It does not open the range; it bounds it. RP-5 still has to retire/widen the other
  two enums (order above), because a partial relation plus an open range silently
  resolves an open mode to `Forward`.
- It does not touch `Substrate` (3 values), which RP-8 explicitly excludes: adding a
  fourth substrate is an RHI-level decision, not an id-range job.
- It says nothing about `PassSemanticEncoding` (the attachment-packing axis), which
  is a new abstraction rather than a closed enum, and remains the real ceiling on the
  G-buffer goal — arbitrary layouts are still not expressible, only arbitrary channel
  names. RP-6's duplicate pair (`ShadingModel` / `RenderTechniquePreset`) is unified
  *under* this ruling, which is exactly why it was worth making before either row
  starts.
