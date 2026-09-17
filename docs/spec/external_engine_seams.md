# External engine seams: physics, animation, audio

Status: normative (step 6 of the completed domain-separation migration; record archived at `docs/outdated/engine_domain_separation_migration.md`).
Supplements the job-system contract in `shs/task/job_system.hpp` (step 4.4)
and the vertical-slice host seam (`shs/app/vertical_slice_host.hpp`).

Purpose: specify how EXTERNAL subsystems (physics, animation, audio, or any
future one) integrate with the rendering library WITHOUT making SHS their
storage, their scheduler, or their programming model. No placeholder
subsystem is added; the host works headless with zero subsystems.

## 1. The seam is data: snapshots in, commands out

  - Subsystem -> renderer: a SNAPSHOT of state (plain values: transforms per
    object, poses, listener pose). The host takes the snapshot and passes
    it through scene/session state before `run_frame`.
  - Renderer -> subsystem (optional): plain value records the subsystem
    polls (e.g. visibility facts). No callbacks into render internals.
  - Scheduling ownership: the SUBSYSTEM owns its tick and snapshot point;
    the HOST owns the frame loop; the renderer owns only its documented
    per-frame state. Independent host instances never share state (pinned:
    interleaved hosts in vertical_slice_tests).

Consequence: the rendering library compiles and runs identically with zero,
one, or many subsystems. This is the "no placeholder subsystems" rule.

## 2. Stable identity across the seams

  - Scene objects: `object_id` (FNV-1a of the name, 0 = empty). Deleting and
    recreating an object with the same name preserves identity by
    construction (step 4.3 policy; pinned in scene_identity_tests and in
    the vertical-slice deletion/recreation pin).
  - Assets: `ResourceRegistry` handles + `generation()` epochs. Handles
    minted before `clear()` are STALE; `SceneResourceView` resolves them to
    nullptr per call and the render SKIPS them (pinned: registry-epoch pin).
    Re-derive via `find_*` after clearing.
  - Projections never cache: `SceneResourceView` resolves per call; pointers
    are valid only until the next registry mutation.

## 3. Snapshot/contract table

| Seam      | Snapshot (in)                          | Stable ID                  | Renderer output (optional)   |
|-----------|----------------------------------------|----------------------------|------------------------------|
| Physics   | poses/velocities per object            | `object_id` + asset handles | visibility/culling facts    |
| Animation | skinned transforms, clip state         | `object_id` + clip IDs     | motion-vector parameters     |
| Audio     | listener + emitter poses               | emitter IDs (subsystem)    | none (renderer is read-only) |

No seam mandates SHS storage, an FSM, ECS, a global event bus, or the
optional app host. Any of those may live on the subsystem side; nothing in
the rendering library depends on the choice.

## 4. Sync/async completion and cancellation at task/asset/backend boundaries

(Supplements the job-system contract, step 4.4; the vertical-slice suite
pins the task-side promise with outstanding work before teardown.)

  - TASK boundary: `IJobSystem::wait_idle()` is the ONLY completion
    guarantee. Destruction DRAINS every job accepted before teardown began
    (pinned: teardown-with-outstanding-work pin), but callers that must
    release memory referenced by jobs use `wait_idle()` first. enqueue()
    after teardown has begun is a caller error.
  - ASSET boundary: asset loading is HOST-owned. Completion is observed by
    the host (future, poll, or wait); the ONLY publication point into
    renderer-visible state is a `ResourceRegistry` mutation BETWEEN frames
    (never during `run_frame` — the registry is read per call during the
    raster). Cancellation = the host stops waiting and never publishes;
    a missing/stale handle degrades to "not drawn this frame" (pinned),
    never a crash.
  - BACKEND boundary: `run_frame` is SYNCHRONOUS by contract (step 6): when
    it returns, this frame's raster and pixel digest are complete. A backend
    may be asynchronous internally only if it exposes its own completion
    primitive (the Vulkan offscreen path does: `vulkan_submit_sync` /
    `execute_offscreen` return after the fence). Cancellation at backend
    level = command-stream rejection BEFORE submission (pinned: preflight
    rejection leaves no partial recording).
  - RULE for future work: any new async renderer task must ship, in the
    same change, a completion query, a cancellation that is safe as a
    no-op, and a teardown-with-outstanding-work test — before the
    concurrency is documented as supported.

## 5. What this document does NOT mandate

  - No SHS-owned physics/animation/audio subsystem exists or is implied.
  - No requirement to store game state in SHS containers.
  - No event bus, no global subsystem registry, no FSM framework.
  - The optional app host remains optional; the vertical-slice host runs
    standalone (headless, value-only, no platform headers).
