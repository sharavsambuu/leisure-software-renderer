# Monadic Pipeline Constitution Amendment Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Adopt C++23 monadic-pipeline assistance into the Domain POD constitutions without losing non-conflicting DOD/VOP law.

**Architecture:** Keep per-command reducers as variant command → variant event streams; promote monadic `std::expected`/`std::optional` pipelines to fallible value/error channels, batch planners, bridges, loaders, and orchestrator/saga tiers.

**Tech Stack:** C++23, CMake/CTest, Domain POD Core 4 (+ orchestrator pattern), `std::expected`, `std::optional`, `std::span`, PMR arenas, boundary linter.

---

## 1. Current context / assumptions

- The repo already has three working constitutional layers:
  - `docs/spec/conventions.md` — Constitution I.
  - `docs/spec/value_oriented_programming.md` — Constitution II, including Core 4, event sourcing, memory/layout laws, and limited monadic adoption.
  - `docs/spec/dod_ecs_architecture.md` — Constitution III.
- Supporting semantic docs:
  - `docs/pods/DOMAIN_GLOSSARY.md`
  - `docs/pods/EVENT_FLOW.md`
  - `docs/backlog/domain_pod_hardening_backlog.md`
  - `tools/check_vop_boundaries.sh`
- Campaign evidence strongly supports:
  - Per-command reducers with data-dependent event counts.
  - Closed enum/event vocabularies with edge name tables.
  - `expected` at fallible planner/compiler seams.
  - Chunk-level monads, not per-element monads.
- The new F-DOD-DDD vision is adopted as direction, but its bundled reducer signature is not yet proven against the above evidence.
- Hobby project: breaking changes are allowed, but deletions must be recoverable and recorded.

## 2. Proposed approach

Adopt by separation of channels:

1. **Retain by default:** every old law stays unless it directly conflicts with the adopted monadic-pipeline vision.
2. **Record conflicts explicitly:** do not silently overwrite dissent or campaign evidence.
3. **Adopt monads where the channel is value/error:** planners, compilers, loaders, bridges, batch stages, orchestrators/sagas.
4. **Retain variants where the channel is command/events:** per-command reducers and replayable event logs.
5. **Make the boundary semantic explicit:** bounded contexts own PODs, error enums, events, and pipelines; cross-context traffic is event-only.

## 3. Conflict register — zero-signal-loss record

### C1: Reducer return shape

- Old/campaign position:
  - House reducer signature: `reduce(State&, span<const Action>, Inputs, pmr::vector<Event>&)`.
  - Evidence: N-events-per-command, conditional 0–2 events, silent pods.
- New vision position:
  - Reducer returns `expected<(State, Events), Error>`.
- Planned resolution:
  - Adopt synthesis first: monadic pipelines assist at planner/orchestrator/batch level.
  - Do **not** migrate per-command reducers to bundled `expected<(State,Events)>` without a replay/event-count spike proving no loss.
  - Preserve this dissent in the constitution amendment notes.

### C2: Event representation

- Old/campaign position:
  - Closed enum/struct events, no strings in facts, name tables at edges.
- New vision risk:
  - Stringly `stage/message` telemetry.
- Planned resolution:
  - Adopt closed events as supreme law.
  - Strings only render facts at edges/logs.

### C3: Parallel runtime

- Old/campaign position:
  - Explicit wait-free spans, arenas, controlled scheduler trajectory.
- New vision risk:
  - Blanket `std::execution::par_unseq`.
- Planned resolution:
  - Retain explicit data-parallel control.
  - Allow standard parallel algorithms only where scheduler/arena behavior is explicit.

### C4: Snapshots

- Old/campaign position:
  - Value-copy snapshots, replay logs, generational handles.
- New vision risk:
  - `memcpy` snapshot claim.
- Planned resolution:
  - Retain value-copy + replay law.
  - State linear-ownership/in-place mutation theorem narrowly.

## 4. Retain list — do not drop these

- Functional core / imperative shell separation.
- Passive PODs, stateless logic, deterministic replay.
- Core 4 pod file structure and zone direction law.
- Cross-domain event-only communication.
- Contiguous memory, SoA/AoS selection by access pattern, PMR arenas.
- Closed command/event vocabularies.
- Parity harness and Tier0/Tier1 evidence.
- Existing pod test-kit replay guarantees.

## 5. Clause-by-clause amendment map

### Constitution I — `docs/spec/conventions.md`

- Change:
  - Promote C++23 from demo-follow-up to lib baseline.
  - Add pod-idiomatic C++23 subset:
    - Allowed broadly: `expected`, monadic `optional`, `span`, `format`/enum formatters.
    - Allowed at defined tiers: ranges in planners, `mdspan` at tile kernels.
    - Restricted: coroutines outside edges, concepts at API rims.
- Retain:
  - Coordinate handedness, NDC, canvas/screen, Jolt bridge, Vulkan viewport rules.

### Constitution II — `docs/spec/value_oriented_programming.md`

- §2 architecture diagram:
  - Add pipeline tier: commands → per-command reducer → state/events, assisted by planner/orchestrator pipelines.
  - Name orchestrator as a pod, not an external controller.
- §3 enforceable rules:
  - Add saga/compensation rule:
    - Prefer validate-before-mutate.
    - Compensators consume fact/event logs, not ad hoc context flags.
    - Every saga failure path must preserve auditability.
  - Add bounded-context rule:
    - Intra-context stages may compose synchronously.
    - Inter-context interaction is event-only through shell/bus.
- §6 Core 4:
  - Change Core 4 → Core 4+1 pattern:
    - Core 4 remains mandatory for every pod.
    - `+1` orchestrator/saga recipe is required only for multi-domain workflows.
    - Every orchestrator must itself be a pod.
  - Add bounded-context ownership table requirements:
    - PODs, error enum, events, pipelines, ubiquitous language.
- §7 memory/layout:
  - Retain all existing laws.
  - Add granularity law wording:
    - Monad at chunk/batch/span level.
    - Never wrap hot-loop scalar elements in `expected`.
- §8 monadic adoption:
  - Expand from “leaf seams only” to tier doctrine:
    - Value/error channels: monadic.
    - Command/event streams: variant-based.
    - Batch/orchestrator pipelines: Kleisli `and_then`/`transform`/`or_else`.
  - Add saga compensation pattern and money-leak regression rationale.
  - Preserve campaign evidence for unbundled event streams.
- §10 linter:
  - Add planned gates:
    - Ban per-element `expected` in hot reducer loops.
    - Ban stringly event facts.
    - Require saga compensators to reference event/fact logs.
- Appendix A:
  - Add F-DOD-DDD correspondence:
    - State, Writer, Either, Kleisli mappings.
    - Explicit boundary where the bundled signature diverges.

### Constitution III — `docs/spec/dod_ecs_architecture.md`

- Change:
  - Adopt ECS mapping table:
    - Entity = ID.
    - Component = Domain POD.
    - System = pure reducer/stage.
    - World/scheduler = saga orchestrator.
  - Narrow in-place-mutation purity theorem to exclusive linear ownership.
- Retain:
  - SoA/AoS, cache, SIMD, allocation, hot/cold separation.
- Add:
  - Scheduler control requirement for Virtual SPU trajectory.

### Glossary and event docs

- `docs/pods/DOMAIN_GLOSSARY.md`:
  - Add bounded-context section.
  - Add orchestrator/saga rows.
  - Point pod rows at final homes.
- `docs/pods/EVENT_FLOW.md`:
  - Add saga event requirements.
  - Keep drift gate coverage for every event struct.

### Backlog and tooling

- `docs/backlog/domain_pod_hardening_backlog.md`:
  - Add constitutional amendment track.
  - Preserve P6 blockers and leftovers.
- `tools/check_vop_boundaries.sh`:
  - Add only gates required by amended law.
  - Do not weaken existing Core 4, zone-direction, purity, or event-drift gates.

---

## 6. Step-by-step plan

### Task 1: Freeze the amendment baseline

**Objective:** Record the exact pre-amendment constitution state.

**Files:**
- Modify: `.hermes/plans/2026-09-16_120707-monadic-pipeline-constitution-amendment.md`

**Step 1: Verify working tree is clean**

Run:

```bash
git status --short
```

Expected: no output.

**Step 2: Record baseline commit**

Run:

```bash
git log --oneline -5
```

Expected: current campaign tip is visible and recorded in the plan.

**Step 3: Commit**

Do not commit code; this task only prepares plan context.

### Task 2: Amend Constitution I for C++23 baseline

**Objective:** Make C++23 explicit and define the pod-idiomatic subset.

**Files:**
- Modify: `docs/spec/conventions.md`

**Step 1: Change baseline wording**

Replace demo-deferred C++23 language with lib-baseline language.

**Step 2: Add allowed/restricted feature tiers**

Include `expected`, monadic `optional`, `span`, formatters, ranges, `mdspan`, coroutines, concepts.

**Step 3: Verify**

Run:

```bash
git diff -- docs/spec/conventions.md
```

Expected: only C++23 baseline/tier changes appear.

**Step 4: Commit**

```bash
git add docs/spec/conventions.md
git commit -m "Amend Constitution I: C++23 lib baseline and pod-idiomatic subset"
```

### Task 3: Add bounded context and saga rules to Constitution II §3

**Objective:** Make domain boundaries and compensation enforceable.

**Files:**
- Modify: `docs/spec/value_oriented_programming.md`

**Step 1: Add bounded-context rule**

Define POD ownership, error enum, events, pipelines, ubiquitous language.

**Step 2: Add saga/compensation rule**

Require validate-before-mutate preference and event-log-based compensation.

**Step 3: Verify**

Run:

```bash
git diff -- docs/spec/value_oriented_programming.md
```

Expected: new numbered rules only; old rules unchanged.

**Step 4: Commit**

```bash
git add docs/spec/value_oriented_programming.md
git commit -m "Amend Constitution II: bounded contexts and saga compensation rules"
```

### Task 4: Change Core 4 to Core 4+1 without weakening Core 4

**Objective:** Add orchestrator pattern while keeping every pod mandatory Core 4.

**Files:**
- Modify: `docs/spec/value_oriented_programming.md`
- Modify: `docs/pods/DOMAIN_GLOSSARY.md`

**Step 1: Define `+1` narrowly**

An orchestrator is required only for multi-domain workflows and must itself be a pod.

**Step 2: Forbid god-object orchestrators**

No non-pod workflow controller may own cross-domain state.

**Step 3: Verify**

Run:

```bash
git diff -- docs/spec/value_oriented_programming.md docs/pods/DOMAIN_GLOSSARY.md
```

Expected: Core 4 intact; orchestrator added as constrained extension.

**Step 4: Commit**

```bash
git add docs/spec/value_oriented_programming.md docs/pods/DOMAIN_GLOSSARY.md
git commit -m "Amend Core 4: add constrained orchestrator extension"
```

### Task 5: Expand §8 into monadic tier doctrine

**Objective:** Replace “leaf seams only” with channel-based monadic law.

**Files:**
- Modify: `docs/spec/value_oriented_programming.md`

**Step 1: State the channel law**

Monads for value/error channels; variants for command/event streams.

**Step 2: Add Kleisli pipeline vocabulary**

Document verifier, transformer, compensator stages.

**Step 3: Preserve campaign dissent**

Record why bundled `expected<(State,Events)>` is not adopted for per-command reducers.

**Step 4: Commit**

```bash
git add docs/spec/value_oriented_programming.md
git commit -m "Amend Constitution II §8: monadic tier doctrine and preserved reducer evidence"
```

### Task 6: Adopt ECS mapping in Constitution III

**Objective:** Formalize Entity/Component/System/World correspondence.

**Files:**
- Modify: `docs/spec/dod_ecs_architecture.md`

**Step 1: Add mapping table**

Entity, Component, System, World/Scheduler.

**Step 2: Narrow mutation theorem**

Allow in-place mutation only under exclusive linear ownership.

**Step 3: Add scheduler-control requirement**

No blanket scheduler surrender on the Virtual SPU path.

**Step 4: Commit**

```bash
git add docs/spec/dod_ecs_architecture.md
git commit -m "Amend Constitution III: ECS correspondence and scheduler control"
```

### Task 7: Update event and glossary semantics

**Objective:** Ban stringly facts and require saga auditability.

**Files:**
- Modify: `docs/pods/DOMAIN_GLOSSARY.md`
- Modify: `docs/pods/EVENT_FLOW.md`

**Step 1: Require closed event payloads**

Enums, IDs, quantities, facts; no prose as state.

**Step 2: Require saga facts**

Every compensated mutation must have a corresponding fact.

**Step 3: Verify drift gate still passes**

Run:

```bash
bash cpp-folders/src/shs-renderer-lib/tools/check_vop_boundaries.sh
```

Expected: boundary check passes.

**Step 4: Commit**

```bash
git add docs/pods/DOMAIN_GLOSSARY.md docs/pods/EVENT_FLOW.md
git commit -m "Amend event law: closed saga facts, no stringly events"
```

### Task 8: Add linter gates only for new law

**Objective:** Mechanically enforce the amendment without weakening old gates.

**Files:**
- Modify: `tools/check_vop_boundaries.sh`

**Step 1: Add hot-loop monad granularity check**

Target per-element `expected` in reducer hot paths.

**Step 2: Add stringly-event check if absent**

Protect closed event payloads.

**Step 3: Run boundary check**

Run:

```bash
bash cpp-folders/src/shs-renderer-lib/tools/check_vop_boundaries.sh
```

Expected: green.

**Step 4: Commit**

```bash
git add cpp-folders/src/shs-renderer-lib/tools/check_vop_boundaries.sh
git commit -m "Enforce monadic amendment boundaries"
```

### Task 9: Prove saga law with a regression spike

**Objective:** Demonstrate that event-log compensation catches the wallet-style leak.

**Files:**
- Create: temporary saga regression test under the existing test layout
- Test: saga rollback preserves both inventory and payment facts

**Step 1: Write failing test**

Model reserve → debit → failure → compensation.

**Step 2: Run test to verify failure**

Run:

```bash
ctest -R 'shs_renderer_vop' --output-on-failure
```

Expected: new saga regression test fails before the fix.

**Step 3: Implement event-log-based compensation**

Do not use ad hoc restored/not-restored flags as source of truth.

**Step 4: Run tests to verify pass**

Run:

```bash
ctest -R 'shs_renderer_vop' --output-on-failure
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add <test-files> <implementation-files>
git commit -m "Prove saga compensation through fact logs"
```

### Task 10: Final constitution/test/linter verification

**Objective:** Prove the amendment did not regress the campaign.

**Files:**
- Modify: `docs/backlog/domain_pod_hardening_backlog.md`

**Step 1: Run boundary check**

Run:

```bash
bash cpp-folders/src/shs-renderer-lib/tools/check_vop_boundaries.sh
```

Expected: green.

**Step 2: Run renderer tests**

Run:

```bash
ctest -R 'shs_renderer_vop' --output-on-failure
```

Expected: green in both configured build trees.

**Step 3: Tick amendment backlog**

Record retained laws, conflicts, and unlock conditions.

**Step 4: Commit**

```bash
git add docs/backlog/domain_pod_hardening_backlog.md
git commit -m "Close monadic constitution amendment with verification"
```

---

## 7. Files likely to change

- `docs/spec/conventions.md`
- `docs/spec/value_oriented_programming.md`
- `docs/spec/dod_ecs_architecture.md`
- `docs/pods/DOMAIN_GLOSSARY.md`
- `docs/pods/EVENT_FLOW.md`
- `docs/backlog/domain_pod_hardening_backlog.md`
- `tools/check_vop_boundaries.sh`
- Temporary saga regression test and minimal implementation files.

## 8. Tests / validation

- `git status --short` clean before and after major amendment commits.
- `bash cpp-folders/src/shs-renderer-lib/tools/check_vop_boundaries.sh` green.
- `ctest -R 'shs_renderer_vop' --output-on-failure` green.
- Parity suite remains unaffected unless a rendering path changes.
- New saga regression test must fail before compensation and pass after.
- Backlog records retained laws, conflicts, and unlock conditions.

## 9. Risks, tradeoffs, and open questions

- Risk: the bundled `expected<(State,Events)>` signature may appear cleaner but destroy per-command auditability.
  - Mitigation: adopt synthesis first; require a spike before migration.
- Risk: orchestrator language may invite god objects.
  - Mitigation: orchestrator must itself be a pod.
- Risk: standard parallel algorithms may surrender scheduler control.
  - Mitigation: allow only explicit scheduler/arena-compatible use.
- Risk: breaking-change boldness may delete useful legacy context.
  - Mitigation: recoverable deletions plus amendment notes.
- Open question: should demos move immediately to C++23, or only on restart?
  - Proposed default: move on restart; record L2 closure then.
- Open question: is there any pod shape that genuinely needs bundled state+events?
  - Current answer: no proven case; the spike in Task 9 is the gate.
