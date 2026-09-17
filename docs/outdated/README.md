# Outdated Docs — Superseded but Preserved

> Zero-signal-loss archive (created 2026-09-16, monadic amendment cleanup).
> Files here were removed from the live tree because they contradict or
> predate current constitutional law. Original bodies are preserved; status
> banners and repaired links may be added. Git history holds byte-identical
> originals and original paths; this folder retains their reasoning and provenance.
>
> Rule: nothing enters here without a row below. Nothing here is law.

| File | Moved from | Superseded by | Why kept |
| :--- | :--- | :--- | :--- |
| `modern_rendering_strategies.md` | `docs/arch/` | Constitution II §8 tier doctrine; `rendering_techniques_curriculum.md` ladder | Generic GPU-technique essay (visibility buffers, 1000-light clustered); zero connection to the software-renderer spine |
| `global_illumination_strategies.md` | `docs/arch/` | `docs/roadmap/global_illumination_roadmap.md` (live track) | Second-person GI essay predating the roadmap; roadmap is the track |
| `compact_rendering_strategies.md` | `docs/arch/` | §7 memory laws (SoA, chunking, streaming stores) | Bandwidth essay absorbed into enforceable §7 law |
| `vop-track-leaf-seams-scope-2026-09-16.md` | `docs/roadmap/value_oriented_programming_first_class_roadmap.md` item 9 | §8 tier doctrine (amendment 2026-09-16) | The exact superseded scope paragraph, preserved verbatim |
| `coroutine-logic-scripts-2026-09-16.md` | `docs/roadmap/coroutine_opportunities.md` §1 | `logic` value FSM + Rule 12 | Gameplay-logic-as-coroutines recommendation, ruled out by edges-only law |
| `extensibility_and_modular_design.md` | `docs/arch/` | KDBA (Const. II §2.1/Rules 8.1/11, Const. III §3) | Tier 3 prescribed inheriting a base `System` class + registry one-liners — banned inheritance trees; Tiers 1–2 live on as prefab composition + Lua edges under pod law |
| `state_orchestration_architecture.md` | `docs/arch/` | KDBA gateway + orchestrator pods (Rules 8.1/11–12) | String-map blackboard + direct cross-system calls (`NavigationSystem.FindPath`, `PhysicsSystem.ApplyImpulse`) — banned cross-domain writes; coordination is gateway sagas with closed vocabularies |
| `event_system_architecture.md` | `docs/arch/` | Typed KDBA gateway + `EVENT_FLOW.md` catalog (Rule 8.1) | Callback immediate-events + untyped central bus ('shouts into the void') — the soup KDBA defeats; tag-component §3 idea survives in ECS chunk practice |
| [scripting_and_ecs_integration.md](scripting_and_ecs_integration.md) | `docs/arch/` | [Constitution III governing clarification](../spec/dod_ecs_architecture.md) and Constitution II §8 | Historical Lua/ECS vocabulary proposal; not implementation guidance |
| [ecs_architecture_and_scripting_roadmap.md](ecs_architecture_and_scripting_roadmap.md) | `docs/roadmap/` | [Constitution III governing clarification](../spec/dod_ecs_architecture.md) and Constitution II §8 | Superseded ECS-backbone plan; preserve rationale, not an active work track |
| [engine_domain_separation_migration.md](engine_domain_separation_migration.md) | `docs/backlog/` | Completed 2026-09-17 (steps 0-7 + same-day accelerated forwarder removal, `c9928d2`); governing law lives in the constitutions; active planning moved to [`constitution_enforcement_plan.md`](../backlog/constitution_enforcement_plan.md) | All seven phases COMPLETE with per-step evidence and the task-completion table snapshot; retained as the migration provenance record, not an active track |
| [kdba_kleisli_migration_plan.md](kdba_kleisli_migration_plan.md) | `docs/backlog/` | Constitution II §8 (the Kleisli doctrine is law); port evidence lives in [`kdba_conformance_backlog.md`](../backlog/kdba_conformance_backlog.md) Run A-C close-outs | K1.1 DoD artifact: the shared `Step`/gateway vocabulary and port order; historical migration context, superseded by the law it shipped |
| [domain_pod_hardening_backlog.md](domain_pod_hardening_backlog.md) | `docs/backlog/` | `kdba_conformance_backlog.md` (KDBA gateway migration directive, 2026-09-16) | FROZEN same-day per its own banner; open work rolled forward to the conformance backlog — kept as read-only provenance, never re-opened |



## Retention policy (2026-09-17)

- Archive by supersession, not age. Current constitutions remain in the active tree and must be internally consistent.
- Keep useful explanations with an explicit **non-normative** status; see the [education index](../education/README.md). Historical analogies do not authorize importing another programming model.
- Reassess unfinished work before archiving a backlog. The [KDBA backlog](../backlog/kdba_conformance_backlog.md) retains open tasks pending semantic review; obsolete signature-only mandates do not authorize a mechanical migration.
- Legacy implementation is migration context, not a design precedent. New work follows current KDBA law in C++23.
