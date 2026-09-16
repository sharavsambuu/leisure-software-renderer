# Superseded: gameplay logic as coroutine scripts (2026-09-16 ruling)

> Provenance: extracted verbatim from
> `docs/roadmap/coroutine_opportunities.md` §1. Ruled out by the monadic
> amendment: gameplay state machines are table-driven value FSMs (`logic`
> pod precedent, Constitution II Rule 12); coroutines live on execution
> edges only (Constitution I §10). Kept so the reasoning is never lost.

## 1. Logic → Coroutine Scripts
Replace the callback-based `StateMachine` with sequential coroutine scripts for "Patrol -> Chase -> Attack" logic.
