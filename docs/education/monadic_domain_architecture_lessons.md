# Monadic Domain Architecture — Terms, Mappings & Gotchas

> Status: education (2026-09-16, monadic amendment). This file *explains*;
> the constitutions *legislate*. Single-source rule (Constitution II §2.2):
> where this file and a numbered rule disagree, the rule wins.
>
> Legislating sources: Constitution II (`docs/spec/value_oriented_programming.md`
> §2/§3/§6/§7/§8/§10/Appendix A), Constitution I §10
> (`docs/spec/conventions.md`), Constitution III §3/§6
> (`docs/spec/dod_ecs_architecture.md`), catalogs
> (`docs/pods/DOMAIN_GLOSSARY.md`, `docs/pods/EVENT_FLOW.md`).
> Living proof: `tests/vop_saga_tests.cpp` (Rule 12 spike).

## 1. Glossary of the new terms

| Term | One-line meaning | Legislated in |
| :--- | :--- | :--- |
| State monad | Gateway as pure transition `(S_old, A) -> (S_new, Events)` | §2.1, A.7 |
| Writer monad | Event/fact accumulation on the caller arena, both rails | §2, A.7 |
| Either / Result monad | `std::expected<T, E>`: value rail vs error rail | §8, A.7 |
| Kleisli arrow | One pipeline stage `Ctx -> expected<Ctx, Err>`, chained | §8, A.7 |
| Railway-oriented programming (ROP) | Linear `.and_then()` / `.transform()` / `.or_else()` chains instead of early-return ladders. **External alias only** — canonical term stays *Kleisli pipeline* / *flat railway composition* (§8 vocabulary alias) | §8 |
| Unit / Pure | Wrapping a raw value into the monad (`expected<T,E>{v}`) | §8 |
| Bind / FlatMap (`.and_then()`) | Chain a fallible stage; short-circuits to the error rail | §8 |
| Map / Functor (`.transform()`) | Infallible pure transform inside the container | §8 |
| Recovery / Catch (`.or_else()`) | Error-rail continuation: the compensator slot | §8, Rule 12 |
| Bounded context | Cohesive pipelines over shared PODs + one error/event language | Rule 11 |
| Ubiquitous language | The one error-enum family + event vocabulary a context speaks | Rule 11 |
| Saga | Multi-domain workflow with compensating (rollback) actions | Rule 12 |
| Compensator | `.or_else()` continuation consuming the fact log, never flags | Rule 12 |
| Orchestrator-is-a-pod | The saga coordinator is itself a Domain Pod (Core 4+1) | §2.1(5), §6.1 |
| Channel law | The signature tells you which channel: `expected` return vs `pmr::vector<Event>&` out-param | §8 |
| Granularity law (anti-monadic-tax) | Monad at chunk/batch/span level, never per hot-loop element | §7.1 |
| Vertical loop fusion | Chained stages inline into one register-resident pass | §7.1 |
| Linear ownership | In-place mutation ≙ purity iff the buffer is exclusively owned | §2.2(4) |
| Effect-describing values | Plans/commands/events as data; the shell interprets them | A.5 |
| Supervision (in miniature) | Bad input keeps state + emits a rejection fact (`PATH_SWAP_REJECTED`) | §6.4, A.6 |
| Wallet-leak shape | Compensating flagged stages while a prior debit leaks | Rule 12 |
| Flag archaeology | Reconstructing undo from ad hoc done-flags instead of facts | Rule 12 |

## 2. The monad mappings (the aha core)

- **Gateway ≅ State monad.** Old state + action span in, new state + events
  out. Deterministic, side-effect-free *decisions*; owned buffers may update
  in place under linear ownership (§2.2(4)) — purity is about decisions, not
  copying.
- **Event log ≅ Writer monad.** Facts accumulate alongside transitions on a
  caller-provided PMR arena. Crucially on *both* rails: failure must never
  destroy the log, which is why the log is an out-param, not a value-channel
  payload (A.7 divergence).
- **`expected` ≅ Either monad.** One value *or* one closed-enum error. The
  compiler enforces handling: unchecked `valid` flags become unrepresentable.
  Error payloads are closed enums — never `std::string` in pod vocabulary.
- **Stage ≅ Kleisli arrow.** Verifier (`and_then`), transformer (`and_then`),
  infallible finalize (`transform`), compensator (`or_else`). A saga reads
  top-to-bottom as one transaction story.
- **Orchestrator ≅ supervision.** Like an actor supervisor, a bad message
  does not crash the pod: state survives, failure is observable as an event.

## 3. The channel law — how to read any signature

```
expected<Plan, ClosedEnum>          → value/error channel: compose monadically
void f(State&, span<const Command>,
       Context, pmr::vector<Event>&) → command/event stream: variant vocabulary
```

Two channels, side by side, never merged. The bundled signature
`expected<(State, Events), Error>` was considered and rejected (A.7):
per-command gateways emit a *data-dependent event count* (N facts per
command, conditional 0–2 emissions, silent pods) that a single-value channel
cannot express. Merging forces failure to eat the log or smuggle it through
the error rail — the exact leak §4 warns about.

## 4. Boundaries as pipeline combinations (the boundary aha)

A domain boundary is not a directory — it is a *pipeline combination* over
shared PODs speaking one language:

- Same PODs + different stage chains = different boundaries. `renderpath`
  recipes compile through validation chains in the rendering context, while
  `input`/`logic` compose tokenizer→FSM chains in the session context.
- Inside one context, stages compose *synchronously* (Kleisli chains).
- Across contexts, only *events* cross, through the shell (Rule 8.1).
- Current contexts are tabled in `DOMAIN_GLOSSARY.md` §8; a new context is
  born by declaring its POD set + error/event language, not by creating a
  directory.

## 5. C++23 field guide — what lives where

| Feature | Tier | Why |
| :--- | :--- | :--- |
| `std::expected` + `and_then`/`transform`/`or_else`/`transform_error` | planners, compilers, loaders, bridges, sagas | honest failure types, linear chains |
| Monadic `std::optional` | edge bridges (`map_action_to_*`) | compose fallible lookups, no get-if noise |
| `std::format` + enum formatters | diagnostics/logging | no more `static_cast<unsigned>` casts |
| `std::ranges` | planners only, PMR-backed | allocation must stay explicit |
| `std::mdspan` | tile kernels (future) | the vocabulary for Morton-swizzled tiles |
| Coroutines | edges only | never inside a gateway body |
| Concepts | API rims | never constraining gateway bodies |

Baseline: every tree is `cxx_std_23` since 2026-09-16 (Constitution I §10, L2 closed). `tl::expected` is the documented fallback where the toolchain
lags — same shape, same laws.

## 6. Gotchas — each bought with a real failure

1. **The wallet leak.** A saga whose error rail carries the whole context
   still leaks: the compensator restored flagged stages while a $100 wallet
   debit passed through, and the audit printed `ROLLED_BACK` over a charged
   wallet. Root cause: undo reconstructed from flags, not facts. Law: the
   compensator consumes the fact log (Rule 12). Pinned by
   `tests/vop_saga_tests.cpp` (RED then GREEN).
2. **The bundled-signature trap.** Looks cleaner, destroys auditability —
   see §3. Smell: an error type that carries the entire context so the
   compensator can "read" it. That is the log begging to be a first-class
   channel.
3. **The monadic tax.** `std::vector<std::expected<...>>` in a hot loop:
   discriminant padding breaks 64-byte alignment, kills auto-vectorization.
   Monad at the chunk; stream flat arrays inside. Linter FAILs it.
4. **Stringly events.** `"Stock reserved successfully."` cannot be asserted
   without string comparison and rots determinism gates. Facts are
   enums/ids/quantities; prose renders at the edge from name tables.
5. **The `par_unseq` sidestep.** Surrenders pinning, scheduler, and arena
   awareness to the implementation — a step sideways off the Virtual SPU
   trajectory. Explicit chunked workers with exclusive spans instead.
6. **The `memcpy` mirage.** Snapshots-as-`memcpy` holds for toy structs, not
   for states with vectors, arenas, and generational handles — there it is a
   corruption vector. Value-copy + replay logs instead.
7. **Flag archaeology.** `inventory_reserved` + `processed_items_count` as
   undo state is the shape `Context::forward_plus` already deleted once.
   Every compensated mutation needs a fact row *before* the saga ships
   (EVENT_FLOW.md saga requirements).
8. **Validate-before-mutate beats compensate.** The cheapest rollback is the
   one never needed: check funds/stock for the whole batch before mutating
   anything (the planner/executor split applied inside the saga).

## 7. What this cures in ECS gameplay (retained diagnosis)

- **Tag-component hell:** a 3-step attack smeared across 5 systems/frames
  becomes one state machine + one deterministic transition in one file.
- **Multi-entity transactions:** saga + `.or_else()` rolls back in-frame —
  *provided* the compensator consumes facts (gotcha 1 is the price of
  forgetting).
- **Rollback netcode / save-load:** pure gateways + value snapshots + command
  logs make re-simulation a function call, not archetype-pool surgery.
- **Smeared logic:** one pipeline tells the whole story
  (`check → deduct → apply → emit`), readable in seconds.
- **Honest limits (not cured):** spatial queries still want trees (the Jolt
  bridge concedes exactly this); runtime-open archetypes still want explicit
  variants (the closed-vocabulary law answers it, at the cost of wildness).

## 8. Where to go next

- Write a fallible planner? Read §8 tiers + the `try_compile` precedent
  (`renderpath.gateway.hpp` detail namespace).
- Cross two contexts? Read Rules 11–12, then the saga spike test.
- Touch a hot loop? Read §7.1 granularity + the chunk-sizing rules.
- Name something new? Read §6.5 vocabulary law, then the glossary.
- Unsure which channel? Re-read §3 above, then A.7 for the divergence proof.
