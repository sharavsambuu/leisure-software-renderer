# KDBA Kleisli Composition — Primer

> Education: constitutions legislate (Const. II §2.2 wins all ties).

Laws live in Constitution II §2.1/§3/§6/§8/§10/A.7, I §10, `DOMAIN_GLOSSARY.md` §8, `EVENT_FLOW.md`. This file teaches the shape; verbatim sources in `kdba_history/`.

## 1. Primitive

`A -> expected<B, DomainError>`: 2–5 lines, pure, `constexpr`-friendly. Flat chains: `.and_then()` bind, `.transform()` map, `.or_else()` compensator, `.transform_error()` boundary translation. House signature: `(State, span<Action>, dt) -> expected<Step{Next, Events}, Err>`. No switch-case monoliths.

## 2. Axioms / laws / dimensions (one line each, full text in §2.1/§3/§8)

Arrows sole unit; PODs passive (contracts method-free, arrows in `reducer.hpp`); transient `SagaContext` vs persistent invariants (zero phantom flags); sealed single-writer boundaries (Commands in, Events out). Five bans: monoliths, phantoms, exceptions, cross-domain writes, in-arrow side effects. Seven dims: state/logic/flow/writes/consistency/layout/time — never mixed.

## 3. C++23 notes

Native `expected` monadics (P2505R5). `DomainError` stays `enum class : uint8_t`, no `std::string` in `E` (~16B register boundary). Named `constexpr` free functions over lambda chains; `static_assert` invariant tests; concepts on API rims only (I §10); no deducing-`this` in contracts. Small non-virtual arrows inline away; success rail hot, error rail cold.

## 4. Guardrails

Chunk/batch-level monads, never per-element (`vector<expected>` FAILs, §7.1). KDBA for transactions/sagas/ingress; flat loops for 60Hz math. Boring 5-second functions only.

## 5. Gotchas

No in-place mutation (return `Step`, boundary commits). No per-entity `TExpected` in Mass chunks. Cross-domain `and_then` only inside orchestrator pods. Failure keeps state + materializes rejection; validate-before-mutate. Verse `<transacts>` is memory-only — still emit facts.
