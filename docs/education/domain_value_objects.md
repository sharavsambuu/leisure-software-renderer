# Domain Value Objects — why "Domain POD" retired (teaching, 2026-09-17)

> **Explanatory, not normative.** The law lives in the terminology annex
> [docs/spec/domain_value_object_law.md](../spec/domain_value_object_law.md);
> this file teaches it. Companion teaching:
> [cpp26_contract_guardrails.md](cpp26_contract_guardrails.md) (contracts over
> DVOs). Style follows `kdba_kleisli_composition.md`: explain properties, cite
> the law, never edit history.

## 1. How the term got here — the honest history

"Domain POD" was coined in the reducer era, when the project's core idea was
"domain data as plain structs, transformed by pure reducer functions". It was an
honest name at the time. Then the architecture outgrew it:

1. **Pure reducer switch-cases became nesting hell** — one function owning the
   Cartesian product of state shapes × actions × failure modes. This is the
   exact problem monadic pipelines were invented for, and the project adopted
   them as KDBA (Kleisli gateways, §8, Rules 8–12).
2. **Names lagged architecture.** `reduce_*` → `*_gateway` was fixed by the
   Pod Identifier Law (§6.6). But "Domain POD" survived in prose, still carrying
   two meanings at once: the *C++* meaning (trivial + standard-layout layout
   guarantee) and the *project* meaning ("our domain data type"). A word can't
   legally mean both, and the C++ one died in C++20 (`std::is_pod` deprecated).
3. **The name obscured the one property that matters most.** A POD is an
   invariant-free bag of bytes — any bit pattern is valid. The architecture's
   whole point is the opposite: data that is *always valid at module edges*,
   enforced by gateways, with zero phantom flags. "Value object" states that
   property in the term itself.

## 2. What the old name promised vs what the architecture is

| The old name promised | What the architecture actually is | KDBA axiom it obscured |
| :--- | :--- | :--- |
| "POD" — any bit pattern is a valid value | validate-before-mutate; invalid ⇒ keep + reject | gateway-enforced invariants at edges |
| "Plain" — no behavior, no law | zero phantom flags; state expressed as structure | Rule 1, §2.1 layout honesty |
| "Domain" — a folder of structs | membership in a bounded context with one ubiquitous language | Rule 11 |

This is the same lesson the Pod Identifier Law annex already recorded for
`reduce_*`: *the name is the repository's navigation surface, and a lying name
is a defect, not a cosmetic issue.*


---

## 3. What a Domain Value Object is

> **Definition.** A **Domain Value Object (DVO)** is a plain, identity-free
> data value that belongs to exactly one bounded context and is always valid
> at every module edge.

Concretely, in this codebase:

- **Home**: Core 1 of a pod — `<Pod>State` / `<Pod>Snapshot` in `<pod>.contract.hpp`.
- **Always valid**: a DVO is produced only by the owning gateway's
  validate-before-mutate step. There is no "half-valid" DVO, and no phantom
  flag apologizing for one (`is_pending`, `needs_rebuild` are banned shapes).
- **No identity**: DVOs are compared and copied by value. Identity (stable IDs,
  generational handles) lives *outside*, in registries owned by the context.
- **Plain aggregate**: trivially copyable, memcpy/persist-friendly, no
  validating constructors *required*. A DVO is not an "invariant type" — the
  invariants are context law enforced by the gateway, not type law enforced by
  a constructor. (This is why C++26 contracts fit DVOs so well — see the
  companion doc: they let invariants live at edges, where this architecture
  already put them.)

The table version of this definition (with the law each property serves) is
Part 2 of the terminology annex.

---

## 4. The three-role model — values, entities, gateways

The rename completes a three-role model that migration step 4 (state/context
ownership) needs:

| Role | What it is | Example |
| :--- | :--- | :--- |
| **Domain Value Object** | stateless, identity-free, always-valid data | `RenderPathRecipe`, scene item values |
| **Entity** | stable identity + authoritative owner + lifecycle | a scene object = stable ID + its owned DVOs |
| **Gateway** | the pure boundary that enforces context invariants over DVOs on behalf of entities | `renderpath_gateway` |

Rule of thumb: *values are data, entities own identity, gateways enforce
invariants.* When step 4 asks "who is the authoritative owner for camera and
render settings?", this model is the answer shape: the DVO is the state, the
entity is (ID + owner + lifecycle), and the gateway is the only writer.


---

## 5. Bounded contexts — why "domain" only means something inside one

A value object outside a bounded context is just a struct. What makes it
*domain* is the context around it (Rule 11): a suite of cohesive Kleisli
pipelines over a shared set of DVOs, bound by **one ubiquitous language** — one
error-enum family, one event vocabulary, one command vocabulary
(`<Pod>Command = variant<...Intent>`, `<Pod>Event = variant<...Fact>`).

Current contexts (glossary §8):

| Context | DVOs | Error / event language |
| :--- | :--- | :--- |
| Rendering | `renderpath`, `frame`, `geometry`, `lighting`, `camera`, `resources`, `sky`, `scene`, `gfx` | `PathSwapRejectionReason` + per-pod `*Event` (`EVENT_FLOW.md`) |
| Session flow | `input`, `logic` | `InputEvent`, `FsmEvent` (+ rejections as facts) |

Why the separation matters (the four payoffs):

1. **Independent evolution** — the rendering context can redesign its plan
   compilation without the session-flow context ever knowing; boundaries are
   the only coupling surface.
2. **One failure language per context** — every rejection is a typed value
   (`PathSwapRejectionReason`), every fact is a variant alternative; saga
   compensation (Rule 12) consumes receipts, never gossip.
3. **No cross-boundary DVO mutation** — Commands in, Events out (Rule 8.1).
   The DVO belongs to the context; foreign code may hold copies, never authority.
4. **Invariants are context law, not type law** — the same data shape in two
   contexts can carry different invariants, because the *gateway* enforces
   them. This is what makes the word "Domain" in "Domain Value Object" mean
   something: the law comes from the context, not the struct.

---

## 6. What changes day-to-day

The terminology annex (Part 5) is the law; the practical digest:

- **Say and write "Domain Value Object" / "DVO"** in prose, comments, commit
  messages, and docs. Introducing "Domain POD" in new or edited live text is a
  review-blocking defect.
- **Archives are never rewritten** (`kdba_history/`, the FROZEN hardening
  backlog, `docs/outdated/`). Read them with the substitution in mind.
- **Live docs migrate at next edit** — no mass rewrite; the live-doc match
  count for "domain pod" is shrink-only (Rule 15 discipline).
- **Code symbols are out of scope** — `Core 4` file names and structural
  "pod" wording stay until a separate, gated ruling.

Review checklist for docs PRs:

1. Does new prose say DVO (not "Domain POD")?
2. Are archive files untouched?
3. If the doc states an invariant, is it attributed to the *gateway/context*,
   not to the type?

---

## 7. Reading order

1. `docs/spec/domain_value_object_law.md` — the terminology annex (law).
2. `docs/education/cpp26_contract_guardrails.md` — how contracts enforce
   gateway invariants over DVOs, and the C++23 emulation bridge.
3. `docs/spec/pod_identifier_law.md` — the previous vocabulary repair
   (`reduce_*` → gateway), for the pattern this rename follows.
4. `docs/education/kdba_kleisli_composition.md` — the pipeline primer.

