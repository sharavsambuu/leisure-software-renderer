# Domain Value Object Law — Terminology Annex to Constitution II

> Status: **normative annex (2026-09-17)**. Companion to
> [Constitution II §2.1](value_oriented_programming.md) ("The KDBA Domain Pod
> Constitution") and [§6.6](value_oriented_programming.md) ("Pod Identifier Law").
> Precedence: §2.1 + §2.2 govern; this annex legislates **terminology only** — it
> changes no include direction, no gate, no code symbol. Where older prose says
> "Domain POD", this annex states what that text means and what replaces it.
> Teaching companion (explanatory): `docs/education/domain_value_objects.md`.

---

## Part 0 — The one-sentence answer

**The term "Domain POD" is retired. The official term is "Domain Value Object"
(DVO).** New prose, comments, commit messages, gate strings and docs must use
"Domain Value Object" or "DVO". "Domain POD" survives only in frozen archives
(Part 5). The architecture is unchanged — this annex renames the word, not the law.

---

## Part 1 — Why the term had to change

### 1.1 "POD" is a dead C++ term that never meant this

"POD" (Plain Old Data) has a precise, **deprecated-in-C++20** technical meaning:
`std::is_pod` — trivially copyable *and* standard-layout. It is a statement about
binary layout, not about domain modeling. Every C++ programmer reading
"Domain POD" imports the layout meaning, which was never what SHS meant. A term
whose standard definition died in C++20 cannot carry a living architecture.

### 1.2 "POD" describes the opposite of what the architecture is

A POD is an *invariant-free bag of bytes* — any bit pattern is a valid value. The
KDBA philosophy is the opposite at every layer:

- Core 1 state is **validated before it is committed** — a rejected snapshot is
  kept-and-rejected, never half-mutated (invalid ⇒ keep + reject).
- Persistent DVOs carry **zero phantom flags** ("is_pending", "needs_rebuild") —
  state is expressed as structure, never as a boolean apology.
- Gateways are the single mutation boundary per pod (Rule 10); invariants hold
  **at module edges**, not by caller discipline.

A name that says "invariant-free" for a design whose entire point is *enforced
invariants at boundaries* will mislead every new reader. Names are design
pressure: call a thing an invariant-free bag and the next contributor will
treat it as one.

### 1.3 The DDD reading was already correct — it just wasn't named

In DDD terms, what SHS called a "Domain POD" is exactly a **value object**:
plain data, no identity, compared and copied by value, always valid. The old
term obscured the one property that matters most — validity — while the new
term states it in the name itself.

---

## Part 2 — Definition

A **Domain Value Object** is a plain, identity-free data value that belongs to
exactly one bounded context and is always valid at every module edge:

| Property | Law it serves | Reality in the tree |
| :--- | :--- | :--- |
| Plain aggregate, trivially copyable | Constitution I; stays memcpy/persist-friendly | Core 1 `<Pod>State`/`<Pod>Snapshot` in `<pod>.contract.hpp` |
| No identity of its own | identity lives in registries / stable IDs | `LogicContext<TStateId>`; scene/resource identity policy (migration step 4) |
| Valid at every edge — no phantom flags | validate-before-mutate | compile → validate → commit-or-reject in every gateway |
| Value semantics | Rule 1 (explicit structs by value) | all planning/query APIs |
| Transitions only through the owning gateway | Rules 8.1, 10 | Core 4 `<pod>_gateway`, one per pod |
| Tier discipline | Rule 5.1 | transient hot-state vs persistent snapshots |

A DVO is **not**: a C++ `is_pod` type (different, deprecated meaning), a reducer
store, an entity, or an invariant-carrying class. Validating constructors are
*not* required — invariants deliberately live at **edges**, not inside the type
(rationale in the education companion).

---

## Part 3 — The three-role model (DVO / entity / gateway)

The rename completes a model that migration step 4 needs. Three roles, three layers:

| Role | What it is | Where it lives | Example |
| :--- | :--- | :--- | :--- |
| **Domain Value Object** | stateless, identity-free, always-valid data | `<pod>.contract.hpp` Core 1 | `RenderPathRecipe`, scene item values |
| **Entity** | stable identity + authoritative owner + lifecycle | registries / identity policy (step 4 rulings) | a scene object = stable ID + its owned DVOs |
| **Gateway** | the pure boundary enforcing context invariants over DVOs on behalf of entities | `<pod>.gateway.hpp` Core 4 | `renderpath_gateway` |

Rule of thumb: *values are data, entities own identity, gateways enforce
invariants.* C++26 contracts (`docs/education/cpp26_contract_guardrails.md`) are
the sanctioned enforcement mechanism of the third role — expressed at module
edges over DVOs, never baked into the DVO types themselves. This is now law:
Constitution II Rule 17 ("Contract Guardrails at Module Edges", 2026-09-17).



---

## Part 4 — Why the term only means something inside a bounded context

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

Consequences the term now makes explicit:

1. **No cross-boundary DVO mutation** — Commands in, Events out (Rule 8.1).
   The DVO belongs to the context; foreign code may hold copies, never authority.
2. **The ubiquitous language names the DVOs** — §6.5/§6.6 keep type names
   greppable against the law; "DVO" inherits that findability rule.
3. **Invariants are context law, not type law** — the same data shape in two
   contexts can carry different invariants, because the *gateway* enforces them.

---

## Part 5 — Terminology migration rules (normative)

- **T1 — New text.** New prose, comments, commit messages, doc files and gate
  strings use "Domain Value Object" / "DVO". Introducing "Domain POD" in new or
  edited live text is a review-blocking defect (Rule N2 findability spirit: the
  vocabulary must grep to one term).
- **T2 — Archives are never rewritten.** Frozen/history documents keep the old
  term byte-identical (Rule N5). Known occurrences at adoption (124 total,
  2026-09-17): `docs/education/kdba_history/*`,
  `docs/outdated/domain_pod_hardening_backlog.md` (FROZEN banner; archived 2026-09-17),
  `docs/outdated/*`, `docs/education/monadic_domain_architecture_lessons.md`,
  demo `docs/pods/*` snapshots. Reading them: mentally substitute
  "Domain Value Object"; do not edit. (The terminology-defining documents
  created with this annex — this file and its teaching companions — quote the
  retired term deliberately, to define the retirement; they are part of the
  new baseline, not violations.)
- **T3 — Live normative docs migrate at next edit**, not by mass rewrite. Each
  live doc still saying "Domain POD" adopts the new term in the same commit as
  its next substantive change; `docs/pods/DOMAIN_GLOSSARY.md` carries the
  standing amendment note at header level.
- **T4 — Shrink-only.** The count of "domain pod" matches in *live* (non-frozen,
  non-terminology) docs is a closed, shrink-only set (Rule 15 discipline).
  Track with:
  `grep -ri "domain pod" --include='*.md' . | grep -v kdba_history | grep -v domain_pod_hardening_backlog | grep -v domain_value_object_law.md`
- **T5 — Code symbols are out of scope.** This annex renames *prose*, not
  identifiers. A future symbol-level rename (e.g. re-homing "pod" wording in
  file names or banners) is a separate, gated decision — never bundled with
  documentation edits.
- **T6 — Legacy alias (recognized, never re-legitimized).** In conversation —
  human speech, commit messages, AI-agent prompts — the retired term is a
  recognized alias: when someone says "Domain POD", they always mean the
  Domain Value Object, and agents/reviewers shall respond with the DVO concept
  and redirect to the current term (T1). This clause exists so accidental use
  never breaks understanding; it does *not* permit the old term in new written
  live text.

---

## Part 6 — What this annex does *not* change

- No header moves, gate edits, or inventory changes (Rule 15 untouched).
- No change to Core 4 structure, §6.1/§6.2 suffix laws, or the §6.6 identifier
  law — "pod" as a *structural* code noun remains until a separate ruling under T5.
- No change to bounded-context membership (glossary §8 stays authoritative).
