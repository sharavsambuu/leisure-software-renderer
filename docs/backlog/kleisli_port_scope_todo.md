# Kleisli port scope — todo list

> **Status: closed 2026-09-19** (KP-0..KP-6 executed; the three ports resolved
> as demand-driven deferrals — §1a). This file re-plans ROP-3.1
> ([`rop_hardening_todo.md`](rop_hardening_todo.md) §4) after a vocabulary audit of
> the six "missing" ports. It is schedule, not law (Constitution II §2.2) and owns
> no item canonically: the per-port DoD stays in ROP-3.1, the `Step` concept stays
> in ROP-3.2, and this file does not tick either on its own behalf — each item
> below closes with its own DoD, and owner-doc census rows update in the same
> commit. It applies the K1.4/ROP-3.3 precedent (empty `variant<monostate>`
> vocabularies are legal; a stub discards no real signal) as a *re-scoring*, not a
> doctrine change: the two-tier rule — value-carrying pass outcomes at hot paths,
> full railway only at gateway rims — is untouched (Constitution II §8).

## 1. What the audit found (2026-09-19)

The ROP-3 census counts vocabulary *files*; the code under them is shells. All six
"missing" domains carry `.command` / `.event` triples whose variants are
`std::monostate`-only with **zero consumers** (grep across `include/shs`,
2026-09-19; `gfx` lives under `render/targets/`, not top level):

| Domain | Vocabulary files | Alternatives | Consumers | Verdict |
|---|---|---|---|---|
| `geometry` | `geometry.command.hpp` / `.event.hpp` | `variant<monostate>` | 0 | **No rim** — pure value-lib (`aabb.hpp`, `convex_cell.hpp`, culling queries); wrapping infallible math in `expected` is the ceremony the doctrine forbids |
| `lighting` | `lighting.command.hpp` / `.event.hpp` | `variant<monostate>` | 0 | **Stale census row** — gateway already retired at step 4.5 per `pod_identifier_law.md` ("pure shading terms remain") |
| `sky` | `sky.command.hpp` / `.event.hpp` | `variant<monostate>` | 0 | **Defer** — generation is pure and infallible in shape; port only if a backend sky-realization edge appears (then it is a mini-`gfx`) |
| `scene` | `scene.command.hpp` / `.event.hpp` | `variant<monostate>` | 0 | **Port** — identity/binding seams (`scene_identity.hpp`, `scene_bindings.hpp`) produce real known-answer refusals |
| `resources` | `resources.command.hpp` / `.event.hpp` | `variant<monostate>` | 0 | **Port** — textbook fallible rim: file I/O under `loaders/`; failure is currently reported by non-railway means (zero `expected` in `resources/`) |
| `gfx` | `render/targets/gfx.command.hpp` / `.event.hpp` | `variant<monostate>` | 0 | **Port — first** — the rhi value layer already speaks `expected` (`VulkanSubmitFailure`, `BackendNotRealized`); the rim names facts that already exist |

Truthful census: **5 landed rims + 3 genuine ports + 2 no-rim rulings + 1 stale
row** — not "5 of 11". This is the ROP-4 lesson again: a census that counts rows
can drift from substance; correct the census before executing against it.

### 1a. Rim rulings (KP-2, 2026-09-19)

The audit sharpened during execution: the shells are not merely empty — they are
**empty by law**, and each shell's own `static_assert` names the arrival path
("land a new intent as a named `apply_*` arrow behind a real gateway first").
With the K1.1 banner's directive (no signature-only ports, no invented vacuous
errors), the rim program resolves per domain as:

| Domain | Rim today | Refusal vocabulary when a rim fires | Demand trigger |
|---|---|---|---|
| `gfx` | none (lawful §6.1 shells) | reuse the ruled rhi value-layer failures (`VulkanSubmitFailure`, `BackendNotRealized`) — the facts already exist | first real alloc/command intent of the R5b registry-edge migration (named in `gfx.command.hpp`) |
| `resources` | none | missing asset / malformed-magic / unsupported version over `loaders/` file I/O | first loader rim consumer as R5b moves the stores to the edge (named in `resources.event.hpp`) |
| `scene` | none | unknown identity / unbound binding / missing element (`scene_identity.hpp`, `scene_bindings.hpp` known-answer refusals) | first binding intent as the R5b store migration supplies it |
| `geometry` | none, stays none | n/a — no trust boundary in evidence (pure value leaf, module-class amendment §6.1) | none; a fallible build/upload edge gets its own ruling |
| `sky` | none | mini-`gfx` shape if a backend sky-realization edge appears | named; not vague |
| `lighting` | retired (step 4.5) | n/a per `pod_identifier_law.md` | none — do not re-open |

**The three port items below are therefore DEFERRED BY RULING, not abandoned:**
they re-open only when their trigger fires, and then ROP-3.1's per-domain DoD +
the `StepShape` pin govern the landing. Inventing an intent today to have
something to port is exactly the ceremony the doctrine and the K1.1 banner
forbid.

## 2. The todos

Order matters: census first (cheap, restores honest reporting), concept second
(precondition for uniform ports), rulings third (a gateway cannot be copied onto
an empty vocabulary), then one commit per port, highest value first:
`gfx` → `resources` → `scene`.

**Outcome (2026-09-19):** executed as written through KP-2 — which resolved the
ports as demand-driven deferrals (§1a). The program closes with the census
corrected and the concept landed; the three port items stay open conditional on
their triggers.

- [x] **KP-0 Census correction (docs-only, commit 1)** — update the ROP-3 census in
      [`rop_hardening_todo.md`](rop_hardening_todo.md) §4 and the ROP-3 row in
      [`remaining_todos_2026-09-18.md`](remaining_todos_2026-09-18.md) Bucket A from
      "5 of 11" to the §1 census, with the audit evidence inline; fix the
      `lighting` row against `pod_identifier_law.md`'s step-4.5 retirement;
      annotate the K1.1 order in
      [`kdba_conformance_backlog.md`](kdba_conformance_backlog.md) as re-planned
      here (inline dispute, never a silent rewrite). DoD: both owner docs amended
      in the same commit; no code touched; `check_doc_paths.py` and
      `check_kdba_boundaries.sh` green. — **DONE 2026-09-19:** owner blockquote
      upgraded to the executed re-scope (ROP-3.1 §4), tracker row extended, K1.1a
      port-order amendment inserted (open, not silent), DOMAIN_GLOSSARY.md §7
      rim-census amendment added (5 rims + 6 lawful-empty); doc-paths +
      kdba-boundary gates green.
- [x] **KP-1 Land the R3 `Step` concept (ROP-3.2, precondition)** — concept +
      `static_assert` at each existing step + negative fixture proving an
      off-shape type is rejected, zero renames. Owned by ROP-3.2; this program
      consumes it so the three ports copy one *enforced* shape instead of
      drifting into three dialects. DoD: ROP-3.2's own. KP-3..KP-5 stay closed
      until it lands. — **DONE 2026-09-19:** `shs::core::StepShape` landed in
      `shs/core/step_shape.hpp`; five definition-site pins (FsmStep, InputStep,
      FrameStep, RenderPathStep, orchestrator's reused InputStep); negative
      fixture = six `static_assert(!StepShape<…>)` rejections + positive pins in
      `tests/core_tests.cpp` (CTest: `shs_renderer_tests` green); zero renames;
      gates green (gateway-rails + twin, include-graph, contract-placement).
- [x] **KP-2 Per-domain vocabulary rulings (docs, commit 2)** — record, per
      candidate, the refusal facts the rim will carry: `gfx` (submission /
      realization refusals, reusing the ruled rhi failure names), `resources`
      (missing asset, malformed/magic mismatch, unsupported version), `scene`
      (unknown identity, unbound binding, missing element); and the exemptions
      with their triggers: `geometry` (N/A — no trust boundary in evidence),
      `sky` (defer — port when a backend sky-realization edge exists), `lighting`
      (stays retired per pod law). DoD: ruling table appended to §1 here;
      `DOMAIN_GLOSSARY.md` rows consistent; docs gates green. — **DONE
      2026-09-19:** ruling table appended as §1a here; the DOMAIN_GLOSSARY.md §7
      rim-census amendment (KP-0) keeps the pod rows consistent; doc-paths +
      kdba-boundary gates green.
- [ ] **KP-3 `gfx` port (one commit)** — gateway rim per ROP-3.1's DoD: gateway +
      kit-extended tests (replay + empty-log + value-equality `operator==` on
      state/events, proving the signature swap is behavior-neutral) + glossary
      rows + `check_gateway_rails.sh` green. Closest to the renderpath pilot —
      the proven shape applies almost verbatim. — **DEFERRED BY RULING (KP-2,
      2026-09-19):** the `gfx` vocabulary is empty by law (§6.1;
      `gfx.command.hpp`'s own `static_assert` names the R5b registry-edge
      migration as the intents' arrival path). There is no rim to port until a
      real intent exists; re-opens on that trigger under ROP-3.1's DoD + the
      `StepShape` pin.
- [ ] **KP-4 `resources` port (one commit)** — same DoD as KP-3;
      `loaders/primitive_import.hpp` is the first rim consumer, and the KP-2
      refusal vocabulary becomes the kit's negative cases. — **DEFERRED BY
      RULING (KP-2, 2026-09-19):** empty by law (§6.1); re-opens when the R5b
      store migration supplies a real loader rim intent.
- [ ] **KP-5 `scene` port (one commit)** — same DoD as KP-3; the refusal facts
      are known-answer tests (unknown instance id, unbound material, missing
      element). — **DEFERRED BY RULING (KP-2, 2026-09-19):** the `scene`
      vocabulary is empty by law (§6.1); re-opens when the R5b store migration
      supplies a real binding intent.
- [x] **KP-6 Close-out census (final commit)** — ROP-3.1 ticks (as re-scoped by
      KP-0) only when 5 landed + 3 ports are resolved rim-or-exempt; update the
      tracker row in the same commit (tracker law: tick here only with the owner
      doc). — **DONE 2026-09-19 (close-out):** ROP-3.1 ticks CLOSED-RESCOPED
      (owner blockquote extended) and the tracker row ticks in the same commit;
      the §9 sweep rows and the §7 R3 "landed" cell carry inline 2026-09-19
      updates; the three ports are resolved as demand-driven deferrals (§1a).
      Final verification: build + CTest + gateway-rails (+twin), include-graph,
      contract-placement, kdba-boundary, doc-paths all green; inventory
      regenerated same commit (Rule 15).

## 3. Not doing (recorded so a later sweep does not re-litigate)

- **`geometry` — no port.** Pure leaf, no effects; an empty result is not a
  refusal. If a fallible build/upload edge ever appears, it gets its own ruling,
  not a retroactive rim.
- **`sky` — no port today.** The defer trigger is named in KP-2 so the exemption
  is checkable, not vague.
- **`lighting` — stays retired** per `pod_identifier_law.md` step 4.5; the census
  row was stale, the state was not.
- **`camera` — stays closed** (ROP-3.3). Do not re-open.
- **No railways in hot loops, no blanket monads** — unchanged (Constitution II
  §8); `PassOutcome` remains the hot-path tier.

## 4. Verification (per commit)

- Full `build/` configure + CTest green.
- `check_kdba_boundaries.sh` green.
- After each port: `check_gateway_rails.sh` + its negative twin green.
- Docs edits: `check_doc_paths.py` + its negative twin green (ROP-4 gate).
- A touched header regenerates the inventory in the same commit (Rule 15).

## 5. Provenance

- Audit: 2026-09-19 — vocabulary grep over `include/shs` (all six triples
  `std::variant<std::monostate>`, zero consumers); loaders surface
  (`resources/loaders/primitive_import.hpp`) and rhi `expected` sites
  (`VulkanSubmitFailure`, `BackendNotRealized`) re-checked the same day.
- Precedents applied: K1.4 camera resolution and ROP-3.3 (empty vocabularies are
  legal; no obligation without a non-empty rim), ROP-4 (census vs substance).
- Supersedes: the K1.1 remaining port order (`geometry → lighting → sky → scene →
  resources → gfx`) as a *plan*, pending KP-0. ROP-3.1's per-domain DoD is
  unchanged and remains the authority for what "ported" means.
