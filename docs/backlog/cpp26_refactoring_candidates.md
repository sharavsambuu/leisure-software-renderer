# C++26 adoption refactoring candidates (standing reference)

> **Status: reference, not a plan.** Nothing here is sanctioned work; each
> item graduates to a proposal + owner ruling individually when its trigger
> holds (one item = one commit, per house convention). Version facts reflect
> the cppreference C++26 compiler-support snapshot checked 2026-09-17 (the
> same snapshot as the toolchain table in `cpp26_contract_guardrails.md`
> §7.1); **re-check versions at execution time** — they drift.

## 0. Trigger: what "widely adopted" means

This doc becomes actionable when **all** of the following hold (checked at
each docs sweep, alongside §7.1):

1. **GCC 16+ is the default dev/CI toolchain** — the 2026-09-17 baseline is
   GCC 13.3.0 (Ubuntu 24.04).
2. **Clang ships `__cpp_contracts`** — until then, every item carrying a
   mixed-toolchain caveat stays gated.
3. The C4.3/P4 native switch has been executed (runbook:
   `cpp26_native_switch_runbook.md`) — contracts refactor first; nothing
   here gates or delays it.

## 1. Candidates

### 1.1 Native contracts — already sanctioned, in the runbook

Do not refactor here independently — execute
`cpp26_native_switch_runbook.md`. In scope at GCC 16 (language + `<contracts>`
library together); Clang: none yet.

### 1.2 `std::expected` — candidate to supersede the house Result railway

House `core/result.hpp` predates `std::expected` (C++23, libstdc++ 12+).
Evaluation points when the trigger fires:

- **Benefit**: one less core primitive to maintain; monadic API
  (`and_then`/`transform`/`or_else`) already shaped like the house railway;
  future stdlib interop.
- **Risks**: the house Result may carry pmr-friendly error storage,
  message formatting, or ergonomic aliases the std type lacks; every gateway
  returning it is a P5 rim — behavior parity must be byte-verified via the
  replay-test pattern before any mass migration.
- **Verdict: pilot, don't mass-migrate.** Try one leaf module first; adopt
  broadly only if the std type covers 100% of the house API surface.

### 1.3 `std::flat_map` / `std::flat_set` — probably keep the house one

C++23 `<flat_map>` (libstdc++ 15+). The registry migration (`2cf4fea`) just
standardized on house `containers::FlatMap`, chosen for pmr-awareness and
its pointer-returning `find` — the stale-handle null rule and the
`ensure_transient_*` rims are built on it. A swap would rewrite pointer-API
code for iterator semantics with no behavioral win.
**Verdict: keep** unless the house container becomes a maintenance burden.
This entry exists so the question has a recorded answer.

### 1.4 Reflection (P2996 + `<meta>`) — highest-value candidate

GCC 16 (language); libstdc++ `<meta>` partial. Kills hand-written
pod/enum machinery:

- enum↔name mapping for closed enums (`contract_kind`,
  `PathSwapRejectionReason`, pass-id tables);
- pod field walking for the DVO/domain checkers (today text-scan gates +
  Python tooling);
- long-term, parts of `inventory_headers.py`'s *output* (though the
  build-level gate itself stays — it is textual by design).

**Verdict: highest-value candidate.** Pilot shape: replace one enum
to-string table with a reflection walk, twin it against the hand-written
version (same discipline as the contract test twins), then graduate.

### 1.5 Expansion statements (P1306) — fail-on-new-enum must survive

Pairs with 1.4: closed-enum arrows (`map_rejection`, `apply_runtime_toggle`,
P5 pins) could iterate the enum instead of hand-maintaining switch cases.
**Caveat:** the exhaustive switch is itself the P5 pin (new enumerator →
compile error). A rewrite that loses that property is a weakening, not a
refactor. GCC 16.

### 1.6 Constexpr exceptions (P3068) — script checks into compile time

GCC 16. Targets: transition-table completeness (currently runtime-asserted
in `render_path_compiler.hpp`), pass-id/placement tables → build-time
failures. Adopt per-table after reflection (1.4) lands — reflection is what
makes the checks writable.

### 1.7 Erroneous behavior / `[[indeterminate]]` (P2795/P3684)

GCC 16. Marks reads of never-written storage as erroneous (diagnosable)
instead of UB — relevant to flat-map pod storage and transient rim maps
where initialization is by convention. **Verdict: opportunistically,**
where an audit already found initialized-by-convention storage; no sweep.

### 1.8 Deliberately out of scope

- **`std::execution` (senders)** — the house model is wait-free rims with
  explicit lifetimes; senders are an architecture change, not a refactor.
- **`std::mdspan::at`**, parallel range algorithms, stdlib hardening
  (P3471) — toolchain-level wins that arrive with the baseline bump for
  free; no code motion proposed.
- **Modules** — no `import` rollout; the include-graph gate and header
  inventory are textual-include based and would need re-ratification.

## 2. Ground rules for graduating a candidate

1. §0's trigger holds; versions re-checked against the current cppreference
   snapshot.
2. Short proposal (expand this doc's entry) + owner ruling — the C0.3/C0.4
   shape.
3. Behavior-preserving, or it is not a refactor; the parity-twin pattern is
   the acceptance net.
4. No new third-party dependencies (ruling of 2026-09-17: stdlib + house).
5. Each adopted item: one commit, gates green, inventory regenerated when a
   header appears or disappears.

