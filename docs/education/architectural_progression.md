# Architectural Progression — how I personally got here (history, 2026-09-17)

> **Explanatory, not normative.** This is a personal history, not law. The law
> lives in the constitutions; the founding documents live hashed in
> [kdba_history/](kdba_history/). This file is the human-readable story of *how
> the ideas were found* — the problems, the aha moments, and the (sometimes
> surprising) moment each home-grown concept turned out to already have a name
> in C++ or in the literature. Companion reading:
> [kdba_kleisli_composition.md](kdba_kleisli_composition.md) (the primer),
> [domain_value_objects.md](domain_value_objects.md),
> [cpp26_contract_guardrails.md](cpp26_contract_guardrails.md).

## 0. The preface — the C purist I used to be

An honest starting point: for most of this project's life I thought the new
C++ features were a bloated cool-kids thing. Classic ANSI C, maybe 1990s C++ —
that was "real" C++ to me. A software renderer seemed to prove it: the GPU
cares about memory and triangles, not about your abstractions. Concepts,
`expected`, variant visits, allocators — noise, overhead, committee fashion.

What changed my mind was not a feature. It was the **concept of the domain**.
The moment I started asking "what does this code *mean*?" instead of "how fast
does this loop run?", every modern feature stopped looking like bloat and
started looking like an answer to a problem I had been re-solving by hand:

- I hand-rolled error codes and out-parameter dances → that was
  `std::expected`'s railway, waiting.
- I wrote tagged unions and dispatch switches by hand → that was
  `std::variant` + `std::visit`, my command/event vocabulary spelled by the
  committee.
- I passed raw pointers with a length and a prayer at every edge → that was
  `std::span`, my edge discipline made a type.
- I wrote "requirements" in comments and hoped reviewers read them → those
  were contracts, and C++26 is making them executable.

The lesson I'd tattoo on the README if laws allowed: **the features were never
bloated; they were answers to questions I hadn't asked yet.** Once the domain
question was asked, the mapping rate exploded — Core 4 made Kleisli obvious,
Kleisli made DVO obvious, DVO made contracts obvious — which is why the last
weeks felt like geometric growth. The concepts were compounding. Each one gave
the next one its vocabulary.


## 1. Pixels before principles (August 2023)

It started the way it should: a pixel, a canvas, a triangle. TGA images, line
drawing, an OBJ monkey rendered as wireframe with rapidobj and glm, then my own
pixel/canvas/color classes and an SDL2 loop. No architecture yet — just the
irreplaceable experience of making the software renderer draw everything myself.

The first real habit I formed was abstraction honesty: renaming classes until
they said what they did ("more semantically meaningful at least that's what I
thought", 2023-08-31). I didn't know it yet, but I was rehearsing for a day
when renaming would become *law* in this repo.

**Aha #1 — input is data, not function calls (2023-09-11).** While building
camera controls I reached for the Command pattern on my own: commands as
objects, processed by command processors, mutating the viewer. I liked it
because it made "what happened" *replayable* — a thing I couldn't articulate
yet, but which three years later became Core 2 (`<Pod>Command`) and the
command/event vocabulary of every bounded context.

**Early dead ends that taught by contrast:** an ECS library "might be useful"
(2023-09-11) — I kept it at arm's length and eventually ruled the opposite
(Constitution III: DOD without the ECS programming model). And `SystemProcess`,
an intermediate class I invented to beat forward-declaration cycles — my first
instinct that *dependencies must flow through a single named seam*. That seam
later grew up to be the gateway.

## 2. The shader era (2023-08-31 → 2023-09-02)

Color spaces, HSB, polar coordinates, Fractal Brownian Motion — the
thebookofshaders pass. It looks like a detour, but it taught me the single most
important property I now demand everywhere: **a good function is a pure
function you can stare at**. A shader maps coordinates to colors, statelessly.
Everything wrong with my later reducer monoliths was wrong because it violated
a property shaders already taught me.

## 3. One header became many — the packaging instinct (2026-02)

When the project came back from hiatus it went through its lib-ification:
mega header split into chunked headers, vcpkg, one assets folder, an
`shs-renderer-lib` identity. The aha was small but durable: **the include graph
is the architecture**. If a header can't be compiled alone, the design isn't
done. That instinct is now mechanical law (self-containment gate, include-graph
check, §6.2's pinned suffixes).

## 4. Rendering paths as data (2026-02-17)

I built forward+, deferred, and then a **dynamic compositional renderer path** —
and the honest discovery was that the interesting part wasn't rendering. It was
that a "render path" is really a *plan*: a precomputed, inspectable value that
the backend executes. Backend-switch sequencing had to be **planned, not decided
during pass execution**. I didn't have the word for it yet, but I had
discovered **decision-as-value** — choosing is data, and doing is mechanical.
Today that idea has a name in this repo: the `PipelineExecutionPlan`, and its
generalization, the Kleisli `Step`.

## 5. Value-Oriented Programming — naming my own instinct (2026-02-24)

I named the migration **Value-Oriented Programming (VOP)** — value-oriented
input action/gateway APIs (`value_commands.hpp`, `value_input_latch.hpp`),
execution planning as values, precomputed execution groups. VOP was my own
coinage, and at first it meant something loose: "make everything a value, plan
before you do."

It only became rigorous when I met the other half of the idea — data-oriented
design — and fused them (2026-08-20, tetris as the test bench). VOP gave the
*decisions* shape; DOD gave the *data* shape. The synthesis became Constitution
III: domain-owned data layout and execution, explicitly **without** importing
the ECS programming model.

## 6. The Domain Pod — tetris taught me Core 4 (2026-08-21 → 09-15)

Refactoring the FPS and tetris demos, I kept re-deriving the same four things
around every piece of state: the data itself, the actions that may touch it,
the single thing that enforces the rules, and the facts it emits. When I
finally wrote that down it became **Core 4: Types (contract), Command,
Gateway, Event — nothing else mutates state.**

The aha I remember most clearly: I was storing *transitional* state —
`is_pending`, `needs_rebuild` flags — and losing to them. A flag is a scar
where a decision should have been. **Phantom flags are state trying to
remember what the pipeline already knows.** Kill the flag, keep the state
valid, and the "pending" thing lives only inside the in-flight computation.
(DDD people call this making invalid states unrepresentable; monad people call
it the transient context of a saga. I found it by being annoyed at booleans.)

## 7. The monadic leap — the founding week (2026-09-16)

This was the big one, and it came from failure. My pure reducer switch-cases
had turned into nesting hell: one function owning the Cartesian product of
state shapes × actions × failure modes. I wrote the manifesto ("Monadic Kleisli
Composition", now hashed on the history shelf) before I knew how much of it
already existed as theory — and that's when the mappings started landing:

- My "every transition is one small pure step" → the **Kleisli arrow**
  `A → expected<B, DomainError>` — a *monad* in disguise. I had reinvented
  monadic composition to escape callback-style nesting, and `and_then` /
  `transform` / `or_else` turned out to be the standard spelling.
- My "domains never touch each other's data" → **DDD bounded contexts** and
  the **Single-Writer principle**.
- My "failure keeps state and emits a fact" → the **saga pattern with
  compensation**, proven in this repo through fact logs.
- My "commands in, events out" → Redux's dispatch/model split, minus the
  reducer (the constitution is explicit: we borrow the *properties*, not the

## 8. Names became law (2026-09-17)

Once the shapes stabilized, I noticed the *names* were lying. `reduce_*` said
reducer; the thing was a gateway enforcing invariants. The fix became the **Pod
Identifier Law**: the name is the repository's only navigation surface, and a
lying name is a defect, not a cosmetic issue.

The same audit killed my own coined term. **"Domain POD"** had silently carried
two meanings — the C++ one (trivial + standard-layout) and my project one ("our
domain data") — and the C++ meaning had died with `std::is_pod` in C++20. The
language itself agreed the term was retired. The replacement, **Domain Value
Object (DVO)**, says the one property that matters in the term itself: *always
valid at every module edge*. That rename law (T1–T6) taught me the meta-lesson
that is now Constitution II §6.6: **in a value-oriented design, names are the
only navigation surface.**

## 9. Contracts — the invariant finally found its home (2026-09-17)

The DVO definition ("always valid at every edge") was being enforced by review
discipline and comments — which is to say, by hope. The aha: this property is
*exactly* what **design by contract** is for, and C++26 (P2900) is about to
make it a language feature. So the law now says: invariants live at module
edges, written as `SHS_PRE`/`SHS_POST`/`SHS_CONTRACT_ASSERT`, checked in debug,
folded into native C++23 `[[assume]]` in release, and migrating mechanically to
native C++26 contracts when the toolchain allows. The `expected` railway stays
the only *domain failure* path — contracts are for unreachable states, never
control flow (my Rule 4.1 determinism instinct again).

And the third pillar snapped into place: **the DVO is the backbone; monadic
pipelines build its logic; contracts guard its edges.** Everything is a Domain
Separation or a Domain Boundary — there is no third category.

## 10. The pattern in the pattern

Looking back, every era follows the same loop, which is the actual personal
discovery behind all of this:

1. **Hit real pain** (nesting hell, phantom flags, lying names, undecided
   rendering paths).
2. **Invent a fix that feels like mine** — commands, plans, pods, gateways,
   sagas, valid values.
3. **Discover the literature already named it** — Command pattern, DDD bounded
   contexts, value objects, monads/Kleisli composition, sagas, design by
   contract — and that modern C++ (C++20→23→26) has been growing the exact
   features to express it: `span`, `variant`, `expected`, `pmr`, concepts,
   `[[assume]]`, and (soon) native contracts.
4. **Write it down as law**, so the discovery survives me.

The mapping table, honestly stated:

| My discovery (before I knew the name) | The name that already existed | The C++ feature that fits it |
| :--- | :--- | :--- |
| "input as objects through processors" (2023) | Command pattern | `std::variant<...Intent>` command sets |
| "data + actions + enforcer + facts" (2026-08) | DDD aggregate + ubiquitous language | Core 4: contract/command/gateway/event headers |
| "flags are scars" (2026-08) | make invalid states unrepresentable; saga transient context | closed enums, `expected` error rail |
| "plan then execute" (2026-02) | decision-as-data; interpreter | `PipelineExecutionPlan` value |
| "one small pure step" (2026-09-16) | Kleisli arrow / monadic composition | `std::expected::and_then/transform/or_else` |
| "domains never touch each other" | bounded context + anti-corruption layer | events out, commands in (§6.2) |
| "data must always be valid at the edge" | DDD value object; design by contract | DVO + C++23 `[[assume]]` → C++26 contracts |
| "names are the navigation surface" | ubiquitous language (DDD) | Pod Identifier Law (§6.6) |
| "the language should carry my discipline" | railway-oriented programming; DbC | C++23 baseline; C++26 `pre/post` (gated on GCC 16) |

Reading order: the hashed founding documents ([kdba_history/](kdba_history/))
are the raw "aha" artifacts; the law lives in the constitutions; this file is
the story in between.

