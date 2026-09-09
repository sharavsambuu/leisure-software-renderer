# PLANNING — visual tools & workflow for Domain POD design

> Why this file exists: Domain POD code is event-trap architecture. The code
> is easy to TEST but hard to READ, because control flow lives in facts and
> reactions instead of call stacks. The industry answer: plan the graph
> FIRST, generate documentation from it, then implement. This doc records the
> tools, when to use each, and how they map to real pod work.
>
> Rule of thumb learned on L4 (STATUS Sessions 32b/33): plan the graph before
> coding. Reverse-engineering a fact-to-consumer map from finished code works,
> but it is cheaper to draw it first.

---

## 1 · The planning workflow (use in this order)

1. **EVENT STORM** — paper / whiteboard / Excalidraw stickies.
   Output: list of FACTS + which cluster owns each.
2. **POD BOUNDARIES** — cluster the facts; clusters ARE pods.
   Output: pod list + a one-line ownership statement per pod.
3. **STATE MACHINES** — Stately/XState sketch for phase-machine pods only.
   Output: states, transitions, guards per machine-pod.
4. **PRODUCERS TABLE** — fill `scripts/generate-event-flow.mjs` PRODUCERS.
   Every fact gets ONE declared owner (an undeclared producer warning is a
   boundary smell).
5. **RED TESTS** — spec pins written BEFORE implementation.
6. **IMPLEMENT** — contracts → reducers → edges.
7. **REGENERATE DOCS** — `node scripts/generate-event-flow.mjs`.

Steps 1–4 cost half a day and prevent weeks of "why does this feel weird"
playtest feedback (the L1→L4 experience — see STATUS Sessions 8b through 28).

---

## 2 · Event Storming (the methodology)

**What it is**: a facilitated brainstorm invented by Alberto Brandolini for
discovering domain events BEFORE designing software. Invented for business
domains; works identically for game mechanics.

**How to run it for a game feature (solo or with friends):**

1. Write every outcome as an orange sticky in PAST TENSE on a rough timeline:
   `Line Cleared`, `Special Locked`, `Combo Escalated`, `Freeze Expired`,
   `Board Emptied`.
2. For each event, add the blue sticky that CAUSED it: `Hard Drop` caused
   `Line Cleared`. These become your COMMANDS.
3. Cluster events that always change together onto yellow stickies — each
   cluster IS a pod. If a sticky seems to belong to two clusters, either the
   boundary is wrong or the event carries two meanings (split it).
4. Pink stickies at the edge = external systems reacting: Sound, Storage,
   Leaderboard. These are your EDGES.

**Why it works for pods**: it discovers OWNERSHIP empirically. You never
decide "powerups should be its own pod" by argument — you observe that
cadence-armed / special-spawned / cycle-completed always move together and
never with score, and the boundary declares itself.

**Mapping in this repo:**

- matrix cluster: Piece Locked · Lines Cleared · Game Over · Special Locked
- progression cluster: Score Changed · Level Up · Combo Streak · Clock Tick
- powerups cluster: Cadence Armed · Cycle Completed
- fx cluster reacts to everything but owns nothing

**Resources**: Brandolini, _Introducing EventStorming_; free summaries —
search "event storming cheat sheet".

---

## 3 · Diagramming tools

### Mermaid (text-in-repo) — ALREADY IN USE

- Lives inside markdown; GitHub renders graphs natively.
- Never rots: the diagram is versioned next to the code it describes.
- Used by `docs/EVENT_FLOW.md` §2 (macro flow graph).
- Best for: macro architecture, fact-flow direction, docs that must stay
  current.
- Limitation: hand-maintained layout; big graphs get messy. Keep one graph
  per concern (flow vs state machines), never one giant map.

### Excalidraw — brainstorm canvas

- Free, open source, hand-drawn style; the JS/TS community favorite.
- Has a Mermaid-to-Excalidraw converter: paste the EVENT_FLOW mermaid block,
  get an editable diagram for rearranging.
- Best for: workflow steps 1–2 (storming, pod-boundary debates) where you
  move boxes constantly.

### draw.io / diagrams.net — formal artifacts

- Free, precise, exports PNG/SVG for polished docs.
- Best for: committed architecture diagrams embedded in README/docs and
  rarely changed.

### PlantUML — legacy text-based

Common in enterprise C++/Java worlds; Mermaid has largely replaced it for new
web projects. Listed here because its state-diagram syntax is still widely
referenced in older architecture docs.

---

## 4 · State machine tools (phase-machine pods only)

### Stately.ai / XState

A statechart editor AND runtime. You drag states and transitions; it emits a
diagram plus executable JS from the same definition — the diagram cannot
drift from behavior because they are the same artifact.

**Which of our pods qualify** (reducer dominated by screen/phase switching):

| Pod              | Machine         | States                             |
| ---------------- | --------------- | ---------------------------------- |
| session          | app flow        | TITLE → PLAYING ↔ PAUSED → RESULTS |
| environment (L5) | overseer phases | CALM → RAIN → BLACKOUT → CRESCENDO |

**Which do NOT qualify**: math/trap-wall pods (matrix, spatial-fx,
progression). Their reducers branch on many independent facts — that is a
trap wall, not a state machine. Forcing them into XState adds ceremony
without clarity.

**Practical note**: adopting XState later is possible per-pod (it is just an
actor consuming events), but L4 shipped fine without it — adopt for game #2's
machine-like pods if diagrams keep drifting from behavior.

---

## 5 · Runtime visibility (what ACTUALLY happened)

Static maps show what CAN happen; runtime logs show what DID:

- The engine already accumulates per-tick event arrays (`mStep.events`) — the
  raw material of Redux-DevTools-style time travel exists.
- Cheap upgrade path: a debug flag dumping `tick#, facts[]` to console or a
  devtools page; replay from there.
- Use case: "why did the combo break?" — find the tick where a sterile lock
  reset comboCount and see exactly which facts arrived that tick.

---

## 6 · LLM-assisted analysis (hybrid workflow)

LLMs answer WHY/HOW questions tools cannot ("is five freeze signals too
many?"), but their recall goes stale as code changes. Ground them:

1. Regenerate `docs/EVENT_FLOW.md` after trap changes — facts must be true.
2. Ask LLM questions WITH the doc in context — grounded reasoning beats raw
   recall, and stale-map errors become visible instead of silent.
3. Let the LLM critique the DESIGN (signal budget, reward topology) — that is
   judgment work, not lookup work. Session 25 (the v2 redesign) shows how
   much a critique pass can change.

Division of labor: **tools own FACTS, humans+LLMs own JUDGMENT.**

---

## 7 · Quick reference — which tool for which question

| Question                         | Tool                                   |
| -------------------------------- | -------------------------------------- |
| What facts exist? Who owns each? | Event storming → PRODUCERS table       |
| How should phase X behave?       | Stately sketch (machine pods only)     |
| Who consumes SPECIAL_LOCKED?     | docs/EVENT_FLOW.md §3 matrix           |
| What does the fx pod do?         | EVENT_FLOW.md §4 trap table            |
| Why did combo break at tick N?   | runtime event dump (future devtools)   |
| Is this design any good?         | LLM critique pass over spec + flow map |

---

## 8 · Worked examples — the workflow applied to other genres

### 8.1 First-person shooter — [FPS_EXAMPLE.md](FPS_EXAMPLE.md)

Entities as IDs in SoA arrays, pod mapping (world/movement/weapon/combat/fx),
the one-click event chain traced pod by pod, projected event-flow graph,
kill-cam sequence capture, and the three FPS-specific traps (aim latency
clock split, camera-as-presentation, real collision math). Includes §2.1:
entity VIEWS vs entity objects — never store references outside a tick.

### 8.2 Missions & higher-level goals — [MISSIONS.md](MISSIONS.md)

How game goals/quests are programmed: the mission system is another pod that
owns only MISSION progress (never duplicated game truth), with two condition
patterns — event traps for kill/count goals, snapshot queries for state-shape
goals. Mission definitions are DATA in level files (scripter ladder applies);
sequencing flows through MISSION_COMPLETE fact chaining, not controllers.
Includes test pins M1–M7.

---

## 9 · Template checklist for a NEW game (condensed)

1. Event-storm all mechanics → fact list + pod clusters
2. Fill PRODUCERS table before writing any reducer
3. Machine-like pods sketched in Stately (optional)
4. Contracts (.contract.js factories) per pod
5. RED tests per feature pin
6. Reducers → engine wiring → edges
7. `node scripts/generate-event-flow.mjs`
8. Spec doc per major system (spec-first saved L4's redesign)
9. STATUS.md entry per session — failures WITH root causes
