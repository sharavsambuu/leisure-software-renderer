# snake/docs/DETAILS.md — Gotchas & Debugging Playbook

> Distilled from the 2026-08-22 sessions (wall→floor remap, fixed camera, controls/mirror/food hunt).
> Read alongside `STATUS.md` (what IS true now) and `docs/spec/conventions.md` (Constitution I).
> Each lesson states: **symptom → root cause → fix → how to detect it again fast.**

---

## 1. Coordinate System & Camera (Constitution I)

### 1.1 The floor mapping is a ROTATION — never a reflection
- **Rule:** grid `(x, y)` → world `(x, 0, -y)`. This is a proper rotation about +X (det=+1).
  `(x, 0, y)` is a REFLECTION: it flips triangle winding → backface culling eats faces and
  lighting goes inside-out.
- **Detection:** scene renders partially/inside-out after a mapping change; analyzer sees a band
  instead of a floor blob.

### 1.2 GLM's `lookAtLH` MIRRORS X in this vendored version ⚠️ (costliest bug of the session)
- **Symptom:** controls feel like you're watching from BEHIND the board; ArrowRight moves the
  snake screen-LEFT; an off-center object appears on the wrong side of center.
- **Root cause:** `/usr/include/glm/ext/matrix_transform.inl` builds the side vector as
  `s = normalize(cross(up, f))` — NOT the textbook `cross(f, up)`. For our front camera
  (`eye = center + (0,13,17)`, looking down/-Z-ish) this yields **s = (-1,0,0)**: the entire
  view renders horizontally mirrored. Y is unaffected (`u = cross(f, s)` still points up-screen),
  which is why vertical checks passed while horizontal ones failed — a nasty half-broken signal.
- **Fix (in place):** hand-rolled view basis in `spatial_fx/snake.plan.hpp`
  (`right = normalize(cross(fwd, up))`, rows `[r; u; f]`, translation column `-dot(axis, eye)`),
  version-independent. **Do NOT switch back to `glm::lookAtLH`.**
- **Coupled consequence:** un-mirroring the view FLIPS screen-space triangle winding. The
  rasterizer's front-face test must flip CCW → **Clockwise** in the same commit, or EVERYTHING
  gets culled (symptom: near-empty frame, one degenerate sliver). View handedness and front-face
  convention are ONE decision, never two.

### 1.3 Screen-space control contract (memorize)
With the current camera + floor mapping:
- grid **+x = screen-RIGHT** ⇒ ArrowLeft/Right are `dx ∓ 1`
- grid **+y = screen-UP** (away from camera) ⇒ ArrowUp/Down are `dy ± 1`
When EITHER the camera side or the floor mapping changes, re-derive this contract from scratch —
do not patch individual arrows (see 2.1).

### 1.4 Perspective foreshortening breaks naive pixel calibration
- Local scale on the floor varies strongly by row: ~30 px/cell at row 9 vs ~47 px/cell as the
  board-wide AVERAGE (near rows stretch, far rows compress). Never calibrate with
  `board_bbox_width / GRID_W` and apply it to a specific row.
- **Correct approach:** solve the actual projection equations for the measured point, or compare
  RELATIVE positions only (see §4).

---

## 2. Pure Reducers / State Machine

### 2.1 Semantics drift when presentation changes
- **Symptom:** UP/DOWN inverted while LEFT/RIGHT worked — maddeningly inconsistent.
- **Root cause:** `snake.action.hpp` kept tetris-WALL semantics (`UP → dy -= 1`) after the demo
  became a FLOOR. Half-migrated conventions are worse than either endpoint.
- **Lesson:** when the coordinate mapping changes, grep every consumer of the old semantics
  (actions, plan, FX emission) and re-derive them together. Encode the new contract as comments
  AT the mapping site AND at the input site.

### 2.2 Identity input must produce identity transition (idle-decay bug)
- **Symptom:** idle snake shrank one segment per frame; soft-wall holds did too.
- **Root cause:** no-intent ticks produced `delta=(0,0)` which fell through to the normal-move
  path: `new_head == head_pos` still executed "move", vacating the tail and duplicating the head.
- **Fix:** reducer returns the snapshot UNCHANGED when there is no intent (and on wall holds).
- **General rule:** in pure reducers, a zero/no-op input must short-circuit BEFORE any
  position bookkeeping. Test: two idle screenshots N frames apart must be pixel-identical.

### 2.3 Head-color lag is cosmetic-only
Body array appends the new head at the END, so index 0 (brightest gradient stop) trails behind
after turns. Accepted cosmetic quirk — do not "fix" by reordering the SoA (movement logic depends
on append-at-head).

---

## 3. Rendering Pipeline

### 3.1 Depth bias can BURY geometry (food invisibility)
- **Symptom:** object completely absent from the frame — zero pixels of its color anywhere.
- **Root cause:** food box top (y=0.45) sat barely above its tile's top (y=0.25); the NDC depth
  gap between those surfaces was SMALLER than the food's own depth_bias (+0.06), so the food lost
  every depth test against the tile beneath it.
- **Rules of thumb:**
  - bias magnitude must be ≪ the smallest real depth separation you rely on;
  - positive bias = LOSE ties (drawn as if farther);
  - prefer real geometric separation (lift the object) over bias hacks;
  - verify presence by counting pixels of the object's exact lit color, not by eyeballing.

### 3.2 Lit colors ≠ base colors (breaks naive color filters)
Lambert shading multiplies base RGB by `diffuse*(warm tint) + ambient*(blue tint)`. A filter
written against BASE colors (e.g. orange `255,180,140`) finds nothing because the lit top face is
`(≈255,190,156)`. Derive expected lit values by hand before writing pixel classifiers.

### 3.3 BMP/surface byte traps (recurring)
- `SDL_PIXELFORMAT_RGBA32` memory bytes are **B,G,R,A** on little-endian — read pixels accordingly.
- SDL_SaveBMP writes bottom-up rows (positive biHeight); loaders must flip Y once.
- These bit BOTH the C++ side and every Python diag script; helpers live in `_diag_snake/*.py`.

---

## 4. Debugging Methodology (pixel forensics playbook)

This session was solved by treating screenshots as measurement instruments:

1. **Relative beats absolute.** Inject ONE synthetic input at a known frame (`--autodrive-*`),
   screenshot before/after, diff bboxes. Direction of CHANGE is calibration-free truth.
2. **Asymmetric landmarks detect mirrors.** Checkerboards are mirror-invisible. The snake's
   head→tail color GRADIENT is a built-in compass: find per-color centroids
   (`snake_ends.py`); head-left-of-tail ⇒ view is mirrored. Any demo should carry one
   deliberately asymmetric element for this purpose.
3. **Solve, don't guess.** When measurements contradict mental math, write out the full
   projection chain (view basis → perspective → viewport) and solve for what world position the
   measured pixels imply. Two competing interpretations (mirror vs offset) resolved cleanly once
   equations were solved against three independent measurements.
4. **Beware degenerate survivors.** After a winding flip, a 1px sliver may pass culling and look
   like "mostly working". Empty-ish frames mean a systematic rejection (culling/winding/clip),
   not partial breakage.
5. **Verify binary freshness by behavior.** A rebuild that changes nothing observable may not
   have recompiled what you think. Confirm via an intentional behavioral delta (e.g. the idle-fix
   made runs A≡B identical — that PROVED the new reducer ran).
6. **Keep the falsification tools.** Every one-off diag script graduated into
   `cpp-folders/_diag_snake/` as a permanent regression suite (see §5).

---

## 5. Verification Tooling (deterministic, display-less)

Composable CLI hooks in `hello_3d_snake.cpp`:
```
--screenshot            render until --frame=N, save BMP, exit
--frame=N               trigger frame (default 60)
<path>                  output BMP path
--autodrive-up          inject ONE synthetic ArrowUp intent at frame 60
--autodrive-right       inject ONE synthetic ArrowRight intent at frame 60
```

Regression scripts (`cpp-folders/_diag_snake/`, stdlib-only Python).
**These are intentionally NOT tracked by git** (repo-root .gitignore: `cpp-folders/_diag_*/`) —
they are dev-local tooling. If missing, recreate them from the table below; each is ~20-60 lines
of stdlib Python using the shared BMP conventions of §3.3:
| Script | Checks |
|---|---|
| `compare_runs.py A B C` | idle@58 ≡ idle@70 (no decay/drift); ArrowUp raises bbox TOP ≥8px |
| `bbox_snake.py imgs…` | snake bbox per image (width/height/position deltas) |
| `snake_ends.py img` | per-gradient-stop centroids — mirror detector |
| `find_food.py img` | food orb presence + centroid (flip detector, uses asymmetric food cell) |
| `analyze_frame.py img` | board reads as wide centered floor blob (not edge-on band) |
| `dump_colors.py img` | full palette histogram — first look at ANY anomaly |

**Metric choice matters:** centroid shift DILUTES L-shaped motion (horizontal arm cancels vertical);
bbox top-edge is the robust signal for "moved up-screen". Prefer structural metrics over averages.

**Golden recipe (all must PASS):**
```bash
cd cpp-folders/build_vcpkg/src/hello-3d-demos/snake
./Hello3DSnake --screenshot /tmp/a.bmp --frame=58
./Hello3DSnake --screenshot /tmp/b.bmp --frame=70
./Hello3DSnake --autodrive-up    --screenshot /tmp/c.bmp --frame=66
./Hello3DSnake --autodrive-right --screenshot /tmp/d.bmp --frame=66
python3 cpp-folders/_diag_snake/compare_runs.py /tmp/a.bmp /tmp/b.bmp /tmp/c.bmp
python3 cpp-folders/_diag_snake/snake_ends.py /tmp/a.bmp     # head RIGHT of tail
python3 cpp-folders/_diag_snake/find_food.py  /tmp/a.bmp     # food present, left of center
python3 cpp-folders/_diag_snake/bbox_snake.py /tmp/a.bmp /tmp/d.bmp  # d shifted RIGHT one pitch
```

---

## 6. Symptom → Root Cause Quick Table

| Symptom | Likely root cause | Section |
|---|---|---|
| One axis of controls inverted, other fine | stale action semantics after mapping change | 2.1 |
| Controls ALL feel reversed / viewed from behind | view mirrored (lookAtLH `cross(up,f)` quirk) | 1.2 |
| Frame nearly EMPTY after touching view/projection | winding flipped → all culled; flip front-face too | 1.2 |
| Object completely invisible, zero pixels of its color | depth_bias > real NDC gap (or buried under neighbor) | 3.1 |
| Idle state decays / drifts without input | zero-delta fell through to move path | 2.2 |
| Pixel positions "impossible" vs hand math | foreshortening mis-calibration OR mirror | 1.4, 4.3 |
| Color filter finds nothing though object visible | filter used base colors, not lit colors | 3.2 |
| Rebuild seems to change nothing | stale binary — prove freshness via intentional delta | 4.5 |

---

*Keep this file alive: any future agent that burns >30 minutes on a non-obvious bug here should
append the pattern above.*