# INPUT & ACCESSIBILITY - interchangeable intent sources, done inclusively

> The input edge is one of the interchangeable intent sources (FPS_EXAMPLE
> section 2). This doc covers adding new sources (touch, gamepad) and the
> accessibility duties that pair with the color language (section 8.2.4).

---

## 1 - Adding an intent source

Every source compiles to the SAME command shapes:

| Source       | Notes                                                                                                                    |
| ------------ | ------------------------------------------------------------------------------------------------------------------------ |
| keyboard-map | shipped; OS-repeat filter + DAS/ARR + first-gesture audio unlock                                                         |
| touch        | virtual stick = MOVE vec; swipe zones / on-screen buttons for discrete intents; multi-touch bookkeeping is the hard part |
| gamepad      | Gamepad API polling in rAF; sticks need deadzone + response curve; triggers are analog (hold-to-charge patterns)         |

Rules unchanged from keyboard-map: sources emit commands, never mutate pods;
OS/auto-repeat filtering at the edge; blur/visibility loss releases all held
states.

## 2 - Remapping (expected by players)

Store bindings as logical-action -> physical-key maps in storage-edge.
Settings page writes the map; input edge reads it. Never hardcode 'A' inside
reducers - they only ever see action names.

## 3 - Accessibility checklist (pairs with section 8.2.4 color audit)

| Need             | Mechanism                                                                                                                      |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| photosensitivity | cap flash frequency/intensity; screenFlash has caps already; add a settings toggle to disable lightning-style effects entirely |
| motion sickness  | camera shake toggle + intensity slider (shake exists as fx value; gate it)                                                     |
| hearing          | visual twins for every sound cue (floater/banner systems already provide most; verify each SND_* has a visible twin)           |
| motor            | hold-to-toggle options for soft drop / hold; adjustable DAS/ARR (constants exist, expose them)                                 |
| cognitive        | A5-style legends; coach content lives pause-side per Session 24 law                                                            |

## 4 - Test pins

- I1: remap persists across reloads and applies immediately
- I2: gamepad stick passes deadzone test; no drift at rest
- I3: every SND_* mapping has a registered visual twin (extends i18n-parity
  style check)
- I4: toggles actually gate their effect (shake=0 produces zero cameraShake)
