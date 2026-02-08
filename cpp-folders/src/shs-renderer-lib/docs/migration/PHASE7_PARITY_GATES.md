# Phase 7 Parity Gates

This file defines repeatable parity gates between:

- `HelloPBRLightShafts` (legacy SOTA baseline)
- `HelloPassPlumbing` (modern plumbing path)

## Capture workflow

Run:

```bash
./cpp-folders/scripts/run_phase7_parity_capture.sh
```

Optional args:

```bash
./cpp-folders/scripts/run_phase7_parity_capture.sh <build_dir> <out_dir>
```

Outputs:

- `legacy/view_0.ppm`, `legacy/view_1.ppm`, `legacy/view_2.ppm`
- `plumbing/view_0.ppm`, `plumbing/view_1.ppm`, `plumbing/view_2.ppm`
- `SHA256SUMS.txt` (if `shasum` exists)

Capture CLI knobs (both demos):

- `--preset <0|1|2>`
- `--capture <absolute_or_relative_output_path>`
- `--capture-after <frame_count>`

## Gate checklist

1. Visual parity (3 fixed camera presets)
- Compare each legacy/plumbing pair for:
  - silhouette stability
  - shadow softness continuity
  - shafts visibility near sun-facing geometry
  - tonemap/exposure consistency

2. Feature parity
- `shadow_map`, `pbr_forward`, `tonemap`, `light_shafts` all active in plumbing path.
- Light shafts toggle still works (`L` key in interactive mode).

3. Build parity
- `HelloPassPlumbing` and `HelloPBRLightShafts` both build cleanly from the same tree.

## Coordinate convention checks

Verify parity with repository convention:

- LH world/view/projection (`+X right`, `+Y up`, `+Z forward`)
- NDC `z` in `[-1, 1]`
- Canvas Y-up, screen Y-down with present-time vertical flip
