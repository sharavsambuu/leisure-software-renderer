#!/usr/bin/env python3
"""t0_parity_suite — Tier0/Tier1 equivalence gate with per-demo envelopes.

Runs each demo pair fresh (SW binary always; Vulkan twin when present),
diffs with t0_parity.py inside an isolated scratch directory, and checks
the result against the pinned envelope (docs/education/
backend_capability_notes.md S3: exact -> tolerance maturity).

Paths are supplied explicitly (CMake passes them at test registration);
no checkout-specific paths are baked in:
  --build-dir DIR         tier0 binary directory (env T0_ADVENTURES_BUILD_DIR)
  --tier1-build-dir DIR   tier1 binary directory (env T1_ADVENTURES_BUILD_DIR)
  --scratch-dir DIR       isolated output dir (default: <build-dir>/t0_parity_out)
  --only NAME             limit to one pair (used by per-pair CTest entries)

Capability vs failure: a missing *_vk binary (GPU-free or slang-free
configure) is a documented capability SKIP. A binary that exists but exits
non-zero, a parity-tool failure, malformed comparator output, or an
envelope breach is a FAIL.

Envelopes (pinned 2026-09-16, Mesa device set in this WSL env):
  01 barycentric : NEAR-EXACT (differ <= 8, all within 1 LSB) — re-pinned
                   2026-09-18: was EXACT on the 2026-09-16 device set; Mesa
                   25.2.8 (llvmpipe, LLVM 20.1.2) now shows 4 px @ 1 LSB.
                   Proven pre-existing device drift, NOT a refactor
                   regression: the pair breaches identically when rebuilt
                   against the pre-R1 library (983925c worktree cross-check,
                   docs/backlog/adventure_demo_baseline_2026-09-18.md).
  02 projection  : TOLERANCE (<=13% differ, >=70% of those within 1 LSB,
                   edge-localized depth flips from affine-vs-perspective
                   NDC-z interpolation; max drift recorded, must not grow)
                   — re-pinned 2026-09-18: 12.33% observed on Mesa 25.2.8
                   vs the 12.0% pin; identical value reproduced on the
                   pre-R1 library (983925c), so device drift again.
  03 depth_blend : TOLERANCE-ONLY, max drift <= 1 LSB
  04 texture     : TOLERANCE-ONLY, max drift <= 1 LSB
  05 stencil     : EXACT (0 differing)
  08 normal map  : TOLERANCE (<=27% differ, >=95% within 1 LSB, max 2)

Re-pins are shrink-only from here: a future device set that beats these
numbers may tighten them; nothing loosens without a fresh investigation.

Exit 0: every checked pair inside its envelope (skips allowed, reported).
Exit 1: any failure (envelope breach, binary failure, tool failure).
A breach means code changed or the device changed — investigate, don't
blindly re-pin (lessons doc S7: stale artifacts lie; this suite always
regenerates before comparing).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "t0_parity.py"

PAIRS = ["01_barycentric", "02_projection", "03_depth_blend",
         "04_texture_sampling", "05_stencil"]
T1_PAIRS = ["08_normal_mapping"]
BIN = {"01_barycentric": "tri_barycentric", "02_projection": "projection",
       "03_depth_blend": "depth_blend", "04_texture_sampling": "texture_sampling",
       "05_stencil": "stencil"}
T1_BIN = {"08_normal_mapping": "normal_mapping"}
T1_PREFIX = "t1_"

ENVELOPE = {"01_barycentric": "near_exact", "02_projection": "tol13",
            "03_depth_blend": "tol1", "04_texture_sampling": "tol1",
            "05_stencil": "exact"}
T1_ENVELOPE = {"08_normal_mapping": "tol27"}  # sampler-precision drift, see rung-08 notes


def run(cmd, cwd):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    except OSError as e:  # vanished binary, bad exec, etc. — a failure, not a crash
        return 127, f"subprocess launch failed: {e}"
    return r.returncode, (r.stdout + r.stderr).strip()


def parse_parity(out):
    differ = pct = within = maxd = None
    for line in out.splitlines():
        if line.startswith("differ_exact="):
            try:
                parts = line.split()
                differ = int(parts[0].split("=")[1])
                pct = float(parts[1].strip("()%"))
                within = int(parts[2].split("=")[1])
                maxd = int(parts[3].split("=")[1])
            except (ValueError, IndexError):
                # malformed comparator line: ignore it; if no valid line was
                # seen the caller reports PARITY TOOL FAILED
                differ = pct = within = maxd = None
    return differ, pct, within, maxd


def check_env(env, differ, pct, within, maxd):
    if env == "exact":
        return (differ == 0, f"exact={'PASS' if differ == 0 else 'BREACH'}")
    if env == "near_exact":
        ok = (differ is not None and maxd is not None
              and differ <= 8 and differ == within and maxd <= 1)
        return (ok, f"near_exact differ={differ} max={maxd}")
    if env == "tol1":
        ok = differ is not None and maxd is not None and maxd <= 1 and differ == within
        return (ok, f"tol1 max={maxd}")
    if env == "tol13":
        ok = (differ is not None and pct is not None and within is not None
              and pct <= 13.0 and (within / max(differ, 1)) >= 0.70 and maxd <= 217)
        return (ok, f"tol13 pct={pct}% within1={within}/{differ} max={maxd}")
    if env == "tol27":
        ok = (differ is not None and pct is not None and within is not None
              and pct <= 27.0 and (within / max(differ, 1)) >= 0.95 and maxd <= 2)
        return (ok, f"tol27 pct={pct}% within1={within}/{differ} max={maxd}")
    return (False, "unknown envelope")


def check_pair(name, binname, prefix, builddir, scratch):
    sw_bin = builddir / f"{prefix}{binname}_sw"
    vk_bin = builddir / f"{prefix}{binname}_vk"
    sw_png = f"{prefix}{name}_sw.png"
    vk_png = f"{prefix}{name}_vk.png"
    if not sw_bin.exists():
        print(f"{name}: FAIL — SW binary missing at {sw_bin} (SW demos always build)")
        return False
    if not vk_bin.exists():
        print(f"{name}: SKIP — Vulkan twin not built (capability: GPU-free or "
              f"slang-free configuration); SW-only smoke still applies")
        return True
    rc1, out1 = run([str(sw_bin), sw_png], cwd=scratch)
    rc2, out2 = run([str(vk_bin), vk_png], cwd=scratch)
    if rc1 != 0 or rc2 != 0:
        print(f"{name}: BINARY FAILED sw={rc1} vk={rc2}")
        if out1:
            print(f"  sw: {out1[-400:]}")
        if out2:
            print(f"  vk: {out2[-400:]}")
        return False
    _, out = run([sys.executable, str(TOOL), sw_png, vk_png], cwd=scratch)
    differ, pct, within, maxd = parse_parity(out)
    if differ is None:
        print(f"{name}: PARITY TOOL FAILED — {out[-200:]}")
        return False
    env = T1_ENVELOPE.get(name, ENVELOPE.get(name))
    ok, note = check_env(env, differ, pct, within, maxd)
    print(f"{name}: differ={differ} ({pct:.2f}%) within1={within} max={maxd} [{note}]")
    return ok


def binname_of(prefix, name):
    table = BIN if prefix == "t0_" else T1_BIN
    return table[name]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build-dir", type=Path, default=None,
                    help="tier0 binary directory (or env T0_ADVENTURES_BUILD_DIR)")
    ap.add_argument("--tier1-build-dir", type=Path, default=None,
                    help="tier1 binary directory (or env T1_ADVENTURES_BUILD_DIR)")
    ap.add_argument("--scratch-dir", type=Path, default=None,
                    help="isolated output directory for regenerated PNGs")
    ap.add_argument("--only", default=None,
                    help="check a single pair (name as in PAIRS/T1_PAIRS)")
    args = ap.parse_args(argv)

    if args.build_dir is None:
        env_dir = os.environ.get("T0_ADVENTURES_BUILD_DIR", "")
        if not env_dir:
            print("t0_parity_suite: --build-dir (or T0_ADVENTURES_BUILD_DIR) "
                  "is required", file=sys.stderr)
            return 2
        args.build_dir = Path(env_dir)
    if args.tier1_build_dir is None and os.environ.get("T1_ADVENTURES_BUILD_DIR"):
        args.tier1_build_dir = Path(os.environ["T1_ADVENTURES_BUILD_DIR"])
    build = args.build_dir.resolve()
    if not build.is_dir():
        print(f"t0_parity_suite: build dir does not exist: {build}", file=sys.stderr)
        return 2
    scratch = (args.scratch_dir or (build / "t0_parity_out")).resolve()
    scratch.mkdir(parents=True, exist_ok=True)

    pairs = [("t0_", n, build) for n in PAIRS]
    if args.tier1_build_dir is not None:
        pairs += [(T1_PREFIX, n, args.tier1_build_dir.resolve()) for n in T1_PAIRS]
    if args.only:
        pairs = [p for p in pairs if p[1] == args.only]
        if not pairs:
            print(f"t0_parity_suite: unknown pair '{args.only}'", file=sys.stderr)
            return 2

    ok_all = True
    for prefix, name, bdir in pairs:
        ok_all = check_pair(name, binname_of(prefix, name), prefix, bdir,
                            scratch) and ok_all
    skipped = any(not (bdir / f"{prefix}{binname_of(prefix, n)}_vk").exists()
                  for prefix, n, bdir in pairs)
    if ok_all:
        tail = " (with capability skips)" if skipped else ""
        print("SUITE: PASS (all envelopes hold)" + tail)
    else:
        print("SUITE: BREACH")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

