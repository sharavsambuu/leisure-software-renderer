#!/usr/bin/env python3
"""t0_parity_suite — Tier0 equivalence gate with per-demo tolerance envelopes.

Runs all 10 tier0 binaries fresh, diffs each pair with t0_parity.py, and
checks the result against the pinned envelope (docs/education/
backend_capability_notes.md S3: exact -> tolerance maturity).

Envelopes (pinned 2026-09-16, Mesa device set in this WSL env):
  01 barycentric : EXACT (0 differing)
  02 projection  : TOLERANCE (<=12% differ, >=70% of those within 1 LSB,
                   edge-localized depth flips from affine-vs-perspective
                   NDC-z interpolation; max drift recorded, must not grow)
  03 depth_blend : TOLERANCE-ONLY, max drift <= 1 LSB
  04 texture     : TOLERANCE-ONLY, max drift <= 1 LSB
  05 stencil     : EXACT (0 differing)

Exit 0: every pair inside its envelope. Exit 1: envelope breach.
A breach means code changed or the device changed — investigate, don't
blindly re-pin (lessons doc S7: stale artifacts lie; this suite always
regenerates before comparing).
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = Path("/home/sharavsambuu/src/dev/leisure-software-renderer/"
             "cpp-folders/build/src/exps-rendering-adventures/"
             "tier0-rasterization-foundations")
TOOL = HERE / "t0_parity.py"

TIER1_BUILD = Path("/home/sharavsambuu/src/dev/leisure-software-renderer/"
             "cpp-folders/build/src/exps-rendering-adventures/"
             "tier1-classic-shading")
PAIRS = ["01_barycentric", "02_projection", "03_depth_blend",
         "04_texture_sampling", "05_stencil"]
T1_PAIRS = ["08_normal_mapping"]
BIN = {"01_barycentric": "tri_barycentric", "02_projection": "projection",
       "03_depth_blend": "depth_blend", "04_texture_sampling": "texture_sampling",
       "05_stencil": "stencil"}
T1_BIN = {"08_normal_mapping": "normal_mapping"}
T1_PREFIX = "t1_"
T1_BUILD_OF = {"08_normal_mapping": TIER1_BUILD}

ENVELOPE = {"01_barycentric": "exact", "02_projection": "tol12",
            "03_depth_blend": "tol1", "04_texture_sampling": "tol1",
            "05_stencil": "exact"}
T1_ENVELOPE = {"08_normal_mapping": "tol27"}  # sampler-precision drift, see rung-08 notes


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def parse_parity(out):
    differ = pct = within = maxd = None
    for line in out.splitlines():
        if line.startswith("differ_exact="):
            parts = line.split()
            differ = int(parts[0].split("=")[1])
            pct = float(parts[1].strip("()%"))
            within = int(parts[2].split("=")[1])
            maxd = int(parts[3].split("=")[1])
    return differ, pct, within, maxd


def check(name, differ, pct, within, maxd):
    return check_env(ENVELOPE[name], differ, pct, within, maxd)


def check_env(env, differ, pct, within, maxd):
    if env == "exact":
        return (differ == 0, f"exact={'PASS' if differ == 0 else 'BREACH'}")
    if env == "tol1":
        ok = differ is not None and maxd is not None and maxd <= 1 and differ == within
        return (ok, f"tol1 max={maxd}")
    if env == "tol12":
        ok = (differ is not None and pct is not None and within is not None
              and pct <= 12.0 and (within / max(differ, 1)) >= 0.70 and maxd <= 217)
        return (ok, f"tol12 pct={pct}% within1={within}/{differ} max={maxd}")
    if env == "tol27":
        ok = (differ is not None and pct is not None and within is not None
              and pct <= 27.0 and (within / max(differ, 1)) >= 0.95 and maxd <= 2)
        return (ok, f"tol27 pct={pct}% within1={within}/{differ} max={maxd}")
    return (False, "unknown envelope")


def check_pair(name, binname, prefix, builddir):
    b = binname
    rc1, _ = run([f"./t1_{b}_sw" if prefix.startswith("t1") else f"./t0_{b}_sw"], cwd=builddir)
    rc2, _ = run([f"./t1_{b}_vk" if prefix.startswith("t1") else f"./t0_{b}_vk"], cwd=builddir)
    if rc1 != 0 or rc2 != 0:
        print(f"{name}: BINARY FAILED sw={rc1} vk={rc2}")
        return False
    _, out = run([sys.executable, str(TOOL),
                  f"{prefix}{name}_sw.png", f"{prefix}{name}_vk.png"], cwd=builddir)
    differ, pct, within, maxd = parse_parity(out)
    if differ is None:
        print(f"{name}: PARITY TOOL FAILED")
        return False
    env = T1_ENVELOPE.get(name, ENVELOPE.get(name))
    ok, note = check_env(env, differ, pct, within, maxd)
    print(f"{name}: differ={differ} ({pct:.2f}%) within1={within} max={maxd} [{note}]")
    return ok


def main():
    ok_all = True
    for name in PAIRS:
        ok_all = check_pair(name, BIN[name], "t0_", BUILD) and ok_all
    for name in T1_PAIRS:
        ok_all = check_pair(name, T1_BIN[name], T1_PREFIX, T1_BUILD_OF[name]) and ok_all
    print("SUITE: PASS (all envelopes hold)" if ok_all else "SUITE: BREACH")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
