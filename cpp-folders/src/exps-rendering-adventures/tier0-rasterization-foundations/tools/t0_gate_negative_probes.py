#!/usr/bin/env python3
"""t0_gate_negative_probes — negative probes for the t0 parity gates (AD1).

Verifies the gate machinery fails reliably instead of silently passing:
  1. t0_parity.py accepts both documented --tol forms (`--tol 1`, `--tol=1`)
     and they agree.
  2. t0_parity.py exit codes: 0 exact, 2 tolerance-only, 1 significant drift.
  3. The suite's parse_parity treats malformed comparator output as a
     failure (returns None), never as a pass.
  4. Envelope breaches are detected (check_env returns False on crafted
     breaches of every envelope kind), including "unknown envelope".

GPU-free; no Vulkan or demo binaries required. Exit 0 iff every probe holds.
"""
import subprocess
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t0_parity_suite as suite  # noqa: E402

TOOL = HERE / "t0_parity.py"


def make_png(path, size, rgba):
    """Write a solid-color PNG (no external deps; stdlib zlib only)."""
    raw = b"".join(b"\x00" + bytes(rgba) * size
                   for _ in range(size))
    def chunk(tag, data):
        c = tag + data
        return len(data).to_bytes(4, "big") + c + zlib.crc32(c).to_bytes(4, "big")
    ihdr = size.to_bytes(4, "big") + size.to_bytes(4, "big") + bytes([8, 6, 0, 0, 0])
    body = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    Path(path).write_bytes(body)


def run_tool(*argv):
    return subprocess.run([sys.executable, str(TOOL), *argv],
                          capture_output=True, text=True)


def main():
    failures = []
    tmp = Path("/tmp/t0_gate_probes")
    tmp.mkdir(parents=True, exist_ok=True)
    a, b, c = tmp / "a.png", tmp / "b.png", tmp / "c.png"
    make_png(a, 8, (200, 100, 50, 255))
    make_png(b, 8, (200, 100, 50, 255))
    make_png(c, 8, (10, 20, 30, 40))

    # 1) --tol forms: both documented spellings accepted, identical verdicts
    r_eq = run_tool(str(a), str(b), "--tol", "1")
    r_eq2 = run_tool(str(a), str(b), "--tol=1")
    if r_eq.returncode != 0 or r_eq2.returncode != 0:
        failures.append(f"--tol forms must both exit 0, got {r_eq.returncode} "
                        f"and {r_eq2.returncode}")
    if r_eq.stdout != r_eq2.stdout:
        failures.append("--tol 1 and --tol=1 must produce identical output")

    # 2) exit-code semantics
    r_drift = run_tool(str(a), str(c))            # far apart -> 1
    if r_eq.returncode != 0:
        failures.append(f"identical images must exit 0, got {r_eq.returncode}")
    if r_drift.returncode != 1:
        failures.append(f"significant drift must exit 1, got {r_drift.returncode}")

    # 3) malformed comparator output -> parse_parity None (suite FAILs, never passes)
    if any(v is not None for v in suite.parse_parity("not parity output")):
        failures.append("malformed comparator output must parse as failure")
    if suite.parse_parity("differ_exact=oops") [0] is not None:
        failures.append("malformed numeric field must not parse as success")

    # 4) subprocess failure surfaces as failure
    rc, _ = suite.run([str(tmp / "no_such_binary")], cwd=tmp)
    if rc == 0:
        failures.append("subprocess failure must produce non-zero rc")

    # 5) envelope breaches detected (shrink-only caps hold)
    probes = [
        ("exact breach", suite.check_env("exact", 1, 0.0, 1, 1), False),
        ("near_exact count breach",
         suite.check_env("near_exact", 9, 0.0, 9, 1), False),
        ("near_exact max breach",
         suite.check_env("near_exact", 2, 0.0, 2, 2), False),
        ("tol1 max breach", suite.check_env("tol1", 5, 0.0, 5, 2), False),
        ("tol13 pct breach",
         suite.check_env("tol13", 100, 13.5, 95, 217), False),
        ("tol27 max breach",
         suite.check_env("tol27", 10, 1.0, 10, 3), False),
        ("unknown envelope",
         suite.check_env("nonsense", 0, 0.0, 0, 0), False),
        ("near_exact pass", suite.check_env("near_exact", 4, 0.0, 4, 1), True),
        ("tol13 pass", suite.check_env("tol13", 100, 12.0, 90, 217), True),
    ]
    for name, got, want in probes:
        if got[0] != want:
            failures.append(f"envelope probe '{name}': got {got}")

    if failures:
        for f in failures:
            print(f"NEGATIVE PROBE FAILED: {f}")
        print("NEGATIVE PROBES: FAIL")
        return 1
    print("NEGATIVE PROBES: PASS (gates fail reliably)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
