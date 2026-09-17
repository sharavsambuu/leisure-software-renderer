#!/usr/bin/env python3
"""Replay-parity gate for the contract guardrails bridge (C1.4).

Compiles nothing itself: CMake hands in the two probe binaries (identical
source, only SHS_CONTRACTS_ENFORCED differs) and this script asserts their
stdout is byte-identical — the bridge must never gate control flow
(Constitution II Rule 4.1). Permanent across the C++26 switch (C4.3).
"""
import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: contract_guardrails_replay_parity_test.py "
            "<enforced-probe> <assume-probe>",
            file=sys.stderr,
        )
        return 2

    digests = []
    for label, exe in (("enforced", sys.argv[1]), ("assume", sys.argv[2])):
        proc = subprocess.run([exe], capture_output=True, text=True)
        if proc.returncode != 0:
            print(
                f"[contract-replay-parity] FAIL: {label} probe exited "
                f"{proc.returncode}: {proc.stderr.strip()}",
                file=sys.stderr,
            )
            return 1
        digests.append(proc.stdout)

    if digests[0] != digests[1]:
        print("[contract-replay-parity] FAIL: replay-relevant output differs", file=sys.stderr)
        print(f"  enforced: {digests[0]!r}", file=sys.stderr)
        print(f"  assume:   {digests[1]!r}", file=sys.stderr)
        return 1

    print(f"[contract-replay-parity] OK: byte-identical replay digest ({digests[0].strip()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
