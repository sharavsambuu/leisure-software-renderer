#!/usr/bin/env python3
"""Transitive include-graph dependency gate (KDBA step 3).

Structural rules over the real include graph (forwarders resolved):

  R1  no include cycles between canonical (non-forwarder) headers
  R2  raw SDK includes (Jolt/SDL3/Assimp/Vulkan) only in integration-tier
      headers (adapter dirs, rhi/, driver-adjacent vk_* execution headers
      and the pass-adapter aggregation pass_adapters.hpp,
      or files feature-guarded with SHS_HAS_<SDK>)
  R3  value-tier headers must not reach adapter/SDK-bearing integration
      headers directly or transitively, unless tracked in
      tools/engine_include_exceptions.json
"""

import json
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # .../shs-renderer-lib
INCLUDE_ROOT = REPO_ROOT / "include/shs"
EXCEPTIONS_PATH = Path(__file__).parent / "engine_include_exceptions.json"
SCHEMA_VERSION = 1
FORWARDER_MARKER = "Compatibility include"
ADAPTER_PATH_RE = re.compile(r"(^|/)adapters/")

SHS_INC_RE = re.compile(r'#include\s+"(shs/[^"]+)"')
SDK_INC_RE = re.compile(r'#include\s+[<"](Jolt/|SDL3/|assimp/|vulkan/vulkan\.h|GL/)')
SDK_TOKEN_RE = re.compile(r"JPH::|SDL_[A-Z]|aiScene|AiMesh|Vk[A-Z]")
GUARD_RE = re.compile(r"#\s*if\s+defined\(SHS_HAS_(JOLT|VULKAN|ASSIMP|SDL)\)")
INTEGRATION_PREFIXES = ("rhi/", "platform/", "app/backend/")
# Driver-adjacent execution helpers: the vk_* driver headers and the
# pass-adapter aggregation header (execution-tier adapter code over the
# software renderer + Jolt-backed culling; no value-tier header may
# transitively reach them). Ruling: pass_adapters.hpp is folded under the
# same driver-adjacent classification as vk_* (see migration step 5).
DRIVER_EXEC_RE = re.compile(r"^renderpath/execution/(vk_|pass_adapters)")


def load_canonical_headers(root=INCLUDE_ROOT):
    headers = {}
    for path in sorted(root.rglob("*.hpp")):
        rel = path.relative_to(root).as_posix()
        text = path.read_text()
        if FORWARDER_MARKER in text:
            continue
        headers[rel] = text
    return headers


def resolve_include(inc, headers):
    rel = inc[len("shs/"):]
    for _ in range(8):
        if rel in headers:
            return rel
        path = INCLUDE_ROOT / rel
        if not path.exists():
            return None
        m = SHS_INC_RE.search(path.read_text())
        if not m:
            return None
        rel = m.group(1)[len("shs/"):]
    return None


def build_graph(headers):
    graph = {}
    for rel, text in headers.items():
        deps = []
        for inc in SHS_INC_RE.findall(text):
            target = resolve_include(inc, headers)
            if target and target != rel:
                deps.append(target)
        graph[rel] = deps
    return graph


def classify(rel, text):
    """Integration tier = adapter paths, rhi/driver, driver-adjacent vk_*
    execution helpers, or feature-guarded (SHS_HAS_*) headers.

    A raw SDK include in an *unguarded, non-adapter* path does NOT upgrade
    the file to integration — it stays value-tier and rule R2 flags it.
    """
    if rel.startswith(INTEGRATION_PREFIXES) or DRIVER_EXEC_RE.match(rel):
        return "integration"
    if ADAPTER_PATH_RE.search(rel):
        return "integration"
    if GUARD_RE.search(text[:2048]):
        return "integration"
    return "value"


def find_cycles(graph):
    cycles = []
    state = {}
    for start in graph:
        if state.get(start):
            continue
        stack = []
        def visit(node):
            state[node] = 1
            stack.append(node)
            for dep in graph.get(node, ()):
                if state.get(dep) == 1:
                    cycles.append(stack[stack.index(dep):] + [dep])
                elif not state.get(dep):
                    visit(dep)
            stack.pop()
            state[node] = 2
        visit(start)
    return cycles


def transitive_reach(graph):
    memo = {}
    def reach(node):
        if node in memo:
            return memo[node]
        memo[node] = set()
        out = set()
        for dep in graph.get(node, ()):
            out.add(dep)
            out |= reach(dep)
        memo[node] = out
        return out
    for node in graph:
        reach(node)
    return memo


def is_dangerous(rel, text):
    """Adapter-tier or SDK-bearing header: the purity danger set."""
    return bool(ADAPTER_PATH_RE.search(rel)) or bool(SDK_INC_RE.search(text))


def load_exceptions(path=EXCEPTIONS_PATH):
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "exceptions": []}
    return json.loads(path.read_text())


def check(root=INCLUDE_ROOT, exceptions=None):
    headers = load_canonical_headers(root)
    graph = build_graph(headers)
    tiers = {rel: classify(rel, text) for rel, text in headers.items()}
    reach = transitive_reach(graph)
    if exceptions is None:
        exceptions = load_exceptions()
    exc = {e["header"]: e for e in exceptions.get("exceptions", [])}

    violations = []

    for cycle in find_cycles(graph):
        violations.append({
            "rule": "R1-header-cycle",
            "header": cycle[0],
            "detail": " -> ".join(cycle),
        })

    for rel, text in headers.items():
        if (tiers[rel] == "value" and SDK_INC_RE.search(text)
                and rel not in exc):
            violations.append({
                "rule": "R2-sdk-include-in-value-tier",
                "header": rel,
                "detail": "raw SDK include in value-tier header",
            })

    for rel, tier in tiers.items():
        if tier != "value":
            continue
        bad = sorted(
            dep for dep in reach.get(rel, ())
            if tiers[dep] == "integration" and is_dangerous(dep, headers[dep])
        )
        if bad and rel not in exc:
            violations.append({
                "rule": "R3-value-reaches-adapter",
                "header": rel,
                "detail": "transitive: " + ", ".join(bad[:6]),
            })
    return violations


def run_self_test():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "shs"
        (root / "adapters").mkdir(parents=True)

        def w(name, text):
            (root / name).write_text(text)

        w("value.hpp", '#pragma once\n#include "shs/adapters/driver.hpp"\n')
        w("adapters/driver.hpp", '#pragma once\n#include <Jolt/Jolt.h>\n')
        w("sdk_user.hpp", '#pragma once\n#include <SDL3/SDL.h>\n')
        w("cy_a.hpp", '#pragma once\n#include "shs/cy_b.hpp"\n')
        w("cy_b.hpp", '#pragma once\n#include "shs/cy_a.hpp"\n')
        w("clean.hpp", "#pragma once\n")

        no_exc = {"schema_version": 1, "exceptions": []}
        rules = sorted(v["rule"] for v in check(root, no_exc))
        assert "R1-header-cycle" in rules, rules
        assert "R2-sdk-include-in-value-tier" in rules, rules
        assert "R3-value-reaches-adapter" in rules, rules

        with_exc = {"schema_version": 1, "exceptions": [
            {"header": "value.hpp", "reason": "tracked"},
            {"header": "sdk_user.hpp", "reason": "tracked"}]}
        rules2 = sorted(v["rule"] for v in check(root, with_exc))
        assert rules2 == ["R1-header-cycle"], rules2

        w("value.hpp", "#pragma once\n")
        w("sdk_user.hpp", "#pragma once\n")
        w("cy_a.hpp", "#pragma once\n")
        w("cy_b.hpp", "#pragma once\n")
        assert not check(root, no_exc)

    print("[include-graph] self-test OK (negative + positive fixtures)")
    return 0


def main(argv):
    if "--self-test" in argv[1:]:
        return run_self_test()
    violations = check()
    if not violations:
        print("[include-graph] OK: acyclic; SDK placement and value-tier "
              "purity rules hold")
        return 0
    for v in violations:
        print(f"[include-graph] FAIL {v['rule']}: {v['header']}")
        print(f"    {v['detail']}")
    print(f"[include-graph] {len(violations)} violation(s); tracked "
          f"exceptions live in tools/engine_include_exceptions.json")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
