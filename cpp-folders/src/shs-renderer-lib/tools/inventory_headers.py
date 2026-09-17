#!/usr/bin/env python3
"""Reproduce the full public-header inventory; proposed paths are NOT approval.

All literal includes, including inactive conditional branches, are inventoried.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from check_header_migration import ROOT, includes

PLANNERS = {
    'frame_graph.hpp', 'pass_contract.hpp', 'pass_contract_registry.hpp',
    'pass_id.hpp', 'render_path_barrier_plan.hpp', 'render_path_capabilities.hpp',
    'render_path_compiler.hpp', 'render_path_interfaces.hpp',
    'render_path_recipe.hpp', 'render_path_resource_plan.hpp',
    'render_path_standard_pass_routing.hpp', 'render_path_presets.hpp',
    'render_composition_presets.hpp', 'render_technique_presets.hpp', 'technique_profile.hpp',
}


def destination(name):
    # The legacy shs/domains/ + shs/execution/ forwarder trees were removed
    # (migration step-7 item 4); only the reviewed rhi/vulkan value/runtime
    # split and the app-context proposal remain as pending proposals.
    if name == 'shs/core/context.hpp':
        return 'shs/app/context.hpp', 'split-values-from-runtime-before-move'
    if name.startswith('shs/rhi/drivers/vulkan/'):
        return name.replace('shs/rhi/drivers/vulkan/', 'shs/rhi/vulkan/runtime/'), 'review-vulkan-runtime-ownership'
    return name, 'already-named-review-dependencies'


def components(graph):
    """Tarjan SCCs over all conditional edges, not just today's build."""
    index, low, stack, active, result = {}, {}, [], set(), []

    def visit(node):
        index[node] = low[node] = len(index)
        stack.append(node)
        active.add(node)
        for dep in graph[node]:
            if dep not in graph:
                continue
            if dep not in index:
                visit(dep)
                low[node] = min(low[node], low[dep])
            elif dep in active:
                low[node] = min(low[node], index[dep])
        if low[node] == index[node]:
            group = []
            while True:
                dep = stack.pop()
                active.remove(dep)
                group.append(dep)
                if dep == node:
                    break
            if len(group) > 1 or node in graph[node]:
                result.append(sorted(group))

    for node in sorted(graph):
        if node not in index:
            visit(node)
    return sorted(result)



def inventory(root):
    scope = 'cpp-folders/src/shs-renderer-lib/include'
    base = root / scope
    manifest = json.loads((root / 'docs/backlog/engine_header_migration_manifest.json').read_text())
    approved = {e['canonical']: e for e in manifest['headers']}
    legacy = {e['old']: e for e in manifest['headers']}
    texts = {p.relative_to(base).as_posix(): p.read_text()
             for p in sorted((base / 'shs').rglob('*')) if p.suffix in ('.hpp', '.h')}
    graph = {name: sorted(set(includes(text))) for name, text in texts.items()}
    consumers = {name: [] for name in texts}
    tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')
    for name in tracked:
        path = root / name
        if path.suffix not in ('.cpp', '.hpp', '.h', '.cc', '.cxx') or not path.is_file():
            continue
        for dep in set(includes(path.read_text())):
            if dep in consumers:
                consumers[dep].append(name)
    headers = []
    for name, text in texts.items():
        target, status = destination(name)
        if name in approved:
            target, status = name, 'approved-canonical'
        elif name in legacy:
            target, status = legacy[name]['canonical'], 'approved-compatibility-forwarder'
        headers.append({
            'current': name, 'proposed_canonical': target, 'review_status': status,
            'proposed_owner': target.split('/')[1],
            'namespace_declarations': sorted(set(re.findall(r'\bnamespace\s+([\w:]+)\s*\{', text))),
            'visibility': 'public-include-tree',
            'direct_dependencies': graph[name],
            'internal_dependencies': [d for d in graph[name] if d.startswith('shs/')],
            'external_dependencies': [d for d in graph[name] if not d.startswith('shs/')],
            'tracked_consumers': sorted(consumers[name]),
            'build_target': 'shs::renderer-values' if name in approved or name in legacy else 'review-values-vs-aggregate',
        })
    destinations = {}
    for h in headers:
        if h['review_status'] != 'approved-compatibility-forwarder':
            destinations.setdefault(h['proposed_canonical'], []).append(h['current'])
    modules = {}
    for h in headers:
        owner = h['proposed_owner']
        modules.setdefault(owner, set())
        for dep in h['internal_dependencies']:
            dep_owner = destination(dep)[0].split('/')[1]
            if dep_owner != owner:
                modules[owner].add(dep_owner)
    module_graph = {key: sorted(value) for key, value in sorted(modules.items())}
    return {
        'schema_version': 1, 'scope': scope,
        'coverage': 'all .h/.hpp public headers; literal includes in all conditional branches',
        'approval': 'proposed paths are inventory only; reviewed migrations live in engine_header_migration_manifest.json',
        'header_count': len(headers), 'headers': headers,
        'missing_internal_includes': sorted({d for deps in graph.values() for d in deps
                                            if d.startswith('shs/') and d not in texts}),
        'header_cycles': components(graph),
        'proposed_module_dependencies': module_graph,
        'proposed_module_cycles': components(module_graph),
        'destination_collisions': {k: v for k, v in sorted(destinations.items()) if len(v) > 1},
        'vulkan_decision': 'Keep value/ and runtime/ separate under rhi/vulkan; no implementation merge or symbol/ABI claim. Review required before relocation.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--write', action='store_true', help='explicitly regenerate the committed inventory')
    args = parser.parse_args()
    path = args.root / 'docs/backlog/engine_header_inventory.json'
    content = json.dumps(inventory(args.root), indent=2, ensure_ascii=False) + '\n'
    if args.write:
        path.write_text(content)
        print(f'Wrote {path}')
        return 0
    if not path.exists() or path.read_text() != content:
        print('Header inventory is stale; review changes and run inventory_headers.py --write', file=sys.stderr)
        return 1
    print('Full header inventory matches source tree')
    return 0


if __name__ == '__main__':
    sys.exit(main())
