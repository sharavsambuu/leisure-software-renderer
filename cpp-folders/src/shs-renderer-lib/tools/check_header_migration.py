#!/usr/bin/env python3
"""Enforce reviewed migrations. Uses only the Python standard library.

Literal includes in all conditional branches count as dependencies. This is a
conservative source check, not a C++ preprocessor or a proof of semantic purity.
"""
import argparse
import json
from pathlib import Path
import re
import sys

INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"\n]+)[>"]', re.MULTILINE)
AMBIENT = re.compile(r'rand\(|srand\(|std::chrono|std::time\(|getenv\(|random_|SDL_|fopen\(')
ROOT = Path(__file__).resolve().parents[4]


def includes(text):
    # Ignore commented-out directives, but retain strings (include operands).
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.DOTALL)
    return INCLUDE.findall(text)


def forwarder(entry):
    return ('#pragma once\n\n'
            '// Compatibility include: definitions live in the '
            f'{entry["owner"]}-owned canonical header.\n'
            f'#include "{entry["canonical"]}"\n')


def validate(root, manifest):
    errors = []
    base = root / manifest['scope']
    entries = manifest['headers']
    canonical = {e['canonical']: e for e in entries}
    legacy = {e['old'] for e in entries}
    if len(canonical) != len(entries) or len(legacy) != len(entries):
        errors.append('duplicate migration mapping')
    if legacy & canonical.keys():
        errors.append('forwarder chain or self mapping')
    for e in entries:
        for key in ('old', 'canonical'):
            p = Path(e[key])
            if p.is_absolute() or '..' in p.parts or not e[key].startswith('shs/'):
                errors.append(f'unsafe {key} path: {e[key]}')
        if e['status'] != 'migrated-with-forwarder':
            errors.append(f'unsupported status: {e["status"]}')
    if errors:
        return errors
    texts = {p.relative_to(base).as_posix(): p.read_text()
             for p in sorted((base / 'shs').rglob('*.hpp'))}
    graph = {name: includes(text) for name, text in texts.items()}
    for name, e in canonical.items():
        if name not in texts:
            errors.append(f'missing canonical header: {name}')
            continue
        if texts.get(e['old']) != forwarder(e):
            errors.append(f'compatibility content drift: {e["old"]}')
        if graph[name] != e['direct_dependencies']:
            errors.append(f'dependency drift: {name}')
        if e.get('policy') != 'pure-leaf':
            errors.append(f'unreviewed policy: {name}')
        if any(not dep.startswith('glm/') for dep in e['direct_dependencies']):
            errors.append(f'pure-leaf policy only permits GLM dependencies: {name}')
        directives = re.findall(r'^\s*#\s*include\b[^\n]*', texts[name], re.MULTILINE)
        if len(directives) != len(graph[name]) or '\\\n' in texts[name]:
            errors.append(f'nonliteral or continued include directive: {name}')
        # Traverse actual includes, not the manifest's asserted dependencies.
        seen = set()
        active = set()

        def visit(node):
            if node in active:
                errors.append(f'include cycle reachable from {name}: {node}')
                return
            if node in seen:
                return
            seen.add(node)
            active.add(node)
            if AMBIENT.search(texts.get(node, '')):
                errors.append(f'ambient side effect reachable from {name}: {node}')
            for dep in graph.get(node, []):
                if dep.startswith('shs/'):
                    # A reviewed pure leaf has NO internal dependencies. Adding
                    # a dependency also to the manifest cannot bypass this law.
                    errors.append(f'pure leaf includes internal header: {node} -> {dep}')
                    if dep not in texts:
                        errors.append(f'missing internal dependency: {dep}')
                    else:
                        visit(dep)
                elif dep not in e['direct_dependencies']:
                    errors.append(f'unapproved transitive dependency: {node} -> {dep}')
            active.remove(node)

        visit(name)
    for name, deps in graph.items():
        for dep in deps:
            if dep in legacy:
                errors.append(f'legacy include in library header: {name} -> {dep}')
        # Preserve legacy-zone enforcement; new named modules are opt-in.
        parts = Path(name).parts
        if parts[1] not in ('domains', 'execution', 'core', 'memory', 'containers', 'rhi'):
            if name not in canonical:
                errors.append(f'unregistered canonical header: {name}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    path = args.manifest or args.root / 'docs/backlog/engine_header_migration_manifest.json'
    try:
        errors = validate(args.root, json.loads(path.read_text()))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors = [f'invalid migration manifest: {exc}']
    for error in errors:
        print(f'[header-migration] FAIL: {error}', file=sys.stderr)
    if not errors:
        print('[header-migration] reviewed canonical headers and compatibility mappings passed')
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
