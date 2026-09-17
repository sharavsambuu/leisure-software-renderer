#!/usr/bin/env python3
"""Isolated positive/negative tests: never mutate the real include tree."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / 'tools/check_header_migration.py'
spec = importlib.util.spec_from_file_location('migration', TOOL)
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


class HeaderMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.entry = {
            'old': 'shs/domains/camera/convention.hpp',
            'canonical': 'shs/camera/convention.hpp',
            'owner': 'camera', 'status': 'migrated-with-forwarder',
            'policy': 'pure-leaf', 'direct_dependencies': ['glm/glm.hpp'],
        }
        self.manifest = {'scope': 'include', 'headers': [self.entry]}
        self.write(self.entry['old'], migration.forwarder(self.entry))
        self.write(self.entry['canonical'], '#pragma once\n#include <glm/glm.hpp>\n')

    def write(self, name, text):
        path = self.root / 'include' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def errors(self):
        return migration.validate(self.root, self.manifest)

    def test_valid_leaf(self):
        self.assertEqual(self.errors(), [])

    def test_duplicate_mapping(self):
        self.manifest['headers'].append(copy.deepcopy(self.entry))
        self.assertIn('duplicate migration mapping', self.errors())

    def test_forwarder_extra_code(self):
        self.write(self.entry['old'], migration.forwarder(self.entry) + 'int state;\n')
        self.assertTrue(any('content drift' in e for e in self.errors()))

    def test_missing_canonical(self):
        (self.root / 'include' / self.entry['canonical']).unlink()
        self.assertTrue(any('missing canonical' in e for e in self.errors()))

    def test_unregistered_header(self):
        self.write('shs/camera/new.hpp', '#pragma once\n')
        self.assertTrue(any('unregistered' in e for e in self.errors()))

    def test_retired_include(self):
        self.write('shs/domains/input/input.hpp', '#include "' + self.entry['old'] + '"\n')
        self.assertTrue(any('legacy include' in e for e in self.errors()))

    def test_driver_and_app_rejected_even_if_manifest_changed(self):
        for dep in ('shs/execution/app/app.hpp', 'shs/rhi/drivers/vulkan/vk_backend.hpp'):
            with self.subTest(dep=dep):
                self.write(dep, '#pragma once\n')
                self.write(self.entry['canonical'], '#include "' + dep + '"\n')
                self.entry['direct_dependencies'] = [dep]
                self.assertTrue(any('pure leaf includes internal' in e for e in self.errors()))

    def test_transitive_cycle_and_side_effect(self):
        self.write(self.entry['canonical'], '#include "shs/core/bridge.hpp"\n')
        self.write('shs/core/bridge.hpp', '#include "shs/core/other.hpp"\n')
        self.write('shs/core/other.hpp', '#include "shs/core/bridge.hpp"\n// std::chrono\n')
        errors = self.errors()
        self.assertTrue(any('cycle reachable' in e for e in errors))
        self.assertTrue(any('ambient side effect' in e for e in errors))

    def test_dependency_drift(self):
        self.write(self.entry['canonical'], '#include <vulkan/vulkan.h>\n')
        self.assertTrue(any('dependency drift' in e for e in self.errors()))

    def test_comment_includes_are_ignored(self):
        self.assertEqual(migration.includes('/* #include "bad.hpp" */\n'
                                          '// #include "bad.hpp"\n'
                                          '# include <glm/glm.hpp>\n'), ['glm/glm.hpp'])

    def test_unsafe_path(self):
        self.entry['canonical'] = '../outside.hpp'
        self.assertTrue(any('unsafe' in e for e in self.errors()))

    def test_sdk_dependency_cannot_be_added_to_manifest(self):
        self.entry['direct_dependencies'] = ['vulkan/vulkan.h']
        self.write(self.entry['canonical'], '#include <vulkan/vulkan.h>\n')
        self.assertTrue(any('only permits GLM' in e for e in self.errors()))

    def test_macro_include_rejected(self):
        self.write(self.entry['canonical'], '#include <glm/glm.hpp>\n#include SDK_HEADER\n')
        self.assertTrue(any('nonliteral' in e for e in self.errors()))

    def test_inventory_cycles_and_destinations(self):
        sys.path.insert(0, str(TOOL.parent))
        self.addCleanup(sys.path.pop, 0)
        import inventory_headers as inventory
        self.assertEqual(inventory.components({'a': ['b'], 'b': ['a'], 'c': []}), [['a', 'b']])
        self.assertEqual(inventory.components({'a': ['b'], 'b': []}), [])
        self.assertEqual(inventory.destination('shs/domains/gfx/edge/rt_registry.hpp')[0],
                         'shs/render/targets/storage/rt_registry.hpp')
        value = inventory.destination('shs/execution/rhi/drivers/vulkan/vk_backend.hpp')[0]
        runtime = inventory.destination('shs/rhi/drivers/vulkan/vk_backend.hpp')[0]
        self.assertNotEqual(value, runtime)

    def test_inventory_cli_exact_output_and_drift(self):
        # Minimal repository fixture exercises the real CLI and complete JSON,
        # independently of the renderer checkout and its developer build cache.
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        scope = 'cpp-folders/src/shs-renderer-lib/include'
        header = self.root / scope / 'shs/core/value.hpp'
        header.parent.mkdir(parents=True)
        header.write_text('#pragma once\n#include <cstdint>\nnamespace shs { }\n')
        docs = self.root / 'docs/backlog'
        docs.mkdir(parents=True)
        (docs / 'engine_header_migration_manifest.json').write_text(
            json.dumps({'scope': scope, 'headers': []}))
        subprocess.run(['git', '-C', str(self.root), 'add', '.'], check=True)
        tool = TOOL.parent / 'inventory_headers.py'
        command = [sys.executable, str(tool), '--root', str(self.root)]
        result = subprocess.run(command + ['--write'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = docs / 'engine_header_inventory.json'
        expected = {
            'schema_version': 1, 'scope': scope,
            'coverage': 'all .h/.hpp public headers; literal includes in all conditional branches',
            'approval': 'proposed paths are inventory only; reviewed migrations live in engine_header_migration_manifest.json',
            'header_count': 1,
            'headers': [{
                'current': 'shs/core/value.hpp', 'proposed_canonical': 'shs/core/value.hpp',
                'review_status': 'already-named-review-dependencies', 'proposed_owner': 'core',
                'namespace_declarations': ['shs'], 'visibility': 'public-include-tree',
                'direct_dependencies': ['cstdint'], 'internal_dependencies': [],
                'external_dependencies': ['cstdint'], 'tracked_consumers': [],
                'build_target': 'review-values-vs-aggregate',
            }],
            'missing_internal_includes': [], 'header_cycles': [],
            'proposed_module_dependencies': {'core': []}, 'proposed_module_cycles': [],
            'destination_collisions': {},
            'vulkan_decision': 'Keep value/ and runtime/ separate under rhi/vulkan; no implementation merge or symbol/ABI claim. Review required before relocation.',
        }
        self.assertEqual(output.read_text(), json.dumps(expected, indent=2, ensure_ascii=False) + '\n')
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        header.write_text('#include <cstddef>\n')
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('stale', result.stderr)

    def test_cli_failure_exit(self):
        path = self.root / 'manifest.json'
        self.write(self.entry['old'], 'broken\n')
        path.write_text(json.dumps(self.manifest))
        result = subprocess.run([sys.executable, str(TOOL), '--root', str(self.root),
                                 '--manifest', str(path)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('content drift', result.stderr)


if __name__ == '__main__':
    unittest.main()
