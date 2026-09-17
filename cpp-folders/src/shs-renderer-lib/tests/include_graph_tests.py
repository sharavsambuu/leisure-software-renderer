#!/usr/bin/env python3
"""Include-graph dependency gate tests: isolated fixtures plus live-tree
checks. Never mutates the real include tree."""
import importlib.util
import json
from pathlib import Path
import unittest

TOOL = Path(__file__).resolve().parents[1] / 'tools/check_include_graph.py'
EXCEPTIONS = Path(__file__).resolve().parents[1] / 'tools/engine_include_exceptions.json'
spec = importlib.util.spec_from_file_location('include_graph', TOOL)
include_graph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(include_graph)


class IncludeGraphGateTests(unittest.TestCase):
    def test_self_test_fixtures(self):
        """Synthetic negative/positive fixtures (cycles, SDK placement,
        value->adapter reachability) all behave as specified."""
        self.assertEqual(include_graph.run_self_test(), 0)

    def test_live_tree_is_green(self):
        """The real include tree passes the gate with tracked exceptions."""
        violations = include_graph.check()
        self.assertEqual(violations, [])

    def test_gate_is_not_vacuous(self):
        """The gate must actually see the real tree (guards against path
        regressions silently loading zero headers)."""
        headers = include_graph.load_canonical_headers()
        self.assertGreater(len(headers), 200)
        tiers = set(include_graph.classify(r, t) for r, t in headers.items())
        self.assertEqual(tiers, {'value', 'integration'})

    def test_exceptions_manifest_shape(self):
        """Every tracked exception names a live header, a known rule, a
        reason and a tracking note."""
        data = json.loads(EXCEPTIONS.read_text())
        self.assertEqual(data['schema_version'], include_graph.SCHEMA_VERSION)
        headers = include_graph.load_canonical_headers()
        for entry in data['exceptions']:
            self.assertIn(entry['header'], headers, entry)
            self.assertIn(entry['rule'], (
                'R1-header-cycle', 'R2-sdk-include-in-value-tier',
                'R3-value-reaches-adapter'), entry)
            self.assertTrue(entry.get('reason', '').strip(), entry)
            self.assertTrue(entry.get('tracking', '').strip(), entry)

    def test_exceptions_are_live(self):
        """Every tracked exception must still be an actual violation without
        exceptions; stale entries (already-fixed violations) are reported
        so the manifest can be pruned."""
        data = json.loads(EXCEPTIONS.read_text())
        live = include_graph.check(
            exceptions={'schema_version': 1, 'exceptions': []})
        live_by_header = {}
        for v in live:
            live_by_header.setdefault(v['header'], set()).add(v['rule'])
        for entry in data['exceptions']:
            header, rule = entry['header'], entry['rule']
            self.assertIn(rule, live_by_header.get(header, set()),
                          f'stale exception (violation no longer fires): '
                          f'{header} / {rule}')


if __name__ == '__main__':
    unittest.main()
