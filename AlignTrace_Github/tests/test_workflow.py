"""Regression checks for input isolation, filtering, and CPU reconstruction."""
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from aligntrace.pipeline import Pipeline
from aligntrace.io import ROOT, load_records, read_json, sha
from aligntrace.prefilter_learned import InputGate10, filter_reads
from aligntrace.shortcut import validate_and_group


class WorkflowTests(unittest.TestCase):
    def test_bundled_checkpoints(self):
        for entry in read_json(ROOT / 'models/MANIFEST.json').values():
            self.assertEqual(sha(ROOT / entry['path']), entry['sha256'])
        for entry in read_json(ROOT / 'protocol/FINETUNED_MODELS.json').values():
            for kind in ('generator', 'reranker'):
                self.assertEqual(sha(ROOT / entry[kind]), entry[kind + '_sha256'])

    def test_inference_discards_reference_and_foreign_labels(self):
        row = dict(id='test', target_length=12, reads=['ACGTACGTACGT'],
                   reference='TTTTTTTTTTTT', foreign_mask=[True])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reads.jsonl'
            path.write_text(json.dumps(row), encoding='utf8')
            loaded = load_records(path)[0]
        self.assertEqual(set(loaded), {'id', 'target_length', 'reads'})
        self.assertEqual(loaded['reads'], row['reads'])

    def test_validation_and_shortcut(self):
        sequence = 'ACGT' * 35
        groups, bypass = validate_and_group([[], [sequence] * 3], [140, 140], True)
        self.assertFalse(groups)
        self.assertEqual(bypass, {1: [sequence]})
        for reads, length in [(['ACGN'], 4), ([''], 4), (['ACGT'], 2)]:
            with self.assertRaises(ValueError):
                validate_and_group([reads], [length])

    def test_learned_filter_retains_inliers(self):
        gate = InputGate10(ROOT / 'models/prefilter/gate.json')
        reads = ['ACGT' * 35] * 6 + ['T' * 140] * 4
        selected, kept, info = filter_reads([reads], gate)
        self.assertTrue(info['evaluated'][0])
        self.assertTrue(info['flagged'][0])
        self.assertEqual(kept, [list(range(6))])
        self.assertEqual(selected, [reads[:6]])

    def test_small_clusters_are_retained(self):
        gate = InputGate10(ROOT / 'models/prefilter/gate.json')
        reads = ['ACGTACGT', 'TTTTTTTT', 'GGGGGGGG']
        selected, kept, _ = filter_reads([reads], gate)
        self.assertEqual(selected, [reads])
        self.assertEqual(kept, [[0, 1, 2]])

    def test_filter_control(self):
        self.assertEqual(list(inspect.signature(Pipeline).parameters),
                         ['filter_on', 'shortcut_on', 'batch_size', 'device', 'generator', 'reranker'])
        self.assertEqual(list(inspect.signature(filter_reads).parameters), ['read_sets', 'gate'])

    def test_cpu_inference_filter_and_shortcut_combinations(self):
        rows = load_records(ROOT / 'examples/demo.jsonl')
        reads = [row['reads'] for row in rows] + [[]]
        lengths = [row['target_length'] for row in rows] + [140]
        for filter_on in (False, True):
            for shortcut_on in (False, True):
                with self.subTest(filter_on=filter_on, shortcut_on=shortcut_on):
                    pipe = Pipeline(filter_on=filter_on, shortcut_on=shortcut_on,
                                    batch_size=1, device='cpu')
                    try:
                        predictions, timing, diagnostics = pipe.predict(reads, lengths)
                        repeated, _, _ = pipe.predict(reads, lengths)
                    finally:
                        pipe.close()
                    self.assertEqual(predictions, repeated)
                    self.assertEqual(predictions[-1], [])
                    self.assertEqual(timing['empty_clusters'], 1)
                    self.assertEqual(len(predictions), len(reads))
                    for index, candidates in enumerate(predictions[:-1]):
                        self.assertTrue(1 <= len(candidates) <= 5)
                        self.assertEqual(len(candidates), len(set(candidates)))
                        self.assertTrue(all(len(s) == lengths[index] and set(s) <= set('ACGT')
                                            for s in candidates))
                        evaluated = filter_on and index not in diagnostics['shortcut_indices']
                        self.assertEqual(diagnostics['prefilter_gate_evaluated'][index], evaluated)
                    if not filter_on:
                        self.assertEqual(diagnostics['kept_indices'],
                                         [list(range(len(group))) for group in reads])


if __name__ == '__main__':
    unittest.main()
