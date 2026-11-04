"""Regression checks for scientific summaries and portable input preparation."""
import io
import unittest

from aligntrace.io import ROOT, iter_binned, read_json
from workflow.reporting import aggregate, paired_depth


class ProtocolTests(unittest.TestCase):
    def test_catalog_matches_paper_scope(self):
        catalog = read_json(ROOT / 'protocol/DATASETS.json')
        self.assertEqual(len(catalog), 93)
        self.assertEqual(sum(len(spec['variants']) for spec in catalog), 141)
        contamination = [spec for spec in catalog if spec['cohort'] == 'contamination_sweep']
        self.assertEqual(len(contamination), 42)
        self.assertEqual({spec['requested_percent'] for spec in contamination}, {0, 10, 20, 30, 40, 50})
        self.assertEqual(sum(spec['n'] for spec in contamination), 25200)

    def test_binned_reader_retains_empty_cluster_and_final_block(self):
        text = 'ACGT\n****\n\nTGCA\n****\nTGCA'
        self.assertEqual(list(iter_binned(io.StringIO(text))), [('ACGT', []), ('TGCA', ['TGCA'])])
        with self.assertRaises(ValueError):
            list(iter_binned(io.StringIO('ACGT\nACGT\n')))

    def test_pooled_accuracy_weights_clusters_and_time(self):
        def entry(n, exact, seconds):
            return dict(mean_seconds=seconds, metrics=dict(n=n, exact_count=exact,
                success_percent=100 * exact / n, mean_ed=(n - exact) / n,
                top5_percent=100 * exact / n, within1_percent=100,
                top5_within1_percent=100, top5_mean_min_ed=(n - exact) / n))
        result = aggregate([entry(1, 1, .01), entry(9, 0, .09)])
        self.assertEqual(result['n'], 10)
        self.assertEqual(result['success_percent'], 10)
        self.assertEqual(result['top5_exact_count'], 1)
        self.assertAlmostEqual(result['mean_ed'], .9)
        self.assertAlmostEqual(result['milliseconds_per_cluster'], 10)

    def test_paired_depth_joins_target_identity_and_counts_transitions(self):
        def row(key, depth, distance):
            return dict(reference_sha256=key, read_count=depth, distances=[distance])
        lower = [row('a', 2, 1), row('b', 3, 0), row('c', 4, 0)]
        higher = [row('c', 4, 0), row('a', 12, 0), row('b', 13, 1)]
        result = paired_depth(lower, higher)
        self.assertEqual(result['all']['failure_to_exact'], 1)
        self.assertEqual(result['all']['exact_to_failure'], 1)
        self.assertEqual(result['additional_reads']['n'], 2)
        self.assertEqual(result['unchanged_reads']['higher_exact_count'], 1)
        with self.assertRaises(ValueError):
            paired_depth(lower, higher[:-1])


if __name__ == '__main__':
    unittest.main()
