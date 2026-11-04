"""Export paper summaries from completed receipts and saved predictions."""
import csv
from collections import defaultdict

from aligntrace.io import open_text, save_json


RATE_METRICS = ('success_percent', 'mean_ed', 'top5_percent', 'within1_percent',
                'top5_within1_percent', 'top5_mean_min_ed')


def aggregate(entries):
    """Weight each dataset by its evaluated cluster count, including abstentions."""
    n = sum(entry['metrics']['n'] for entry in entries)
    metrics = dict(n=n, exact_count=sum(entry['metrics']['exact_count'] for entry in entries))
    for key in RATE_METRICS:
        metrics[key] = sum(entry['metrics'][key] * entry['metrics']['n'] for entry in entries) / n
    for name, rate in [('top5_exact_count', 'top5_percent'),
                       ('within1_count', 'within1_percent'),
                       ('top5_within1_count', 'top5_within1_percent')]:
        metrics[name] = sum(round(entry['metrics'][rate] * entry['metrics']['n'] / 100)
                            for entry in entries)
    metrics['milliseconds_per_cluster'] = 1000 * sum(entry['mean_seconds'] for entry in entries) / n
    return metrics


def paired_depth(low, high):
    """Join saved depth variants by reference hash, never by shuffled row order."""
    def index(rows):
        indexed = {row['reference_sha256']: row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError('Paired-depth analysis requires unique references')
        return indexed

    lower, higher = index(low), index(high)
    if lower.keys() != higher.keys():
        raise ValueError('Paired-depth variants must contain the same references')
    subsets = defaultdict(list)
    for key, left in lower.items():
        right = higher[key]
        added = right['read_count'] - left['read_count']
        if added < 0:
            raise ValueError('Higher-depth variant has fewer reads')
        item = (left['distances'][0] == 0, right['distances'][0] == 0, added)
        subsets['all'].append(item)
        subsets['additional_reads' if added else 'unchanged_reads'].append(item)
    result = {}
    for name, pairs in subsets.items():
        n = len(pairs)
        before, after = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
        result[name] = dict(n=n, lower_exact_count=before, higher_exact_count=after,
            lower_success_percent=100 * before / n, higher_success_percent=100 * after / n,
            failure_to_exact=sum(not a and b for a, b, _ in pairs),
            exact_to_failure=sum(a and not b for a, b, _ in pairs),
            change_percentage_points=100 * (after - before) / n,
            added_reads=sum(p[2] for p in pairs),
            minimum_added_reads=min(p[2] for p in pairs),
            maximum_added_reads=max(p[2] for p in pairs))
    return result


def write_summaries(receipts, out, smoke=False):
    """Write only the selected comparisons; record coverage for partial runs."""
    import json

    groups = defaultdict(list)
    by_id = {receipt['id']: receipt for receipt in receipts}
    strata = []
    coverage = []
    for receipt in receipts:
        spec = receipt['spec']
        cohort = 'synthetic' if spec['cohort'] == 'synthetic_depth' else spec['cohort']
        for variant, entry in receipt['variants'].items():
            groups[(cohort, variant, spec.get('requested_percent'))].append(entry)
            coverage.append(dict(dataset_id=spec['id'], variant=variant, n=entry['metrics']['n']))
            for kind, partitions in entry['strata'].items():
                for value, metrics in partitions.items():
                    row = dict(dataset_id=spec['id'], variant=variant, stratum=kind, value=value,
                               **metrics)
                    if kind == 'target_length':
                        row['training_seen_length'] = int(value) in (80, 120, 140, 180)
                    strata.append(row)
    pooled = [dict(cohort=cohort, variant=variant, requested_percent=percent,
                   dataset_count=len(entries), **aggregate(entries))
              for (cohort, variant, percent), entries in groups.items()]
    paired = {}
    if not smoke:
        for channel in ('burst8_error', 'burst10_error'):
            ids = ['synthetic_depth__' + channel + suffix
                   for suffix in ('_low_seq_depth', '_high_seq_depth')]
            if not all(ident in by_id for ident in ids):
                continue
            banks = []
            for ident in ids:
                entry = by_id[ident]['variants']['synthetic']
                with open_text(out / entry['predictions']) as handle:
                    banks.append([json.loads(line) for line in handle if line.strip()])
            paired[channel] = paired_depth(*banks)
    save_json(out / 'TABLE_SUMMARIES.json', dict(smoke=smoke, coverage=coverage,
        pooled=pooled, paired_depth=paired,
        note='Aggregates include only the listed selected comparisons. Cluster counts are evaluation instances; targets recur across conditions. Stratum metrics reuse the same predictions and have no separate timing measurement.'))
    fields = ['dataset_id', 'variant', 'stratum', 'value', 'training_seen_length',
              'n', 'exact_count', *RATE_METRICS]
    with (out / 'stratified_metrics.csv').open('w', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(strata)
