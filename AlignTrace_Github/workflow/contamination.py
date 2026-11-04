"""Paired additive contamination: retain inliers and add independent donor reads."""
from __future__ import annotations

import collections
import hashlib
import math
import random
from .inputs import read_rows

# Fixed simulation stream identifiers, retained to reproduce the released data.
STREAM_KEYS = {'burst4_error': 'test_user_burst4',
 'dt4dds_deep_pcr': 'test_dt4dds_deep_pcr',
 'framed_paper_05': 'test_framed_paper_05',
 'iid_high_error': 'test_user_iid_high',
 'iid_low_error': 'test_user_iid_low',
 'storalator_long_deletion_stress': 'test_storalator_long_deletion_stress',
 'storalator_nanopore_minion': 'test_storalator_nanopore_minion'}
SEED = 2026103019


def source_bank(path, name):
    """Select all 600 length-140 targets, retaining source order and profile."""
    bank = []
    for row in read_rows(path):
        meta = row['meta']
        reference, reads = row['reference'], row['reads']
        if (hashlib.sha256(reference.encode()).hexdigest()!=meta['reference_sha256']
                or len(reads)!=meta['n_reads']
                or not len(reference)==row['target_length']==meta['reference_length']):
            raise ValueError('Contamination source content differs from its metadata')
        if row['target_length'] != 140:
            continue
        if meta['truth']['contaminant_count']!=0:
            raise ValueError('Additive contamination requires an uncontaminated source bank')
        bank.append(dict(reference=reference, reads=reads, family_id=meta['family_id'],
                         profile=meta['profile'], source=name,
                         source_row=meta['record_index'], target_length=140))
    if len(bank)!=600 or len({r['reference'] for r in bank})!=600:
        raise ValueError('Expected 600 unique length-140 contamination targets')
    return bank


def contaminate(bank, percent):
    """Generate paired mixtures using fixed per-family random streams."""
    if not 0<=percent<100:
        raise ValueError('Contamination percentage must be in [0,100)')
    profiles = collections.defaultdict(list)
    for row in bank:
        profiles[row['profile']].append(row)
    result = []
    for row in bank:
        seed_string = f'{SEED}:{STREAM_KEYS[row["source"]]}:{row["family_id"]}'
        seed = int(hashlib.sha256(seed_string.encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
        n = len(row['reads'])
        # Reuse the same rounding draw and donor stream at every level.
        # Changing draw order changes the frozen benchmark inputs.
        offset = rng.random()
        added = math.floor(n * percent / (100 - percent) + offset)
        entries = [(rng.random(), read, False, row['family_id']) for read in row['reads']]
        donors = [d for d in profiles[row['profile']] if d['reference'] != row['reference']]
        if not donors:
            raise ValueError('No foreign donor available in the matching profile')
        for _ in range(added):
            donor = rng.choice(donors)
            read = rng.choice(donor['reads'])
            entries.append((rng.random(), read, True, donor['family_id']))
        entries.sort(key=lambda x: x[0])
        output = dict(row)
        output.update(id=f'{row["source"]}:{row["family_id"]}:p{percent:02}',
                      reads=[e[1] for e in entries], foreign_mask=[e[2] for e in entries],
                      donors=[e[3] for e in entries], inlier_count=n, foreign_count=added,
                      requested_percent=percent, actual_percent=100 * added / len(entries))
        result.append(output)
    return result


def validate(rows, bank, previous=None):
    """Check labels, retained inliers and nested additions, preserving multiplicity."""
    if len(rows)!=len(bank) or (previous is not None and len(previous)!=len(bank)):
        raise ValueError('Contamination banks must contain the same targets')
    for index, (row, original) in enumerate(zip(rows, bank)):
        if not len(row['reads']) == len(row['foreign_mask']) == len(row['donors']):
            raise ValueError('Contamination labels do not match read count')
        if sum(row['foreign_mask']) != row['foreign_count']:
            raise ValueError('Contamination count does not match labels')
        inliers = [s for s, foreign in zip(row['reads'], row['foreign_mask']) if not foreign]
        if collections.Counter(inliers) != collections.Counter(original['reads']):
            raise ValueError('Contamination must retain every genuine read')
        if not all(d != row['family_id'] for d, foreign in zip(row['donors'], row['foreign_mask']) if foreign):
            raise ValueError('A donor must come from another target family')
        if previous is not None:
            # Donor/read tuples handle duplicate strings and repeated donor draws correctly.
            old = collections.Counter(zip(previous[index]['reads'], previous[index]['foreign_mask'], previous[index]['donors']))
            new = collections.Counter(zip(row['reads'], row['foreign_mask'], row['donors']))
            if old-new:
                raise ValueError(f'Non-nested contamination additions: {row["id"]}')
