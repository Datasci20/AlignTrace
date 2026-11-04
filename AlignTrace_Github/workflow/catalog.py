"""Dataset selectors and protocol settings."""
import fnmatch
from pathlib import PurePosixPath


def aliases(spec):
    names = {spec[key] for key in ('id', 'name') if spec.get(key)}
    if '__' in spec['id']:
        names.add(spec['id'].split('__', 1)[1])
    for source in spec.get('sources', []):
        filename = PurePosixPath(source['path'].replace('\\', '/')).name
        for suffix in ('.metadata.jsonl', '.summary.json', '.jsonl.gz', '.jsonl', '.txt'):
            if filename.endswith(suffix):
                names.add(filename[:-len(suffix)])
                break
    return names


def select_datasets(catalog, cohort='all', patterns=None):
    def matches(spec):
        if cohort != 'all' and spec['cohort'] != cohort and not (
            cohort == 'synthetic' and spec['cohort'] == 'synthetic_depth'
        ):
            return False
        return not patterns or any(fnmatch.fnmatchcase(name, pattern)
                                   for name in aliases(spec) for pattern in patterns)
    return [spec for spec in catalog if matches(spec)]


def describe(spec):
    filters = ['learned' if variant == 'learned_filter' else 'off' for variant in spec['variants']]
    detail = (f"name={spec['name']} clusters={spec['n']} "
              f"shortcut={'on' if spec['shortcut'] else 'off'} filter={','.join(filters)}")
    if spec['cohort'] == 'contamination_sweep':
        detail += f" requested_contamination={spec['requested_percent']}%"
    return detail
