"""Prepare and verify the frozen benchmark inputs using the standard library."""
from __future__ import annotations
import gzip
import hashlib
import itertools
import json
from pathlib import Path
from aligntrace.io import ROOT,iter_binned,read_json,sha
def normalized(row):
    return {key:row[key] for key in ('id','reference','reads','target_length')}
def record_digest(rows):
    h=hashlib.sha256()
    for r in rows:
        h.update((json.dumps(normalized(r),separators=(',',':'),ensure_ascii=True)+'\n').encode())
    return h.hexdigest()
def read_rows(path):
    with gzip.open(path,'rt',encoding='utf8') as handle:
        return [json.loads(line) for line in handle if line.strip()]
def binned(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        for reference,reads in iter_binned(handle):
            yield dict(id=None,reference=reference,reads=reads)

def read_real(spec,root):
    paths=[root/s['path'] for s in spec['sources']]
    if len(paths)==1:
        rows=list(binned(paths[0]))
    else:
        refs=[s.strip() for s in paths[0].read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        groups=[]; current=None
        with paths[1].open(encoding='utf-8-sig') as handle:
            for line in handle:
                line=line.strip()
                if line.startswith('CLUSTER ') or (line and set(line)=={'='}):
                    if current is not None: groups.append(current)
                    current=[]
                elif line:
                    if current is None: raise ValueError('Read before cluster header')
                    current.append(line)
        if current is not None: groups.append(current)
        if len(refs)!=len(groups): raise ValueError(f'Reference/cluster count mismatch: {spec["name"]}')
        rows=[dict(reference=ref,reads=reads) for ref,reads in zip(refs,groups)]
    for i,row in enumerate(rows): row.update(id=i,target_length=spec['target_length'])
    return rows

def read_synthetic(spec,root):
    text,metadata=[root/s['path'] for s in spec['sources']]
    rows=[]
    with metadata.open(encoding='utf8') as handle:
        for i,(row,line) in enumerate(itertools.zip_longest(binned(text),handle)):
            if row is None or line is None: raise ValueError('Synthetic TXT/metadata length mismatch')
            meta=json.loads(line)
            if meta['record_index']!=i or len(row['reads'])!=meta['n_reads']:
                raise ValueError(f'Synthetic record index/read count mismatch: {text}, row {i}')
            if hashlib.sha256(row['reference'].encode()).hexdigest()!=meta['reference_sha256']:
                raise ValueError(f'Synthetic reference hash mismatch: {text}, row {i}')
            ident='case_'+hashlib.sha256(meta['id'].encode()).hexdigest()[:24] if spec['cohort']=='synthetic_depth' else i
            row.update(id=ident,target_length=meta['reference_length'],meta=meta)
            rows.append(row)
    return rows

def validate(rows,spec):
    if len(rows)!=spec['n']: raise ValueError(f'Cluster count differs: {spec["id"]}')
    for r in rows:
        if len(r['reference'])!=r['target_length'] or set(r['reference'])-set('ACGT'):
            raise ValueError(f'Invalid reference or length in {spec["id"]}')
        if any(not x or set(x)-set('ACGT') for x in r['reads']):
            raise ValueError(f'Invalid read in {spec["id"]}')
    actual=record_digest(rows)
    if actual!=spec['ordered_input_sha256']:
        raise ValueError(f'Input content/order does not match the frozen benchmark: {spec["id"]}')
    return actual

def prepare(specs,dataset_root,out,catalog):
    """Rebuild only selected datasets; validate every read, target, ID and order."""
    from .contamination import source_bank,contaminate,validate as validate_contamination
    out=Path(out).resolve();dataset_root=Path(dataset_root).resolve()
    out.mkdir(parents=True,exist_ok=True)
    all_specs={s['id']:s for s in catalog}
    with gzip.open(ROOT/'protocol/real_test_indices.json.gz','rt',encoding='utf8') as h: splits=json.load(h)
    verified=set(); complete={}
    def ensure(spec):
        if spec['id'] in complete: return complete[spec['id']]
        path=out/'data'/spec['cohort']/(spec['name']+'.jsonl.gz')
        for item in spec.get('sources',[]):
            source=dataset_root/item['path']
            if source not in verified:
                if not source.is_file(): raise FileNotFoundError(f'Required dataset file: {source}')
                if sha(source)!=item['sha256']: raise ValueError(f'Source hash differs: {source}')
                verified.add(source)
        if path.exists():
            rows=read_rows(path);validate(rows,spec)
        else:
            cohort=spec['cohort']
            if cohort=='real': rows=read_real(spec,dataset_root)
            elif cohort=='real_test75':
                full=ensure(all_specs['real__'+spec['name']])
                bank=read_rows(out/full['path'])
                rows=[bank[i] for i in splits[spec['name']]]
            elif cohort=='contamination_sweep':
                parent_id=spec['source_dataset_id']
                parent=ensure(all_specs[parent_id])
                bank=source_bank(out/parent['path'],spec['channel'])
                rows=contaminate(bank,spec['requested_percent'])
                validate_contamination(rows,bank)
            else: rows=read_synthetic(spec,dataset_root)
            validate(rows,spec)
            path.parent.mkdir(parents=True,exist_ok=True)
            # Stream compression to avoid a second large uncompressed copy in RAM.
            temp=path.with_suffix('.tmp')
            try:
                with temp.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0,compresslevel=3) as zipped:
                    for row in rows: zipped.write((json.dumps(row,separators=(',',':'))+'\n').encode())
                temp.replace(path)
            finally:
                if temp.exists(): temp.unlink()
        item={**spec,'path':path.relative_to(out).as_posix(),'sha256':sha(path)}
        complete[spec['id']]=item
        print('INPUT VERIFIED',spec['id'],f"name={spec['name']}",len(rows),item['path'],flush=True)
        return item
    return [ensure(spec) for spec in specs]
