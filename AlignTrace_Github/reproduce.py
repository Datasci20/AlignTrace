"""Reproduce AlignTrace benchmark results from the accompanying dataset folders."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
from workflow.inputs import prepare,read_json,sha
from workflow.catalog import select_datasets,describe

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--datasets-root',type=Path,default=ROOT.parent,help='Parent containing Real Datasets and Synthetic Datasets')
    ap.add_argument('--output',type=Path,default=ROOT.parent/'AlignTrace_Reproduced',help='Generated inputs and new results; keep outside the source package')
    ap.add_argument('--cohort',choices=['all','synthetic','real','real_test75','contamination_sweep'],default='all')
    ap.add_argument('--dataset',action='append',help='Dataset name or ID; wildcard patterns allowed; repeat to select several')
    ap.add_argument('--list-datasets',action='store_true',help='List matching names, IDs and settings without writing files or running inference')
    ap.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    ap.add_argument('--prepare-only',action='store_true',help='Verify/reconstruct inputs using only the Python standard library')
    ap.add_argument('--smoke',action='store_true',help='First 32 clusters of the first selected dataset; separate output required')
    args=ap.parse_args();out=args.output.resolve()
    if out==ROOT or out.is_relative_to(ROOT/'models') or out.is_relative_to(ROOT/'protocol'):
        ap.error('Choose a separate output directory, not the package root/models/protocol')
    catalog=read_json(ROOT/'protocol/DATASETS.json')
    selected=select_datasets(catalog,args.cohort,args.dataset)
    if not selected: ap.error('No datasets match the selection')
    if args.list_datasets:
        for spec in selected: print(spec['id'],describe(spec),flush=True)
        print(f'{len(selected)} matching datasets. No files written; no inference run.')
        return
    if args.smoke: selected=selected[:1]
    out.mkdir(parents=True,exist_ok=True)
    config_path=out/'RUN_CONFIG.json'
    # A changed split, model map, or protocol must never reuse old receipts.
    fingerprint_paths=['protocol/DATASETS.json','protocol/PROTOCOL.json',
        'protocol/real_test_indices.json.gz','protocol/FINETUNED_MODELS.json',
        'protocol/EXPECTED_RESULTS.json','protocol/REFERENCE_TIMING.json','models/MANIFEST.json']
    config=dict(smoke=args.smoke,device=args.device,
                protocol_hashes={path:sha(ROOT/path) for path in fingerprint_paths})
    if config_path.exists() and read_json(config_path)!=config:
        ap.error('Output directory belongs to a different protocol/device/smoke run; choose a fresh --output')
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    manifest=prepare(selected,args.datasets_root,out,catalog)
    (out/'DATASETS.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.prepare_only:
        print(f'All {len(manifest)} selected datasets verified against frozen input content. No inference run.');return
    os.environ.setdefault('NUMBA_NUM_THREADS','16')
    os.environ.setdefault('NUMBA_CACHE_DIR',str(out/'cache/numba'))
    from workflow import benchmark_core as benchmark
    benchmark.HERE=out
    # Initialize the threading layer before importing Numba, as in prediction.
    from aligntrace.pipeline import ALIGN_THREADS,DECODE_THREADS
    import torch,numpy,numba
    from importlib.metadata import version
    if args.device=='cuda' and not torch.cuda.is_available(): ap.error('CUDA unavailable; install an appropriate PyTorch wheel or use --device cpu')
    cpu_ids=benchmark.affinity()
    models=read_json(ROOT/'protocol/FINETUNED_MODELS.json')
    code_paths=list(ROOT.glob('*.py'))+list((ROOT/'aligntrace').glob('*.py'))+list((ROOT/'workflow').glob('*.py'))
    code={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(code_paths)}
    # Validate checkpoints even when all selected result receipts can resume.
    model_manifest=read_json(ROOT/'models/MANIFEST.json')
    for kind in ('generator','reranker','prefilter_gate'):
        m=model_manifest[kind]
        if sha(ROOT/m['path'])!=m['sha256']: raise ValueError(f'Checkpoint checksum mismatch: {kind}')
    for model in models.values():
        for kind in ('generator','reranker'):
            if sha(ROOT/model[kind])!=model[kind+'_sha256']: raise ValueError('Fine-tuned checkpoint checksum mismatch')
    expected=read_json(ROOT/'protocol/EXPECTED_RESULTS.json')
    reference_timing=read_json(ROOT/'protocol/REFERENCE_TIMING.json')
    checks=[]; completed=[]; started=time.perf_counter()
    environment=dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,numba=numba.__version__,rapidfuzz=version('rapidfuzz'),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,cpu_affinity=cpu_ids,torch_threads=torch.get_num_threads(),align_threads=ALIGN_THREADS,decode_threads=DECODE_THREADS,numba_threading_requested=os.environ.get('NUMBA_THREADING_LAYER'))
    (out/'ENVIRONMENT.json').write_text(json.dumps(environment,indent=2)+'\n')
    for spec in manifest:
        r=benchmark.run(spec,args,models,code)
        completed.append(r)
        for variant,v in r['variants'].items():
            old=expected[spec['id']][variant]
            deltas={k:v['metrics'][k]-old['metrics'][k] for k in old['metrics'] if k in v['metrics'] and isinstance(old['metrics'][k],(int,float))}
            reference_seconds=reference_timing[spec['id']][variant]['mean_seconds']
            checks.append(dict(dataset_id=spec['id'],variant=variant,n=v['metrics']['n'],smoke=args.smoke,output_digest_matches=None if args.smoke else v['output_digest']==old['output_digest'],metric_deltas=None if args.smoke else deltas,metrics=v['metrics'],mean_seconds=v['mean_seconds'],sd_seconds=v['sd_seconds'],gpu_timing_isolated=r['gpu_timing_isolated'],reference_mean_seconds=None if args.smoke else reference_seconds,runtime_ratio=None if args.smoke or args.device!='cuda' or r['gpu_timing_isolated'] is not True else v['mean_seconds']/reference_seconds))
    summary=dict(complete=True,datasets=len(manifest),method_dataset_cells=len(checks),smoke=args.smoke,seconds=time.perf_counter()-started,comparisons=checks,note='Hardware and library differences may change numerical ties and runtimes.')
    fields=['dataset_id','variant','n','success_percent','mean_ed','top5_percent','within1_percent','top5_within1_percent','top5_mean_min_ed','mean_seconds','sd_seconds','gpu_timing_isolated','reference_mean_seconds','runtime_ratio','output_digest_matches']
    with (out/'dataset_metrics.csv').open('w',encoding='utf8',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=fields);writer.writeheader()
        for c in checks: writer.writerow({key:(c['metrics'].get(key) if key in c['metrics'] else c.get(key)) for key in fields})
    from workflow.reporting import write_summaries
    write_summaries(completed,out,smoke=args.smoke)
    (out/'COMPARISON.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'Complete: {len(manifest)} datasets; {len(checks)} method/dataset results. See {out / "COMPARISON.json"}')
    if any(c['gpu_timing_isolated'] is False for c in checks):
        print('GPU isolation was not verified for some timings; runtime ratios are omitted. See gpu_checks in the result receipts.')
    if not args.smoke and any(not c['output_digest_matches'] for c in checks):
        print('Some output digests differ from the reference run; inspect metric_deltas and ENVIRONMENT.json.')

if __name__=='__main__':main()
