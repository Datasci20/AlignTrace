"""Benchmark evaluation with saved ranked predictions and synchronized timing."""
import datetime, gc, gzip, hashlib, json, os, random
import statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
from aligntrace.io import read_json, save_json, sha, load_records
from workflow.catalog import describe


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def affinity():
    if sys.platform != 'win32': return None
    import ctypes
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    handle=k.GetCurrentProcess(); available=ctypes.c_size_t(); system=ctypes.c_size_t()
    if not k.GetProcessAffinityMask(handle, ctypes.byref(available), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    ids=[i for i in range(64) if available.value & (1 << i)][:16]
    if not k.SetProcessAffinityMask(handle, sum(1 << i for i in ids)):
        raise ctypes.WinError(ctypes.get_last_error())
    return ids


def gpu_snapshot():
    # Record contention outside timed intervals so timing can be interpreted.
    try:
        proc = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                              capture_output=True, text=True)
    except FileNotFoundError:
        return dict(available=False, error='nvidia-smi is not installed or not on PATH')
    if proc.returncode: return dict(available=False, error=proc.stderr.strip())
    foreign=[]
    for line in proc.stdout.splitlines():
        pid=line.split(',')[0].strip()
        if pid.isdigit() and int(pid)!=os.getpid(): foreign.append(line)
    return dict(available=True, foreign_compute=foreign, raw=proc.stdout.strip())


def paths(variant, spec, model_map):
    if variant == 'fine_tuned':
        m=model_map[spec['name']]
        for kind in ('generator','reranker'):
            if sha(PACKAGE/m[kind])!=m[kind+'_sha256']:
                raise ValueError(f'Fine-tuned checkpoint checksum mismatch: {spec["name"]}/{kind}')
        return dict(generator=PACKAGE/m['generator'],reranker=PACKAGE/m['reranker'])
    return {}


def run(spec, args, model_map, code_hashes):
    import torch
    from aligntrace.pipeline import Pipeline
    from aligntrace.metrics import distance, summarize
    dest = HERE/'results'/(spec['id']+'.json')
    if dest.exists():
        existing=read_json(dest)
        if (not existing.get('complete') or existing.get('code_hashes')!=code_hashes
                or existing.get('input_sha256')!=spec['sha256']
                or existing.get('spec')!=spec or existing.get('device')!=args.device
                or set(existing.get('variants',{}))!=set(spec['variants'])):
            raise ValueError('Existing receipt uses different data/code/settings; choose a new experiment folder')
        for result in existing['variants'].values():
            prediction=HERE/result['predictions']
            if not prediction.is_file() or sha(prediction)!=result['predictions_sha256']:
                raise ValueError(f'Saved prediction file is missing or changed: {prediction}')
        print('RESUME',spec['id'],flush=True); return existing
    wall_start=time.perf_counter(); source=HERE/spec['path']
    if sha(source)!=spec['sha256']:
        raise ValueError(f'Prepared input checksum mismatch: {source}')
    t=time.perf_counter(); rows=load_records(source,training=True); load_seconds=time.perf_counter()-t
    if len(rows)!=spec['n']:
        raise ValueError(f'Prepared input cluster count mismatch: {spec["id"]}')
    reads=[r['reads'] for r in rows]; lengths=[r['target_length'] for r in rows]
    if args.smoke:
        rows=rows[:32]; reads=reads[:32]; lengths=lengths[:32]
    pipes={}; results={}; checks=[]
    try:
        for variant in spec['variants']:
            print('RUN',spec['id'],variant,describe(spec),
                  f"active_filter={'learned' if variant=='learned_filter' else 'off'}",flush=True)
            t=time.perf_counter()
            pipe=Pipeline(filter_on=variant=='learned_filter', shortcut_on=spec['shortcut'],
                          batch_size=64, device=args.device, **paths(variant,spec,model_map))
            pipes[variant]=pipe; setup_seconds=time.perf_counter()-t
            checks.append(gpu_snapshot() if args.device=='cuda' else {})
            pred, first, diagnostic=pipe.predict(reads,lengths)
            checks.append(gpu_snapshot() if args.device=='cuda' else {})
            if len(pred)!=len(rows):
                raise ValueError('Inference changed the evaluation denominator')
            t=time.perf_counter(); distances=[]; missing=0
            for row, out in zip(rows,pred):
                if len(out)!=len(set(out)) or len(out)>5:
                    raise ValueError('Expected at most five distinct ranked candidates')
                if not all(len(s)==row['target_length'] and not(set(s)-set('ACGT')) for s in out):
                    raise ValueError('Candidate alphabet/length does not match the task')
                if not out:
                    if row['reads']:
                        raise ValueError('Only empty clusters may abstain')
                    missing+=1
                ds=[distance(row['reference'],s) for s in out]
                distances.append(ds+[ds[-1] if ds else row['target_length']]*(5-len(ds)))
            met=summarize(distances); met['abstentions']=missing
            # These are scoring views of the same predictions, outside timing.
            strata={}
            for label,values in [('target_length',lengths),('read_depth',[len(r) for r in reads])]:
                strata[label]={str(value):summarize([d for d,v in zip(distances,values) if v==value])
                               for value in sorted(set(values))}
            strata['depth_group']={name:summarize([d for d,r in zip(distances,reads) if predicate(len(r))])
                for name,predicate in [('below_10',lambda n:n<10),('equal_10',lambda n:n==10),('above_10',lambda n:n>10)]
                if any(predicate(len(r)) for r in reads)}
            score_seconds=time.perf_counter()-t
            folder=HERE/('smoke_predictions' if args.smoke else 'predictions')/spec['id']; folder.mkdir(parents=True,exist_ok=True)
            pp=folder/(variant+'.jsonl.gz'); t=time.perf_counter(); shortcuts=set(diagnostic['shortcut_indices'])
            with gzip.open(pp,'wt',encoding='utf8',compresslevel=3) as f:
                for i,(r,out,ds) in enumerate(zip(rows,pred,distances)):
                    record=dict(id=r['id'],reference_sha256=hashlib.sha256(r['reference'].encode()).hexdigest(),
                        target_length=r['target_length'],read_count=len(r['reads']),candidates=out,distances=ds,
                        kept_indices=diagnostic['kept_indices'][i],shortcut=i in shortcuts,
                        gate_probability=diagnostic['prefilter_gate_probability'][i],
                        gate_flagged=diagnostic['prefilter_gate_flagged'][i])
                    f.write(json.dumps(record,separators=(',',':'))+'\n')
            results[variant]=dict(metrics=met,strata=strata,predictions=str(pp.relative_to(HERE)),predictions_sha256=sha(pp),
                output_digest=digest(pred), first_call_seconds=first['wall'], first_call_stages=first,
                model_setup_seconds=setup_seconds,score_seconds=score_seconds,write_seconds=time.perf_counter()-t,
                generator_sha256=sha(pipe.generator_path),reranker_sha256=sha(pipe.reranker_path),
                gate_sha256=sha(pipe.prefilter_checkpoint) if pipe.prefilter_checkpoint else None,
                filter_on=pipe.filter_on,shortcut_on=pipe.shortcut_on,repetitions=[])
            print('SCORE',spec['id'],variant,round(met['success_percent'],4),'ED',round(met['mean_ed'],5),
                  'FIRST',round(first['wall'],3),flush=True)
        orders=[]; rng=random.Random(261031)
        for repeat in range(1 if args.smoke else 3):
            order=list(pipes);rng.shuffle(order);orders.append(order)
            for variant in order:
                checks.append(gpu_snapshot() if args.device=='cuda' else {})
                pred,tm,_=pipes[variant].predict(reads,lengths)
                checks.append(gpu_snapshot() if args.device=='cuda' else {})
                if digest(pred)!=results[variant]['output_digest']:
                    raise ValueError(f'Predictions differ between timing repetitions: {spec["id"]}/{variant}')
                results[variant]['repetitions'].append(tm)
            print('REPEAT',spec['id'],repeat+1,{v:round(results[v]['repetitions'][-1]['wall'],4) for v in order},flush=True)
        for v,r in results.items():
            reps=r['repetitions']; times=[x['wall'] for x in reps]
            r.update(mean_seconds=statistics.mean(times),median_seconds=statistics.median(times),
                sd_seconds=statistics.stdev(times) if len(times)>1 else 0,
                milliseconds_per_cluster=1000*statistics.mean(times)/len(rows),
                stages={k:statistics.mean(x.get(k,0) for x in reps) for k in set().union(*(x.keys() for x in reps))})
        result=dict(complete=True,dataset=spec['name'],cohort=spec['cohort'],id=spec['id'],n=len(rows),device=args.device,
            input_path=spec['path'],input_sha256=spec['sha256'],code_hashes=code_hashes,
            variants=results,load_seconds=load_seconds,experiment_wall_seconds=time.perf_counter()-wall_start,
            timing_orders=orders,gpu_checks=checks,
            gpu_timing_isolated=all(c.get('available') and not c.get('foreign_compute') for c in checks) if args.device=='cuda' else None,
            finished_utc=now(),spec=spec)
        save_json(HERE/('smoke.json' if args.smoke else 'results/'+spec['id']+'.json'),result)
        return result
    finally:
        for pipe in pipes.values(): pipe.close()
        pipes.clear();gc.collect();torch.cuda.empty_cache()
