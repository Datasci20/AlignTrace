"""Reconstruct DNA with optional learned filtering and exact-read shortcut."""
import argparse,json,time
from pathlib import Path
from aligntrace.io import load_records,open_text,save_json,sha

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--length',type=int,help='Known reference/design length; required for binned TXT, otherwise JSONL target_length is used')
    ap.add_argument('--filter',choices=['off','on'],default='off',help='Enable the bundled learned read filter before alignment')
    ap.add_argument('--shortcut-mode',dest='shortcut',choices=['off','on'],default='off',help='Enable the exact-read counting shortcut before filtering')
    ap.add_argument('--batch-size',type=int,default=64);ap.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    ap.add_argument('--generator',type=Path,help='Optional compatible generator checkpoint')
    ap.add_argument('--reranker',type=Path,help='Optional compatible reranker checkpoint')
    args=ap.parse_args();receipt=Path(str(args.output)+'.receipt.json')
    if args.output.exists() or receipt.exists():raise FileExistsError('Choose a new output path')
    start=time.perf_counter();rows=load_records(args.input,args.length)
    from aligntrace.pipeline import Pipeline
    from aligntrace.prefilter_learned import GATE_THRESHOLD,READ_SUPPORT_RATIO
    pipe=Pipeline(filter_on=args.filter=='on',shortcut_on=args.shortcut=='on',batch_size=args.batch_size,
                  device=args.device,generator=args.generator,reranker=args.reranker)
    try:
        pp,t,d=pipe.predict([r['reads'] for r in rows],[r['target_length'] for r in rows])
        args.output.parent.mkdir(parents=True,exist_ok=True);short=set(d['shortcut_indices'])
        with open_text(args.output,'wt') as h:
            for i,(row,pred,kk) in enumerate(zip(rows,pp,d['kept_indices'])):
                h.write(json.dumps(dict(id=row['id'],target_length=row['target_length'],candidates=pred,kept_indices=kk,shortcut=i in short,
                    prefilter_gate_probability=d['prefilter_gate_probability'][i],prefilter_gate_flagged=d['prefilter_gate_flagged'][i],
                    prefilter_gate_evaluated=d['prefilter_gate_evaluated'][i]))+'\n')
        save_json(receipt,dict(package='AlignTrace',reference_bases_used=False,filter=args.filter,
            learned_gate_threshold=GATE_THRESHOLD if args.filter=='on' else None,
            read_support_ratio=READ_SUPPORT_RATIO if args.filter=='on' else None,
            prefilter_checkpoint_sha256=sha(pipe.prefilter_checkpoint) if pipe.prefilter_checkpoint else None,
            shortcut=args.shortcut,decoder='fast',batch_size=args.batch_size,device=args.device,candidate_budget=20,neural_outputs=5,
            input_sha256=sha(args.input),generator_sha256=sha(pipe.generator_path),reranker_sha256=sha(pipe.reranker_path),timing=t,
            total_load_infer_write_seconds=time.perf_counter()-start,
            note='Single call includes first-use JIT/CUDA graph capture. Shortcut may return 1..5 observed strings. Stage times overlap; prefilter_* subtimes are included in pre_filter.'))
    finally:pipe.close()
    print(args.output)
if __name__=='__main__':main()
