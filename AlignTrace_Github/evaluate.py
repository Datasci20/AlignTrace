"""Score saved predictions separately; reference bases never enter inference."""
import argparse,json
from pathlib import Path
from aligntrace.io import load_records,open_text,save_json,sha

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True,help='Labeled dataset used for prediction')
    ap.add_argument('--predictions',type=Path,required=True)
    ap.add_argument('--length',type=int)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new metrics output path')
    from aligntrace.metrics import distance,summarize
    rows=load_records(a.input,a.length,training=True)
    with open_text(a.predictions) as h:pred=[json.loads(line) for line in h if line.strip()]
    if len(rows)!=len(pred):raise ValueError('Dataset/prediction row count mismatch')
    distances=[];empty_clusters=0
    for row,p in zip(rows,pred):
        if row['id']!=p['id'] or row['target_length']!=p['target_length']:
            raise ValueError('Predictions must match dataset order, IDs and target lengths')
        candidates=p['candidates']
        if not candidates:
            if row['reads']:raise ValueError('Only a zero-read cluster may have no candidates')
            # Score an abstention as the empty prediction (ED = target length),
            # retaining the complete input denominator for every accuracy metric.
            distances.append([row['target_length']]*5);empty_clusters+=1
            continue
        if not 1<=len(candidates)<=5 or len(set(candidates))!=len(candidates):
            raise ValueError('Expected one to five distinct ranked candidates')
        if any(len(s)!=row['target_length'] or set(s)-set('ACGT') for s in candidates):
            raise ValueError('Prediction alphabet/length does not match the declared task')
        d=[distance(row['reference'],s) for s in candidates]
        distances.append(d+[d[-1]]*(5-len(d)))
    # Repeating the last distance only rectangularizes short shortcut lists;
    # it cannot change their first prediction or best available prediction.
    result=summarize(distances)
    result.update(input_sha256=sha(a.input),predictions_sha256=sha(a.predictions),
        empty_clusters=empty_clusters,
        empty_cluster_policy='No candidates; counted as failure with ED equal to target length; retained in all denominators',
        top5_definition='Best of up to five returned candidates; no oracle reordering')
    save_json(a.output,result)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
