"""Training/evaluation labels only. Never imported by the inference pipeline."""
import numpy as np
from rapidfuzz.distance.Levenshtein import distance
def summarize(d):
    d=np.asarray(d);a=d[:,0];b=d[:,:5].min(1)
    return dict(n=len(d),exact_count=int((a==0).sum()),success_percent=float(100*(a==0).mean()),
        mean_ed=float(a.mean()),top5_percent=float(100*(b==0).mean()),
        within1_percent=float(100*(a<=1).mean()),top5_within1_percent=float(100*(b<=1).mean()),top5_mean_min_ed=float(b.mean()))
