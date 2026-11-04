"""Learned cluster gate and 5-mer read selection using fixed inference settings."""
import time
import numpy as np
from .io import read_json
from .reads import prepare_batch
from .prefilter_evidence import evidence,retained_indices

# Calibrated with the released gate; the public method control is filter_on.
GATE_THRESHOLD = 0.5
READ_SUPPORT_RATIO = 0.8

def probabilities(z,model):
    # Feature normalization and weights use float64 in the trained feature order.
    x=np.clip((z-np.asarray(model['mean']))/np.asarray(model['std']),-8,8)
    h=x@np.asarray(model['w1']).T+np.asarray(model['b1']);h=h/(1+np.exp(-np.clip(h,-40,40)))
    y=h@np.asarray(model['w2']).T+np.asarray(model['b2'])
    return 1/(1+np.exp(-np.clip(y[:,0],-40,40)))

class InputGate10:
    def __init__(self,path):
        ck=read_json(path)
        shapes=dict(mean=(10,),std=(10,),w1=(12,10),b1=(12,),w2=(1,12),b2=(1,))
        self.model={key:np.asarray(ck[key],np.float64) for key in shapes}
        if any(value.shape!=shapes[key] or not np.isfinite(value).all() for key,value in self.model.items()):
            raise ValueError('InputGate10 checkpoint dimensions/values are invalid')
        if np.any(self.model['std']<=0):raise ValueError('InputGate10 normalization std must be positive')
    def predict_probability(self,z):return probabilities(z,self.model)

def filter_reads(read_sets,gate):
    start=time.perf_counter();cache=prepare_batch(read_sets);packed_at=time.perf_counter()
    z,scores=evidence(cache.packed,cache.lengths,cache.counts);features_at=time.perf_counter()
    p=gate.predict_probability(z);flags=p>=GATE_THRESHOLD;gate_at=time.perf_counter()
    indices,sizes=retained_indices(scores,cache.counts,flags,READ_SUPPORT_RATIO)
    kept=[r[:int(n)] for r,n in zip(indices.tolist(),sizes)]
    selected=[[reads[i] for i in ids] for reads,ids in zip(read_sets,kept)]
    info=dict(probabilities=p.tolist(),flagged=flags.tolist(),evaluated=[True]*len(read_sets),
        timing=dict(prefilter_pack=packed_at-start,prefilter_evidence=features_at-packed_at,
        prefilter_gate_MLP=gate_at-features_at,prefilter_read_selection=time.perf_counter()-gate_at))
    return selected,kept,info
