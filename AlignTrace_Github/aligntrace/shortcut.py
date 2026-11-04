"""Validate inputs and optionally return frequent reads of the target length."""
from collections import defaultdict
def validate_and_group(read_sets,lengths,enabled=False):
    if len(read_sets)!=len(lengths):raise ValueError('One target length per cluster required')
    groups=defaultdict(list);bypass={}
    for i,(reads,target) in enumerate(zip(read_sets,lengths)):
        if isinstance(target,bool) or int(target)!=target or not 3<=int(target)<=4096:
            raise ValueError('Known target length must be an integer in 3..4096')
        target=int(target)
        if not isinstance(reads,(list,tuple)) or any(not isinstance(s,str) or not 1<=len(s)<=4096 for s in reads):
            raise ValueError(f'Invalid reads in cluster {i}')
        # A zero-read cluster is an explicit abstention, never a fabricated
        # sequence or a dropped evaluation row. It is not a count shortcut.
        if not reads:continue
        if ''.join(reads).encode('ascii').translate(None,b'ACGT'):
            raise ValueError(f'Only ACGT reads accepted, cluster {i}')
        if target==140 and min(map(len,reads))>180:raise ValueError(f'No admissible length-140 anchor in cluster {i}')
        counts={}
        if enabled:
            for s in reads:
                if len(s)==target:counts[s]=counts.get(s,0)+1
        if counts and (max(counts.values())>=3 or 5*max(counts.values())>=3*len(reads)):
            # Stable ties follow first occurrence in the supplied read order.
            bypass[i]=sorted(counts,key=counts.get,reverse=True)[:5]
        else:groups[target].append(i)
    return groups,bypass
