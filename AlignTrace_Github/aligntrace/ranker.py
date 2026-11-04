"""Rank generator proposals using read support, path scores, and pool context.

The checkpoint consumes 22 features: 5/7/9-mer support summaries, generator
scores, read-depth/length statistics, and support centered within the pool.
"""
import numpy as np
import torch
from torch import nn
from numba import njit,prange
FEATURE_NAMES=('k5_mean', 'k5_std', 'k5_min', 'k5_unsupported', 'k7_mean', 'k7_std', 'k7_min', 'k7_unsupported', 'k9_mean', 'k9_std', 'k9_min', 'k9_unsupported', 'generator_score_per_base', 'generator_score_gap', 'read_count_div10', 'read_length_std_per_base', 'pool_centered_k5_mean', 'pool_centered_k5_unsupported', 'pool_centered_k7_mean', 'pool_centered_k7_unsupported', 'pool_centered_k9_mean', 'pool_centered_k9_unsupported')
# Serialized checkpoint identifier; retain it when loading the released weights.
SCHEMA="Budget22_K20_no_biology_20261031"

@njit(cache=True, inline="always")
def _slot(h, capacity):
    h = h ^ (h >> 10)
    return (h * 2654435761) & (capacity - 1)
@njit(cache=True, inline="always")
def _candidate_table(candidates, lengths, k):
    """Build complete candidate vocabulary before reading evidence.

    Hashed tables rebuild at <=50% occupancy; unsuccessful read lookups
    therefore terminate. Growth never discards candidates or word keys.
    k5 directly addresses its complete vocabulary and needs no growth.
    """
    nc, width = candidates.shape
    capacity = 1024 if k == 5 else 128
    if k != 5:
        required = 4 * width
        while capacity < required:
            capacity *= 2
    mask = (1 << (2 * k)) - 1
    while True:
        keys = np.full(capacity, -1, np.int32)
        slots = np.full((nc, width), -1, np.int32)
        unique = 0
        grow = False
        for j in range(nc):
            code = 0
            for position in range(lengths[j]):
                code = ((code << 2) | int(candidates[j, position])) & mask
                if position + 1 < k:
                    continue
                if k == 5:
                    slot = code
                else:
                    slot = _slot(code, capacity)
                    while keys[slot] != -1 and keys[slot] != code:
                        slot = (slot + 1) & (capacity - 1)
                if keys[slot] == -1:
                    if k != 5 and unique >= capacity // 2:
                        grow = True
                        break
                    keys[slot] = code
                    unique += 1
                slots[j, position] = slot
            if grow:
                break
        if not grow:
            return keys, slots
        capacity *= 2
@njit(cache=True, parallel=True, nogil=True)
def kernel(reads, lengths, read_counts, stats, candidates, clens, scores, target):
    n, nc, width = candidates.shape
    output = np.zeros((n,nc,22),np.float32)
    for i in prange(n):
        nr = read_counts[i]
        for ik in range(3):
            k = 5 + 2*ik
            mask = (1 << (2*k))-1
            keys,slots = _candidate_table(candidates[i],clens[i],k)
            capacity = len(keys)
            counts = np.zeros(capacity,np.int32)
            seen = np.full(capacity,-1,np.int32)
            for r in range(nr):
                code = 0
                for position in range(lengths[i,r]):
                    code = ((code << 2) | (int(reads[i,r,position])-1)) & mask
                    if position+1 < k: continue
                    if k==5: slot=code
                    else:
                        slot=_slot(code,capacity)
                        while keys[slot]!=-1 and keys[slot]!=code:
                            slot=(slot+1)&(capacity-1)
                    if keys[slot]==code and seen[slot]!=r:
                        counts[slot]+=1
                        seen[slot]=r
            for j in range(nc):
                if clens[i,j]==0: continue
                total,square,minimum,zero=0.,0.,1.,0.
                windows=max(1,clens[i,j]-k+1)
                for position in range(k-1,clens[i,j]):
                    count=counts[slots[j,position]]
                    support=count/max(nr,1)
                    total+=support
                    square+=support*support
                    minimum=min(minimum,support)
                    zero+=count==0
                off=4*ik
                output[i,j,off]=total/windows
                output[i,j,off+1]=np.sqrt(max(0.,square/windows-(total/windows)**2))
                output[i,j,off+2]=minimum
                output[i,j,off+3]=zero/windows
        maximum=-1e30
        for j in range(nc):
            if clens[i,j]>0: maximum=max(maximum,scores[i,j])
        for j in range(nc):
            if clens[i,j]==0: continue
            output[i,j,12]=scores[i,j]/target
            output[i,j,13]=max(-50.,scores[i,j]-maximum)/10.
            output[i,j,14]=nr/10.
            output[i,j,15]=stats[i,1]/target
        for f in range(6):
            base=(f//2)*4+(3 if f%2 else 0)
            total,count=0.,0
            for j in range(nc):
                if clens[i,j]>0:
                    total+=output[i,j,base]
                    count+=1
            for j in range(nc):
                output[i,j,16+f]=output[i,j,base]-total/max(count,1)
    return output
def packed_features(cache,frontier):
    a,lens,sc=frontier.packed()
    x=kernel(cache.packed,cache.lengths,cache.counts,cache.statistics,
             a,lens,sc,frontier.target)
    return x,sc

class Budget22(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim=22
        self.gap_index=13
        self.net=nn.Sequential(nn.Linear(22,32),nn.SiLU(),nn.Linear(32,16),
                               nn.SiLU(),nn.Linear(16,1))
        self.scale=nn.Parameter(torch.tensor(0.))
        self.register_buffer('mean',torch.zeros(22))
        self.register_buffer('std',torch.ones(22))

    def forward(self,x):
        residual=10*x[:,:,self.gap_index]*(.1+1.8*self.scale.sigmoid())
        return residual+self.net(((x-self.mean)/self.std).clamp(-8,8)).squeeze(-1)

def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('schema')!=SCHEMA or ck.get('dim')!=22:
        raise ValueError('Expected a compatible Budget22 checkpoint with 22 inputs.')
    if tuple(ck['feature_names'])!=FEATURE_NAMES:
        raise ValueError('Checkpoint feature ordering does not match Budget22.')
    model=Budget22()
    model.load_state_dict(ck['model'])
    return model.eval()
