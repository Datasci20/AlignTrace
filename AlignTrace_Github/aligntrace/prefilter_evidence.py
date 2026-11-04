"""Ten cluster features from leave-one-read-out 5-mer support and read lengths."""
import numpy as np
from numba import njit,prange

FEATURE_NAMES=('log1p_read_count','support_mean','support_std','support_min',
    'support_lower_quartile_order_statistic','support_upper_median',
    'support_upper_quartile_order_statistic','support_max',
    'fraction_below_0_6_upper_median','read_length_coefficient_of_variation')

@njit(cache=True,parallel=True,nogil=True)
def evidence(packed,lengths,counts):
    B,R,T=packed.shape;scores=np.zeros((B,R),np.float32);z=np.zeros((B,10),np.float32)
    for b in prange(B):
        nr=counts[b];table=np.zeros(1024,np.int32);seen=np.full(1024,-1,np.int32)
        for r in range(nr):
            code=0
            for t in range(lengths[b,r]):
                code=((code<<2)|(int(packed[b,r,t])-1))&1023
                if t>=4 and seen[code]!=r:table[code]+=1;seen[code]=r
        for r in range(nr):
            code=0;total=0.
            for t in range(lengths[b,r]):
                code=((code<<2)|(int(packed[b,r,t])-1))&1023
                if t>=4:total+=(table[code]-1)/max(nr-1,1)
            scores[b,r]=total/max(1,lengths[b,r]-4)
        # The gate was fitted with upper order statistics. Read selection below
        # uses the ordinary median; keep these two definitions in feature order.
        order=np.sort(scores[b,:nr]);med=order[nr//2]
        z[b,0]=np.log1p(nr);z[b,1]=order.mean();z[b,2]=order.std();z[b,3]=order[0]
        z[b,4]=order[nr//4];z[b,5]=med;z[b,6]=order[min(nr-1,3*nr//4)];z[b,7]=order[-1]
        z[b,8]=(order<.6*med).sum()/nr;z[b,9]=lengths[b,:nr].std()/max(1.,lengths[b,:nr].mean())
    return z,scores

@njit(cache=True,nogil=True,parallel=True)
def retained_indices(scores,counts,flags,ratio):
    B,R=scores.shape;indices=np.full((B,R),-1,np.int32);sizes=counts.copy()
    for i in prange(B):
        n=counts[i];kept=0
        if flags[i] and n>=4:
            median=np.float32(np.median(scores[i,:n]));threshold=np.float32(ratio)*median
            for j in range(n):
                if scores[i,j]>=threshold:indices[i,kept]=j;kept+=1
        if kept>=max(2,(n+1)//2):sizes[i]=kept
        else:
            for j in range(n):indices[i,j]=j
    return indices,sizes
