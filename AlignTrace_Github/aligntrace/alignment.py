"""Align reads to a read-derived anchor and build 32 features per output slot.

A 7-mer support score selects the anchor. Each anchor base has two preceding
insertion slots and one backbone slot, plus two trailing insertion slots.
Two optimal traceback tie policies expose alignment ambiguity to the generator.
"""
import numpy as np
from numba import njit,prange,set_num_threads
from .reads import PreparedReads

@njit(cache=True)
def compact_traceback(a, b, dp, width, alternate=False):
    n, m = len(a), len(b)
    tok = np.zeros(3*n+2, np.uint8)
    count = np.zeros(n+1, np.int32)
    i, j = n, m
    while i or j:
        col = j-i+width+1
        v = int(dp[i, col])
        diag = i > 0 and j > 0 and dp[i-1, col] + (a[i-1] != b[j-1]) == v
        delete = i > 0 and dp[i-1, col+1] + 1 == v
        insert = j > 0 and dp[i, col-1] + 1 == v
        if (diag and not alternate) or (diag and not delete and not insert):
            tok[3*(i-1)+2] = b[j-1]
            i -= 1; j -= 1
        elif (insert and alternate) or (insert and not delete):
            tok[3*i+1] = tok[3*i]
            tok[3*i] = b[j-1]
            count[i] += 1; j -= 1
        else:
            i -= 1
    return tok, count > 2, int(dp[n,m-n+width+1])
@njit(cache=True)
def exact_matrix(a, b, initial_width=16):
    # Unit-cost ED <= width certifies that the optimum fits inside this band.
    # Otherwise recompute the full matrix before extracting either traceback.
    n, m = len(a), len(b)
    width = max(initial_width, abs(n - m))
    while True:
        dp = np.full((n + 1, 2 * width + 3), 15000, np.int16)
        for i in range(min(n, width) + 1):
            dp[i, width + 1 - i] = i
        for j in range(min(m, width) + 1):
            dp[0, width + 1 + j] = j
        for i in range(1, n + 1):
            for j in range(max(1, i - width), min(m, i + width) + 1):
                col = j - i + width + 1
                dp[i, col] = min(dp[i-1, col] + (a[i-1] != b[j-1]),
                                 dp[i-1, col+1] + 1, dp[i, col-1] + 1)
        if dp[n, m-n+width+1] <= width or width >= max(n, m):
            return dp, width
        width = max(n, m)

@njit(cache=True)
def _identity(a):
    n = len(a)
    tok = np.zeros(3*n+2, np.uint8)
    for i in range(n):
        tok[3*i+2] = a[i]
    return tok, np.zeros(n+1, np.bool_), 0, tok.copy()

@njit(cache=True)
def _same(a, b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True

@njit(cache=True)
def exact_align_pair(a, b):
    if _same(a, b):
        return _identity(a)
    dp, width = exact_matrix(a, b, 16)
    t, v, d = compact_traceback(a, b, dp, width, False)
    at, _, _ = compact_traceback(a, b, dp, width, True)
    return t, v, d, at
@njit(cache=True)
def choose(reads, lens, target):
    nr = len(lens)
    last_seen = np.full(16384, -1, np.int32)
    counts = np.zeros(16384, np.int16)
    keys = np.zeros((nr,reads.shape[1]), np.int32)
    nk = np.zeros(nr, np.int32)
    for r in range(nr):
        h = 0
        for j in range(lens[r]):
            h = ((h<<2) | (int(reads[r,j])-1)) & 16383
            if j >= 6 and last_seen[h] != r:
                last_seen[h] = r; counts[h] += 1
                keys[r,nk[r]] = h; nk[r] += 1
    chosen = -1; best = -1e20
    for r in range(nr):
        if target == 140 and lens[r] > 180: continue
        support = 0.
        for k in range(nk[r]): support += counts[keys[r,k]]
        score = support/(max(1,nk[r])*nr) - (3.5/target)*abs(lens[r]-target)
        if score > best: best = score; chosen = r
    return chosen
@njit(parallel=True,nogil=True,cache=True)
def extract(packed,lens,counts,width,target,read_stats):
    B,R,T=packed.shape
    fs=np.zeros((B,3*width+2,32),np.float32)
    sl=np.zeros(B,np.int32)
    for b in prange(B):
        nr=counts[b]
        selected=choose(packed[b,:nr],lens[b,:nr],target)
        n=lens[b,selected];a=packed[b,selected,:n]
        S=3*n+2;sl[b]=S
        ts=np.empty((nr,S),np.uint8);ats=np.empty_like(ts)
        local=np.empty((nr,n+1),np.float32)
        weights=np.zeros(nr,np.float64);costs=np.zeros(nr,np.float64);over=np.zeros((nr,n+1),np.bool_)
        for r in range(nr):
            t,v,d,at=exact_align_pair(a,packed[b,r,:lens[b,r]])
            ts[r]=t;ats[r]=at;costs[r]=d;over[r]=v
            weights[r]=max(.02,np.exp(-d/(max(1,target)*.08)))
            errors=np.zeros(n+1,np.float32)
            for p in range(n):errors[p]=(t[3*p+2]!=a[p])*1.+(t[3*p]!=0)*1.+(t[3*p+1]!=0)*1.
            errors[n]=(t[3*n]!=0)*1.+(t[3*n+1]!=0)*1.
            for p in range(n+1):
                lo=max(0,p-3);hi=min(n,p+3)
                local[r,p]=np.mean(errors[lo:hi+1])
        sw=weights.sum();meanlen=read_stats[b,2];varlen=read_stats[b,3];mc=np.mean(costs);vc=np.std(costs)
        # Columns: raw/weighted blank+base frequencies (0:10), anchor (10:14),
        # slot/position/depth (14:18), entropy (18), length/edit statistics
        # (19:23), insertion overflow (23), alternate frequencies (24:29),
        # traceback disagreement (29), local errors (30), anchor length (31).
        for s in range(S):
            p=s//3;back=s%3==2
            for r in range(nr):
                t=ts[r,s];at=ats[r,s]
                fs[b,s,t]+=1./nr;fs[b,s,5+t]+=weights[r]/sw;fs[b,s,24+at]+=1./nr
                fs[b,s,29]+=(t!=at)*1./nr;fs[b,s,30]+=local[r,p]/nr
                if not back:fs[b,s,23]+=over[r,p]*1./nr
            if back:fs[b,s,10+a[p]-1]=1.
            fs[b,s,14]=not back;fs[b,s,15]=back;fs[b,s,16]=p/max(1,n);fs[b,s,17]=np.log1p(nr)/np.log(11.)
            for t in range(5):
                v=fs[b,s,5+t]
                if v>0:fs[b,s,18]-=v*np.log(v)/np.log(5.)
            fs[b,s,19]=meanlen/target;fs[b,s,20]=varlen/target;fs[b,s,21]=mc/target;fs[b,s,22]=vc/target;fs[b,s,31]=n/target
    return fs,sl

def extract_batch(cache,target_len,workers=6):
    if not isinstance(cache,PreparedReads):raise TypeError('Expected prepared reads')
    set_num_threads(workers)
    if target_len==140 and any(not any(len(s)<=180 for s in rs) for rs in cache.read_sets):
        raise ValueError('Length-140 alignment requires at least one read <=180 bases')
    f,sl=extract(cache.packed,cache.lengths,cache.counts,max(180,int(cache.lengths.max())),target_len,cache.statistics)
    return dict(features=f,lengths=sl)
