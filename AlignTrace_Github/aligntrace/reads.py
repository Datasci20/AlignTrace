"""ACGT packing and read-length statistics."""
from dataclasses import dataclass
import numpy as np
from numba import njit,prange
LUT=np.full(256,255,np.uint8)
for i,b in enumerate(b"ACGT",1):LUT[b]=i

@njit(parallel=True,nogil=True,cache=True)
def scatter(raw,starts,sizes,places,dest):
    for i in prange(len(sizes)):
        for j in range(sizes[i]): dest[places[i],j]=raw[starts[i]+j]

def pack(groups):
    counts=np.asarray([len(x) for x in groups],np.int32)
    if not len(counts) or np.any(counts==0): raise ValueError('Nonempty clusters required')
    strings=[s for group in groups for s in group]
    sizes=np.asarray(list(map(len,strings)),np.int32)
    if sizes.min()<1 or sizes.max()>4096: raise ValueError('Read length must be 1..4096')
    raw=LUT[np.frombuffer(''.join(strings).encode('ascii'),np.uint8)]
    if np.any(raw==255): raise ValueError('Only A/C/G/T reads supported')
    R=int(counts.max());T=int(sizes.max())
    starts=np.r_[0,np.cumsum(sizes,dtype=np.int64)[:-1]]
    places=np.arange(len(strings))+np.repeat(np.arange(len(groups))*R-np.r_[0,np.cumsum(counts)[:-1]],counts)
    dest=np.zeros((len(groups)*R,T),np.uint8)
    lengths=np.zeros(len(groups)*R,np.int32);lengths[places]=sizes
    scatter(raw,starts,sizes,places,dest)
    return dest.reshape(len(groups),R,T),lengths.reshape(len(groups),R),counts
@njit(parallel=True, nogil=True, cache=True)
def _statistics(lengths, counts, statistics):
    # Columns 0:2 feed the reranker; 2:4 feed the generator lattice. They
    # represent the same moments but retain the original reduction orders
    # used for the released checkpoints and frozen candidate digests.
    for i in prange(len(counts)):
        nr = counts[i]
        mean_len = 0.
        for r in range(lengths.shape[1]):
            if lengths[i, r] > 0:mean_len += lengths[i, r]
        mean_len /= max(nr, 1)
        var_len = 0.
        for r in range(nr):var_len += (lengths[i, r] - mean_len) ** 2
        statistics[i, 0] = mean_len
        statistics[i, 1] = np.sqrt(var_len / max(nr, 1))
        statistics[i, 2] = np.mean(lengths[i, :nr])
        statistics[i, 3] = np.std(lengths[i, :nr])

@dataclass
class PreparedReads:
    packed: np.ndarray
    lengths: np.ndarray
    counts: np.ndarray
    statistics: np.ndarray
    read_sets: tuple

def prepare_batch(read_sets):
    groups=tuple(tuple(g) for g in read_sets)
    packed,lengths,counts=pack(groups)
    stats=np.empty((len(groups),4),np.float64)
    _statistics(lengths,counts,stats)
    return PreparedReads(packed,lengths,counts,stats,groups)
