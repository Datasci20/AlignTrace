"""Generate distinct fixed-length proposals from blank/A/C/G/T slot scores.

A greedy path of the requested length supplies substitutions and paired
deletion/insertion moves. Other emission counts use length-aware beam search.
Distinct completion fills a short proposal list; it is part of this decoder.
"""
import math
import numpy as np
from numba import njit,prange
from .beam import _beam, _complete_distinct

@njit(cache=True,parallel=True,nogil=True)
def initialize(logits,lengths,target):
    B,S,_=logits.shape; M=3*target+257
    lp=np.zeros_like(logits); greedy=np.zeros((B,target),np.uint8); positions=np.zeros((B,target),np.int32)
    desc=np.full((B,M,4),-1,np.int32); values=np.full((B,M),-1e30,np.float32)
    frontier=np.zeros((B,M),np.int32); limit=np.zeros(B,np.int32); routes=np.zeros(B,np.uint8)
    beamseq=np.zeros((B,24,target),np.uint8); beampos=np.zeros((B,24,target),np.int32)
    for b in prange(B):
        n=lengths[b]; tokens=np.zeros(n,np.uint8); pos=np.zeros(n,np.int32); count=0; score=0.
        for s in range(n):
            mx=np.max(logits[b,s]); total=0.
            for t in range(5):total+=math.exp(logits[b,s,t]-mx)
            norm=mx+math.log(total)
            for t in range(5):
                v=logits[b,s,t]-norm;lp[b,s,t]=v
            token=np.argmax(lp[b,s])
            score+=lp[b,s,token]
            if token:tokens[count]=token;pos[count]=s;count+=1
        if count==target:
            greedy[b]=tokens[:target];positions[b]=pos[:target]
            at=1;desc[b,0,0]=0;values[b,0]=score
            for t in range(target):
                for base in range(1,5):
                    if base==tokens[t]:continue
                    desc[b,at,0]=1;desc[b,at,1]=t;desc[b,at,2]=base
                    values[b,at]=score+lp[b,pos[t],base]-lp[b,pos[t],tokens[t]];at+=1
            blank=np.full(n,-1e30,np.float32);dele=np.empty(target,np.float32);emitted=np.zeros(n,np.uint8)
            for t in range(target):emitted[pos[t]]=1;dele[t]=lp[b,pos[t],0]-lp[b,pos[t],tokens[t]]
            for s in range(n):
                if not emitted[s]:blank[s]=max(lp[b,s,1],lp[b,s,2],lp[b,s,3],lp[b,s,4])-lp[b,s,0]
            bo=np.argsort(blank);do=np.argsort(dele)
            for di in range(min(8,target)):
                t=do[target-di-1]
                for bi in range(min(8,n-target)):
                    s=bo[n-bi-1]
                    for base in range(1,5):
                        desc[b,at,0]=2;desc[b,at,1]=t;desc[b,at,2]=s;desc[b,at,3]=base
                        values[b,at]=score+dele[t]+lp[b,s,base]-lp[b,s,0];at+=1
            limit[b]=at;frontier[b,:at]=np.argsort(-values[b,:at],kind='mergesort')
        else:
            routes[b]=1
            seq,ps,sc,alive=_beam(lp[b],n,target,24)
            beamseq[b,:alive]=seq[:alive];beampos[b,:alive]=ps[:alive]
            limit[b]=alive;values[b,:alive]=sc[:alive];frontier[b,:alive]=np.argsort(-sc[:alive],kind='mergesort')
    return lp,greedy,positions,desc,values,frontier,limit,routes,beamseq,beampos

@njit(cache=True,parallel=True,nogil=True)
def resume(lp,lengths,greedy,gpos,desc,values,frontier,limit,routes,beamseq,beampos,
           requested,cursors,counts,candidates,paths,scores):
    B,cap,L=candidates.shape
    for b in prange(B):
        want=requested[b]
        while counts[b]<want and cursors[b]<limit[b]:
            index=frontier[b,cursors[b]];cursors[b]+=1
            seq=np.empty(L,np.uint8);pos=np.empty(L,np.int32)
            if routes[b]:seq[:]=beamseq[b,index];pos[:]=beampos[b,index]
            else:
                kind=desc[b,index,0]
                if kind<2:
                    seq[:]=greedy[b];pos[:]=gpos[b]
                    if kind==1:seq[desc[b,index,1]]=desc[b,index,2]
                else:
                    removed=desc[b,index,1];inserted=desc[b,index,2];base=desc[b,index,3];at=0;done=False
                    for t in range(L):
                        if t==removed:continue
                        if not done and inserted<gpos[b,t]:seq[at]=base;pos[at]=inserted;at+=1;done=True
                        seq[at]=greedy[b,t];pos[at]=gpos[b,t];at+=1
                    if not done:seq[at]=base;pos[at]=inserted
            duplicate=False
            for j in range(counts[b]):
                same=True
                for t in range(L):
                    if candidates[b,j,t]!=seq[t]:same=False;break
                if same:duplicate=True;break
            if duplicate:continue
            j=counts[b];candidates[b,j]=seq;paths[b,j]=pos;scores[b,j]=values[b,index];counts[b]+=1
        if counts[b]<want:
            counts[b]=_complete_distinct(lp[b],lengths[b],L,candidates[b],paths[b],scores[b],counts[b],want)
    return counts

class Frontier:
    """Materialize up to twenty proposals and retain their generator path scores."""
    def __init__(self,logits,lengths,target,capacity=20):
        self.lengths=np.ascontiguousarray(lengths);self.target=target
        self.state=initialize(np.ascontiguousarray(logits),self.lengths,target)
        n=len(lengths);self.counts=np.zeros(n,np.int32);self.cursors=np.zeros(n,np.int32)
        self.candidates=np.zeros((n,capacity,target),np.uint8);self.paths=np.zeros((n,capacity,target),np.int32)
        self.scores=np.full((n,capacity),-1e30,np.float32)
    def extend(self,requested):
        requested=np.asarray(requested,np.int32)
        assert requested.shape==self.counts.shape and np.all(requested>=self.counts) and requested.max()<=self.candidates.shape[1]
        resume(*self.state[:1],self.lengths,*self.state[1:10],requested,self.cursors,self.counts,self.candidates,self.paths,self.scores)
        assert np.array_equal(self.counts,requested)
        return self
    def packed(self):
        k=int(self.counts.max());a=self.candidates[:,:k]-np.uint8(1)
        lens=np.where(np.arange(k)[None,:]<self.counts[:,None],self.target,0).astype(np.int32)
        return np.ascontiguousarray(a),lens,np.ascontiguousarray(self.scores[:,:k])
    def strings(self,order=None):
        if order is None:order=np.tile(np.arange(int(self.counts.max())),(len(self.counts),1))
        order=np.asarray(order)
        chosen=self.candidates[np.arange(len(self.counts))[:,None],order]
        encoded=np.frombuffer(b'_ACGT',np.uint8)[chosen]
        return encoded.view('S'+str(self.target)).reshape(order.shape).astype(str).tolist()
