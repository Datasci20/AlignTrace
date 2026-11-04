"""DNA reconstruction with optional learned filtering and exact-read shortcut."""
from .runtime import configure
ALIGN_THREADS,DECODE_THREADS=configure()
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch,numba
from .io import ROOT,read_json,sha
from .model import Mixed64
from .graphs import ExactShapeGraphs
from .reads import prepare_batch
from .alignment import extract_batch
from .decoder import Frontier
from .ranker import load as load_ranker,packed_features
from .prefilter_learned import InputGate10,filter_reads as learned_filter
from .shortcut import validate_and_group

class Pipeline:
    """Optional count shortcut -> optional read filter -> generator -> reranker.

    The two method switches are independent. Batch size controls scheduling;
    target lengths are supplied task information. Reference bases and donor
    labels are never accepted by this interface.
    """
    def __init__(self,filter_on=False,shortcut_on=False,batch_size=64,device='cuda',generator=None,reranker=None):
        if batch_size<1:raise ValueError('batch_size must be positive')
        self.device=torch.device(device)
        if self.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; install a compatible PyTorch wheel/driver or select --device cpu')
        self.filter_on=filter_on;self.shortcut_on=shortcut_on;self.batch_size=batch_size
        self.prefilter_checkpoint=None;self.prefilter_gate=None
        manifest=read_json(ROOT/'models/MANIFEST.json')
        if filter_on:
            entry=manifest['prefilter_gate']
            self.prefilter_checkpoint=ROOT/entry['path']
            if sha(self.prefilter_checkpoint)!=entry['sha256']:
                raise ValueError('Bundled prefilter gate checksum mismatch')
            self.prefilter_gate=InputGate10(self.prefilter_checkpoint)
        self.generator_path=ROOT/manifest['generator']['path'] if generator is None else generator
        self.reranker_path=ROOT/manifest['reranker']['path'] if reranker is None else reranker
        for key,path,supplied in [('generator',self.generator_path,generator),('reranker',self.reranker_path,reranker)]:
            if supplied is None and sha(path)!=manifest[key]['sha256']:raise ValueError(f'Bundled {key} checksum mismatch')
        ck=torch.load(self.generator_path,map_location='cpu',weights_only=False)
        model=Mixed64();model.load_state_dict(ck['model_state']);model.to(self.device).eval()
        self.model=ExactShapeGraphs(model,max_graphs=128) if self.device.type=='cuda' else model
        self.ranker=load_ranker(self.reranker_path)
        self.align_worker=ThreadPoolExecutor(1);self.model_worker=ThreadPoolExecutor(1)
    def close(self):self.align_worker.shutdown();self.model_worker.shutdown()
    def sync(self):
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
    @torch.inference_mode()
    def predict(self,read_sets,lengths):
        self.sync();start=time.perf_counter();times=defaultdict(float)
        tick=time.perf_counter();groups,bypass=validate_and_group(read_sets,lengths,self.shortcut_on)
        times['validation_and_shortcut']=time.perf_counter()-tick
        empty_indices=[i for i,r in enumerate(read_sets) if not r]
        output=[[] if not r else None for r in read_sets];kept=[list(range(len(r))) for r in read_sets]
        gate_probability=[None]*len(read_sets);gate_flagged=[None]*len(read_sets);gate_evaluated=[False]*len(read_sets)
        for i,ss in bypass.items():output[i]=ss
        # Batch equal target lengths together. A CPU alignment worker prepares
        # the next batch while a model worker and the current decoder proceed.
        for L,ids in sorted(groups.items()):
            sets=[read_sets[i] for i in ids];B=self.batch_size
            def align(j):
                numba.set_num_threads(ALIGN_THREADS);rs=sets[j:j+B];t={};tick=time.perf_counter()
                info=dict(probabilities=[None]*len(rs),flagged=[None]*len(rs),evaluated=[False]*len(rs),timing={})
                if self.filter_on:
                    rs,kk,info=learned_filter(rs,self.prefilter_gate)
                else:kk=[list(range(len(r))) for r in rs]
                t['pre_filter']=time.perf_counter()-tick
                t.update(info['timing'])
                tick=time.perf_counter();cache=prepare_batch(rs);t['read_preparation']=time.perf_counter()-tick
                tick=time.perf_counter();f=extract_batch(cache,L,ALIGN_THREADS);t['alignment_and_lattice']=time.perf_counter()-tick
                return cache,f,t,kk,info
            @torch.inference_mode()
            def infer(future,j):
                cache,f,t,kk,info=future.result()
                nxt=self.align_worker.submit(align,j+B) if j+B<len(sets) else None
                tick=time.perf_counter();S=int(f['lengths'].max())
                b={'features':torch.from_numpy(np.ascontiguousarray(f['features'][:,:S].astype(np.float16))).to(self.device),'lengths':torch.from_numpy(f['lengths']).to(self.device)}
                with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):logits=self.model(b)
                logits=logits.float().cpu().numpy();t['transfer_CNN_output']=time.perf_counter()-tick
                return cache,f,t,kk,info,logits,nxt
            future=self.model_worker.submit(infer,self.align_worker.submit(align,0),0)
            for j in range(0,len(sets),B):
                cache,f,t,kk,info,logits,nxt=future.result()
                if nxt is not None:future=self.model_worker.submit(infer,nxt,j+B)
                for key,value in t.items():times[key]+=value
                ii=ids[j:j+B]
                for i,k in zip(ii,kk):kept[i]=k
                for i,p,flag,ev in zip(ii,info['probabilities'],info['flagged'],info['evaluated']):
                    gate_probability[i]=p;gate_flagged[i]=flag;gate_evaluated[i]=ev
                if np.any(f['lengths']<L):raise ValueError('Declared target length exceeds available alignment slots')
                numba.set_num_threads(DECODE_THREADS)
                # Every neural cluster uses the same twenty-proposal budget.
                # Rank distinct strings by learned evidence, then return five.
                tick=time.perf_counter();state=Frontier(logits,f['lengths'],L);times['decoder_initialize']+=time.perf_counter()-tick
                tick=time.perf_counter();state.extend(np.full(len(ii),20,np.int32));times['decoder_materialize']+=time.perf_counter()-tick
                tick=time.perf_counter();x,_=packed_features(cache,state);times['support_and_features']+=time.perf_counter()-tick
                tick=time.perf_counter();score=self.ranker(torch.from_numpy(x)).numpy();times['reranker_MLP']+=time.perf_counter()-tick
                tick=time.perf_counter();pred=state.strings(np.argsort(-score,axis=1,kind='stable')[:,:5]);times['sort_and_strings']+=time.perf_counter()-tick
                for i,p in zip(ii,pred):output[i]=p
        self.sync();times['wall']=time.perf_counter()-start
        times['clusters']=len(read_sets);times['shortcut_clusters']=len(bypass)
        times['empty_clusters']=len(empty_indices)
        times['removed_reads']=sum(len(r)-len(k) for r,k in zip(read_sets,kept))
        times['prefilter_gate_evaluated_clusters']=sum(gate_evaluated)
        times['prefilter_gate_flagged_clusters']=sum(v is True for v in gate_flagged)
        times['prefilter_changed_clusters']=sum(len(r)!=len(k) for r,k in zip(read_sets,kept))
        return output,dict(times),dict(kept_indices=kept,shortcut_indices=sorted(bypass),empty_indices=empty_indices,
            prefilter_gate_probability=gate_probability,prefilter_gate_flagged=gate_flagged,
            prefilter_gate_evaluated=gate_evaluated)
