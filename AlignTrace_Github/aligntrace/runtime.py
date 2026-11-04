"""Thread limits, deterministic backend settings, and a relocatable JIT cache."""
import os
from pathlib import Path
def configure():
    # Caller may redirect caches for read-only shared installations.
    os.environ.setdefault('NUMBA_CACHE_DIR',str(Path(__file__).resolve().parents[1]/'.cache/numba'))
    os.environ.setdefault('NUMBA_THREADING_LAYER','omp')
    import torch,numba
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    if hasattr(torch.backends,'cuda'):torch.backends.cuda.matmul.allow_tf32=False
    if hasattr(torch.backends,'cudnn'):
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
    return min(6,numba.config.NUMBA_NUM_THREADS),min(8,numba.config.NUMBA_NUM_THREADS)
