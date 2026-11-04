"""CUDA graph cache indexed by exact input shape."""
import torch

class ExactShapeGraphs(torch.nn.Module):
    """Reuse a captured generator call for each exact batch/slot shape."""
    def __init__(self,model,max_graphs=128):
        super().__init__()
        self.inner=model
        self.graphs={}
        self.max_graphs=max_graphs
        self.pool=torch.cuda.graph_pool_handle()

    @torch.inference_mode()
    def forward(self,b):
        x,n=b['features'],b['lengths']
        key=(tuple(x.shape),x.dtype,n.dtype)
        if key not in self.graphs:
            if len(self.graphs)>=self.max_graphs:
                return self.inner(b)
            static={'features':x.clone(),'lengths':n.clone()}
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                for _ in range(3):out=self.inner(static)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=self.pool),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                out=self.inner(static)
            self.graphs[key]=(graph,static,out)
        graph,static,out=self.graphs[key]
        static['features'].copy_(x)
        static['lengths'].copy_(n)
        graph.replay()
        return out
