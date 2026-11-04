"""Mixed64 convolutional sequence model."""
import torch
from torch import nn

class Block(nn.Module):
    def __init__(self,width,dilation):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.conv=nn.Conv1d(width,width,5,padding=2*dilation,dilation=dilation,groups=width)
        self.mix=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))
    def forward(self,h,mask):
        z=(self.norm(h)*mask).transpose(1,2)
        return (h+self.mix(self.conv(z).transpose(1,2)))*mask

class Mixed64(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(32,64)
        self.blocks=nn.ModuleList([Block(64,d) for d in (1,2,4,8,16)])
        self.output=nn.Sequential(nn.LayerNorm(64),nn.Linear(64,5))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)
        self.prior_strength=nn.Parameter(torch.tensor(1.))
    def forward(self,b):
        f=b['features'].float()
        mask=torch.arange(f.shape[1],device=f.device)[None,:]<b['lengths'][:,None]
        expanded=mask.unsqueeze(-1);h=self.input(f)*expanded
        for block in self.blocks:h=block(h,expanded)
        # Empirical aligned blank/base frequencies, not biological constraints.
        prior=(.5*f[:,:,:5]+.5*f[:,:,5:10]+.002).log()
        logits=self.output(h)+self.prior_strength.clamp(.1,3.)*prior
        logits=logits.masked_fill(~expanded,-10000.)
        logits[:,:,0]=torch.where(mask,logits[:,:,0],0.)
        return logits
