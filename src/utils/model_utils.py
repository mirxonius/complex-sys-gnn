import torch
from torch_scatter import scatter

class Mean(torch.nn.Module):
    def __init__(self, dim = 0,keepdim=False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self,x):
        return torch.mean(x,dim=self.dim,keepdim=self.keepdim)
    
class MeanOnGraph(torch.nn.Module):
    def __init__(self, dim=0,keepdim=False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
    def forward(self,node_features,batch_index=None):
        if batch_index is None:
            return torch.mean(node_features,dim=self.dim,keepdim=self.keepdim)
        return scatter(src=node_features,dim=self.dim,index=batch_index,reduce="mean")