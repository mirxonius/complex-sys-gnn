import torch
from e3nn import o3, nn, math, io
from e3nn.util.jit import compile_mode
from torch_scatter import scatter


@compile_mode("script")
class O3EquivConv(torch.nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self):
        pass




@compile_mode("script")
class O3AttentionLayer(torch.nn.Module):
    def __init__(self,
                 input_irreps:str|o3.Irreps,
                 key_irreps:str|o3.Irreps,
                 query_irreps:str|o3.Irreps,
                 value_irreps:str|o3.Irreps                
                 ) -> None:
        super().__init__()

        
    def forward(self,
                
                ):
        #Encode vector features
        # Typically edge lengths, or in the case
        # of the Galilleian or Poincare group relative velocity magnitudes
        # soft_one_hot_linspace(x,start=0.0,end,number=number_of_basis)        
        pass
    