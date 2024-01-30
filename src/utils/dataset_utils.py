import torch


class RadnomNoise(object):
    def __init__(self,input_dim:int=3,mean:float=0.0,std:float=0.01):
        self.mean = mean
        self.std = std
        self.input_dim=input_dim
    def __call__(self,x:torch.Tensor):
        return x + self.mean+self.std*torch.randn(size=(1,self.input_dim))
