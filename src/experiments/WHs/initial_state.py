import torch
from copy import deepcopy
from src.core.configuration import Config

class InitialState:
    def __init__(self,cfg:Config,N:int,M:int) -> None:
        self.cfg = cfg
        
        m_init = torch.bernoulli(torch.ones(N)*self.cfg.density_init_m)
        T_init = torch.rand(N)*(self.cfg.Tmax+0.5-self.cfg.Tmin)+self.cfg.Tmin-0.1
        who_switched = torch.zeros((N,))
        state = torch.stack([m_init,T_init,who_switched])
        self.state = torch.repeat_interleave(state[:,:,None],M,dim=2)
        self.characteristics = {"":""}
        if hasattr(self.cfg,"heterogeneous") and self.cfg.heterogeneous:
            self.characteristics = self.get_characteristics(N)
    
    def get_characteristics(self,N:int):
        characteristics = dict()
        if hasattr(self.cfg,"max_vol") and hasattr(self.cfg,"min_vol"):
            characteristics["vol"] = torch.rand(N)*(self.cfg.max_vol-self.cfg.min_vol) + self.cfg.min_vol
        if hasattr(self.cfg,"max_height") and hasattr(self.cfg,"min_height"):
            characteristics["height"] = torch.rand(N)*(self.cfg.max_height-self.cfg.min_height) + self.cfg.min_height
        if hasattr(self.cfg,"max_EI") and hasattr(self.cfg, "min_EI"):
            characteristics["EI"] = torch.rand(N)*(self.cfg.max_EI-self.cfg.min_EI) + self.cfg.min_EI
        if hasattr(self.cfg, "max_powerRes") and hasattr(self.cfg, "min_powerRes"):
            characteristics["powerRes"] = torch.rand(N)*(self.cfg.max_powerRes-self.cfg.min_powerRes) + self.cfg.min_powerRes
        for k in characteristics.keys():
            characteristics[k] = characteristics[k].reshape(-1,1)
        return characteristics
    
    def from_new_state(self,state,M):
        self.state = torch.repeat_interleave(state[:,:,None],M,dim=2)
    
    def cut(self):
        self.state = self.state[:,:,:1]
    
    def extend(self, N_ratio: int) -> "InitialState":
        if not isinstance(N_ratio, int):
            raise ValueError("N_ratio is not an integer value, found : " + str(N_ratio))
        if N_ratio == 1:
            return self
        new_obj = deepcopy(self)
        new_obj.state = torch.repeat_interleave(new_obj.state,N_ratio,1)
        if hasattr(self.cfg,"heterogeneous") and self.cfg.heterogeneous:
            for k in new_obj.characteristics.keys():
                print(k,new_obj.characteristics[k])
                new_obj.characteristics[k] = torch.repeat_interleave(new_obj.characteristics[k],N_ratio,1)
        return new_obj