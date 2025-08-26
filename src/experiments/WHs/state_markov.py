import torch

from src.core.configuration import Config
from src.experiments.WHs.temperature_markov import TMarkov
from src.experiments.WHs.initial_state import InitialState

class SMarkov:
    def __init__(self, cfg: Config, tmarkov: TMarkov):
        self.cfg = cfg
        self.tmarkov = tmarkov
    
    def nextMMarkov(self,m,T,i,Time):
        who_switched = torch.zeros_like(m)
        Thigh_mask = (T>=self.cfg.Tmax)
        m[Thigh_mask] = 0
        
        Tlow_mask = (T<=self.cfg.Tmin)
        m[Tlow_mask] = 1
        
        Tswitch =  torch.any(i == Time, axis=2)
        m[Tswitch & (~Tlow_mask) & (~Thigh_mask)] = 1-m[Tswitch & (~Tlow_mask) & (~Thigh_mask)]
        who_switched[Tswitch & (~Tlow_mask) & (~Thigh_mask)] = 1

        if self.cfg.KLQ_sto:
            Tother_mask = (~(Tlow_mask & Thigh_mask))
            
            m_activate = (m==1)
            m_desactivate = (m==0)
            
            prob_activate = torch.exp(-self.cfg.A*(self.cfg.Tmax-T)/(self.cfg.Tmax-self.cfg.Tmin))
            prob_desactivate = torch.exp(-self.cfg.A*(T-self.cfg.Tmin)/(self.cfg.Tmax-self.cfg.Tmin))
            
            a = torch.rand(T.shape)
            do_change = (((a<prob_activate) & m_activate) | ((a < prob_desactivate) & m_desactivate))
            m[do_change & Tother_mask] = 1 - m[do_change & Tother_mask]
    
        return m, who_switched


    def TrajMarkov2FromX(self, N,M,init_array:InitialState,Time,tinit=0,tend=100000,deterministic=False, validation=False):
        nb_step = min(self.cfg.Nt,tend)-tinit
        Traj=torch.zeros((3,nb_step,N,M))
        Traj[:,0,:,:]= init_array.state
        Traj[0,0],Traj[2,0] = self.nextMMarkov(Traj[0,0],Traj[1,0],0,Time)
        for i in range(1,nb_step):
            m,T,_=torch.clone(Traj[:,i-1])
            Traj[1,i] = self.tmarkov.nextT(i+tinit-1,m,T,characteristics=init_array.characteristics,deterministic=deterministic, validation=validation)
            Traj[0,i],Traj[2,i] = self.nextMMarkov(m,Traj[1,i],i,Time)
        return Traj
    
    def DrawX(self,N,M=1, init_array=None, nb_time_left=0, tinit=0,tend=100000,deterministic=False, return_attributes=False):
        if init_array is None:
            init_array = InitialState(self.cfg,N,M)
            
        # first 1 for 0 time (only a 0) ; second 1 for one trajectory according to this time
        
        if isinstance(nb_time_left,int):
            if nb_time_left == 0:
                Time = - torch.ones((N,1,1))
            else:
                Time=torch.randint(0,self.cfg.Nt,(N,M,nb_time_left))
        elif torch.is_tensor(nb_time_left):
            nb_max_time = torch.max(nb_time_left)
            nb_max_time = 2
            if nb_max_time == 0:
                Time = - torch.ones((N,1,1))
            else:
                Time=torch.randint(0,self.cfg.Nt,(N,M,nb_max_time))
                for k in range(nb_max_time):
                    k_time_left = (nb_time_left <= k).nonzero()
                    Time[k_time_left,:,k] = -1
        else:
            raise TypeError("Not the right type for nb_time_left, expected int of torch tensor and got : "+str(type(nb_time_left)))
        traj = self.TrajMarkov2FromX(N,M,init_array,Time, tinit=tinit,tend=tend, deterministic=deterministic)
        if return_attributes:
            return traj,Time,init_array
        return traj