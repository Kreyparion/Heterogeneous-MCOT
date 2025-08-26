import torch
import time
import matplotlib.pyplot as plt

from src.core.configuration import Config

from src.experiments.WHs.model import Model
from src.experiments.WHs.lbda_update import MPCCoupledLambda
from src.experiments.WHs.initial_state import InitialState




class MPCCoupledModel(Model):
    def __init__(self, cfg, smarkov, lbda: MPCCoupledLambda):
        super().__init__(cfg, smarkov, lbda)
        
    
    def MPC(self,N,M, nb_switches, N_ratio=1, tknowledge = 100000, verbose= False):
        self.lbda.EVlbda.init_MPC()
        nb_time_left = torch.ones(N,dtype=torch.int)*nb_switches
        init_with_x = InitialState(self.cfg,N,M)
        new_signal = torch.zeros(self.cfg.Nt)
        switches_used = 0
        for i in range(self.cfg.Nt):
            
            rangeR = torch.arange((i+1)//2,self.cfg.Nt2)
            if i != self.cfg.Nt-1:
                self.lbda.EVlbda.prepare_step_MPC(rangeR, add_new_vehicle=(i%2==0))
            extanded_nb_time_left = torch.repeat_interleave(nb_time_left,N_ratio,0)
            extanded_init_with_x = init_with_x.extend(N_ratio)
            self.learn(N*N_ratio,M,init_array=extanded_init_with_x,nb_time_left=extanded_nb_time_left,tinit=i,tend=i+tknowledge)
            G,y = self.validate(N,M,nb_time_left=nb_time_left,init_array=init_with_x,tinit=i,tend=i+tknowledge,tend2=i+2,return_y=True)
            if i != self.cfg.Nt-1:
                if i%2==0:
                    self.lbda.EVlbda.end_step_MPC(rangeR)
                select_m = y[0,0,:,0]
                select_T = y[1,1,:,0]
                select_w = y[2,0,:,0]
                chosen_init = torch.stack([select_m,select_T,select_w])
                init_with_x.from_new_state(chosen_init,M)
                nb_time_left -= (select_w == 1).type(torch.int)
                switches_used += torch.sum(select_w)
                if verbose:
                    print("At time step ", i, ", Nb time left : ", torch.sum(nb_time_left).item(), " / ", nb_switches*N)
            new_signal[i] = torch.sum(y[0,0,:,0],0)/N
            # if i == 0:
            #     self.lbda.plotResult(new_signal[:,None,None])
            #     self.lbda.plot()
        return new_signal[:,None,None]