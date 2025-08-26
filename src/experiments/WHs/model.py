import torch
import time
import matplotlib.pyplot as plt

from src.core.configuration import Config
from src.experiments.WHs.state_markov import SMarkov
from src.experiments.WHs.lbda_update import LbdaUpdate, EVLbdaUpdate, MPCEVLbdaUpdate, MPCCoupledLambda
from src.experiments.WHs.initial_state import InitialState
from src.core.utils import choice_along_first_axis

class Model:
    def __init__(self,cfg:Config, smarkov: SMarkov, lbda: LbdaUpdate):
        self.cfg = cfg
        self.smarkov = smarkov
        self.lbda = lbda
    
    def learn(self,N,M,init_array=None,nb_time_left=0,tinit=0,tend=100000, deterministic=False, verbose=False):
        rangeR = torch.arange(tinit, min(tend,self.cfg.Nt))
        score = 100000 # big enough
        k = 0
        while k<self.cfg.K and score >0.001:
            y=self.smarkov.DrawX(N,M,init_array=init_array, nb_time_left=nb_time_left, tinit=tinit, tend=tend, deterministic=deterministic)
            G,score = self.lbda.update(N,M,y,rangeR=rangeR)
        #if verbose:
            k += 1
        print("Score : ", score.item()," achieved in ",k, " times")
        if tinit==0:
            self.lbda.plotResult(G)
            self.lbda.plot()
        return G
    
    def validate(self,N,M,init_array=None,nb_time_left=0,tinit=0,tend=100000,tend2=100000, return_y=False):
        rangeR = torch.arange(tinit, min(tend,self.cfg.Nt))
        y,Time,init_with_x = self.smarkov.DrawX(N,M,init_array=init_array,nb_time_left=nb_time_left,tinit=tinit,tend=tend,return_attributes=True)
        weights = self.lbda.get_weigths(y,rangeR=rangeR)
        chosen_traj = choice_along_first_axis(weights,N,M)
        chosen_time = (Time[chosen_traj])[:,None,:]
        # Generate trajectories with M = 1
        init_with_x.cut()
        y_random = self.smarkov.TrajMarkov2FromX(N,M,init_with_x,chosen_time,tinit=tinit,tend=tend2,validation=True)
        new_signal = torch.sum(y_random[0,:],1, keepdim=True)
        if return_y:
            return new_signal/N, y_random
        return new_signal/N
    
    def MPC(self,N,M, nb_switches, N_ratio=1, tknowledge = 100000, verbose= False):
        nb_time_left = torch.ones(N,dtype=torch.int)*nb_switches
        init_with_x = InitialState(self.cfg,N,M)
        new_signal = torch.zeros(self.cfg.Nt)
        for i in range(self.cfg.Nt):
            extanded_nb_time_left = torch.repeat_interleave(nb_time_left,N_ratio,0)
            extanded_init_with_x = init_with_x.extend(N_ratio)
            self.learn(N*N_ratio,M,init_array=extanded_init_with_x,nb_time_left=extanded_nb_time_left,tinit=i,tend=i+tknowledge)
            G,y = self.validate(N,M,nb_time_left=nb_time_left,init_array=init_with_x,tinit=i,tend=i+tknowledge,tend2=i+2,return_y=True)
            if i != self.cfg.Nt-1:
                select_m = y[0,0,:,0]
                select_T = y[1,1,:,0]
                select_w = y[2,0,:,0]
                chosen_init = torch.stack([select_m,select_T,select_w])
                init_with_x.from_new_state(chosen_init,M)
                nb_time_left -= (select_w == 1).type(torch.int)
                if verbose:
                    print("At time step ", i, ", Nb time left : ", torch.sum(nb_time_left).item(), " / ", nb_switches*N)
            new_signal[i] = torch.sum(y[0,0,:,0],0)/N
        return new_signal[:,None,None]

class EVModel:
    def __init__(self,cfg:Config, lbda: EVLbdaUpdate):
        self.cfg = cfg
        self.lbda = lbda

    def learn(self):
        for k in range(100):
            G,score = self.lbda.update()
            print("Step " + str(k) + " : " + str(score.item()))
            if k%100 == 99:
                self.lbda.plot()
                self.lbda.plotResult(G)
        return G


class MPCEVModel:
    def __init__(self,cfg:Config, lbda: MPCEVLbdaUpdate):
        self.cfg = cfg
        self.lbda = lbda
        
    def learn(self):
        self.lbda.init_MPC()
        self.lbda.prepare_step_MPC(0)
        Gtrain = self.learn_step_MPC(0)
        return Gtrain

    def learn_step_MPC(self,rangeR):
        score = 2
        k = 0
        while k<200 and score >0:
            G,score = self.lbda.update(rangeR)
            #print("Step " + str(k) + " : " + str(score.item()))
            k +=1
            #if k%100 == 99:
        #self.lbda.plot()
        #self.lbda.plotResult(G, rangeR)
        print(str(score.item()))
        return G
    

    def MPC(self):
        self.lbda.init_MPC()
        for tOn in range(self.cfg.Nt2):
            rangeR = torch.arange(tOn,self.cfg.Nt2)
            self.lbda.prepare_step_MPC(rangeR)
            Gtrain = self.learn_step_MPC(rangeR)
            self.lbda.end_step_MPC(rangeR)
            

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
            if i == 0:
                self.lbda.plotResult(new_signal[:,None,None])
                self.lbda.plot()
        return new_signal[:,None,None]