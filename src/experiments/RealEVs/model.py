import torch
import time
import matplotlib.pyplot as plt

from src.core.configuration import Config
from src.experiments.RealEVs.lbda_update import MPCEVLbdaUpdate



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
        # self.lbda.plot()
        # self.lbda.plotResult(G, rangeR)
        print("Score : ", score.item()," achieved in ",k, " times")
        return G
    

    def MPC(self):
        self.lbda.init_MPC()
        for tOn in range(self.cfg.Nt2):
            rangeR = torch.arange(tOn,self.cfg.Nt2)
            self.lbda.prepare_step_MPC(rangeR)
            Gtrain = self.learn_step_MPC(rangeR)
            self.lbda.end_step_MPC(rangeR)
            if tOn == 0:
                self.lbda.plotResult(Gtrain)
                self.lbda.plot()
        return Gtrain
         