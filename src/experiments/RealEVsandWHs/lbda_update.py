import torch
import matplotlib.pyplot as plt

from src.core.configuration import Config
from src.core.utils import write_to_txt

from src.experiments.WHs.lbda_update import LbdaUpdate
from src.experiments.RealEVs.lbda_update import MPCEVLbdaUpdate

class MPCCoupledLambda(LbdaUpdate):
    EVlbda: MPCEVLbdaUpdate
    def __init__(self, cfg, r, rnom, mode, online_coef=0, lbda0=None, forward_max_grad=0.8, backward_max_grad=0.8):
        super().__init__(cfg, r, rnom, mode, online_coef, lbda0, forward_max_grad, backward_max_grad)
        self.EV_consumption = 6
        self.real_nb_WH = 2.2*1000
        self.EVlbda = MPCEVLbdaUpdate(cfg,r,rnom,mode,online_coef, lbda0, forward_max_grad, backward_max_grad)
        self.G = torch.clone(self.rnom[:])*self.real_nb_WH 
        self.GL = self.EVlbda.Nom * self.EV_consumption
        self.nominal = 0.5*torch.clone(self.rnom[:,0,0])*self.real_nb_WH 
        self.nominal += 0.5*torch.repeat_interleave(self.GL,2,0)
        
    def update(self,N,M,y,rangeR=None):
        self.G = super().total_consumption(y,rangeR=rangeR)*self.real_nb_WH /N

        self.GL = self.EVlbda.ProdFM(self.EVlbda.fT[:,None], self.EVlbda.muFinal)
        # self.GL = self.EVlbda.gradJLFromMu(rangeR[::2]//2, self.EVlbda.MuLambda(rangeR[::2]//2))*(self.EVlbda.N1t+self.EVlbda.N2t)
        
        if rangeR[0]%2 == 0:
            self.GL+= self.EVlbda.gradJLFromMu((rangeR[::2])//2, self.EVlbda.MuLambda((rangeR[::2])//2))*(self.EVlbda.N1t+self.EVlbda.N2t)
        elif len(rangeR) != 1:
            self.GL[(rangeR[1::2])//2]+= (self.EVlbda.gradJLFromMu((rangeR[1::2])//2, self.EVlbda.MuLambda((rangeR[1::2])//2))*(self.EVlbda.N1t+self.EVlbda.N2t))[(rangeR[1::2])//2]

        self.GL *= self.EV_consumption
        GL = torch.repeat_interleave(self.GL,2,0)[rangeR]

        
        total = torch.clone(self.G)*0.5
        total += GL[:,None,None]*0.5
        return total,self.update_both_lambda(total,rangeR=rangeR)
    
    def update_both_lambda(self, s, rangeR=None):
        score = self.update_lbda(s,rangeR)
        # score2 = self.EVlbda.update_lbda(s,rangeR)
        if self.mode == "gradient control":
            self.EVlbda.lbda = torch.clone(self.lbda[:,::2,0,0])
            self.EVlbda.onlineLbda = torch.clone(self.onlineLbda)
        elif self.mode == "tracking":
            self.EVlbda.lbda = torch.clone(self.lbda[::2,0,0])
            self.EVlbda.onlineLbda = torch.clone(self.onlineLbda)
        return score
    
    def plotResult(self, G=None,rangeR=None):
        Td=torch.linspace(0,24,self.cfg.Nt)
        plt.plot(Td.cpu(),self.nominal.cpu(),label='Nominal consumption')
        plt.plot(Td.cpu(),self.r[:,0,0].cpu(),label="Reference")
        if G is None:
            plt.plot(Td.cpu(),0.5*self.G[:,0,0].cpu(),label="WH consumption")
            write_to_txt("Cons_WHs_end",Td,0.5*self.G[:,0,0])
            total = torch.clone(self.G[:,0,0])*0.5
        else:
            totG = torch.clone(G)*self.real_nb_WH
            totG[-len(self.G):] = self.G
            plt.plot(Td.cpu(),0.5*totG[:,0,0].cpu(),label="WH consumption")
            write_to_txt("Cons_WHs_end",Td,0.5*totG[:,0,0])
            total = torch.clone(totG[:,0,0])*0.5
        plt.plot(Td.cpu(),0.5*torch.repeat_interleave(self.GL,2,0).cpu(),label="EV consumption")
        write_to_txt("Cons_EVs_end",Td,0.5*torch.repeat_interleave(self.GL,2,0))
        total += 0.5*torch.repeat_interleave(self.GL,2,0)
        write_to_txt("Total_end",Td,total)
        plt.plot(Td.cpu(),total.cpu(),label="Total consumption")
        plt.xlabel("Time")
        plt.ylabel("Aggregated consumption")
        plt.grid()
        plt.legend()
        plt.show()

