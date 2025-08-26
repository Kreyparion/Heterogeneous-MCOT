import torch

from src.experiments.WHs.drains import Drains
from src.core.configuration import Config

class TMarkov:
    def __init__(self, cfg:Config, drains: Drains):
        self.cfg = cfg
        self.drains = drains
    
    def nextT(self,t,m,T,characteristics=None, deterministic=False, validation=False):
        vol = torch.tensor(self.cfg.vol)    # Volume in m3
        height = self.cfg.height            # Height in m
        EI = self.cfg.EI                    # Thickness of isolation (in m)
        powerRes = self.cfg.powerRes        # Heat resistance power in W
        if characteristics is not None:
            for key in characteristics.keys():
                if key == "vol":
                    vol = characteristics["vol"]
                elif key == "height":
                    height = characteristics["height"]
                elif key == "EI":
                    EI = characteristics["EI"]
                elif key == "powerRes":
                    powerRes = characteristics["powerRes"]

        Sect = vol/height
        ray = torch.sqrt(Sect/3.14)         # Radius in m
        EI4 = EI/4
        coefLoss = self.cfg.CI/EI4 * 2 * 3.14 * ray
        e_unit = vol * self.cfg.rho * self.cfg.capWater
        lossH = (coefLoss*60)/(self.cfg.capWater * self.cfg.rho * Sect)
        Pmax = 60*powerRes
        loss=-lossH*(T-self.cfg.T_amb)
        heating=m*Pmax/ e_unit
        drain_heat=-self.drains.f(t,m.shape[0],deterministic=deterministic,validation=validation)/(vol*self.cfg.rho)
        return T+self.cfg.dt*(loss+heating)+drain_heat