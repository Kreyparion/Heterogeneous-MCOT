import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import yaml
from copy import deepcopy

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))

from src.core.configuration import Config
from src.core.utils import write_to_txt

from src.experiments.WHs.drains import Drains
from src.experiments.WHs.temperature_markov import TMarkov
from src.experiments.WHs.state_markov import SMarkov

from src.experiments.RealEVsandWHs.lbda_update import MPCCoupledLambda

from src.experiments.RealEVsandWHs.model import MPCCoupledModel


def main():
    with open(os.path.join(os.path.dirname(__file__),"config.yaml")) as file:
        config = yaml.safe_load(file)
    cfg = Config(config)
    
    drains = Drains(cfg, device)
    
    tmarkov = TMarkov(cfg,drains)
    smarkov = SMarkov(cfg, tmarkov)
    R1=torch.mean(smarkov.DrawX(100000, deterministic=False)[0],1,keepdim=True)
    
    Td=torch.linspace(0,cfg.Nt,cfg.Nt)[:,None,None]
    # R2=0.24-torch.sin(2*1.8*torch.pi*(Td)/cfg.Nt)*0.15
    # R2[83:] = 0.20-torch.sin(2*2*torch.pi*(Td[83:]-3)/cfg.Nt)*0.04
    # R2*=400
    
    Td=torch.linspace(0,cfg.Nt,cfg.Nt)[:,None,None]
    R2=780 + Td*0
    # R2=500+torch.sin(torch.pi*(Td-20)/cfg.Nt)*200
    # R2=640-torch.sin(2*1.31*torch.pi*(Td+10)/cfg.Nt)*400
    # R2=600-torch.sin(2*1.31*torch.pi*(Td+10)/cfg.Nt)*300
    # R2=620-torch.sin(2*1*torch.pi*(Td+15)/cfg.Nt)*300
    # R2=570-torch.sin(2*0.9*torch.pi*(Td+40)/cfg.Nt)*300
    # R2=580-torch.sin(2*2*torch.pi*(Td+5)/cfg.Nt)*350
    write_to_txt("Ref_max",Td[:,0,0]/6,R2[:,0,0])
    lbda = MPCCoupledLambda(cfg,R2,R1,"gradient control", forward_max_grad=10000, backward_max_grad=10000)

    model = MPCCoupledModel(cfg,smarkov,lbda)
    lbda.plotResult()
    lbda.plot()
    
    Gtot = model.MPC(1000,200, nb_switches=8, verbose=True)
    lbda.plotResult(Gtot)
    lbda.plot()
    
if __name__ == "__main__":
    main()