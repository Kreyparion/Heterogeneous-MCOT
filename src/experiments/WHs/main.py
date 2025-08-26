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
from src.experiments.WHs.lbda_update import LbdaUpdate,DeepLambda 
from src.experiments.WHs.temperature_markov import TMarkov
from src.experiments.WHs.state_markov import SMarkov

from src.experiments.WHs.model import Model

def main():
    with open(os.path.join(os.path.dirname(__file__),"config.yaml")) as file:
        config = yaml.safe_load(file)
    cfg = Config(config)
    
    drains = Drains(cfg, device)
    
    tmarkov = TMarkov(cfg,drains)
    smarkov = SMarkov(cfg, tmarkov)
    R1=torch.mean(smarkov.DrawX(100000, deterministic=False)[0],1,keepdim=True)
    
    Td=torch.linspace(0,cfg.Nt,cfg.Nt)[:,None,None]
    # R2=0.29-torch.sin(2*2*torch.pi*(Td+3)/cfg.Nt)*0.15
    # R2[72:] = 0.2-torch.sin(2*2*torch.pi*(Td[72:]-10)/cfg.Nt)*0.07
    R2 = 0*Td + 0.3
    lbda = LbdaUpdate(cfg,R2,R1,"gradient control",forward_max_grad=100000,backward_max_grad=100000)
    # lbda = DeepLambda(cfg,R2,R1,"tracking")
    model = Model(cfg,smarkov,lbda)
    
    Gtot = model.MPC(1000,100, nb_switches=8, verbose=True, tknowledge=1000)
    lbda.plotResult(Gtot)
    lbda.plot()
 
    
if __name__ == "__main__":
    main()