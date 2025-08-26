import torch
import yaml

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))

from src.core.configuration import Config
from src.core.utils import write_to_txt

from src.experiments.RealEVs.lbda_update import MPCEVLbdaUpdate

from src.experiments.RealEVs.model import  MPCEVModel


def main():
    with open(os.path.join(os.path.dirname(__file__),"config.yaml")) as file:
        config = yaml.safe_load(file)
    cfg = Config(config)

    Td=torch.linspace(0,cfg.Nt2,cfg.Nt2)
    R2 = 190+Td*0
    #R2[0:12] = 0
    lbda = MPCEVLbdaUpdate(cfg,R2,None,"gradient control", forward_max_grad=10000, backward_max_grad=10000)
    lbda.plotResult(lbda.Nom)
    model = MPCEVModel(cfg,lbda)
    G = model.MPC()
    print(torch.sum(G))
    lbda.plotResult(G)
    lbda.plot()
    
    
if __name__ == "__main__":
    main()