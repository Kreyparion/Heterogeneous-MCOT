
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.core.configuration import Config

def store_drains(Nt):
    files = os.listdir('datas/drain_trajs')
    drains = torch.zeros((7,len(files),Nt))
    for j,filename in enumerate(files):
        df_drain = pd.read_csv('datas/drain_trajs/' + filename, sep=',')
        for i in range(7):
            drain_traj = torch.asarray(df_drain.loc[:, 'drain'])[Nt *i:Nt * (i+1)]
            drains[i,j] = drain_traj
    for i in range(7):
        traj = torch.mean(drains[i],0)
        plt.plot(torch.arange(Nt).cpu(),traj.cpu())
    plt.legend(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    plt.show()
    torch.save(drains,"datas/drains.pt")
        
def get_drains(Nt,dt):
    drains = torch.load("datas/drains.pt")
    for i in range(7):
        traj = torch.mean(drains[i],0)
        plt.plot(torch.arange(Nt).cpu()*dt/60,traj.cpu())
    plt.legend(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    plt.show()

def drain_corr(Nt,dt):
    drains = torch.load("datas/drains.pt")
    fig, axs = plt.subplots(2, 2)
    for i in range(5):
        traj = drains[i][0]
        axs[0,0].scatter(torch.arange(Nt).cpu()*dt/60,traj.cpu())
        traj = drains[i][1]
        axs[0,1].scatter(torch.arange(Nt).cpu()*dt/60,traj.cpu())
        traj = drains[i][2]
        axs[1,0].scatter(torch.arange(Nt).cpu()*dt/60,traj.cpu())
        traj = drains[i][3]
        axs[1,1].scatter(torch.arange(Nt).cpu()*dt/60,traj.cpu())
    print(drains.shape)
    plt.legend(["Monday","Tuesday","Wednesday","Thursday","Friday"])
    plt.show()

class Drains:
    def __init__(self, config: Config, device, plot=False):
        self.config = config
        drains = torch.load("datas/drains.pt").to(device)
        drains = drains[:5].reshape((-1,config.Nt))
        random_permutation = torch.randperm(drains.shape[0])
        shuffled_drains = drains[random_permutation]    
        train_part = 0.5
        cut = int(shuffled_drains.shape[0]*train_part)
        
        self.train_drains = shuffled_drains[:cut]/self.config.capWater 
        self.mean_drain = torch.mean(self.train_drains,0)       
        self.test_drains = shuffled_drains[cut:]/self.config.capWater  
    
            
            
        if plot:
            plt.plot(torch.arange(config.Nt).cpu()*config.dt/60,self.mean_drain.cpu())
            plt.show()

    def f(self,t,N, deterministic=False, validation=False):
        if deterministic:
            return self.mean_drain[t]
        if validation:
            indexes = torch.arange(N)
            return self.test_drains[indexes,t][:,None]
        if N>self.train_drains.shape[0]:
            indexes = torch.randint(0,self.train_drains.shape[0],(N,))
        else:
            indexes = torch.arange(N)
        return self.train_drains[indexes,t][:,None]
        