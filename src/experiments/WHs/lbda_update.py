import torch
import matplotlib.pyplot as plt

from src.core.configuration import Config
from src.core.utils import write_to_txt, choice_along_first_axis2

class ParamsLambda(torch.nn.Module):
    def __init__(self,lbda0):
        super(ParamsLambda, self).__init__()
        self.params = torch.nn.Parameter(lbda0)
        
    def forward(self):
        return self.params
        

class DeepLambda():
    def __init__(self,cfg: Config,r,rnom,mode,lbda0=None,):
        super(DeepLambda, self).__init__()
        self.cfg = cfg
        self.r = r
        self.rnom = rnom
        self.mode = mode
        
        if lbda0 is None:
            if self.mode == "tracking":
                lbda0 = torch.ones((self.cfg.Nt,1,1))*0.01
        self.lbda = ParamsLambda(lbda0)
    
        self.optimizer = torch.optim.SGD(self.lbda.parameters(),0.4)
    
    def cost(self,y):
        return 1/self.cfg.epsilon
    
    def expval(self,y,rangeR=None):
        if rangeR is None:
            rangeR = torch.arange(self.cfg.Nt)
        if self.mode == "tracking":
            return torch.exp(torch.sum(self.lbda()[rangeR]*(y[0]-self.r[rangeR]),0,keepdim=True)-self.cost(y))

    def get_weigths(self,y,rangeR=None):
        if rangeR is None:
            rangeR = torch.arange(self.cfg.Nt)
        exp_val = self.expval(y,rangeR)
        s2=torch.sum(exp_val,2,keepdim=True)
        return exp_val[0]/s2[0]
    
    def update(self,N,M,y,rangeR=None):
        if rangeR is None:
            rangeR = torch.arange(self.cfg.Nt)
        exp_val = self.expval(y,rangeR)
        s1=torch.sum((y[0])*exp_val,2,keepdim=True)
        s2=torch.sum(exp_val,2,keepdim=True)
        s=torch.sum(s1/s2,1,keepdim=True)
        loss=torch.sum(torch.log(s2/M),1,keepdim=True)/N
        # loss = torch.sum(torch.square(s/N-self.r[rangeR]))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return s/N,torch.exp(loss)
    
    def plot(self):
        if self.mode == "tracking":
            plt.plot(self.lbda().detach().cpu()[:,0,0])
            plt.grid()
            plt.show()
        elif self.mode == "tracking_and_online":
            plt.plot(self.lbda()[0].detach().cpu()[:,0,0])
            plt.grid()
            plt.show()
        elif self.mode == "gradient control":
            plt.plot(self.lbda().detach().cpu()[0,:,0,0])
            plt.plot(self.lbda().detach().cpu()[1,:,0,0])
            plt.plot(self.lbda().detach().cpu()[2,:,0,0])
            plt.legend(["over a cap","forward grad","backward grad"])
            plt.title("Lambda")
            plt.grid()
            plt.show()
    
    def plotResult(self,G,rangeR=None):
        if rangeR is None:
            rangeR = torch.arange(self.cfg.Nt)
        Td=torch.linspace(0,24,self.cfg.Nt).cpu()
        plt.plot(Td,self.rnom.cpu()[:,0,0],label='Nominal')
        plt.plot(Td[rangeR.cpu()],self.r.detach().cpu()[rangeR.cpu()][:,0,0],label="Reference")
        plt.plot(Td,G.detach().cpu()[:,0,0],label='Aggregated consumption')
        plt.legend()
        plt.grid()
        plt.xlabel("Time (in h)")
        plt.ylabel("Power (normalized)")
        plt.show()
    
        

class LbdaUpdate:
    def __init__(self,cfg: Config,r,rnom,mode,online_coef=0,lbda0=None,forward_max_grad=0.8,backward_max_grad=0.8):
        self.r = r
        self.rnom = rnom
        self.cfg = cfg
        self.mode = mode
        self.lbda = lbda0
        self.forward_max_grad = forward_max_grad
        self.backward_max_grad = backward_max_grad
        self.online_coef = online_coef
        self.onlineLbda = torch.tensor(0.01)
        if lbda0 is None:
            if self.mode == "tracking":
                self.lbda = torch.ones((self.cfg.Nt,1,1))*0.01
            elif self.mode == "gradient control":
                self.lbda = torch.ones((3,self.cfg.Nt,1,1))*0.01
    
    def cost(self,y):
        return torch.sum(y[2],0,keepdim=True)/self.cfg.epsilon
        return 1/self.cfg.epsilon
    
    def expval(self,y,rangeR=None):
        if rangeR is None:
            rangeR = torch.arange(self.cfg.Nt)
        if self.mode == "tracking":
            S = self.online_coef*self.onlineLbda*(y[0,:1]) + torch.sum(self.lbda[rangeR]*(y[0]),0,keepdim=True)-self.cost(y)
        elif self.mode == "online":
            S = self.online_coef*self.onlineLbda*(y[0,:1])-self.cost(y)
        elif self.mode == "gradient control":
            S = self.online_coef*self.onlineLbda*(y[0,:1]) + torch.sum(self.lbda[0,rangeR]*(y[0]),0,keepdim=True) + torch.sum(self.lbda[1,rangeR][:-1]*((y[0,1:]-y[0,:-1])),0,keepdim=True) + torch.sum(self.lbda[2,rangeR][:-1]*((y[0,:-1]-y[0,1:])),0,keepdim=True) -self.cost(y)
        # S = torch.clip(S,-500,500)
        # S = torch.minimum(S,700+S*0)
        maximum_in_M = torch.max(S,dim=2,keepdim=True).values
        exp_val = torch.exp(S+500-maximum_in_M)
        
        if torch.any(torch.isnan(exp_val)):
            raise ValueError("NaN in exp")
        elif torch.any(torch.isinf(exp_val)):
            print("In S, there is a NaN ? : ", torch.any(torch.isnan(S)),"there is a inf : ",torch.any(torch.isinf(S)))
            print("Lambda max ", torch.max(self.lbda),", lambda min ",torch.min(self.lbda))
            print("S max ", torch.max(S),", S min ",torch.min(S))
            print("In exp, there is a NaN ? : ", torch.any(torch.isnan(exp_val)),"there is a inf : ",torch.any(torch.isinf(exp_val)))
            raise ValueError("Inf in exp")
        return exp_val

    def total_consumption(self,y,rangeR=None, return_weights=False):
        if rangeR is None:
            rangeR = torch.arange(self.cfg.Nt)
        exp_val = self.expval(y,rangeR)
        s1=torch.sum((y[0])*exp_val,2,keepdim=True)
        s2=torch.sum(exp_val,2,keepdim=True)
        s=torch.sum(s1/s2,1,keepdim=True)
        if return_weights:
            return s,exp_val[0]/s2[0]
        return s
    
    def get_weigths(self,y,rangeR=None):
        if rangeR is None:
            rangeR = torch.arange(self.cfg.Nt)
        exp_val = self.expval(y,rangeR)
        s2=torch.sum(exp_val,2,keepdim=True)
        return exp_val[0]/s2[0]
    
    def update(self,N,M,y,rangeR=None):
        G = self.total_consumption(y,rangeR=rangeR)/N
        return G,self.update_lbda(G,rangeR=rangeR)
    
    def update_lbda(self,s,rangeR=None):
        if rangeR is None:
            rangeR = torch.arange(self.cfg.Nt)
        self.onlineLbda-=self.cfg.lr*(s[0]-self.r[rangeR][0])[0,0]
        if self.mode == "online":
            return torch.abs(s[0]-self.r[rangeR][0])[0,0]
        if self.mode == "tracking":
            self.lbda[rangeR]-=self.cfg.lr*(s-self.r[rangeR])
            return torch.linalg.norm(s[0]-self.r[rangeR[0]])
        elif self.mode == "gradient control":
            self.lbda[0,rangeR]-=self.cfg.lr*(s-self.r[rangeR])
            self.lbda[1,rangeR[:-1]]-=self.cfg.lr*(s[1:]-s[:-1]-self.forward_max_grad)
            self.lbda[2,rangeR[:-1]]-=self.cfg.lr*(s[:-1]-s[1:]-self.backward_max_grad)
            self.onlineLbda = torch.minimum(self.onlineLbda,self.onlineLbda*0)
            self.lbda = torch.minimum(self.lbda,self.lbda*0)
            return torch.linalg.norm(torch.maximum(s-self.r[rangeR],s*0)) + torch.linalg.norm(torch.maximum(s[1:]-s[:-1]-self.forward_max_grad,s[1:]*0)) + torch.linalg.norm(torch.maximum(s[:-1]-s[1:]-self.backward_max_grad,s[:1]*0))
    
    def plot(self):
        if self.mode == "tracking":
            plt.plot(self.lbda.cpu()[:,0,0])
            plt.grid()
            plt.show()
        elif self.mode == "tracking_and_online":
            plt.plot(self.lbda[0].cpu()[:,0,0])
            plt.grid()
            plt.show()
        elif self.mode == "gradient control":
            plt.plot(self.lbda.cpu()[0,:,0,0])
            plt.plot(self.lbda.cpu()[1,:,0,0])
            plt.plot(self.lbda.cpu()[2,:,0,0])
            plt.legend(["over a cap","forward grad","backward grad"])
            plt.title("Lambda")
            plt.grid()
            plt.show()
    
    def plotResult(self,G,title : str = None,rangeR=None):
        if rangeR is None:
            rangeR = torch.arange(self.cfg.Nt)
            
        Td=torch.linspace(0,24,self.cfg.Nt)
        plt.plot(Td.cpu(),self.rnom.cpu()[:,0,0]*1100,label='Nominal')
        if self.mode == "gradient control":
            has_constraint = (self.r[rangeR][:,0,0] < 1)
            plt.plot(Td[rangeR[has_constraint]].cpu(),self.r[rangeR[has_constraint]][:,0,0].cpu()*1100,label="Reference")
        else:
            plt.plot(Td[rangeR].cpu(),self.r[rangeR][:,0,0].cpu()*1100,label="Reference")
        plt.plot(Td.cpu(),G[:,0,0].cpu()*1100,label='Aggregated consumption')
        if title is not None:
            plt.title(title)
        plt.legend()
        plt.grid()
        plt.xlabel("Time (in h)")
        plt.ylabel("Power (kW)")
        plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class MPCEVLbdaUpdate:
    def __init__(self,cfg: Config,r,rnom,mode,online_coef=0,lbda0=None,forward_max_grad=0.8,backward_max_grad=0.8):
        
        self.r = r
        self.rnom = rnom
        self.cfg = cfg
        self.mode = mode
        
        self.forward_max_grad = forward_max_grad
        self.backward_max_grad = backward_max_grad
        self.online_coef = online_coef
        self.onlineLbda = torch.tensor(0.01)
        
        
        NTime=24
        dt=self.cfg.dt2
        Nt=self.cfg.Nt2
        self.Nt = Nt
        self.Td=torch.linspace(0,NTime,Nt)
        
        if self.mode == "gradient control":
            lbd0=torch.zeros((3,Nt)) # Starting lbda
        elif self.mode == "tracking":
            lbd0=torch.zeros(Nt) # Starting lbda
        self.lbda = lbd0

        ###Import Data

        DataTransaction=pd.read_excel('./datas/elaadnl_open_ev_datasets.xlsx',sheet_name="open_transactions")
        DataTransaction['WeekDay'] = DataTransaction['UTCTransactionStart'].dt.dayofweek
        DataTransaction['ConnectedTime4'] = ((DataTransaction['ConnectedTime']*60)//dt)
        DataTransaction['ChargeTime4'] = ((DataTransaction['ChargeTime']*60)//dt)

        def Pow(lP):
            return(np.select([lP < 2, (2 <= lP) & (lP < 5), (5 <= lP) & (lP <= 10), lP > 10],[0, 1, 2, 3]))

        DataTransaction['Start4'] = DataTransaction['UTCTransactionStart'].dt.hour*(60//dt)+((DataTransaction['UTCTransactionStart'].dt.minute)//dt)
        DataTransaction['Stop4'] = DataTransaction['UTCTransactionStart'].dt.hour*(60//dt)+((DataTransaction['UTCTransactionStop'].dt.minute)//dt)
        DataTransactionOld=DataTransaction
        DataTransaction['MaxPower'] = Pow(DataTransaction['MaxPower'])

        DataTransaction['Day'] = DataTransaction['UTCTransactionStart'].dt.day_of_year
        DataTransaction=DataTransaction.drop(['TransactionId', 'ChargePoint', 'Connector', 'UTCTransactionStart','UTCTransactionStop', 'StartCard', 'ConnectedTime', 'ChargeTime','TotalEnergy'],axis=1)
        # print(DataTransaction)
        DataTransaction = DataTransaction[(DataTransaction["ConnectedTime4"]<Nt) & (DataTransaction["ChargeTime4"]<Nt) & (DataTransaction["Start4"]<Nt) & (DataTransaction["Start4"]+DataTransaction["ConnectedTime4"]-DataTransaction["ChargeTime4"]<Nt) ]

        WeekendData = DataTransaction[DataTransaction['WeekDay'].isin([5, 6])]
        WeekdayData = DataTransaction[~DataTransaction['WeekDay'].isin([5, 6])]

        self.rate=0.165
        TrainDay,TestDay=train_test_split(np.unique(WeekdayData.Day), test_size=self.rate, random_state=0)

        WeekdayTest = WeekdayData[WeekdayData['Day'].isin(TestDay)]
        WeekdayTrain = WeekdayData[WeekdayData['Day'].isin(TrainDay)]
        WeekdayTest=WeekdayTest.drop(['Day'],axis=1)
        WeekdayTrain=WeekdayTrain.drop(['Day'],axis=1)


        self.dtTrain=WeekdayTrain
        self.dtTest=WeekdayTest
        print(self.dtTest)

        cT=torch.zeros((Nt,Nt))
        for ta in range(0,Nt):
            for tc in range(0,Nt):
                cT[ta,tc]+=(ta*NTime/Nt-tc*NTime/Nt)**2
                
        self.cT = cT


        fT=torch.zeros((Nt,Nt,Nt))
        for tch in range(Nt):
            for tc in range(Nt):
                fT[tc,tch,tc:min(Nt,tc+tch)]=1
                
        fT = torch.swapaxes(fT,0,2)

        

        ### Definition of mu1 and mu2

        nu0=torch.zeros((Nt,Nt,Nt))

        for row in self.dtTrain.itertuples(index=False):
            if row[4] <Nt and row[2] < Nt and row[3]<Nt:
                nu0[int(row[4]),int(row[2]),int(row[3])]+=1
        nu0=nu0/torch.sum(nu0)
        #nu0 = 0*nu0

        nuReal=torch.zeros((Nt,Nt,Nt))

        for row in self.dtTest.itertuples(index=False):
            if row[4] <Nt and row[2] < Nt and row[3]<Nt:
                nuReal[int(row[4]),int(row[2]),int(row[3])]+=1
        nuReal=nuReal/torch.sum(nuReal)

        muReal=torch.zeros((Nt,Nt,Nt,Nt))

        for t in range(0,Nt):
            muReal[t,:,:,t]=nuReal[t]

        self.Pred=self.ProdFM(fT[:,None],muReal)*len(self.dtTest)
        
        
        muReal = torch.tensor([])


        mu1=torch.zeros((Nt,Nt,Nt,Nt))

        for t in range(0,Nt):
            mu1[t,:,:,t]=nu0[t]
        N=len(self.dtTrain)*self.rate/(1-self.rate)
        print(N)
        print(len(self.dtTest))
        self.muFinal=torch.zeros((self.cfg.Nt2,self.cfg.Nt2,self.cfg.Nt2,self.cfg.Nt2))
        self.nb_hist = 2.3
        self.dur_hist = 25
        for i in range(self.dur_hist):
            self.muFinal[0,i,i,0] =self.nb_hist
            mu1[0,i,i,0] = self.nb_hist/N

        self.Nom=self.ProdFM(fT[:,None],mu1)*len(self.dtTest)
        
        # from RealDataEV2_torch_same_p import PredictedConsumption, NominalConsumption, mu1 as m, fT as ffff
        # print(torch.all(self.Pred == PredictedConsumption))
        # print(torch.all(self.Nom == NominalConsumption))
        # print(torch.all(mu1 ==m))
        # print(torch.all(fT ==ffff))
        
        mu1 = torch.tensor([])
        self.fT = fT
        self.nu0 = nu0
        
        
        self.listX=torch.zeros((0,3), dtype=int)
        
    
    def ProdFM(self,f,m):
        return(torch.tensordot(f,torch.sum(m,1),dims=[(1,2,3),(0,1,2)]))
    
    def gradJLFromMu(self,rangeR,MuL):
        return(self.ProdFM(self.fT[:,None,:,rangeR],MuL))
    
    def RangeR(self,tOn,rangeR):
        l=[]
        for i in range(tOn,self.Nt):
            if i in rangeR:
                l.append(i)
        return(l)
    
    def update(self, rangeR=None):
        if rangeR is None:
            rangeR = torch.arange(self.Nt)
        MuL=self.MuLambda(rangeR)
        
        GL=self.gradJLFromMu(rangeR, MuL)
        plt.plot(self.Td.cpu(),GL.cpu())
        plt.show()
        return self.update_lbda(GL,rangeR=rangeR)

    
    def update_lbda(self, s,rangeR=None):
        if rangeR is None:
            rangeR = torch.arange(self.Nt)
        if self.mode == "gradient control":
            self.lbda[0,rangeR]-=3*(s[rangeR]-self.r_updated[rangeR])
            self.lbda[1,rangeR[:-1]]-=3*(s[rangeR][1:]-s[rangeR][:-1]-self.forward_max_grad)
            self.lbda[2,rangeR[:-1]]-=3*(s[rangeR][:-1]-s[rangeR][1:]-self.backward_max_grad)
            self.lbda = torch.minimum(self.lbda, 0*self.lbda)
        elif self.mode == "tracking":
            self.lbda[rangeR]-=3*(s[rangeR]-self.r_updated[rangeR])
        return s, torch.linalg.norm(torch.maximum(s[rangeR]-self.r_updated[rangeR],rangeR*0))
    
    def InitT(self,tOn,listX,N):
        #tOn = rangeR[0]
        nu_0=torch.zeros((self.Nt,self.Nt,self.Nt))
        mu_2=torch.zeros((self.Nt,self.Nt,self.Nt,self.Nt-tOn))
        N1t=len(listX)
        N2t=torch.sum(self.nu0[tOn+1:])*N
        if N1t+N2t != 0:
            nu_0[0:tOn+1]=self.nu0[0:tOn+1]*0
            nu_0[tOn+1:]=self.nu0[tOn+1:]*N2t/(N1t+N2t)
            nu_0[listX[:,0],listX[:,1],listX[:,2]] += 1/(N1t+N2t)

            for ta_minus_tOn in range(-tOn,0):
                for tco_minus_tch in range(0,self.Nt-tOn-ta_minus_tOn):
                    all_values_tch =torch.arange(self.Nt)
                    all_values_tco = all_values_tch + tco_minus_tch
                    mask = (all_values_tco < self.Nt)
                    mu_2[ta_minus_tOn + tOn,all_values_tco[mask],all_values_tch[mask],:(ta_minus_tOn+tco_minus_tch+1)]=1
                    
                for tco_minus_tch in range(self.Nt-tOn-ta_minus_tOn,self.Nt):
                    all_values_tch =torch.arange(self.Nt)
                    all_values_tco = all_values_tch + tco_minus_tch
                    mask = (all_values_tco < self.Nt)
                    mu_2[ta_minus_tOn + tOn,all_values_tco[mask],all_values_tch[mask],:(self.Nt-tOn)]=1
            
            for ta_minus_tOn in range(0,self.Nt-tOn):
                for tco_minus_tch in range(0,self.Nt-tOn-ta_minus_tOn):
                    all_values_tch =torch.arange(self.Nt)
                    all_values_tco = all_values_tch + tco_minus_tch
                    mask = (all_values_tco < self.Nt)
                    mu_2[ta_minus_tOn + tOn,all_values_tco[mask],all_values_tch[mask],ta_minus_tOn:(ta_minus_tOn+tco_minus_tch+1)]=1
                    
                for tco_minus_tch in range(self.Nt-tOn-ta_minus_tOn,self.Nt):
                    all_values_tch =torch.arange(self.Nt)
                    all_values_tco = all_values_tch + tco_minus_tch
                    mask = (all_values_tco < self.Nt)
                    mu_2[ta_minus_tOn + tOn,all_values_tco[mask],all_values_tch[mask],ta_minus_tOn:(self.Nt-tOn)]=1
            
            mu_2_sums = torch.sum(mu_2,3)
            mask = (mu_2_sums !=0)
            mu_2[mask,:] = mu_2[mask,:]*nu_0[mask][:,None]/mu_2_sums[mask][:,None]
        
        # cT=torch.zeros((self.Nt,self.Nt))
        # for tc in range(0,self.Nt):
        #     cT[:,tc]=(int(tc == tOn) + 1)/2
                
        # self.cT = cT
        
        return(nu_0,mu_2)

    
    def MuLambda(self,rangeR):
        if self.mode == "tracking":
            total_lbda = torch.clone(self.lbda[rangeR])
            total_lbda[0] += self.onlineLbda * self.online_coef
            S=torch.tensordot(total_lbda,self.fT[rangeR,None][:,:,:,rangeR],dims=[[0],[0]])[:,None]-self.cT[:,None,None,rangeR]/self.cfg.epsilon2
        elif self.mode == "gradient control":
            S=torch.tensordot(self.lbda[0,rangeR],self.fT[rangeR,None][:,:,:,rangeR],dims=[[0],[0]])[:,None]-self.cT[:,None,None,rangeR]/self.cfg.epsilon2
            #S=torch.sum(self.lbda[0,:,None,None,None]*self.fT,axis=0) + torch.sum(self.lbda[1,:-1,None,None,None]*(self.fT[1:] - self.fT[:-1]),axis=0) + torch.sum(self.lbda[2,:-1,None,None,None]*(self.fT[:-1] - self.fT[1:]),axis=0)
        # S = torch.clip(S,-500,500)
        # S = torch.minimum(S,700+S*0)
        maximum_in_M = torch.max(S,dim=-1,keepdim=True).values
        exp_val = torch.exp(S+500-maximum_in_M)
        B=exp_val*self.mu_2
        UB=torch.sum(B,-1)
        mask = (UB > 1e-270)
        divi = torch.zeros_like(self.nu_0)

        divi[mask] = self.nu_0[mask] / UB[mask]

        MuL=B*(divi[:,:,:,None])
        if torch.any(torch.isnan(MuL)):
            print("In S, there is a NaN ? : ", torch.any(torch.isnan(S)),"there is a inf : ",torch.any(torch.isinf(S)))
            print("Lambda max ", torch.max(self.lbda),", lambda min ",torch.min(self.lbda))
            print("S max ", torch.max(S),", S min ",torch.min(S))
            print("In exp, there is a NaN ? : ", torch.any(torch.isnan(torch.exp(-S/self.cfg.epsilon))),"there is a inf : ",torch.any(torch.isinf(torch.exp(-S/self.cfg.epsilon))))
            print("In B, there is a NaN ? : ", torch.any(torch.isnan(B)),"there is a inf : ",torch.any(torch.isinf(B)))
            print("In divi, there is a NaN ? : ", torch.any(torch.isnan(divi)),"there is a inf : ",torch.any(torch.isinf(divi)))
            raise ValueError("NAN on MUL")
        return MuL
    
    def VehiclesAtStepT(self,t):
        #return torch.zeros((0,3), dtype=int)
        return torch.tensor(self.dtTest[self.dtTest['Start4']==t.item()][["Start4","ConnectedTime4","ChargeTime4"]].to_numpy()).int()
    

    def plotResult(self,G,rangeR= None):
        if rangeR is None:
            rangeR = torch.arange(self.Nt)
        plt.plot(self.Td.cpu(),self.Nom.cpu(),label='Nominal consumption')
        plt.plot(self.Td.cpu(),self.Pred.cpu(),label="Predicted consumption")
        plt.plot(self.Td.cpu(),self.ProdFM(self.fT[:,None], self.muFinal).cpu(),label="Optimized consumption")
        plt.plot(self.Td[rangeR].cpu(),self.r[rangeR].cpu(),label="Constraint")
        plt.xlabel("Time")
        plt.ylabel("Aggregated consumption")
        plt.grid()
        plt.legend()
        plt.show()
        
    def plot(self):
        if self.mode == "gradient control":
            plt.plot(torch.arange(self.Nt).cpu(),self.lbda[0].cpu())
            plt.plot(torch.arange(self.Nt).cpu(),self.lbda[1].cpu())
            plt.plot(torch.arange(self.Nt).cpu(),self.lbda[2].cpu())
            plt.show()
        elif self.mode == "tracking":
            plt.plot(torch.arange(self.Nt).cpu(),self.lbda.cpu())
            plt.show()
        
    def init_MPC(self):
        self.muFinal=torch.zeros((self.cfg.Nt2,self.cfg.Nt2,self.cfg.Nt2,self.cfg.Nt2))
        for i in range(self.dur_hist):
            self.muFinal[0,i,i,0] =self.nb_hist
        self.listX=torch.zeros((0,3), dtype=int)

    def prepare_step_MPC(self,rangeR, add_new_vehicle=True):
        t = rangeR[0]
        N=len(self.dtTrain)*self.rate/(1-self.rate)
        if add_new_vehicle:
            self.listX = torch.cat([self.VehiclesAtStepT(t),self.listX],0)
        self.N1t=len(self.listX)
        self.N2t=torch.sum(self.nu0[t+1:])*N
        # r=(R2-ProdFM(fT, muFinal)-rHist)/(N1t+N2t)
        # self.r_updated=(self.r-self.ProdFM(self.fT[:,None], self.muFinal))/(N1t+N2t)
        self.nu_0,self.mu_2=self.InitT(t, self.listX,N)
    
    def end_step_MPC(self, rangeR):
        t = rangeR[0]
        MuL=self.MuLambda(rangeR)
        if len(self.listX) != 0:
            mul_v = MuL[self.listX[:,0],self.listX[:,1],self.listX[:,2]]
            sum_mul_V = torch.sum(mul_v,1)
            if torch.any(sum_mul_V == 0):
                print(sum_mul_V)
                print(self.listX)
                print(mul_v)
                raise ValueError
            tc = choice_along_first_axis2(mul_v/sum_mul_V[:,None])
            chosen = (tc == 0)
            print(tc)
            self.muFinal[self.listX[chosen,0],self.listX[chosen,1],self.listX[chosen,2],t] += 1
            self.listX = self.listX[~chosen]


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




class EVLbdaUpdate:
    def __init__(self,cfg: Config,r,rnom,mode,online_coef=0,lbda0=None,forward_max_grad=0.8,backward_max_grad=0.8):
        
        self.r = r
        self.rnom = rnom
        self.cfg = cfg
        self.mode = mode
        
        self.forward_max_grad = forward_max_grad
        self.backward_max_grad = backward_max_grad
        self.online_coef = online_coef
        self.onlineLbda = torch.tensor(0.01)

    
        ### Global Parameters 

        v=0.25 #in h^(-1) charging speed for the EVs; An EV arriving with 0% of battery will be fully charged in 4h.

        NTime=8 #Number of hours
        Nt=6*NTime #Number of points for the discretization in time
        self.Nt = Nt
        Ns=20 #Number of points for the discretization in space

    
        self.Td=(torch.linspace(0,NTime,Nt)+9)*6 #Discretization of time

        if self.mode == "gradient control":
            lbd0=torch.zeros((3,Nt)) # Starting lbda
        elif self.mode == "tracking":
            lbd0=torch.zeros(Nt) # Starting lbda
        self.lbda = lbd0


        cT=torch.zeros((Nt,Nt)) #Discretization of the cost function c in the 1S-RCMOT problem
        for tx in range(Nt):
            for ty in range(Nt):
                cT[tx,ty]=(tx*NTime/Nt-ty*NTime/Nt)**2
        self.cT = cT

        fT=torch.zeros((Nt,Nt,Ns,Nt)) #Discretization of the general moment f in the 1S-RCMOT problem, representing the global consumption
        
        for s in range(Ns):
            for tc in range(Nt):
                end = min(Nt,int(tc+Nt*(1-s/(Ns-1))/(v*NTime)))
                fT[tc:end,:,s,tc]+=1
        self.fT = fT

        ### Definition of mu1 and mu2

        mu0=torch.zeros((Nt,Ns)) #Initial law for the arrival of EVs. 

        for t in range(int(Nt/4)):
            for s in range(int(Ns*0.05),int(Ns*0.95)):
                mu0[t,s]=1
        mu0=mu0/torch.sum(mu0)
        
        self.mu0 = mu0

        mu1=torch.zeros((Nt,Ns,Nt))

        for t in range(0,Nt):
            for s in range(0,Ns):
                mu1[t,s,t]=mu0[t,s]
                
        self.mu1 = mu1

        mu2=torch.zeros((Nt,Ns,Nt))

        for t in range(0,Nt):
            for s in range(0,Ns):
                for tc in range(t,int(Nt*(NTime-(1-s/Ns)/v)/NTime)):
                    mu2[t,s,tc]=1
                if torch.sum(mu2[t,s])!=0:
                    mu2[t,s]=mu2[t,s]*mu0[t,s]/(torch.sum(mu2[t,s]))
        self.mu2 = mu2
    
    def gradJL(self):
        if self.mode == "tracking":
            S=torch.sum(self.lbda[:,None,None,None]*self.fT,axis=0)
        elif self.mode == "gradient control":
            S=torch.sum(self.lbda[0,:,None,None,None]*self.fT,axis=0) + torch.sum(self.lbda[1,:-1,None,None,None]*(self.fT[1:] - self.fT[:-1]),axis=0) + torch.sum(self.lbda[2,:-1,None,None,None]*(self.fT[:-1] - self.fT[1:]),axis=0)
        B=torch.tensordot(torch.exp(S/self.cfg.epsilon)*self.mu2,torch.exp(-self.cT/self.cfg.epsilon),dims=[[2],[1]])
        fB=torch.tensordot(self.fT*torch.exp(S/self.cfg.epsilon)*self.mu2,torch.exp(-self.cT/self.cfg.epsilon),dims=[[3],[1]])
        B = torch.repeat_interleave(B[None],self.Nt,0)
        mask = (B !=0)
        c = torch.zeros_like(fB)
        c[mask] = fB[mask]/B[mask]
        if torch.any(torch.isnan(c)):
            print(torch.any(torch.isnan(fB)))
            print(torch.any(torch.isnan(B)))
            print(torch.max(S))
            print(torch.min(S))
            print(torch.exp(torch.max(S)))
            print(torch.exp(torch.min(S)))
            print(torch.max(self.lbda))
            print(torch.any(torch.isnan(torch.exp(S/self.cfg.epsilon)*self.mu2)))
            raise ValueError("Got a NaN")
        return(torch.tensordot(c,self.mu1,dims=[(1,2,3),(0,1,2)]))

    def update(self, rangeR=None):
        if rangeR is None:
            rangeR = torch.arange(self.Nt)
        GL=self.gradJL()
        return self.update_lbda(GL,rangeR=rangeR)
        
    
    def update_lbda(self, s,rangeR=None):
        if rangeR is None:
            rangeR = torch.arange(self.Nt)
        if self.mode == "gradient control":
            self.lbda[0,rangeR]-=0.4*(s-self.r[rangeR])
            self.lbda[1,rangeR[:-1]]-=0.4*(s[1:]-s[:-1]-self.forward_max_grad)
            self.lbda[2,rangeR[:-1]]-=0.4*(s[:-1]-s[1:]-self.backward_max_grad)
            self.lbda = torch.minimum(self.lbda, 0*self.lbda)
        elif self.mode == "tracking":
            self.lbda[rangeR]-=0.4*(s[rangeR]-self.r[rangeR])
        return s, torch.linalg.norm(torch.maximum(s-self.r[rangeR],s*0))
    
    def MuLambda(self):
        if self.mode == "tracking":
            S=torch.sum(self.lbda[:,None,None,None]*self.fT,axis=0)
        elif self.mode == "gradient control":
            S=torch.sum(self.lbda[0,:,None,None,None]*self.fT,axis=0) + torch.sum(self.lbda[1,:-1,None,None,None]*(self.fT[1:] - self.fT[:-1]),axis=0) + torch.sum(self.lbda[2,:-1,None,None,None]*(self.fT[:-1] - self.fT[1:]),axis=0)
        B=torch.tensordot(torch.exp(S/self.cfg.epsilon)*self.mu2,torch.exp(-self.cT/self.cfg.epsilon),dims=[[2],[1]])
        #B = torch.repeat_interleave(B[None],self.Nt,0)
        mask = (B !=0)
        c = torch.zeros_like(self.mu1)
        c[mask] = self.mu1[mask]/B[mask]
        MuL=self.mu2*torch.exp(S/self.cfg.epsilon)*torch.tensordot(c,torch.exp(-self.cT/self.cfg.epsilon),dims=[[2],[0]])
        return MuL

    def plotResult(self,G, rangeR=None):
        plt.plot(self.Td.cpu(),torch.tensordot(self.fT,self.mu1,dims=[(1,2,3),(0,1,2)]).cpu(),label='Nominal consumption')
        plt.plot(self.Td.cpu(),G.cpu(),label="Optimized consumption")
        plt.plot(self.Td.cpu(),self.r.cpu(),label="Constraint")
        plt.xlabel("Time")
        plt.ylabel("Aggregated consumption")
        plt.grid()
        plt.legend()
        plt.show()

    def plot(self):
        if self.mode == "gradient control":
            plt.plot(torch.arange(self.Nt).cpu(),self.lbda[0].cpu())
            plt.plot(torch.arange(self.Nt).cpu(),self.lbda[1].cpu())
            plt.plot(torch.arange(self.Nt).cpu(),self.lbda[2].cpu())
            plt.show()
        elif self.mode == "tracking":
            plt.plot(torch.arange(self.Nt).cpu(),self.lbda.cpu())
            plt.show()
    
    def plot_explaination(self):
        # plt.imshow(self.mu2[3].cpu(),cmap="Blues",vmin=0,vmax=0.009,origin="lower",extent=[9,17,0,1], aspect='auto')
        # plt.colorbar()
        # plt.xlabel("Time (in h)")
        # plt.ylabel("State of charge at the arrival")
        # plt.show()
        
        mulG=self.MuLambda()
        plt.imshow(mulG[6].cpu(),cmap="Blues",origin="lower",extent=[9,17,0,1], aspect='auto')
        plt.colorbar()
        plt.xlabel("Time (in h)")
        plt.ylabel("State of charge at the arrival")
        plt.show()

class CoupledLambda(LbdaUpdate):
    def __init__(self, cfg, r, rnom, mode, online_coef=0, lbda0=None, forward_max_grad=0.8, backward_max_grad=0.8):
        super().__init__(cfg, r, rnom, mode, online_coef, lbda0, forward_max_grad, backward_max_grad)
        self.EVlbda = EVLbdaUpdate(cfg,r[54:102],rnom,mode,online_coef, lbda0, forward_max_grad, backward_max_grad)
        self.G = torch.clone(self.rnom[:])
        self.GL = torch.tensordot(self.EVlbda.fT,self.EVlbda.mu1,dims=[(1,2,3),(0,1,2)])
        self.nominal = 0.5*torch.clone(self.rnom[:,0,0])
        self.nominal[54:102] += 0.5*torch.tensordot(self.EVlbda.fT,self.EVlbda.mu1,dims=[(1,2,3),(0,1,2)])
        
    def update(self,N,M,y,rangeR=None):
        self.G = super().total_consumption(y,rangeR=rangeR)/N
        self.GL = self.EVlbda.gradJL()
        total = torch.clone(self.G)*0.5
        total[54:102] += self.GL[:,None,None]*0.5
        return total,self.update_both_lambda(total,rangeR=rangeR)
    
    def update_both_lambda(self, s, rangeR=None):
        score = self.update_lbda(s,rangeR)
        #score2 = self.EVlbda.update_lbda(s,rangeR)
        if self.mode == "gradient control":
            self.EVlbda.lbda = self.lbda[:,54:102,0,0]
        elif self.mode == "tracking":
            self.EVlbda.lbda = self.lbda[54:102,0,0]
        return score
    
    def plotResult(self, rangeR=None):
        Td=torch.linspace(0,24,self.cfg.Nt)
        plt.plot(Td.cpu(),self.nominal.cpu(),label='Nominal consumption')
        plt.plot(Td.cpu(),self.r[:,0,0].cpu(),label="Reference")
        plt.plot(Td.cpu(),0.5*self.G[:,0,0].cpu(),label="WH consumption")
        plt.plot(Td[54:102].cpu(),0.5*self.GL.cpu(),label="EV consumption")
        total = torch.clone(self.G[:,0,0])*0.5
        total[54:102] += self.GL*0.5
        plt.plot(Td.cpu(),total.cpu(),label="Total consumption")
        plt.xlabel("Time")
        plt.ylabel("Aggregated consumption")
        plt.grid()
        plt.legend()
        plt.show()