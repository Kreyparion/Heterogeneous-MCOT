import torch
import matplotlib.pyplot as plt

from src.core.configuration import Config
from src.core.utils import write_to_txt, choice_along_first_axis2

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
        DataTransaction=DataTransaction.drop(['TransactionId', 'ChargePoint', 'Connector', 'UTCTransactionStart','UTCTransactionStop', 'StartCard', 'ConnectedTime', 'ChargeTime','TotalEnergy','Day'],axis=1)
        # print(DataTransaction)
        DataTransaction["Start4"] -= 6
        DataTransaction = DataTransaction[(DataTransaction["ConnectedTime4"]<Nt) & (DataTransaction["ChargeTime4"]<Nt) & (DataTransaction["Start4"]<Nt) & (DataTransaction["Start4"]>=0) & (DataTransaction["Start4"]+DataTransaction["ConnectedTime4"]<Nt) ]

        WeekendData = DataTransaction[DataTransaction['WeekDay'].isin([5, 6])]
        WeekdayData = DataTransaction[~DataTransaction['WeekDay'].isin([5, 6])]

        
        
        self.rate=0.18
        TrainDay,TestDay=train_test_split(WeekdayData.index, test_size=self.rate, random_state=0)
        WeekdayTest = WeekdayData.loc[TestDay]
        WeekdayTrain = WeekdayData.loc[TrainDay]
        # WeekdayTest=WeekdayTest.drop(['Day'],axis=1)
        # WeekdayTrain=WeekdayTrain.drop(['Day'],axis=1)


        self.dtTrain=WeekdayTrain
        self.dtTest=WeekdayTest
 
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
            nu0[int(row[4]),int(row[2]),int(row[3])]+=1
        nu0=nu0/torch.sum(nu0)
        #nu0 = 0*nu0

        nuReal=torch.zeros((Nt,Nt,Nt))

        for row in self.dtTest.itertuples(index=False):
            if row[4] <Nt and row[2] < Nt and row[3]<Nt:
                nuReal[int(row[4]),int(row[2]),int(row[3])]+=1
        nuReal=nuReal/torch.sum(nuReal)
        self.nuReal = nuReal

        # muReal=torch.zeros((Nt,Nt,Nt,Nt))

        # for t in range(0,Nt):
        #     muReal[t,:,:,t]=nuReal[t]

        # self.Pred=self.ProdFM(fT[:,None],muReal)*len(self.dtTest)
        
        
        # muReal = torch.tensor([])


        mu1=torch.zeros((Nt,Nt,Nt,Nt))

        for t in range(0,Nt):
            mu1[t,:,:,t]=nu0[t]
        N=len(self.dtTrain)*self.rate/(1-self.rate)
        self.muFinal=torch.zeros((self.cfg.Nt2,self.cfg.Nt2,self.cfg.Nt2,self.cfg.Nt2))
        # self.nb_hist = 2.3
        # self.dur_hist = 25
        # for i in range(self.dur_hist):
        #     self.muFinal[0,i,i,0] =self.nb_hist
        #     mu1[0,i,i,0] += self.nb_hist/N

        self.Nom=self.ProdFM(fT[:,None],mu1)*N
        
        # from RealDataEV2_torch_same_p import PredictedConsumption, NominalConsumption, mu1 as m, fT as ffff
        # print(torch.all(self.Pred == PredictedConsumption))
        # print(torch.all(self.Nom == NominalConsumption))
        # print(torch.all(mu1 ==m))
        # print(torch.all(fT ==ffff))
        
        mu1 = torch.tensor([])
        self.fT = fT
        self.nu0 = nu0
        
        
        self.listX=torch.zeros((0,3), dtype=int)
        
        all_test_vehicles = torch.tensor(self.dtTest[["Start4","ConnectedTime4","ChargeTime4"]].to_numpy()).int()
        length = torch.arange(self.cfg.Nt2)
        
    
    
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
        
        GL = self.ProdFM(self.fT[:,None], self.muFinal)
        GL+=self.gradJLFromMu(rangeR, MuL)*(self.N1t+self.N2t)
        # plt.plot(self.Td.cpu(),GL.cpu())
        # plt.show()
        return GL,self.update_lbda(GL[rangeR],rangeR=rangeR)

    def update_lbda(self,s,rangeR=None):
        if rangeR is None:
            rangeR = torch.arange(self.Nt)
        self.onlineLbda-=2*self.cfg.lr*(s[0]-self.r[rangeR][0])
        if self.mode == "online":
            return torch.abs(s[0]-self.r[rangeR][0])
        if self.mode == "tracking":
            self.lbda[rangeR]-=2*self.cfg.lr*(s-self.r[rangeR])
            return torch.linalg.norm(s[0]-self.r[rangeR[0]])
        elif self.mode == "gradient control":
            self.lbda[0,rangeR]-=2*self.cfg.lr*(s-self.r[rangeR])
            self.lbda[1,rangeR[:-1]]-=2*self.cfg.lr*(s[1:]-s[:-1]-self.forward_max_grad)
            self.lbda[2,rangeR[:-1]]-=2*self.cfg.lr*(s[:-1]-s[1:]-self.backward_max_grad)
            self.onlineLbda = torch.minimum(self.onlineLbda,self.onlineLbda*0)
            self.lbda = torch.minimum(self.lbda,self.lbda*0)
            return torch.linalg.norm(torch.maximum(s-self.r[rangeR],s*0)) + torch.linalg.norm(torch.maximum(s[1:]-s[:-1]-self.forward_max_grad,s[1:]*0)) + torch.linalg.norm(torch.maximum(s[:-1]-s[1:]-self.backward_max_grad,s[:1]*0))

    
    # def update_lbda(self, s,rangeR=None):
    #     if rangeR is None:
    #         rangeR = torch.arange(self.Nt)
    #     if self.mode == "gradient control":
    #         self.lbda[0,rangeR]-=3*(s[rangeR]-self.r_updated[rangeR])
    #         self.lbda[1,rangeR[:-1]]-=3*(s[rangeR][1:]-s[rangeR][:-1]-self.forward_max_grad)
    #         self.lbda[2,rangeR[:-1]]-=3*(s[rangeR][:-1]-s[rangeR][1:]-self.backward_max_grad)
    #         self.lbda = torch.minimum(self.lbda, 0*self.lbda)
    #     elif self.mode == "tracking":
    #         self.lbda[rangeR]-=3*(s[rangeR]-self.r_updated[rangeR])
    #     return s, torch.linalg.norm(torch.maximum(s[rangeR]-self.r_updated[rangeR],rangeR*0))
    
    def InitT(self,tOn,listX,N):
        #tOn = rangeR[0]
        nu_0=torch.zeros((self.Nt,self.Nt,self.Nt))
        mu_2=torch.zeros((self.Nt,self.Nt,self.Nt,self.Nt-tOn))
        N1t=len(listX)
        N2t=torch.sum(self.nu0[tOn+1:])*N
        if N1t+N2t != 0:
            nu_0[0:tOn+1]=self.nu0[0:tOn+1]*0
            nu_0[tOn+1:]=self.nu0[tOn+1:]*N2t/(N1t+N2t)
            for k in range(len(listX)):
                nu_0[listX[k,0],listX[k,1],listX[k,2]] += 1/(N1t+N2t)

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
        plt.plot(self.Td.cpu(),self.Nom.cpu()*3,label='Nominal consumption')
        write_to_txt("EVs_only_nom", self.Td.cpu(),self.Nom.cpu()*3)
        # plt.plot(self.Td.cpu(),self.Pred.cpu(),label="Predicted consumption")
        plt.plot(self.Td.cpu(),G.cpu()*3,label="Optimized consumption")
        write_to_txt("EVs_only_cons", self.Td.cpu(),G.cpu()*3)
        plt.plot(self.Td[rangeR].cpu(),self.r[rangeR].cpu()*3,label="Constraint")
        write_to_txt("EVs_only_r", self.Td.cpu(),G.cpu()*3)
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
        # for i in range(self.dur_hist):
        #     self.muFinal[0,i,i,0] = self.nb_hist
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
            #print(tc)
            for k in range(len(chosen)):
                if chosen[k]:
                    self.muFinal[self.listX[k,0],self.listX[k,1],self.listX[k,2],t] += 1
            self.listX = self.listX[~chosen]
            
        #     tot_listX = torch.sum(self.listX[:,2])
        #     length = torch.arange(self.cfg.Nt2)
        #     tot_muFinal = torch.sum(self.muFinal*length[None,None,:,None])
        #     all_left_test_vehicles = torch.tensor(self.dtTest[self.dtTest['Start4']>t.item()][["Start4","ConnectedTime4","ChargeTime4"]].to_numpy()).int()

        #     print(torch.sum(all_left_test_vehicles[:,2]) + tot_muFinal+tot_listX)
        
        # else:
        #     length = torch.arange(self.cfg.Nt2)
        #     tot_muFinal = torch.sum(self.muFinal*length[None,None,:,None])
        #     all_left_test_vehicles = torch.tensor(self.dtTest[self.dtTest['Start4']>t.item()][["Start4","ConnectedTime4","ChargeTime4"]].to_numpy()).int()

        #     print(torch.sum(all_left_test_vehicles[:,2]) + tot_muFinal)
