import math

class Config:
    vol: float
    height: float
    EI: float
    powerRes: int
    rho: int
    capWater: int
    CI: float
    
    T_floor: float
    T_amb: float
    Tmax: float
    Tmin: float
    
    TEnd: int
    dt: int
    def __init__(self, configuration):
        for k, v in configuration.items():
            setattr(self, k, v)
        
        self.Nt = int(self.TEnd/self.dt)
        self.Nt2 = int(self.TEnd2/self.dt2)
        self.EI4 = self.EI/4
        self.Pmax = 60*self.powerRes    # Energy injected during one timestep (in minutes)
        self.Sect = self.vol/self.height
        self.ray = math.sqrt(self.Sect/3.14)    # Radius in m
        self.coefLoss = self.CI/self.EI4 * 2 * 3.14 * self.ray   # loss coeff in W/(K m)
        self.lossH = (self.coefLoss*60)/(self.capWater * self.rho * self.Sect)  # Fraction of heat loss by minutes
        self.e_unit = self.vol * self.rho * self.capWater               #J.K-1