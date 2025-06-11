import numpy as np
import random as random
import torch
import torch.nn as nn 
import pandas as pd
import torchdiffeq as ode 
import matplotlib.pyplot as plt
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.linear_model import Ridge
import joblib 
from scipy import sparse
from utils.cubic_spline import CSpline
import argparse
from scipy.optimize import fsolve
from scipy.linalg import inv
from scipy.linalg import qr
from utils.reservoir_model_RC import Reservoir

seed = 0
np.random.seed(seed)
random.seed(seed)

class Sepi(nn.Module):  # myModel
    def __init__(self, Win, A, leak, b, gamma):
        super(Sepi, self).__init__()
        self.csp = None
        self.Win = Win
        self.A = A
        self.leak = leak
        self.b = b
        self.gamma = gamma
        
    def forward(self, t, r):
        leak = self.leak; Win = self.Win; A = self.A; b = self.b; csp=self.csp; gamma=self.gamma
        p = torch.cat((csp[0].fit(t),csp[1].fit(t),csp[2].fit(t)),axis=0)
        guy = (leak-1)/gamma * r + (1-leak)/gamma * torch.tanh(torch.mm(Win,p)+torch.mm(A,r)+b)
        return(guy)
    
class Pred(nn.Module):  # myModel
    def __init__(self, Win, A, leak, b, gamma):
        super(Pred, self).__init__()
        self.readout = None
        self.Win = Win
        self.A = A
        self.leak = leak
        self.b = b
        self.gamma = gamma
        
    def forward(self, t, r):
        leak = self.leak; Win = self.Win; A = self.A; b = self.b; readout = self.readout; gamma=self.gamma
        x = torch.tensor(readout.predict(r.t())).t()
        guy = (leak-1)/gamma * r + (1-leak)/gamma * torch.tanh(torch.mm(Win,x)+torch.mm(A,r)+b)
        return(guy)
    
def initialize_internal_weights(n_internal_units, connectivity, spectral_radius):
    
    print("initialize_internal_weights")
    # Generate sparse, uniformly distributed weights.
    internal_weights = sparse.rand(n_internal_units,
                                   n_internal_units,
                                   density=connectivity).todense()

    # Ensure that the nonzero values are uniformly distributed in [-0.5, 0.5]
    internal_weights[np.where(internal_weights > 0)] -= 0.5
    
    # Adjust the spectral radius.
    E, _ = np.linalg.eig(internal_weights)
    e_max = np.max(np.abs(E))
    internal_weights /= np.abs(e_max)/spectral_radius       
    
    return internal_weights

def equ(x, rh):
    g = np.zeros(3)
    g[0] = 10*(x[1]-x[0])
    g[1] = x[0]*(rh-x[2])-x[1]
    g[2] = x[0]*x[1]-8/3*x[2]
    return g

def disLorenz(rho, dt, sigma):
    x0 = np.ones(3)*7 
    #h = 0.001
    h = dt
    nlp = int(dt//h)
    L = np.zeros((len(rho),3))
    L[0] = x0
    for i in range(len(rho)-1):
        for j in range(nlp):
            x0 = equ(x0,rho[i])*h + x0
        noise_add = sigma*x0*np.random.normal(0,1,3)
        x0 = x0 + noise_add
        L[i+1] = x0
        
    return(L)


dt = 0.02
tpoints = 20000
ts = torch.arange(tpoints) * dt
# bifurcation rh 24.74
#rho = 10 + 20* torch.exp((ts-ts[-1])/10)
rho = 10*torch.ones(tpoints)
#rho = np.linspace(50,20,tpoints)
method = "euler"
sigma = 0.001

ts_lorenz = disLorenz(rho, dt, sigma)
X = ts_lorenz

print("Data successfully generated!")


# ================ RC 超参数设置 ====================
n_internal_units = 200
connectivity = 0.05
spectral_radius = 0.85
V = 3
input_scaling = 0.1
leak = 0.0
b = 1.0

internal_weights = initialize_internal_weights(n_internal_units, connectivity, spectral_radius)
A = torch.tensor(internal_weights)
Win = torch.tensor((2.0*np.random.random(size=(n_internal_units, V)) - 1.0)*input_scaling)

sepi = Sepi(Win, A, leak, b, dt)
forc = Pred(Win, A, leak, b, dt)


# ================== RC 多步预测 =====================
csp = []
for i in range(V):
    csp.append(CSpline(ts, X[:,i]))
sepi.csp = csp
n = n_internal_units
warm_up = 100
r0 = torch.rand(n,1)
ri = ode.odeint(sepi, r0, ts, rtol=1e-6, atol=1e-8, method=method)
#ri = ode.odeint(sepi, r0, ts, rtol=1e-6, atol=1e-8, method=method, options={'step_size': dt})
readout = Ridge(alpha=1e-5)

def system(r, tilde_A, tilde_B):
    return (leak-1)/dt * r + (1-leak)/dt * np.tanh(np.dot(tilde_A,r) + tilde_B[:,0])

def jac(r, tilde_A, tilde_B):
    diag = np.diag((1 - np.tanh(np.dot(tilde_A,r[:,None]) + tilde_B)**2)[:,0])
    #diag = np.diag((1 - r**2))
    return (leak-1)/dt * np.eye(len(r)) + (1-leak)/dt * np.dot(diag,tilde_A)

ds = [200,500,1000,2000,3000,4000]
num = 50
DEJs = np.zeros((len(ds), num))
for dsi in range(len(ds)):
    d = ds[dsi]
    for mlei in range(num):
        inir = np.random.rand()*(1-d/tpoints)
        # train
        nst = int(X.shape[0]*inir)
        ned = int(X.shape[0]*(inir+d/tpoints))
        X_b = ri[nst:ned,:,0] 
        Y_b = X[nst:ned,:]
        readout.fit(X_b,Y_b)
        
        # Calculate the domainent eigenvalue
        Wout = readout.coef_
        bias = readout.intercept_[:,None]
        tilde_A = A + np.dot(Win,Wout)
        tilde_B = np.dot(Win,bias) + b
        
        r_ini = X_b[-1]
        r_eq = fsolve(system, r_ini, fprime=jac, args=(tilde_A, tilde_B))
        #r_star = rt[-1][:,None]
        Jr = torch.tensor(jac(r_eq, tilde_A, tilde_B))
        evals_predr, evecs_predr = torch.linalg.eig(Jr)
        ind = torch.argmax(evals_predr.real)
        val = evals_predr[ind]; vec = evecs_predr[:,ind] 
        pred_xeq = readout.predict(r_eq[None,:])
        
        print('d =', d, mlei, "Domainent engenvalue:", val.real)
        DEJs[dsi,mlei] = val.real

# Calculate the groundtruth
rh = rho[0]
true_xeq = torch.tensor([np.sqrt((rh-1)*8/3),np.sqrt((rh-1)*8/3),rh-1])
#true_xeq = torch.tensor([0,0,0])
Jx = torch.tensor([[-10,10,0],[rh-true_xeq[2],-1,-1*true_xeq[0]],[true_xeq[1],true_xeq[0],-8/3]])
evals_x, evecs_x = torch.linalg.eig(Jx)
ind = torch.argmax(evals_x.real)
val = evals_x[ind]
print("lead_x", val)

# 开始画箱线图
plt.figure(figsize=(12, 6))  # 设置图像大小
plt.boxplot(DEJs.T, labels=ds)  # 创建箱线图
plt.ylim(-3,2)

# 设置图表标题和坐标轴标签
plt.xlabel("Window length")
plt.ylabel("MLE")

# 显示图像
plt.show()

DEJs_pd = pd.DataFrame(DEJs)
DEJs_pd.to_csv('results/roubust_lorenz_d_DEJ.csv')
plt.savefig("results/roubust_lorenz_d_DEJ.pdf")





