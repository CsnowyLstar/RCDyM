import numpy as np
import random as random
import torch
import torch.nn as nn 
import pandas as pd
import torchdiffeq as ode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.linear_model import Ridge
import joblib 
import argparse
from scipy.linalg import eig
from utils.RC_EWS import RC_EWM
from scipy import signal
from scipy import sparse
from utils.cubic_spline import CSpline
from scipy.optimize import fsolve

################################################################
###  (1) Hyperparameter setting                              ###
################################################################
def args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--connectivity', type=float, default=0.05)
    parser.add_argument('--spectral_radius', type=float, default=0.85)
    parser.add_argument('--input_scaling', type=float, default=0.1)
    parser.add_argument('--leak', type=float, default=0.0)
    parser.add_argument('--b', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1e-5)
    parser.add_argument('--warm_up', type=int, default=200)
    parser.add_argument('--method', type=str, default='euler') 
    args = parser.parse_args(args=[])
    return(args)

args = args()
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
        p = csp[0].fit(t)
        for i in range(len(csp)-1):
            p = torch.cat((p,csp[i+1].fit(t)),axis=0)
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

def system(r, tilde_A, tilde_B, dt):
    return (args.leak-1)/dt * r + (1-args.leak)/dt * np.tanh(np.dot(tilde_A,r) + tilde_B[:,0])

def jac(r, tilde_A, tilde_B, dt):
    diag = np.diag((1 - np.tanh(np.dot(tilde_A,r[:,None]) + tilde_B)**2)[:,0])
    return (args.leak-1)/dt * np.eye(len(r)) + (1-args.leak)/dt * np.dot(diag,tilde_A)

################################################################
###  (2) Data generation                                     ###
################################################################
def gen_data(rho, dt, sigma, V):
    
    def equ(X, V, ff):
        f = np.zeros(X.shape)
        for Vj in range(V):
            f[Vj] = X[Vj-1]*(X[Vj+1-V]-X[Vj-2]) -X[Vj] + ff
        return(f)
    
    x0 = np.random.rand(V)
    #h = 0.01
    h = dt
    nlp = int(dt//h)
    L = np.zeros((len(rho),V))
    L[0] = x0
    for i in range(len(rho)-1):
        for j in range(nlp):
            x0 = equ(x0,V,rho[i])*h + x0
        noise_add = sigma*x0*np.random.normal(0,1,V)
        x0 = x0 + noise_add
        L[i+1] = x0
    
    return(L)

dts = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
num = 50
tra_T = 100
V = 5
DEJs = np.zeros((len(dts), num))

ddt = 0.005
ttpoints = 50000
tts = torch.arange(ttpoints) * ddt 
FF_bifurcation = 0.1*torch.ones(ttpoints)
method = "euler"
sigma = 0.005
tts_lorenz96 = gen_data(FF_bifurcation, ddt, sigma, V)
XX = tts_lorenz96

print("Data successfully generated!")
    
for dti in range(len(dts)):
    dt = dts[dti]
    d = int(tra_T/dt)
    inte = int(dt/ddt)
    ts = tts[::inte]
    F_bifurcation = FF_bifurcation[::inte] 
    X = XX[::inte]
    
    # ================ RC 超参数设置 ====================
    internal_weights = initialize_internal_weights(args.n, args.connectivity, args.spectral_radius)
    A = torch.tensor(internal_weights)
    Win = torch.tensor((2.0*np.random.random(size=(args.n, V)) - 1.0)*args.input_scaling)
    
    sepi = Sepi(Win, A, args.leak, args.b, dt)
    forc = Pred(Win, A, args.leak, args.b, dt)
    
    # ================== RC 多步预测 =====================
    csp = []
    for i in range(V):
        csp.append(CSpline(ts, X[:,i]))
    sepi.csp = csp
    n = args.n 
    r0 = torch.rand(n,1)
    ri = ode.odeint(sepi, r0, ts, rtol=1e-6, atol=1e-8, method=method)
    readout = Ridge(alpha=1e-5)
    
    for mlei in range(num):
        # train
        inir = int(np.random.rand()*(X.shape[0]-d-args.warm_up))+args.warm_up
        nst = inir
        ned = inir + d
        X_b = ri[nst:ned,:,0] 
        Y_b = X[nst:ned,:]
        readout.fit(X_b,Y_b)
        
        # Calculate the domainent eigenvalue
        Wout = readout.coef_
        bias = readout.intercept_[:,None]
        tilde_A = A + np.dot(Win,Wout)
        tilde_B = np.dot(Win,bias) + args.b
        
        r_ini = X_b[-1]
        r_eq = fsolve(system, r_ini, fprime=jac, args=(tilde_A, tilde_B, dt))
        #r_star = rt[-1][:,None]
        Jr = torch.tensor(jac(r_eq, tilde_A, tilde_B, dt))
        evals_predr, evecs_predr = torch.linalg.eig(Jr)
        ind = torch.argmax(evals_predr.real)
        val = evals_predr[ind]; vec = evecs_predr[:,ind] 
        pred_xeq = readout.predict(r_eq[None,:])
        
        print('dt =', dt, mlei, "Domainent engenvalue:", val.real.numpy())
        DEJs[dti,mlei] = val.real.numpy()
    print('Average value', np.median(DEJs[dti,:]))

'''
# Calculate the groundtruth
rh = F_bifurcation[0]
true_xeq = torch.tensor([np.sqrt((rh-1)*8/3),np.sqrt((rh-1)*8/3),rh-1])
#true_xeq = torch.tensor([0,0,0])
Jx = torch.tensor([[-10,10,0],[rh-true_xeq[2],-1,-1*true_xeq[0]],[true_xeq[1],true_xeq[0],-8/3]])
evals_x, evecs_x = torch.linalg.eig(Jx)
ind = torch.argmax(evals_x.real)
val = evals_x[ind]
print("lead_x", val)
'''

# 开始画箱线图
plt.figure(figsize=(12, 6))  # 设置图像大小
plt.boxplot(DEJs.T, labels=dts)  # 创建箱线图

# 设置图表标题和坐标轴标签
plt.xlabel("Window length")
plt.ylabel("DEJ")

# 显示图像
plt.show()

DEJs_pd = pd.DataFrame(DEJs)
DEJs_pd.to_csv('results/roubust_lorenz96_dt_DEJ.csv')
plt.savefig("results/roubust_lorenz96_dt_DEJ.pdf")
