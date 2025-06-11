import numpy as np
import random as random
import torch
import torch.nn as nn 
import pandas as pd
import torchdiffeq as ode
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import Ridge
import joblib 
import argparse
from scipy.linalg import eig
from utils.RC_EWS import RC_EWM

################################################################
###  (1) Hyperparameter setting                              ###
################################################################
def args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--connectivity', type=float, default=0.05)
    parser.add_argument('--spectral_radius', type=float, default=0.85)
    parser.add_argument('--input_scaling', type=float, default=0.1)
    parser.add_argument('--leak', type=float, default=0.0)
    parser.add_argument('--b', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1e-5)
    parser.add_argument('--warm_up', type=int, default=600)
    parser.add_argument('--method', type=str, default='euler') 
    args = parser.parse_args(args=[])
    return(args)

args = args()
seed = 0
np.random.seed(seed)
random.seed(seed)

################################################################
###  (2) Data generation                                     ###
################################################################
def gen_data(rho, dt, sigma):
    
    def equ(x,rh):
        g = torch.zeros(3)
        g[0] = 10*(x[1]-x[0])
        g[1] = x[0]*(rh-x[2])-x[1]
        g[2] = x[0]*x[1]-8/3*x[2]
        return g
    
    x0 = np.ones(3)*8 
    for j in range(500):
        x0 = equ(x0,rho[0])*dt + x0
            
    #h = 0.001
    h = dt
    nlp = int(dt//h)
    L = np.zeros((len(rho),3))
    L[0] = x0
    for i in range(len(rho)-1):
        for j in range(nlp):
            x0 = equ(x0,rho[i])*h + x0
        #noise_add = sigma*x0*np.random.normal(0,1,3)
        noise_add = sigma*np.random.normal(0,1,3)
        x0 = x0 + noise_add
        L[i+1] = x0
    
    return(L)

def get_lorenz_rk4(rho,dt,sigma):
    leng = len(rho)
    v = np.zeros((leng,3))
    v[0] = np.ones(3)*8 
    
    f = lambda v_i, rho: np.array([
        10 * (v_i[1] - v_i[0]),
        v_i[0] * (rho - v_i[2]) - v_i[1],
        v_i[0] * v_i[1] - 8/3 * v_i[2]
    ])
    
    for i in range(leng-1):
        vi = v[i,:]
        k1 = dt * f(vi, rho[i])
        k2 = dt * f(vi + 0.5*k1, rho[i])
        k3 = dt * f(vi + 0.5*k2, rho[i])
        k4 = dt * f(vi + k3, rho[i])
        noise = sigma*vi*np.random.normal(0,1,3)
        v[i+1,:] = vi + (k1/6 + k2/3 + k3/3 + k4/6) + noise
    
    return(v)

#dt = 0.01
dt = 0.01
tpoints = 20000
ts = np.arange(tpoints)*dt
#F_bifurcation = np.linspace(2,23,tpoints)
F_bifurcation = np.linspace(2,23,tpoints)
sigma = 0.001
ts_hopf = gen_data(F_bifurcation, dt, sigma)
X = ts_hopf
X_ori = X.copy()
print("ts.shape:",ts.shape,"X.shape:",X.shape)
print("Data successfully generated!")

# detrend
tp = 16000
ifdetrend = True
detrend_method = 1
vi = 2
fig = plt.figure(figsize=(12,5))
plt.plot(ts/dt,X_ori[:,vi],'b')
if ifdetrend and detrend_method==1:
    step_de = 100
    X_detrend = X_ori.copy()
    for i in range(step_de,len(ts)-step_de):
        for vi in range(X.shape[-1]):
            X_detrend[i,vi] = X_ori[i,vi] - np.mean(X_ori[i-step_de:i+step_de,vi])
    for i in range(step_de):
        for vi in range(X.shape[-1]):
            X_detrend[i,vi] = X_ori[i,vi] - np.mean(X_ori[0:step_de,vi])
    j = tp - step_de
    for i in range(j,len(ts)):
        for vi in range(X.shape[-1]):
            X_detrend[i,vi] = X_ori[i,vi] - np.mean(X_ori[j-step_de:j+step_de,vi])
    plt.plot(ts/dt,X_detrend[:,vi],'r')
    X = X_detrend
plt.scatter([tp], [-15], marker='^')


################################################################
###  (3) Calculate the RC EWS                                ###
################################################################
window = 2000
step = 500
index = 'max_eigenvalue' # select from ['max_eigenvalue','max_floquet','max_lyapunov']
continuous = False
isdetrend = False
RCDI = RC_EWM(X, ts, window, step, args, continuous, isdetrend)
max_evals, tm = RCDI.calculate(index)
max_evals = (max_evals - 1)/dt

# Calculate ground truth
def true_jac(s,p):
    Jx = torch.tensor([[-10,10,0],[p-s[2],-1,-1*s[0]],[s[1],s[0],-8/3]])
    return(Jx)

jxs = np.zeros_like(tm)
for i in range(len(tm)):
    j = int(tm[i]/dt)
    rh = F_bifurcation[j]
    true_xeq = torch.tensor([np.sqrt((rh-1)*8/3),np.sqrt((rh-1)*8/3),rh-1])
    Jx = true_jac(true_xeq,rh)
    evals_predm, evecs_predm = eig(Jx)
    mi = np.argmax(np.real(evals_predm))
    jxs[i] = evals_predm[mi].real

################################################################
###  (4) Draw and save                                       ###
################################################################
dele = 6
degree = 1
tm_ind = tm[:-dele]/dt-0.5*window
coefficients = np.polyfit(tm_ind, max_evals[:-dele,0], degree)
polynomial = np.poly1d(coefficients)
tm_fine = np.linspace(min(tm_ind), max(tm_ind), 500)
max_evals_fitted = polynomial(tm_fine)

fig = plt.figure(figsize=(21,20))
ls = 60
ax1 = fig.add_subplot(2,1,1)
ax1.plot(ts/dt,X_ori[:,:],alpha=0.5)
ax1.tick_params(labelsize=ls)
ax1.tick_params(axis='y', colors='blue')
ax1.set_ylabel(r"$s$",size=ls,color='blue')

ax12 = ax1.twinx()
ax12.plot(tm[:-dele]/dt-0.5*window,jxs[:-dele],'ko',markersize=8)
ax12.plot(tm_ind,max_evals[:-dele,0],'rx',markersize=15)
ax12.plot(tm_fine,max_evals_fitted,'r-',linewidth=8,alpha=0.6)
#plt.ylim(-1,0.2)
plt.ylim(top=0.2)
ax12.tick_params(labelsize=ls)
ax12.tick_params(axis='y', colors='red')
ax12.set_ylabel("RCDI (GT)",size=ls,color='red')

ax3 = fig.add_subplot(2,1,2)
ax3.plot(ts,F_bifurcation,'k-',linewidth=2.0)
ax3.tick_params(labelsize=ls)
ax3.set_xlabel(r"$t$",size=ls)
ax3.set_ylabel(r"$p$",size=ls)

plt.savefig("results/lorenz_fc.pdf")

# save
X_pd = pd.DataFrame(X)
index_pd = pd.DataFrame(max_evals)
jxs_pd = pd.DataFrame(jxs)
ts_pd = pd.DataFrame(ts)
tm_pd = pd.DataFrame(tm)
ve = 'c' if continuous else 'd'
X_pd.to_csv('results/lorenz_fc_data'+ve+'.csv')
index_pd.to_csv('results/lorenz_fc_index'+ve+'.csv')
jxs_pd.to_csv('results/lorenz_fc_jxs'+ve+'.csv')
ts_pd.to_csv('results/lorenz_fc_ts'+ve+'.csv')
tm_pd.to_csv('results/lorenz_fc_tm'+ve+'.csv')

