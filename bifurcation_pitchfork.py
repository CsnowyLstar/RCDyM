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
from utils.RC_EWS import RC_EWM

################################################################
###  (1) Hyperparameter setting                              ###
################################################################
def args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--connectivity', type=float, default=0.05)
    parser.add_argument('--spectral_radius', type=float, default=0.85)
    parser.add_argument('--input_scaling', type=float, default=1.0)
    parser.add_argument('--leak', type=float, default=0.0)
    parser.add_argument('--b', type=float, default=3.0)
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
def gen_data(dt=0.1, F=1, t_start=1, t_end=10000, x0=-1, sigma=0, noise_on=0):
    if len(F)==1:
        F = np.repeat(F,t_end)
    
    def diff(ut, x):
        out = 0.5 + ut*x -x**3
        return(out)
    
    L = np.zeros(t_end+1-t_start)
    
    #h = 0.001
    h = dt
    nlp = int(dt//h)
    L[0] = x0 
    for i in range(L.shape[0]-1):
        for j in range(nlp):
            x0 = x0 + diff(F[i], L[i])*h
        L_noise_add = sigma*L[i]*np.random.normal() if noise_on!=0 else 0
        x0 = x0 + L_noise_add
        L[i+1] = x0
        
    return(L)

dt = 0.1
tpoints = 20000
ts = np.arange(tpoints)*dt
#F_bifurcation = np.linspace(2.0,1.10,tpoints)
F_bifurcation = np.linspace(1.65,1.15,tpoints)
sigma = 0.01
ts_pitchfork = gen_data(F=F_bifurcation, sigma=sigma, noise_on=1, t_end=tpoints)
X = ts_pitchfork[:,None]
print("Data successfully generated!")
X_ori = X.copy()

################################################################
###  (3) Calculate the RC EWS                                ###
################################################################
window = 1000
step = 300
index = 'max_eigenvalue' # select from ['max_eigenvalue','max_floquet','max_lyapunov']
continuous = True
RCDyM = RC_EWM(X, ts, window, step, args, continuous)
max_evals, tm = RCDyM.calculate(index)

# Calculate ground truth
def true_jac(s,p):
    Jx = p-3*s**2
    return(Jx)

jxs = np.zeros_like(tm)
for i in range(len(tm)):
    j = int(tm[i]/dt-0.5*window)
    jxs[i] = true_jac(X_ori[j,0],F_bifurcation[j])

################################################################
###  (4) Draw and save                                       ###
################################################################
dele = 4
degree = 3
tm_ind = tm[:-dele]/dt-0.5*window
coefficients = np.polyfit(tm_ind, max_evals[:-dele,0], degree)
polynomial = np.poly1d(coefficients)
tm_fine = np.linspace(min(tm_ind), max(tm_ind), 500)
max_evals_fitted = polynomial(tm_fine)

fig = plt.figure(figsize=(21,20))
ls = 60
ax1 = fig.add_subplot(2,1,1)
ax1.plot(ts/dt,X_ori[:,0])
ax1.tick_params(labelsize=ls)
ax1.tick_params(axis='y', colors='blue')
ax1.set_ylabel(r"$s$",size=ls,color='blue')

ax12 = ax1.twinx()
ax12.plot(tm[:-dele]/dt-0.5*window,jxs[:-dele],'ko',markersize=8)
ax12.plot(tm_ind,max_evals[:-dele,0],'rx',markersize=15)
ax12.plot(tm_fine,max_evals_fitted,'r-',linewidth=8,alpha=0.6)
#plt.ylim(0.96,1.0)
ax12.tick_params(labelsize=ls)
ax12.tick_params(axis='y', colors='red')
ax12.set_ylabel("RCDyM (GT)",size=ls,color='red')

ax3 = fig.add_subplot(2,1,2)
ax3.plot(ts,F_bifurcation,'k-',linewidth=2.0)
ax3.tick_params(labelsize=ls)
ax3.set_xlabel(r"$t$",size=ls)
ax3.set_ylabel(r"$p$",size=ls)

plt.savefig("results/bifurcation_pitchfork.png")

# save
X_pd = pd.DataFrame(X)
index_pd = pd.DataFrame(max_evals)
jxs_pd = pd.DataFrame(jxs)
ts_pd = pd.DataFrame(ts)
tm_pd = pd.DataFrame(tm)
ve = 'c' if continuous else 'd'
X_pd.to_csv('results/pitchfork_data'+ve+'.csv')
index_pd.to_csv('results/pitchfork_index'+ve+'.csv')
jxs_pd.to_csv('results/pitchfork_jxs'+ve+'.csv')
ts_pd.to_csv('results/pitchfork_ts'+ve+'.csv')
tm_pd.to_csv('results/pitchfork_tm'+ve+'.csv')
