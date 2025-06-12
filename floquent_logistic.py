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
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--connectivity', type=float, default=0.03)
    parser.add_argument('--spectral_radius', type=float, default=0.85)
    parser.add_argument('--input_scaling', type=float, default=0.8)
    parser.add_argument('--leak', type=float, default=0.0)
    parser.add_argument('--b', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1e-5)
    parser.add_argument('--warm_up', type=int, default=1000)
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
    x0 = np.ones(1)*0.5
    
    def equ(x,rh):
        g = rh*x*(1-x)
        return g
    
    L = np.zeros((len(rho),1))
    L[0] = x0
    for i in range(len(rho)-1):
        #noise_add = sigma*L[i]*np.random.normal(0,1,1)
        noise_add = sigma*np.random.normal(0,1,1)
        L[i+1] = equ(L[i],rho[i]) + noise_add
        
    return(L)

dt = 0.01
tpoints = 50000
ts = np.arange(tpoints)*dt
F_bifurcation = np.linspace(3.44,3.57,tpoints)
sigma = 0.003
ts_hopf = gen_data(F_bifurcation, dt, sigma)
X = ts_hopf
print("Data successfully generated!")


################################################################
###  (3) Calculate the RC EWS                                ###
################################################################
window = 2000
step = 500
iscontinuous = False
isdetrend = False

num = 1; inds_MFM = []
for i in range(num):
    RCDyM = RC_EWM(X, ts, window, step, args, iscontinuous, isdetrend)
    
    MFM, tm = RCDyM.get_max_floquet_dis(Tmin=1, threshold=0.85)
    inds_MFM.append(MFM)

################################################################
###  (4) Draw                                                ###
################################################################
fig = plt.figure(figsize=(20,12))
ls = 25; ms=15
font1 = {'weight':'normal','size':ls}
ax1 = fig.add_subplot(2,1,1)
ax1.plot((ts/dt)[::1],X[::1,0],'bx',label='data',alpha=0.1)
ax1.tick_params(labelsize=ls)
ax1.tick_params(axis='y', colors='k')
ax1.set_ylabel(r"$s$",size=ls,color='k')
ax1.legend(prop=font1)

ax12 = ax1.twinx()
for i in range(len(inds_MFM)):
    max_evals = inds_MFM[i]
    #ind = max_evals[:,0]
    ind = np.sqrt(max_evals[:,0]**2 + max_evals[:,1]**2)
    ax12.plot(tm/dt,ind,'gx',markersize=ms,label='|MFM|')
ax12.set_ylim(0.0,1.2)
ax12.tick_params(labelsize=ls)
ax12.tick_params(axis='y', colors='red')
ax12.set_ylabel("RCDyM",size=ls,color='red')
ax12.legend(prop=font1)

ax2 = fig.add_subplot(2,1,2)
ax2.plot(ts/dt, F_bifurcation)
ax2.plot(6400,3.6,'r^',markersize=15)
ax2.plot(42000,3.6,'r^',markersize=15)
plt.savefig("results/logistic.png")

# save
X_pd = pd.DataFrame(X)
index_MFM_pd = pd.DataFrame(inds_MFM[0])
ts_pd = pd.DataFrame(ts)
tm_pd = pd.DataFrame(tm)
X_pd.to_csv('results/logistic_data.csv')
ts_pd.to_csv('results/logistic_ts.csv')
tm_pd.to_csv('results/logistic_tm.csv')

