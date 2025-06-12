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
    parser.add_argument('--n', type=int, default=300)
    parser.add_argument('--connectivity', type=float, default=0.03)
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
    x0 = np.ones(2)*2
    
    def equ(x,rh):
        g = np.zeros(2)
        g[0] = rh*x[0] - x[1] - x[0]*(x[0]**2+x[1]**2)
        g[1] = rh*x[1] + x[0] - x[1]*(x[0]**2+x[1]**2)
        return g
    
    L = np.zeros((len(rho),2))
    L[0] = x0
    for i in range(len(rho)-1):
        #noise_add = sigma*L[i]*np.random.normal(0,1,2)
        noise_add = sigma*np.random.normal(0,1,2)
        L[i+1] = L[i] + equ(L[i],rho[i])*dt + noise_add
        
    return(L)

dt = 0.01
tpoints = 20000
ts = np.arange(tpoints)*dt
F_bifurcation = np.linspace(2,-0.3,tpoints)
sigma = 0.01
ts_hopf = gen_data(F_bifurcation, dt, sigma)
X = ts_hopf
print("Data successfully generated!")


################################################################
###  (3) Calculate the RC EWS                                ###
################################################################
window = 2000
step = 300

iscontinuous = True
 
index = 'max_floquet' # select from ['max_eigenvalue','max_floquet','max_lyapunov']
num = 1 
inds_rcdi = []
for i in range(num):
    RCDyM = RC_EWM(X, ts, window, step, args, iscontinuous)
    ews, tm = RCDyM.calculate(index)
    inds_rcdi.append(ews)

################################################################
###  (4) Draw                                                ###
################################################################
fig = plt.figure(figsize=(20,10))
ls = 25; ms=15
font1 = {'weight':'normal','size':ls}
ax1 = fig.add_subplot(2,1,1)
ax1.plot(ts/dt,X[:,0],'k',label='data')
ax1.tick_params(labelsize=ls)
ax1.tick_params(axis='y', colors='k')
ax1.set_ylabel(r"$s$",size=ls,color='k')
ax1.legend(prop=font1)

ax12 = ax1.twinx()
for i in range(len(inds_rcdi)):
    max_evals = inds_rcdi[i]
    ax12.plot(tm/dt,np.sqrt(max_evals[:,0]**2 + max_evals[:,1]**2),'gx',markersize=ms,label='|MFM|')
    #ax12.plot(tm/dt,np.sqrt(max_evals[:,0]),'gx',markersize=ms,label='|MFM|')
ax12.tick_params(labelsize=ls)
ax12.tick_params(axis='y', colors='red')
ax12.set_ylabel("RCDyM",size=ls,color='red')
ax12.legend(prop=font1)
plt.savefig("results/hopf_pf.png")

sf = 5
js = np.linspace(window+args.warm_up,ts[-1]/dt,sf) 
for i in range(sf):
    j = int(js[i])
    ax = fig.add_subplot(2,sf,i+1+sf)
    ax.plot(X[j-window:j,0],X[j-window:j,1],'k-',linewidth=2.0, \
            label='t='+str(j)+'\n p='+str(round(F_bifurcation[j],2)))
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.legend(prop=font1)
    ax.tick_params(labelsize=ls)
    ax.set_xlabel(r"$s_1$",size=ls)
    if i>0:
        ax.set_yticklabels([])
    if i==0:
        ax.set_ylabel(r"$s_2$",size=ls)
        
# save
X_pd = pd.DataFrame(X)
index_rcdi_pd = pd.DataFrame(inds_rcdi[0])
ts_pd = pd.DataFrame(ts)
tm_pd = pd.DataFrame(tm)
X_pd.to_csv('results/hopf_data.csv')
index_rcdi_pd.to_csv('results/hopf_rcdi.csv')
ts_pd.to_csv('results/hopf_ts.csv')
tm_pd.to_csv('results/hopf_tm.csv')




