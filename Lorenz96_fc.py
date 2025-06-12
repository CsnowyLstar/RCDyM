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

################################################################
###  (1) Hyperparameter setting                              ###
################################################################
def args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--connectivity', type=float, default=0.05)
    parser.add_argument('--spectral_radius', type=float, default=0.85)
    parser.add_argument('--input_scaling', type=float, default=0.1)
    parser.add_argument('--leak', type=float, default=0.0)
    parser.add_argument('--b', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1e-5)
    parser.add_argument('--warm_up', type=int, default=2000)
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
def gen_data(rho, dt, sigma, V):
    
    def equ(X, V, ff):
        f = np.zeros(X.shape)
        for Vj in range(V):
            f[Vj] = X[Vj-1]*(X[Vj+1-V]-X[Vj-2]) -X[Vj] + ff
        return(f)
    
    #x0 = np.ones(V)
    x0 = np.random.rand(V)
    h = 0.01
    #h = dt
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

dt = 0.02
tpoints = 20000
ts = np.arange(tpoints) * dt
#F_bifurcation = 10 + 20* torch.exp((ts-ts[-1])/10)
#F_bifurcation = 1.0*torch.ones(tpoints)
F_bifurcation = np.linspace(0.1,1.1,tpoints)
method = "euler"
sigma = 0.005

V = 20
ts_lorenz96 = gen_data(F_bifurcation, dt, sigma, V)
X = ts_lorenz96
print("Data successfully generated!")


################################################################
###  (3) Calculate the RC EWS                                ###
################################################################
window = 2000
step = 400
iscontinuous = True
isdetrend = True

index = 'max_eigenvalue' # select from ['max_eigenvalue','max_floquet','max_lyapunov']
num = 1; inds_DEJ = []
for i in range(num):
    RCDyM = RC_EWM(X, ts, window, step, args, iscontinuous, isdetrend)
    max_evals, tm = RCDyM.calculate(index)
    inds_DEJ.append(max_evals)
    
    
################################################################
###  (4) Draw                                                ###
################################################################
fig = plt.figure(figsize=(20,12))
ls = 25; ms=15
font1 = {'weight':'normal','size':ls}
ax1 = fig.add_subplot(2,1,1)
ax1.plot(ts/dt, X[:,:], alpha=0.5)
ax1.tick_params(labelsize=ls)
ax1.tick_params(axis='y', colors='k')
ax1.set_ylabel(r"$s$",size=ls,color='k')
ax1.legend(prop=font1)

ax12 = ax1.twinx()
#ax12.plot(tm/dt,jxs,'ko')
ed = -1; xs = tm[:ed]; ys = inds_DEJ[0][:ed,0]
for i in range(len(inds_DEJ)):
    max_evals = inds_DEJ[i]
    ax12.plot(tm[:ed]/dt,max_evals[:ed,0],'rx',markersize=ms,label='Re(DEJ)')
    if i>0:
        xs = np.concatenate((xs,tm[:ed]),axis=0)
        ys = np.concatenate((ys,max_evals[:ed,0]),axis=0)
ax12.plot([-200,tpoints+200],[0,0],color='gray', linewidth=6, linestyle='--')
coefficients = np.polyfit(xs, ys, 1)
poly_func = np.poly1d(coefficients)
ax12.plot([tm[0]/dt,tm[ed]/dt],[poly_func(tm[0]),poly_func(tm[ed])],'r-',linewidth=5)

ax12.tick_params(labelsize=ls)
ax12.tick_params(axis='y', colors='red')
ax12.set_ylabel("RCDyM",size=ls,color='red')
ax12.legend(prop=font1)
ax12.set_ylim(-0.75,0.1)
plt.savefig("results/lorenz96_fc.png")

sf = 5
js = np.linspace(window+args.warm_up,ts[-1]/dt,sf) 
for i in range(sf):
    j = int(js[i])
    #ax = fig.add_subplot(2,sf,i+1+sf)
    ax = fig.add_subplot(2,sf,i+1+sf, projection='3d')
    ax.plot(X[j-window:j,0],X[j-window:j,1],X[j-window:j,2],'k-',linewidth=2.0, \
            label='t='+str(j)+'\n p='+str(round(F_bifurcation[j],2)))
    #ax.set_xlim(-25,25)
    #ax.set_ylim(-25,25)
    ax.legend(prop=font1)
    ax.tick_params(labelsize=ls)
    ax.set_xlabel(r"$s_1$",size=ls)
    if i==0:
        ax.set_ylabel(r"$s_2$",size=ls)
plt.savefig("L96.pdf")