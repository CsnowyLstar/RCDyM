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
    parser.add_argument('--n', type=int, default=800)
    parser.add_argument('--connectivity', type=float, default=0.001)
    parser.add_argument('--spectral_radius', type=float, default=0.6)
    parser.add_argument('--input_scaling', type=float, default=0.5)
    parser.add_argument('--leak', type=float, default=0.0)
    parser.add_argument('--b', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=1e-4)
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
#ds_num = [38,37,36,35,34,33]
#ds_num = [36, 35.5, 35, 34.5, 34, 33.5, 33]
#ds_num = [35.5, 35, 34.5, 34, 33.5]
ds_num = np.linspace(35.8,33.7,10)

dt = 0.25
h = 0.05
tp_num = 10000
tpoints = tp_num * len(ds_num)
T = int(dt*tpoints)
ts = np.arange(tpoints) * dt
deltas = np.ones(int(T/h))

F_bifurcation = deltas[::int(T/dt/tpoints)]
method = "euler"

ds = np.load('dataset/d_dat.npy')
X = np.load('dataset/ks_dat.npy')
N = 64
xs = ds[0]*np.transpose(np.conj(np.arange(-N/2+1, N/2+1))) / N

print("Data successfully generated!")

################################################################
###  (3) Calculate the RC EWS                                ###
################################################################
window = 10000
step = 3000
index = 'max_lyapunov' # select from ['max_eigenvalue','max_floquet','max_lyapunov']

iscontinuous = False
RCDyM = RC_EWM(X, ts, window, step, args, iscontinuous, ispde=True)
max_floquets, tm = RCDyM.calculate(index)

rcdym = max_floquets

################################################################
###  (4) Draw                                                ###
################################################################
fig = plt.figure(figsize=(20,12))
ls = 25; it = 1
ax1 = fig.add_subplot(2,1,1)
tt_mesh, xx_mesh = np.meshgrid(ts[::it]/dt, xs)
for i in range(len(ds_num)):
    xx_meshi = ds_num[i]*np.transpose(np.conj(np.arange(-N/2+1, N/2+1))) / N
    xx_mesh[:,i*tp_num:(i+1)*tp_num] = xx_meshi[:,None].repeat(tp_num,1)
true = X.transpose()
ax1.pcolormesh(tt_mesh, xx_mesh, true[:,::it], cmap='RdYlBu')
ax1.tick_params(labelsize=ls)
ax1.tick_params(axis='y', colors='blue')
ax1.set_ylabel(r"$s$",size=ls,color='blue')

ax12 = ax1.twinx()
ax12.plot(tm/dt,rcdym,'rx', markersize=20)
#plt.ylim(-0.0,1.1)
#plt.ylim(-0.01,0.1)
plt.ylim(0.03,0.1)
ax12.tick_params(labelsize=ls)
ax12.tick_params(axis='y', colors='red')
ax12.set_ylabel("RCDyM",size=ls,color='red')

ax2 = fig.add_subplot(2,1,2)
ax2.plot(true[it,:],'b-',linewidth=2.0)

ax22 = ax2.twinx()
ax22.plot(ds,'k-',linewidth=2.0)
ax22.tick_params(labelsize=ls)
ax22.set_xlabel(r"$t$",size=ls)
ax22.set_ylabel(r"$p$",size=ls)

plt.savefig("results/KS_cp.png")