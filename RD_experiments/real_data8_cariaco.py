import numpy as np
import random as random
import torch
import torch.nn as nn 
import pandas as pd
import torchdiffeq as ode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy import sparse
from utils.cubic_spline import CSpline
import argparse
from scipy import signal
from utils.RC_EWS import RC_EWM
from utils.Baseline_EWS import BL_EWS

################################################################
###  (1) Hyperparameter setting                              ###
################################################################
def args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--connectivity', type=float, default=0.05)
    parser.add_argument('--spectral_radius', type=float, default=0.65)
    parser.add_argument('--input_scaling', type=float, default=0.1)
    parser.add_argument('--leak', type=float, default=0.0)
    parser.add_argument('--b', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=1e-2)
    parser.add_argument('--warm_up', type=int, default=200)
    parser.add_argument('--method', type=str, default='euler') 
    args = parser.parse_args(args=[])
    return(args)

args = args()
seed = 0
np.random.seed(seed)
random.seed(seed)


################################################################
###  (2) Read data                                           ###
################################################################
data = pd.read_csv('real_data/cariaco.txt', delimiter='\t').values[:6500]

ts = data[:,0]/1000
ts = ts - ts[0]
X = data[:,1][:,None]
dt = (ts[-1]-ts[0])/(len(ts)-1)
tpoints = len(ts)

################################################################
###  (3) Calculate the RC EWS                                ###
################################################################
window = 500
step = 40
index = 'max_eigenvalue' # select from ['max_eigenvalue','max_floquet','max_lyapunov']

num = 1
ns = np.arange(num)*3 + 60
DEJ_cons = []; tm_cons = [] 
for ni in range(num):
    args.n = ns[ni]
    iscontinuous = True
    isdetrend = False
    RCDyM_con = RC_EWM(X, ts, window, step, args, iscontinuous, isdetrend)
    DEJ_con, tm_con = RCDyM_con.calculate(index)
    DEJ_cons.append(DEJ_con); tm_cons.append(tm_con)


################################################################
###  (5) Draw and save                                       ###
################################################################
fig = plt.figure(figsize=(20,18))
ls = 40; ms = 15; tp = 10.5
ax1 = fig.add_subplot(3,1,1)
ax1.plot(ts,X,'b')
ax1.tick_params(labelsize=ls)
ax1.tick_params(axis='y', colors='b')
ax1.set_ylabel(r"$s$",size=ls,color='b')
ax1.set_xticks([])
ax1.set_xlim(ts[0],ts[-1])

ax2 = fig.add_subplot(3,1,2)
ind = 0
for ni in range(num):
    tm_con = tm_cons[ni]; DEJ_con = DEJ_cons[ni]
    i = next((i for i in range(len(tm_con)) if tm_con[i] > tp), len(tm_con)-1)
    ax2.plot(tm_con[:i],DEJ_con[:i,0],'r*',markersize=ms)
    if ind == 0:
        xs = tm_con[:i]; ys=DEJ_con[:i,0]; ind = 1
    else:
        xs = np.concatenate((xs,tm_con[:i]),axis=0)
        ys = np.concatenate((ys,DEJ_con[:i,0]),axis=0)
coefficients = np.polyfit(xs, ys, 1)
poly_func = np.poly1d(coefficients)
ax2.plot([tm_con[0],tm_con[i]],[poly_func(tm_con[0]),poly_func(tm_con[i])],'r-',linewidth=5)
ax2.tick_params(labelsize=ls)
ax2.tick_params(axis='y', colors='red')
ax2.set_ylabel("RCDyM_con",size=ls,color='red')
ax2.set_xticks([])
ax2.set_xlim(ts[0],ts[-1])
plt.savefig("results/r8.png")

moln = 'real_data8'
X_pd = pd.DataFrame(X)
index_pd = pd.DataFrame(np.sqrt(DEJ_con[:,0]**2 + DEJ_con[:,1]**2))
ts_pd = pd.DataFrame(ts)
tm_pd = pd.DataFrame(tm_con)
X_pd.to_csv('results/X_'+moln+'_RCDyM_.csv')
index_pd.to_csv('results/index_'+moln+'_RCDyM_.csv')
ts_pd.to_csv('results/ts_'+moln+'_RCDyM_.csv')
tm_pd.to_csv('results/tm_'+moln+'_RCDyM_.csv')