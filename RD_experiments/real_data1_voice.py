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
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--connectivity', type=float, default=0.05)
    parser.add_argument('--spectral_radius', type=float, default=0.95)
    parser.add_argument('--input_scaling', type=float, default=0.1)
    parser.add_argument('--leak', type=float, default=0.0)
    parser.add_argument('--b', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1e-5)
    parser.add_argument('--warm_up', type=int, default=800)
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
data = np.loadtxt('real_data/voice.txt', delimiter='\t')
ts = data[:,0]-data[0,0]
X = data[:,1][:,None]
dt = ts[-1]/(len(ts)-1)
tpoints = len(ts)
tp = 0.23

print("X.shape:",X.shape)


################################################################
###  (3) Calculate the RC EWS                                ###
################################################################
window = 1000
step = 100
index = 'max_eigenvalue' # select from ['max_eigenvalue','max_floquet','max_lyapunov']

num = 1
ns = [400]
DEJ_cons = []; tm_cons = [] 
DEJ_diss = []; tm_diss = []
for ni in range(num):
    args.n = ns[ni]
    iscontinuous = False
    RCDyM_dis = RC_EWM(X, ts, window, step, args, iscontinuous)
    DEJ_dis, tm_dis = RCDyM_dis.calculate(index)
    DEJ_diss.append(DEJ_dis); tm_diss.append(tm_dis)


################################################################
###  (4) Draw and save                                       ###
################################################################
fig = plt.figure(figsize=(20,18))
ls = 40; ms = 15
ax1 = fig.add_subplot(3,1,1)
ax1.plot(ts,X,'b')
ax1.tick_params(labelsize=ls)
ax1.tick_params(axis='y', colors='b')
ax1.set_ylabel(r"$s$",size=ls,color='b')
ax1.set_xticks([])
ax1.set_xlim(ts[0],ts[-1])

ax3 = fig.add_subplot(3,1,2)
ind = 0
isrealpart = False
for ni in range(num):
    tm_dis = tm_diss[ni]; DEJ_dis = DEJ_diss[ni]
    i = next((i for i in range(len(tm_dis)) if tm_dis[i] > tp), None)
    DEJ_di = DEJ_dis[:i,0] if isrealpart else np.sqrt(DEJ_dis[:i,0]**2 + DEJ_dis[:i,1]**2)
    ax3.plot(tm_dis[:i],DEJ_di,'m*',markersize=ms)
    if ind == 0:
        xs = tm_dis[:i]; ys=DEJ_di; ind = 1
    else:
        xs = np.concatenate((xs,tm_dis[:i]),axis=0)
        ys = np.concatenate((ys,DEJ_di),axis=0)
coefficients = np.polyfit(xs, ys, 1)
poly_func = np.poly1d(coefficients)
ax3.plot([tm_dis[0],tm_dis[i]],[poly_func(tm_dis[0]),poly_func(tm_dis[i])],'r-',linewidth=8,alpha=0.6)
ax3.tick_params(labelsize=ls)
ax3.tick_params(axis='y', colors='m')
ax3.set_ylabel("RCDyM_dis",size=ls,color='m')
ax3.set_xlim(ts[0],ts[-1])
ax3.set_xlabel('Time',size=ls)

moln = 'real_data1'
X_pd = pd.DataFrame(X)
index_pd = pd.DataFrame(np.sqrt(DEJ_dis[:,0]**2 + DEJ_dis[:,1]**2))
ts_pd = pd.DataFrame(ts)
tm_pd = pd.DataFrame(tm_dis)
X_pd.to_csv('results/X_'+moln+'_RCDyM_.csv')
index_pd.to_csv('results/index_'+moln+'_RCDyM_.csv')
ts_pd.to_csv('results/ts_'+moln+'_RCDyM_.csv')
tm_pd.to_csv('results/tm_'+moln+'_RCDyM_.csv')

plt.savefig("results/r1.png")
