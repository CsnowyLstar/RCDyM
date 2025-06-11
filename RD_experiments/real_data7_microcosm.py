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
    parser.add_argument('--b', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1e-5)
    parser.add_argument('--warm_up', type=int, default=500)
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
def del_nan(X,ts):
    Xn = []
    tn = []
    for t in range(X.shape[0]):
        if X[t,0] < 10000:
            Xn.append(X[t,0])
            tn.append(ts[t])
    Xn = np.array(Xn)[:,None]
    tn = np.array(tn)
    return(Xn,tn)
        

# read data
#data = np.loadtxt('real_data/microcosm.txt', delimiter='\t')
data = pd.read_csv('real_data/microcosm.txt', delimiter='\t').values

ts = data[:,0]
X_ini = data[:,1][:,None]
X_ini,ts = del_nan(X_ini,ts)
dt = (ts[-1]-ts[0])/(len(ts)-1)
tpoints = len(ts)

X = X_ini.copy()
trp = [[431,480],[1502,1506],[3150,3190],[4250,4310],[5400,5440],[6800,6850]]

tp = 25

################################################################
###  (3) Calculate the RC EWS                                ###
################################################################
window = 200
step = 20
index = 'max_eigenvalue' # select from ['max_eigenvalue','max_floquet','max_lyapunov']

num = 1
ns = [90,100,110]
DEJ_cons = []; tm_cons = [] 
DEJ_diss = []; tm_diss = []
for ni in range(num):
    args.n = ns[ni]
    for trpi in range(len(trp) + 1):
        if trpi == 0:
            Xseg = X[:trp[trpi][0],:]
            tseg = ts[:trp[trpi][0]]
        elif trpi == len(trp):
            Xseg = X[trp[trpi-1][1]:,:]
            tseg = ts[trp[trpi-1][1]:]
        else:
            Xseg = X[trp[trpi-1][1]:trp[trpi][0],:]
            tseg = ts[trp[trpi-1][1]:trp[trpi][0]]
            
        iscontinuous = True
        isdetrend = False
        #RCDI_con = RC_EWM(X, ts, window, step, args, iscontinuous, isdetrend)
        RCDI_con = RC_EWM(Xseg, tseg, window, step, args, iscontinuous, isdetrend)
        DEJ_coni, tm_coni = RCDI_con.calculate(index)
            
        iscontinuous = False
        isdetrend = False
        #RCDI_dis = RC_EWM(X, ts, window, step, args, iscontinuous, isdetrend)
        RCDI_dis = RC_EWM(Xseg, tseg, window, step, args, iscontinuous, isdetrend)
        DEJ_disi, tm_disi = RCDI_dis.calculate(index)
        
        if trpi == 0:
            DEJ_con = DEJ_coni; tm_con = tm_coni 
            DEJ_dis = DEJ_disi; tm_dis = tm_disi 
        else:
            DEJ_con = np.concatenate((DEJ_con,DEJ_coni),axis=0)
            DEJ_dis = np.concatenate((DEJ_dis,DEJ_disi),axis=0)
            tm_con = np.concatenate((tm_con,tm_coni),axis=0)
            tm_dis = np.concatenate((tm_dis,tm_disi),axis=0)
    
    DEJ_cons.append(DEJ_con); tm_cons.append(tm_con)
    DEJ_diss.append(DEJ_dis); tm_diss.append(tm_dis)

################################################################
###  (5) Draw and save                                       ###
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
ax2.set_ylabel("RCDI_con",size=ls,color='red')
ax2.set_xticks([])
ax2.set_xlim(ts[0],ts[-1])
#ax2.set_ylim(bottom=-80,top=None)

ax3 = fig.add_subplot(3,1,3)
ind = 0
for ni in range(num):
    tm_dis = tm_diss[ni]; DEJ_dis = DEJ_diss[ni]
    i = next((i for i in range(len(tm_dis)) if tm_dis[i] > tp), len(tm_con)-1)
    ax3.plot(tm_dis[:i],DEJ_dis[:i,0],'m*',markersize=ms)
    if ind == 0:
        xs = tm_dis[:i]; ys=DEJ_dis[:i,0]; ind = 1
    else:
        xs = np.concatenate((xs,tm_dis[:i]),axis=0)
        ys = np.concatenate((ys,DEJ_dis[:i,0]),axis=0)
coefficients = np.polyfit(xs, ys, 1)
poly_func = np.poly1d(coefficients)
ax3.plot([tm_dis[0],tm_dis[i]],[poly_func(tm_dis[0]),poly_func(tm_dis[i])],'m-',linewidth=5)
ax3.tick_params(labelsize=ls)
ax3.tick_params(axis='y', colors='m')
ax3.set_ylabel("RCDI_dis",size=ls,color='m')
ax3.set_xlim(ts[0],ts[-1])
ax3.set_xlabel('Time',size=ls)

moln = 'real_data7'
X_pd = pd.DataFrame(X)
index_pd = pd.DataFrame(np.sqrt(DEJ_dis[:,0]**2 + DEJ_dis[:,1]**2))
ts_pd = pd.DataFrame(ts)
tm_pd = pd.DataFrame(tm_dis)
X_pd.to_csv('results/X_'+moln+'_RCDI_.csv')
index_pd.to_csv('results/index_'+moln+'_RCDI_.csv')
ts_pd.to_csv('results/ts_'+moln+'_RCDI_.csv')
tm_pd.to_csv('results/tm_'+moln+'_RCDI_.csv')
