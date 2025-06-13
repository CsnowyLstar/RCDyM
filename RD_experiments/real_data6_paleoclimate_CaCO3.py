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
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--connectivity', type=float, default=0.05)
    parser.add_argument('--spectral_radius', type=float, default=0.1)
    parser.add_argument('--input_scaling', type=float, default=0.1)
    parser.add_argument('--leak', type=float, default=0.0)
    parser.add_argument('--b', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--warm_up', type=int, default=100)
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
# Function to do linear interpolation on data prior to transition
def interpolate(df):
    df_prior = df[df['Age']>=df['Transition'].iloc[0]].copy()
    df_later = df[df['Age']<df['Transition'].iloc[0]].copy()
    t_inter_vals = np.linspace(df_prior['Age'].iloc[0], df_prior['Age'].iloc[-1], len(df_prior))
    df_inter = pd.DataFrame( {'Age':t_inter_vals,'Inter':True} )
    df2=pd.concat([df_prior,df_inter]).set_index('Age')
    df2=df2.interpolate(method='index')
    df_inter = df2[df2['Inter']==True][['Proxy','Transition']].reset_index()
    return df_inter, df_later

df = pd.read_csv('real_data/paleoclimate.csv')
list_records = df['Record'].unique()
dic_bandwidth = {'End of greenhouse Earth':25, 'End of Younger Dryas':100, 'End of glaciation I':25,
                 'Bolling-Allerod transition':25, 'End of glaciation II':25, 'End of glaciation III':10,
                 'End of glaciation IV':50, 'Desertification of N. Africa':10}

record = list_records[0]
df_select = df[df['Record']==record]
df_inter, df_later = interpolate(df_select)
# Series for computing EWS
series1 = df_inter['Proxy'].values
series2 = df_later['Proxy'].values

ts1 = df_inter['Age'].values
ts1 = -ts1
ts1 = (ts1 - ts1[0])/1e6

X = np.concatenate((series1,series2),axis=0)[:,None]

dt = ts1[-1]/(len(ts1)-1)
ts2 = np.arange(1,1+X.shape[0]-ts1.shape[0])*dt + ts1[-1]
ts = np.concatenate((ts1,ts2), axis=0)
tpoints = len(ts)

################################################################
###  (3) Calculate the RC EWS                                ###
################################################################
window = 30
step = 4
index = 'max_eigenvalue' # select from ['max_eigenvalue','max_floquet','max_lyapunov']

num = 1
ns = np.arange(num)*3 + 40
DEJ_cons = []; tm_cons = [] 
DEJ_diss = []; tm_diss = []
for ni in range(num):
    args.n = ns[ni]
    iscontinuous = False
    isdetrend = True
    RCDyM_dis = RC_EWM(X, ts, window, step, args, iscontinuous, isdetrend)
    DEJ_dis, tm_dis = RCDyM_dis.calculate(index)
    DEJ_diss.append(DEJ_dis); tm_diss.append(tm_dis)


################################################################
###  (5) Draw and save                                       ###
################################################################
fig = plt.figure(figsize=(20,18))
ls = 40; ms = 15; tp = 5.8
ax1 = fig.add_subplot(3,1,1)
ax1.plot(ts,X,'b')
ax1.tick_params(labelsize=ls)
ax1.tick_params(axis='y', colors='b')
ax1.set_ylabel(r"$s$",size=ls,color='b')
ax1.set_xticks([])
ax1.set_xlim(ts[0],ts[-1])

ax3 = fig.add_subplot(3,1,2)
ind = 0
for ni in range(num):
    tm_dis = tm_diss[ni]; DEJ_dis = DEJ_diss[ni]
    i = next((i for i in range(len(tm_dis)) if tm_dis[i] > tp), None)
    if ind == 0:
        xs = tm_dis[:i]; ys=DEJ_dis[:i,0]; ind = 1
    else:
        xs = np.concatenate((xs,tm_dis[:i]),axis=0)
        ys = np.concatenate((ys,DEJ_dis[:i,0]),axis=0)
'''        
coefficients = np.polyfit(xs, ys, 1)
poly_func = np.poly1d(coefficients)
ax3.plot([tm_dis[0],tm_dis[i]],[poly_func(tm_dis[0]),poly_func(tm_dis[i])],'m-',linewidth=5)
ax3.tick_params(labelsize=ls)
ax3.tick_params(axis='y', colors='m')
ax3.set_ylabel("RCDyM_dis",size=ls,color='m')
ax3.set_xlim(ts[0],ts[-1])
ax3.set_xlabel('Time',size=ls)
plt.savefig("results/r6.png")
'''
# Filter out points where ys > 1.1
mask = ys <= 1.02
filtered_xs = xs[mask]
filtered_ys = ys[mask]
ax3.plot(filtered_xs, filtered_ys,'m*',markersize=ms)
coefficients = np.polyfit(filtered_xs, filtered_ys, 1)
poly_func = np.poly1d(coefficients)
ax3.plot([tm_dis[0], tm_dis[i]], [poly_func(tm_dis[0]), poly_func(tm_dis[i])], 'm-', linewidth=5)
ax3.tick_params(labelsize=ls)
ax3.tick_params(axis='y', colors='m')
ax3.set_ylabel("RCDyM_dis", size=ls, color='m')
ax3.set_xlim(ts[0], ts[-1])
ax3.set_xlabel('Time', size=ls)
plt.savefig("results/r6.png")

moln = 'real_data6'
X_pd = pd.DataFrame(X)
index_pd = pd.DataFrame(np.sqrt(DEJ_dis[:,0]**2 + DEJ_dis[:,1]**2))
ts_pd = pd.DataFrame(ts)
tm_pd = pd.DataFrame(tm_dis)
X_pd.to_csv('results/X_'+moln+'_RCDyM_.csv')
index_pd.to_csv('results/index_'+moln+'_RCDyM_.csv')
ts_pd.to_csv('results/ts_'+moln+'_RCDyM_.csv')
tm_pd.to_csv('results/tm_'+moln+'_RCDyM_.csv')
