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
import argparse
from scipy import signal
from utils.RC_EWS import RC_EWM
import xarray as xr
import matplotlib.dates as mdates

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
    parser.add_argument('--alpha', type=float, default=1e-5)
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
dataset_salinity = xr.open_dataset("real_data/Salinity/salinity_dmean.nc")
lat = dataset_salinity.variables['lat'][:]
lon = dataset_salinity.variables['lon'][:]
salinity = dataset_salinity.variables['salinity'][:]
time = dataset_salinity.variables['time'][:]

filt_salinity = salinity[:, (lat >= 54) & (lat <= 62), (lon >= 360-62) & (lon <= 360-26)]
snn2 = np.mean(np.mean(filt_salinity,axis=-1),axis=-1).data[:,None]

dataset_sst = xr.open_dataset("real_data/SST/HadSST.4.0.1.0_median_amoc.nc")
sst = dataset_sst.variables['tos'][:].data[:,None]
#time_sst = dataset_sst.variables['time_bnds'][:]
sst[266] = (sst[265]+sst[267])/2
dataset_sst.close()

isyear = False
ischecknan = True
#X = sst.copy()
#X = np.concatenate((snn2, sn, ss), axis=-1)
X = np.concatenate((snn2, sst[sst.shape[0]-snn2.shape[0]:]), axis=-1)
if isyear:
    num = X.shape[0]//12 * 12
    X = np.mean(X[:num].reshape(12,-1,X.shape[-1]),axis=0)
if ischecknan:
    for i in range(X.shape[0]):
        if np.isnan(X[i,0]):
            X[i,0] = (X[i-1,0] + X[i+1,0])/2
            if np.isnan(X[i,0]):
                X[i,0] = (X[i-2,0] + X[i+2,0])/2
ts = np.double(np.arange(X.shape[0]))
dt = (ts[-1]-ts[0])/(len(ts)-1)
tpoints = len(ts)

X_ori = X.copy()
X_pd = pd.DataFrame(X)
ts_pd = pd.DataFrame(ts)
X_pd.to_csv('real_data/AMOC/amocX.csv')
ts_pd.to_csv('real_data/AMOC/amocTS.csv')

################################################################
###  (3) Calculate the RC EWS                                ###
################################################################
window = 500
step = 25
index = 'max_eigenvalue' # select from ['max_eigenvalue','max_floquet','max_lyapunov']

iscontinuous = False
RCDI = RC_EWM(X, ts, window, step, args, iscontinuous)
DEJ, tm = RCDI.calculate(index)
tm_ori = tm.copy()

DEJ_modulus = np.sqrt(DEJ[:,0]**2+DEJ[:,1]**2)
DEJ_modulus = np.array(DEJ_modulus-1.0)/1.0
DEJ_modulus_ori = DEJ_modulus.copy() 

indices = np.where(DEJ_modulus_ori < 0.01)[0]
print(DEJ_modulus_ori.shape, indices.shape) 
DEJ_modulus = DEJ_modulus_ori[indices]
tm = tm_ori[indices] 

################################################################
###  (4) Draw and save                                       ###
################################################################
degree = 1
num_month = 2072
fig = plt.figure(figsize=(24,16))
ls = 40; ms = 15
plt.rc('font', size=ls, family='Times New Roman')  # 设置全局字体

ax1 = fig.add_subplot(2,1,1)
ax1.plot(ts,X[:,0], 'orange', linewidth=3)
ax1.tick_params(labelsize=ls)
ax1.tick_params(axis='y', colors='orange')
ax1.set_ylabel("Salinity, ‰",size=ls,color='orange')

ax12 = ax1.twinx()
ax12.plot(ts,X[:,1], 'purple', linewidth=3, alpha=0.8)
ax12.tick_params(labelsize=ls)
ax12.tick_params(axis='y', colors='purple')
ax12.set_ylabel("SST, °C",size=ls,color='purple')
ax12.set_xticks([])
ax12.set_xlim(0,num_month) 
ax12.set_xlabel('Year')

ax2 = fig.add_subplot(2,1,2)
ax2.plot(tm, DEJ_modulus, 'm*', markersize=ms, label="RCDI indicator (DEJ)")
coefficients = np.polyfit(tm, DEJ_modulus, degree)
polynomial = np.poly1d(coefficients)
tm_fine = np.linspace(min(tm), max(tm), 500)
max_evals_fitted = polynomial(tm_fine)
tm_fine2 = np.linspace(max(tm), (0.0-coefficients[1])/coefficients[0], 100); print("t1",(0.0-coefficients[1])/coefficients[0]/12+1900)
max_evals_fitted2 = polynomial(tm_fine2)
ax2.plot(tm_fine,max_evals_fitted,'r-',linewidth=8,alpha=0.6,label="Regression line 1")
ax2.plot(tm_fine2,max_evals_fitted2,'r--',linewidth=4,alpha=1.0,label="Possible future trend 1")

sta = 18
coefficients = np.polyfit(tm[sta:], DEJ_modulus[sta:], degree)
polynomial = np.poly1d(coefficients)
tm_fine = np.linspace(min(tm[sta:]), max(tm[sta:]), 500)
max_evals_fitted = polynomial(tm_fine)
tm_fine2 = np.linspace(max(tm[sta:]), (0.0-coefficients[1])/coefficients[0], 100); print("t2",(0.0-coefficients[1])/coefficients[0]/12+1900)
max_evals_fitted2 = polynomial(tm_fine2)
ax2.plot(tm_fine,max_evals_fitted,'b-',linewidth=8,alpha=0.6,label="Regression line 2")
ax2.plot(tm_fine2,max_evals_fitted2,'b--',linewidth=4,alpha=1.0,label="Possible future trend 2")

ax2.plot([0,2300],[0,0], 'k--', linewidth=3)
#ax2.plot([0,2300],[0,0], 'k--')
ax2.tick_params(labelsize=ls)
ax2.tick_params(axis='y')
ax2.set_ylabel("RCDI",size=ls)
ax2.set_xticks([])
ax2.set_xlim(ts[0],ts[-1])

ax2.set_xticks(np.arange(num_month)[::(12*12)])
ax2.set_xticklabels(np.arange(1900,1900+num_month//12+1,12),rotation=0) 
ax2.set_xlim(0,num_month) 
ax2.set_xlabel('Time (year)')

plt.legend()
plt.savefig("results/AMOC.pdf")





