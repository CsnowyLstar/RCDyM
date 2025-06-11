import numpy as np
import random as random
import torch
import torch.nn as nn 
import pandas as pd
import torchdiffeq as ode
import matplotlib.pyplot as plt
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
import random
from utils.reservoir_model_RC import Reservoir
from sklearn.linear_model import Ridge
from scipy.signal import find_peaks
from scipy.linalg import eig, qr
import joblib 
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy import stats
import warnings
import argparse

seed = 0
np.random.seed(seed)
random.seed(seed)

################################################################
###  (1) Hyperparameter setting                              ###
################################################################
def args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--connectivity', type=float, default=0.001)
    parser.add_argument('--spectral_radius', type=float, default=0.85)
    parser.add_argument('--input_scaling', type=float, default=0.5)
    parser.add_argument('--leak', type=float, default=0.0)
    parser.add_argument('--b', type=float, default=2.0)
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
def gen_ks(h,T,tpoints,d,Nx,deltas,sigma,u,x,N):
    xs = x[::int(N//Nx)]
    tmax = T
    nmax = round(tmax/h)
    
    k = np.transpose(np.conj(np.concatenate((np.arange(0, N/2), np.array([0]), np.arange(-N/2+1, 0)))))*(2*np.pi/d) 
    
    Xs = np.zeros((tpoints,Nx))#Used to record the status at all times
    nplt = int((tmax/tpoints)/h)
    ux = u[np.arange(0,len(u),int(N/Nx))]
    v = np.fft.fft(u)
    uu = np.array([ux])
    tt = 0
    g = -0.5j*k
    
    for n in range(1, nmax): 
        rh = deltas[n-1]
        L = k**2 - rh * k**4
        E = np.exp(h*L)
        E_2 = np.exp(h*L/2)
        M = 16
        r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5) / M) 
        LR = h*np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], N, axis=0)
        Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
        f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
        f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
        f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
        
        rifftv = np.real(np.fft.ifft(v))
        Nv = g*np.fft.fft(rifftv**2)
        a = E_2*v + Q*Nv
        riffta = np.real(np.fft.ifft(a))
        Na = g*np.fft.fft(riffta**2)
        b = E_2*v + Q*Na
        rifftb = np.real(np.fft.ifft(b))
        Nb = g*np.fft.fft(rifftb**2)
        c = E_2*a + Q*(2*Nb-Nv)
        rifftc = np.real(np.fft.ifft(c))
        Nc = g*np.fft.fft(rifftc**2)
        v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
        if n%nplt == 0:
            #print(n//nplt,"/",tpoints)
            u = np.real(np.fft.ifft(v))
            # add noise
            #noise = (np.random.rand(128)-0.5)*2*sigma
            noise = np.random.randn(128)*sigma
            u = u + noise
            v = np.fft.fft(u)
            ux = u[np.arange(0,len(u),int(N/Nx))]
            uu = np.append(uu, np.array([ux]), axis=0)
            tt = np.hstack((tt, n))
    if True not in np.isnan(uu):
        Xs[:,:] = uu
    return(Xs,xs,u)


dt =  0.05
h = 0.05
tpoints = 20000
T = int(dt*tpoints)
ts = np.arange(tpoints) * dt
deltas = np.linspace(0.076,0.076,int(T/h))
F_bifurcation = deltas[::int(T/dt/tpoints)]
method = "euler"
sigma = 0.04

d = 2*np.pi; Nx = 64; N=128
x = d*np.transpose(np.conj(np.arange(-N/2+1, N/2+1))) / N
u = np.sin(x/4)*10
nu = 2000
X_ini,xs_ini,u = gen_ks(h,h*nu,nu,d,Nx,np.ones(nu)*deltas[0],sigma,u,x,N)
X_ini,xs_ini,u = gen_ks(h,T,tpoints,d,Nx,deltas,sigma,u,x,N)
X = X_ini
xs = xs_ini
print("Data successfully generated!")


# ================ RC 超参数设置 ====================
def args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--connectivity', type=float, default=0.01)
    parser.add_argument('--spectral_radius', type=float, default=0.85)
    parser.add_argument('--input_scaling', type=float, default=0.1)
    parser.add_argument('--leak', type=float, default=0.0)
    parser.add_argument('--b', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=1e-5)
    parser.add_argument('--warm_up', type=int, default=2000)
    parser.add_argument('--method', type=str, default='euler') 
    args = parser.parse_args(args=[])
    return(args)

args = args()

reservoir = Reservoir(n_internal_units=args.n,
                      spectral_radius=args.spectral_radius,
                      leak=args.leak,
                      connectivity=args.connectivity,
                      input_scaling=args.input_scaling,
                      noise_level=0.0,
                      sigma_b = args.b,
                      isks = True)
readout = Ridge(alpha=args.alpha, fit_intercept=True)
b = reservoir.sigma_b
n = reservoir._n_internal_units
l = reservoir.leak

# ================== RC 多步预测 =====================
#warnings.filterwarnings('ignore', category=FutureWarning)
isnonr = False
warm_up = args.warm_up; V = X.shape[1]
ri = reservoir.get_states(X, n_drop=0)
rp = ri.copy()
if isnonr:
    rp = np.concatenate((rp, rp**2), axis = 1)


# train
nst = int(X.shape[0]*0.05)
ned = int(X.shape[0]*0.80)
X_b = rp[nst:ned,:] 
Y_b = X[nst:ned,:]
readout.fit(X_b,Y_b)
print("Training complete")

# Prediction and Calculate the MLE
pl = 500 
start = ned
rt = np.zeros((pl,args.n)) 
preds = np.zeros((pl,V))
previous_state = ri[start,:][None,:]
current_input = X[start:start+1,:]
for jpl in range(pl):
    rt[jpl] = previous_state[0]
    preds[jpl] = current_input[0]
    previous_state = reservoir._compute_next_state(previous_state,current_input)
    rp = previous_state.copy()
    if isnonr:
        rp = np.concatenate((rp, rp**2), axis = 1)
    current_input = readout.predict(rp)
    
rp = ri.copy()
if isnonr:
    rp = np.concatenate((rp, rp**2), axis = 1)
preds2 = readout.predict(rp[start:start+pl])
print("Complete the prediction")


# draw
'''
pre = preds.copy()
mr = 1.0
cmap = 'RdYlBu'
fig, ax = plt.subplots(4, 1, figsize=(30,12))
fig.set_tight_layout(True)
tt_mesh, xx_mesh = np.meshgrid(ts, xs)
true = X.transpose()
ax[0].pcolormesh(tt_mesh, xx_mesh, true, vmin=true.min()*mr, vmax=true.max()*mr, cmap=cmap)

ax[1].pcolormesh(tt_mesh[:,start:start+pl], xx_mesh[:,start:start+pl], true[:,start:start+pl], vmin=true.min()*mr, vmax=true.max()*mr, cmap=cmap)

pred_draw = np.zeros((V,tpoints))
pred_draw[:,start:start+pl] = pre.transpose()
ax[2].pcolormesh(tt_mesh[:,start:start+pl], xx_mesh[:,start:start+pl], pred_draw[:,start:start+pl], vmin=true.min()*mr, vmax=true.max()*mr, cmap=cmap)

error = np.zeros((V,tpoints))
error[:,start:start+pl] = (pre - X[start:start+pl]).transpose()
ax[3].pcolormesh(tt_mesh[:,start:start+pl], xx_mesh[:,start:start+pl], error[:,start:start+pl], vmin=true.min()*mr, vmax=true.max()*mr, cmap=cmap)

print("########### Final results ############")
print(np.mean(np.abs(error[:,start:start+pl])))
'''

pre = preds.copy()
mr = 1.0
cmap = 'RdYlBu'
fig, ax = plt.subplots(2, 1, figsize=(20,8))
fig.set_tight_layout(True)
tt_mesh, xx_mesh = np.meshgrid(ts, xs)
true = X.transpose()
ax[0].pcolormesh(tt_mesh[:,start:start+pl], xx_mesh[:,start:start+pl], true[:,start:start+pl], vmin=true.min()*mr, vmax=true.max()*mr, cmap=cmap)

pred_draw = np.zeros((V,tpoints))
pred_draw[:,start:start+pl] = pre.transpose()
ax[1].pcolormesh(tt_mesh[:,start:start+pl], xx_mesh[:,start:start+pl], pred_draw[:,start:start+pl], vmin=true.min()*mr, vmax=true.max()*mr, cmap=cmap)

print("########### Final results ############")


ind = 8
fig, ax = plt.subplots(3, 1, figsize=(20,8))
ax[0].plot(X[start:start+pl, ind])
ax[1].plot(preds[:,ind], 'r--')
ax[2].plot(X[:start, ind])
plt.show()