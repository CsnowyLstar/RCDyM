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
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--connectivity', type=float, default=0.05)
    parser.add_argument('--spectral_radius', type=float, default=0.85)
    parser.add_argument('--input_scaling', type=float, default=0.1)
    parser.add_argument('--leak', type=float, default=0.8)
    parser.add_argument('--b', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=1e-5)
    parser.add_argument('--warm_up', type=int, default=2000)
    parser.add_argument('--method', type=str, default='euler') 
    args = parser.parse_args(args=[])
    return(args)

args = args()

seed = 1
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

dt =  0.04
h = 0.01
tpoints = 20000
T = int(dt*tpoints)
ts = np.arange(tpoints) * dt
#deltas = np.linspace(0.076,0.082,int(T/h))
deltas = np.linspace(0.076,0.0816,int(T/h))
F_bifurcation = deltas[::int(T/dt/tpoints)]
method = "euler"
sigma = 1e-5

d = 2*np.pi; Nx = 64; N=128
x = d*np.transpose(np.conj(np.arange(-N/2+1, N/2+1))) / N
u = np.sin(x/4)*10
nu = 2000
X_ini,xs_ini,u = gen_ks(h,h*nu,nu,d,Nx,np.ones(nu)*deltas[0],sigma,u,x,N)
X_ini,xs_ini,u = gen_ks(h,T,tpoints,d,Nx,deltas,sigma,u,x,N)
X = X_ini
xs = xs_ini
print("Data successfully generated!")

################################################################
###  (3) Calculate the RC EWS                                ###
################################################################
window = 2000
step = 400
index = 'max_floquet' # select from ['max_eigenvalue','max_floquet','max_lyapunov']

iscontinuous = False
RCDI = RC_EWM(X, ts, window, step, args, iscontinuous)
max_floquets, tm = RCDI.calculate(index)


rcdi = np.ones(max_floquets.shape[0])
for i in range(len(rcdi)):
    rcdi[i] = min(np.abs(max_floquets[i,0]), 1)

################################################################
###  (4) Draw                                                ###
################################################################
fig = plt.figure(figsize=(20,12))
ls = 40; it = 1
ax1 = fig.add_subplot(2,1,1)
tt_mesh, xx_mesh = np.meshgrid(ts[::it]/dt, xs)
true = X.transpose()
ax1.pcolormesh(tt_mesh, xx_mesh, true[:,::it], cmap='RdYlBu')
ax1.tick_params(labelsize=ls)
ax1.tick_params(axis='y', colors='blue')
ax1.set_ylabel(r"$s$",size=ls,color='blue')

ax12 = ax1.twinx()
ax12.plot(tm/dt,rcdi,'rx', markersize=20)
plt.ylim(-0.0,1.1)
#plt.ylim(-0.1,0.03)
ax12.tick_params(labelsize=ls)
ax12.tick_params(axis='y', colors='red')
ax12.set_ylabel("RCDI",size=ls,color='red')

ax2 = fig.add_subplot(2,1,2)
ax2.plot(true[8,:],'b-',linewidth=2.0)
ax2.set_xlim(0,tpoints)
ax2.tick_params(labelsize=ls)
ax2.set_xlabel("Time",size=ls)
ax2.set_ylabel(r"$s$",size=ls)

ax22 = ax2.twinx()
ax22.plot(F_bifurcation[::4],'k-',linewidth=2.0)
ax22.tick_params(labelsize=ls)
ax22.set_xlabel(r"$t$",size=ls)
ax22.set_ylabel(r"$p$",size=ls)

#plt.savefig("results/KSV3.pdf")