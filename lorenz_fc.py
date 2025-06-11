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
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--connectivity', type=float, default=0.05)
    parser.add_argument('--spectral_radius', type=float, default=0.85)
    parser.add_argument('--input_scaling', type=float, default=0.1)
    parser.add_argument('--leak', type=float, default=0.0)
    parser.add_argument('--b', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1e-5)
    parser.add_argument('--warm_up', type=int, default=1000)
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
    
    def equ(x,rh):
        g = torch.zeros(3)
        g[0] = 10*(x[1]-x[0])
        g[1] = x[0]*(rh-x[2])-x[1]
        g[2] = x[0]*x[1]-8/3*x[2]
        return g
    
    x0 = np.ones(3)*8 
    for j in range(500):
        x0 = equ(x0,rho[0])*dt + x0
            
    #h = 0.005
    h = dt
    nlp = int(dt//h)
    L = np.zeros((len(rho),3))
    L[0] = x0
    for i in range(len(rho)-1):
        for j in range(nlp):
            x0 = equ(x0,rho[i])*h + x0
        noise_add = sigma*x0*np.random.normal(0,1,3)
        x0 = x0 + noise_add
        L[i+1] = x0
    
    return(L)

def get_lorenz_rk4(rho,dt,sigma):
    leng = len(rho)
    v = np.zeros((leng,3))
    v[0] = np.ones(3)*8 
    
    f = lambda v_i, rho: np.array([
        10 * (v_i[1] - v_i[0]),
        v_i[0] * (rho - v_i[2]) - v_i[1],
        v_i[0] * v_i[1] - 8/3 * v_i[2]
    ])
    
    for i in range(leng-1):
        vi = v[i,:]
        k1 = dt * f(vi, rho[i])
        k2 = dt * f(vi + 0.5*k1, rho[i])
        k3 = dt * f(vi + 0.5*k2, rho[i])
        k4 = dt * f(vi + k3, rho[i])
        noise = sigma*vi*np.random.normal(0,1,3)
        v[i+1,:] = vi + (k1/6 + k2/3 + k3/3 + k4/6) + noise
    
    return(v)

dt = 0.01
tpoints = 20000
ts = np.arange(tpoints)*dt
F_bifurcation = np.linspace(2,23,tpoints)
sigma = 0.002
ts_hopf = gen_data(F_bifurcation, dt, sigma)
X = ts_hopf
print("Data successfully generated!")


################################################################
###  (3) Calculate the RC EWS                                ###
################################################################
window = 2000
step = 500
iscontinuous = True
isdetrend = True

index = 'max_eigenvalue' # select from ['max_eigenvalue','max_floquet','max_lyapunov']
num = 1; inds_DEJ = []
for i in range(num):
    RCDI = RC_EWM(X, ts, window, step, args, iscontinuous, isdetrend)
    max_evals, tm = RCDI.calculate(index)
    inds_DEJ.append(max_evals)
    
    
################################################################
###  (4) Draw                                                ###
################################################################
def true_jac(s,p):
    Jx = torch.tensor([[-10,10,0],[p-s[2],-1,-1*s[0]],[s[1],s[0],-8/3]])
    return(Jx)

jxs = np.zeros_like(tm)
for i in range(len(tm)):
    j = int(tm[i]/dt)
    rh = F_bifurcation[j]
    true_xeq = torch.tensor([np.sqrt((rh-1)*8/3),np.sqrt((rh-1)*8/3),rh-1])
    Jx = true_jac(true_xeq,rh)
    evals_predm, evecs_predm = eig(Jx)
    mi = np.argmax(np.real(evals_predm))
    jxs[i] = evals_predm[mi].real

fig = plt.figure(figsize=(20,12))
ls = 25; ms=15
font1 = {'weight':'normal','size':ls}
ax1 = fig.add_subplot(2,1,1)
ax1.plot(ts/dt, X[:,:], label='data', alpha=0.5)
ax1.tick_params(labelsize=ls)
ax1.tick_params(axis='y', colors='k')
ax1.set_ylabel(r"$s$",size=ls,color='k')
ax1.legend(prop=font1)

ax12 = ax1.twinx()
ax12.plot(tm/dt,jxs,'ko')
ed = -7; xs = tm[:ed]; ys = inds_DEJ[0][:ed,0]
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
ax12.set_ylabel("RCDI",size=ls,color='red')
ax12.legend(prop=font1)
ax12.set_ylim(-1.5,0.5)

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

# save
X_pd = pd.DataFrame(X)
index_DEJ_pd = pd.DataFrame(inds_DEJ[0])
#index_MLE_pd = pd.DataFrame(inds_MLE[0])
ts_pd = pd.DataFrame(ts)
tm_pd = pd.DataFrame(tm)
X_pd.to_csv('results/lorenz_data.csv')
index_DEJ_pd.to_csv('results/lorenz_DEJ.csv')
#index_MLE_pd.to_csv('results/lorenz_MLE.csv')
ts_pd.to_csv('results/lorenz_ts.csv')
tm_pd.to_csv('results/lorenz_tm.csv')




