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
from scipy.linalg import eig, qr
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
    parser.add_argument('--b', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=1e-5)
    parser.add_argument('--warm_up', type=int, default=500)
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
            
    h = 0.001
    #h = dt
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

dt = 0.02
tpoints = 20000
ts = np.arange(tpoints)*dt
F_bifurcation = np.linspace(50,20,tpoints)
#F_bifurcation = np.linspace(155,166,tpoints)
sigma = 0.001
ts_hopf = get_lorenz_rk4(F_bifurcation, dt, sigma)
X = ts_hopf
print("Data successfully generated!")


################################################################
###  (3) Calculate the RC EWS                                ###
################################################################
window = 2000
step = 400
iscontinuous = True
index = 'max_lyapunov' # select from ['max_eigenvalue','max_floquet','max_lyapunov']
num = 1; inds_MLE = []
for i in range(num):
    RCDyM = RC_EWM(X, ts, window, step, args, iscontinuous)
    max_evals, tm = RCDyM.calculate(index)
    if index == 'max_lyapunov':
        max_evals = max_evals[:,None]
    inds_MLE.append(max_evals)
    
    
################################################################
###  (4) calculate the ground truth of MLE                   ###
################################################################
nwin = (len(ts)-args.warm_up-window)//step
tm = (np.arange(nwin)*step + window + args.warm_up)*dt 

def equa(x,rh):
    g = torch.zeros(3)
    g[0] = 10*(x[1]-x[0])
    g[1] = x[0]*(rh-x[2])-x[1]
    g[2] = x[0]*x[1]-8/3*x[2]
    return g

# Calculate the Lyapunov exponent of the original system
num_lyaps = 1
delta, RR = np.linalg.qr(np.random.rand(3, num_lyaps)) 
pl = 100000
h = 0.001
norm_time = 10
X_ii = np.zeros((num_lyaps, pl // norm_time)) + 0j
true_MLEs = np.zeros(nwin)
for wi in range(nwin):
    x = X[args.warm_up+wi*step+window]
    for i in range(pl):
        rh = F_bifurcation[args.warm_up+wi*step+window]
        
        Jx = torch.tensor([[-10,10,0],[rh-x[2],-1,-1*x[0]],[x[1],x[0],-8/3]])
        delta = Jx @ delta * h + delta
        if i % norm_time == 0:
            QQ,RR = qr(delta, mode='economic')
            delta = QQ
            X_ii[:,i//norm_time] = np.log(np.diag(RR)+0j)
        
        x = equa(x,rh)*h + x
    
    lex = np.sum(X_ii, axis=1).real / (pl * h)
    print("Lyapunov exponent of x:", lex)
    true_MLEs[wi] = np.max(lex)
    
    
################################################################
###  (5) Draw                                                ###
################################################################
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
ax12.plot(tm/dt,true_MLEs,'ko')
ed = -1; xs = tm[:ed]; ys = inds_MLE[0][:ed,0]
for i in range(len(inds_MLE)):
    max_evals = inds_MLE[i]
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
ax12.set_ylim(0.5,1.5)

sf = 5
js = np.linspace(window+args.warm_up,ts[-1]/dt,sf) 
for i in range(sf):
    j = int(js[i])
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
plt.savefig("results/lorenz_cf.png")

# save
X_pd = pd.DataFrame(X)
index_DEJ_pd = pd.DataFrame(inds_MLE[0])
#index_MLE_pd = pd.DataFrame(inds_MLE[0])
ts_pd = pd.DataFrame(ts)
tm_pd = pd.DataFrame(tm)
X_pd.to_csv('results/lorenz_data.csv')
index_DEJ_pd.to_csv('results/lorenz_DEJ.csv')
#index_MLE_pd.to_csv('results/lorenz_MLE.csv')
ts_pd.to_csv('results/lorenz_ts.csv')
tm_pd.to_csv('results/lorenz_tm.csv')


