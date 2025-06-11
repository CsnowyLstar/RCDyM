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
from utils.Baseline_EWS import BL_EWS

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
tpoints = 50000
ts = np.arange(tpoints)*dt
F_bifurcation = np.linspace(2,23,tpoints)
#F_bifurcation = np.linspace(155,166,tpoints)
sigma = 0.002
ts_hopf = gen_data(F_bifurcation, dt, sigma)
X = ts_hopf
tp = 35000
print("Data successfully generated!")



################################################################
###  (3) Calculate the RC EWS                                ###
################################################################
window = 3000
step = 2000
index = 'max_eigenvalue' # select from ['max_eigenvalue','max_floquet','max_lyapunov']
iscontinuous = True
isdetrend = True
RCDI = RC_EWM(X, ts, window, step, args, iscontinuous, isdetrend)
max_evals, tm = RCDI.calculate(index)


################################################################
###  (4) Calculate the baselines                             ###
################################################################
step = 1000
indexs = ['Variance','autocorrelation','skewness']
windows = [2000, 2000, 2000]
baselines = []
tbs = []
for i in range(len(indexs)):
    window = windows[i]
    index = indexs[i]
    EWS = BL_EWS(X, ts, window, step, args)
    ews, tb = EWS.calculate(index)
    
    baselines.append(ews)
    tbs.append(tb)
    
index = 'DEV'; window = 500
EWS = BL_EWS(X, ts, window, step, args)
DEV_ews, DEV_tb = EWS.get_DEV(E=3, tau=1, theta=0.5)

index = 'deep_learning'; window = 2000
EWS = BL_EWS(X, ts, window, step, args)
DL_ews, DL_tb = EWS.calculate(index='Variance')


################################################################
###  (5) Draw and save                                       ###
################################################################
def true_jac(s,p):
    Jx = (1-0.1*s)*np.exp(0.75-0.1*s) - (2*p*s*0.75**2)/((s**2+0.75**2)**2)
    return(Jx)

jxs = np.zeros_like(tm)
for i in range(len(tm)):
    j = int(tm[i]/dt)
    jxs[i] = true_jac(X[j,0],F_bifurcation[j])
    

fig = plt.figure(figsize=(15,25))
ls = 25
def draw(fi, td, ews, label, istrue):
    i = next((i for i in range(len(td)) if td[i] > ts[tp]), None)
    td = td[:i]
    ews = ews[:i]
    ax1 = fig.add_subplot(6,1,fi)
    ax1.plot(ts/dt,X)
    ax1.tick_params(labelsize=ls)
    ax1.tick_params(axis='y', colors='blue')
    ax1.set_ylabel(r"$s$",size=ls,color='blue')
    ax12 = ax1.twinx()
    if istrue:
        i = next((i for i in range(len(tm)) if tm[i] > ts[tp]), None)
        ax12.plot(tm[:i]/dt, jxs[:i], 'ko')
    ax12.plot(td/dt, ews, 'ro')
    ax12.tick_params(labelsize=ls)
    ax12.tick_params(axis='y', colors='red')
    ax12.set_ylabel(label, size=ls, color='red')

draw(1, tm, max_evals[:,0], 'RCDI', False)
draw(2, tbs[0], baselines[0], indexs[0], False)
draw(3, tbs[1], baselines[1], indexs[1], False)
draw(4, tbs[2], baselines[2], indexs[2], False)
draw(5, DEV_tb, DEV_ews, 'DEV', False)
draw(6, DL_tb, DL_ews, 'DL', False)


# save
moln = 'lorenz'
X_pd = pd.DataFrame(X)
index_pd = pd.DataFrame(max_evals)
jxs_pd = pd.DataFrame(jxs)
ts_pd = pd.DataFrame(ts)
tm_pd = pd.DataFrame(tm)
X_pd.to_csv('results/X_'+moln+'_RCDI_'+str(sigma)+'.csv')
index_pd.to_csv('results/index_'+moln+'_RCDI_'+str(sigma)+'.csv')
jxs_pd.to_csv('results/jxs_'+moln+'_RCDI_'+str(sigma)+'.csv')
ts_pd.to_csv('results/ts_'+moln+'_RCDI_'+str(sigma)+'.csv')
tm_pd.to_csv('results/tm_'+moln+'_RCDI_'+str(sigma)+'.csv')

for i in range(len(tbs)):
    tb = pd.DataFrame(tbs[i])
    ews = pd.DataFrame(baselines[i])
    tb.to_csv('results/tb_'+moln+'_'+indexs[i]+'_'+str(sigma)+'.csv')
    ews.to_csv('results/index_'+moln+'_'+indexs[i]+'_'+str(sigma)+'.csv')

DEV_tb_pd = pd.DataFrame(DEV_tb)
DEV_ews_pd = pd.DataFrame(DEV_ews)
tb.to_csv('results/tb_'+moln+'_DEV_'+str(sigma)+'.csv')
ews.to_csv('results/index_'+moln+'_DEV_'+str(sigma)+'.csv')

DL_tb_pd = pd.DataFrame(DL_tb)
DL_ews_pd = pd.DataFrame(DL_ews)
tb.to_csv('results/tb_'+moln+'_deep_learning_'+str(sigma)+'.csv')
ews.to_csv('results/index_'+moln+'_deep_learning_'+str(sigma)+'.csv')




