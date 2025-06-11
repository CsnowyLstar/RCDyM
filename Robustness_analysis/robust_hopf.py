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
from scipy.linalg import eig
from utils.Baseline_EWS import BL_EWS

################################################################
###  (1) Hyperparameter setting                              ###
################################################################
def args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--max_eigenvalue', type=str, default='True') 
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--connectivity', type=float, default=0.05)
    parser.add_argument('--spectral_radius', type=float, default=0.85)
    parser.add_argument('--input_scaling', type=float, default=0.1)
    parser.add_argument('--leak', type=float, default=0.0)
    parser.add_argument('--b', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1e-5)
    parser.add_argument('--warm_up', type=int, default=600)
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
    x0 = torch.ones(2)*7
    
    def equ(x,rh):
        g = np.zeros(2)
        g[0] = rh*x[0] - x[1] - x[0]*(x[0]**2+x[1]**2)
        g[1] = rh*x[1] + x[0] - x[1]*(x[0]**2+x[1]**2)
        return g
    
    L = np.zeros((len(rho),2))
    L[0] = x0
    for i in range(len(rho)-1):
        #noise_add = sigma*L[i]*torch.randn(2)
        noise_add = sigma*np.random.normal(0,1,2)
        L[i+1] = L[i] + equ(L[i],rho[i])*dt + noise_add
        
    return(L)

dt = 0.01
tpoints = 50000
ts = np.arange(tpoints)*dt
F_bifurcation = np.linspace(-2,0.5,tpoints)
sigma = 0.05
ts_hopf = gen_data(F_bifurcation, dt, sigma)
X = ts_hopf
print("Data successfully generated!")
tp = int(16500 * tpoints/20000)

################################################################
###  (3) Calculate the RC EWS                                ###
################################################################
window = 3000
step = 1000
index = 'max_eigenvalue' # select from ['max_eigenvalue','max_floquet','max_lyapunov']
iscontinuous = True
isdetrend = False
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

index = 'DEV'; window = 2000
EWS = BL_EWS(X, ts, window, step, args)
DEV_ews, DEV_tb = EWS.get_DEV(E=1, tau=1, theta=1, isabs=False)

index = 'deep_learning'; window = 2000
EWS = BL_EWS(X, ts, window, step, args)
DL_ews, DL_tb = EWS.calculate(index)

################################################################
###  (5) Draw and save                                       ###
################################################################
def true_jac(s,p):
    Jx = np.array([[p-3*s[0]**2-s[1]**2,-1-2*s[0]*s[1]],[1-2*s[0]*s[1],p-s[0]**2-3*s[1]**2]])
    return(Jx)

jxs = np.zeros_like(tm)
for i in range(len(tm)):
    j = int(tm[i]/dt)
    Jx = true_jac(X[j],F_bifurcation[j])
    evals_predm, evecs_predm = eig(Jx)
    mi = np.argmax(np.abs(evals_predm))
    jxs[i] = evals_predm[mi].real
    
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
    #plt.ylim(a, b)
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
moln = 'hopf'
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
DEV_tb_pd.to_csv('results/tb_'+moln+'_DEV_'+str(sigma)+'.csv')
DEV_ews_pd.to_csv('results/index_'+moln+'_DEV_'+str(sigma)+'.csv')

DL_tb_pd = pd.DataFrame(DL_tb)
DL_ews_pd = pd.DataFrame(DL_ews)
DL_tb_pd.to_csv('results/tb_'+moln+'_deep_learning_'+str(sigma)+'.csv')
DL_ews_pd.to_csv('results/index_'+moln+'_deep_learning_'+str(sigma)+'.csv')