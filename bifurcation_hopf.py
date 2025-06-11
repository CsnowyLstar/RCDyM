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
from scipy.linalg import eig
from utils.RC_EWS import RC_EWM

################################################################
###  (1) Hyperparameter setting                              ###
################################################################
def args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--max_eigenvalue', type=str, default='True') 
    parser.add_argument('--n', type=int, default=100)
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
    x0 = torch.ones(2)*0.25
    
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

dt = 0.05
tpoints = 20000
ts = np.arange(tpoints)*dt
F_bifurcation = np.linspace(-2,0.5,tpoints)
sigma = 0.005
ts_hopf = gen_data(F_bifurcation, dt, sigma)
X = ts_hopf
X_ori = X.copy()
print("ts.shape:",ts.shape,"X.shape:",X.shape)
print("Data successfully generated!")
X_ori = X.copy()

################################################################
###  (3) Calculate the RC EWS                                ###
################################################################
window = 2000
step = 300
index = 'max_eigenvalue' # select from ['max_eigenvalue','max_floquet','max_lyapunov']
continuous = True
RCDI = RC_EWM(X, ts, window, step, args, continuous)
max_evals, tm = RCDI.calculate(index)

# Calculate ground truth
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


################################################################
###  (4) Draw and save                                       ###
################################################################
dele = 10
degree = 1
tm_ind = tm[:-dele]/dt-0.5*window
coefficients = np.polyfit(tm_ind, max_evals[:-dele,0], degree)
polynomial = np.poly1d(coefficients)
tm_fine = np.linspace(min(tm_ind), max(tm_ind), 500)
max_evals_fitted = polynomial(tm_fine)

fig = plt.figure(figsize=(21,20))
ls = 60
ax1 = fig.add_subplot(2,1,1)
ax1.plot(ts/dt,X_ori[:,0])
ax1.tick_params(labelsize=ls)
ax1.tick_params(axis='y', colors='blue')
ax1.set_ylabel(r"$s$",size=ls,color='blue')

ax12 = ax1.twinx()
ax12.plot(tm[:-dele]/dt-0.5*window,jxs[:-dele],'ko',markersize=8)
ax12.plot(tm_ind,max_evals[:-dele,0],'rx',markersize=15)
ax12.plot(tm_fine,max_evals_fitted,'r-',linewidth=8,alpha=0.6)
plt.ylim(top=0.2)
ax12.tick_params(labelsize=ls)
ax12.tick_params(axis='y', colors='red')
ax12.set_ylabel("RCDI (GT)",size=ls,color='red')

ax3 = fig.add_subplot(2,1,2)
ax3.plot(ts,F_bifurcation,'k-',linewidth=2.0)
ax3.tick_params(labelsize=ls)
ax3.set_xlabel(r"$t$",size=ls)
ax3.set_ylabel(r"$p$",size=ls)

plt.savefig("results/bifurcation_hopf.pdf")

# save
X_pd = pd.DataFrame(X)
index_pd = pd.DataFrame(max_evals)
jxs_pd = pd.DataFrame(jxs)
ts_pd = pd.DataFrame(ts)
tm_pd = pd.DataFrame(tm)
ve = 'c' if continuous else 'd'
X_pd.to_csv('results/hopf_data'+ve+'.csv')
index_pd.to_csv('results/hopf_index'+ve+'.csv')
jxs_pd.to_csv('results/hopf_jxs'+ve+'.csv')
ts_pd.to_csv('results/hopf_ts'+ve+'.csv')
tm_pd.to_csv('results/hopf_tm'+ve+'.csv')

