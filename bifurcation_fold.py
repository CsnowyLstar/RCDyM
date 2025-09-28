import numpy as np
import random as random
#import torch
#import torch.nn as nn 
import pandas as pd
#import torchdiffeq as ode
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
    parser.add_argument('--max_eigenvalue', type=str, default='True') 
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--connectivity', type=float, default=0.05)
    parser.add_argument('--spectral_radius', type=float, default=0.85)
    parser.add_argument('--input_scaling', type=float, default=0.1)
    parser.add_argument('--leak', type=float, default=0.0)
    parser.add_argument('--b', type=float, default=4.0)
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
def gen_data(r=0.75, b=0.1, h=0.75, p=2, F=1, t_start=1, t_end=10000, x0=7.5, sigma=0, noise_on=0):
    if len(F)==1:
        F = np.repeat(F,t_end)
    
    def ricker(N,r,b,h,p,F,noise_add=0):
        out = N*np.exp(r-b*N) - F*(N**p/(N**p+h**p)) + noise_add
        return(out)
    
    L = np.zeros(t_end+1-t_start)
    
    L[0] = x0 
    for i in range(L.shape[0]-1):
        L_noise_add = sigma*L[i]*np.random.normal() if noise_on!=0 else 0
        L[i+1] = ricker(N=L[i], noise_add=L_noise_add, r=r, b=b, h=h, p=p, F=F[i])
    
    return(L)

dt = 0.01
tpoints = 20000
ts = np.arange(tpoints)*dt
F_bifurcation = np.linspace(1,2,tpoints)
ts_noy_meir = gen_data(F=F_bifurcation, sigma=0.01, noise_on=1, t_end=tpoints)
X = ts_noy_meir[:,None]
print("Data successfully generated!")

# detrend
tp = 16200
ifdetrend = True
X_ori = X.copy()
if ifdetrend:
    fig = plt.figure(figsize=(12,5))
    plt.plot(ts/dt,X_ori[:,:],'b')
    step_de = 100
    X_detrend = X_ori.copy()
    for i in range(step_de,len(ts)-step_de):
        X_detrend[i,0] = X_ori[i,0] - np.mean(X_ori[i-step_de:i+step_de,0])
    for i in range(step_de):
        X_detrend[i,0] = X_ori[i,0] - np.mean(X_ori[0:step_de,0])
    j = tp - step_de
    for i in range(j,len(ts)):
        X_detrend[i,0] = X_ori[i,0] - np.mean(X_ori[j-step_de:j+step_de,0])
    plt.plot(ts/dt,X_detrend[:,0],'r')
    plt.scatter([tp], [-0.5], marker='^')
    X = X_detrend


################################################################
###  (3) Calculate the RC EWS                                ###
################################################################
window = 2000; step = 200
#window = 1000; step = 500
index = 'max_eigenvalue' # select from ['max_eigenvalue','max_floquet','max_lyapunov']
continuous = False
num = 1; inds = []
for i in range(num):
    RCDyM = RC_EWM(X, ts, window, step, args, continuous)
    max_evals, tm = RCDyM.calculate(index)
    inds.append(max_evals)

# Calculate ground truth
def true_jac(s,p):
    Jx = (1-0.1*s)*np.exp(0.75-0.1*s) - (2*p*s*0.75**2)/((s**2+0.75**2)**2)
    return(Jx)

jxs = np.zeros_like(tm)
for i in range(len(tm)):
    j = int(tm[i]/dt-0.5*window)
    jxs[i] = true_jac(X_ori[j,0],F_bifurcation[j])

time = (tm-ts[0])/dt-0.5*window
time_tp = len(time) - 1
for ti in range(len(time)):
    if time[ti]>tp:
        time_tp = ti
        break
print(time_tp)


from scipy.optimize import minimize

def objective(coeffs, x, y, degree, alpha):
    design_matrix = np.vander(x, degree+1, increasing=True) 
    y_pred = np.dot(design_matrix, coeffs)
    data_fit = np.sum((y_pred - np.array(y)) ** 2)
    regularization = alpha * np.sum(coeffs[1:]**2)
    return data_fit + regularization

def monotonic_polyfit_nonnegative(x_data, y_data, degree, alpha=0.1):
    coeffs_init = np.polyfit(x_data, y_data, degree)[::-1] 
    #coeffs_init = np.ones(degree + 1) * (1e-5)
    
    bounds = [(None, None)] 
    bounds.extend([(0, None) for _ in range(degree)]) 

    result = minimize(objective, coeffs_init, args=(x_data, y_data, degree, alpha), 
                      bounds=bounds, method='SLSQP', options={'maxiter': 1000, 'ftol': 1e-8})
    if not result.success:
        print("warning:", result.message)
        return coeffs_init

    return result.x


################################################################
###  (4) Refined low-order polynomial                                                                      ###
################################################################
time = (tm-ts[0])/dt-0.5*window
time_tp = len(time) - 1
for ti in range(len(time)):
    if time[ti]>tp:
        time_tp = ti
        break
print(time_tp)

from scipy.optimize import minimize

def objective(coeffs, x, y, degree, alpha):
    design_matrix = np.vander(x, degree+1, increasing=True)
    y_pred = np.dot(design_matrix, coeffs)
    data_fit = np.sum((y_pred - np.array(y)) ** 2)
    regularization = alpha * np.sum(coeffs[1:]**2)
    return data_fit + regularization

def monotonic_polyfit_nonnegative(x_data, y_data, degree, alpha=0.1):
    coeffs_init = np.polyfit(x_data, y_data, degree)[::-1]
    #coeffs_init = np.ones(degree + 1) * (1e-5)
    
    bounds = [(None, None)]
    bounds.extend([(0, None) for _ in range(degree)])

    result = minimize(objective, coeffs_init, args=(x_data, y_data, degree, alpha), 
                      bounds=bounds, method='SLSQP', options={'maxiter': 1000, 'ftol': 1e-8})
    if not result.success:
        print("warning:", result.message)
        return coeffs_init

    return result.x
    

################################################################
###  (5) Draw and save                                       ###
################################################################
adv = 32
obt = time_tp - adv
cst = obt

tm_ind = (tm[:obt]-ts[0])/dt-0.5*window
x_data = tm_ind[obt-cst:]
y_data = max_evals[obt-cst:obt, 0]

degree = 4; alpha = 0.01
coefficients = monotonic_polyfit_nonnegative(x_data/tpoints, y_data, degree, alpha)
print(coefficients)
polynomial = np.poly1d(coefficients[::-1])
tm_fine = np.linspace(min(tm_ind[obt-cst:]), max(tm_ind[obt-cst:]), 500)
max_evals_fitted = polynomial(tm_fine/tpoints)
tm_extra = np.linspace(max(tm_ind[obt-cst:]), tp+850, 500)
extrapolation_pre = polynomial(tm_extra/tpoints)
print("coefficients", coefficients)


fig = plt.figure(figsize=(27,26.57))
ls = 60
ax1 = fig.add_subplot(2,1,1)
ax1.plot((ts/dt)[50:],X_ori[50:,0])
ax1.tick_params(labelsize=ls)
ax1.tick_params(axis='y', colors='blue')
ax1.set_ylabel(r"$s$",size=ls,color='blue')

ax12 = ax1.twinx()
ax12.plot(tm_fine, max_evals_fitted, 'b-', linewidth=10, alpha=0.6)
ax12.plot(tm_extra, extrapolation_pre, 'r-', linewidth=10, alpha=0.6)
ax12.plot(tm[:time_tp]/dt-0.5*window,jxs[:time_tp],'ko',markersize=8)
ax12.plot(tm_ind,max_evals[:obt,0],'rx',markersize=15)
#plt.ylim(-2.05,0.05)
ax12.tick_params(labelsize=ls)
ax12.tick_params(axis='y', colors='red'); ax12.set_ylabel("RCDyM (GT)",size=ls,color='red')

ax3 = fig.add_subplot(2,1,2)
ax3.plot(ts,F_bifurcation,'k-',linewidth=2.0)
ax3.tick_params(labelsize=ls)
ax3.set_xlabel(r"$t$",size=ls)
ax3.set_ylabel(r"$p$",size=ls)

plt.savefig("results/bifurcation_fold.png")

# save
X_pd = pd.DataFrame(X)
index_pd = pd.DataFrame(max_evals)
jxs_pd = pd.DataFrame(jxs)
ts_pd = pd.DataFrame(ts)
tm_pd = pd.DataFrame(tm)
ve = 'c' if continuous else 'd'
X_pd.to_csv('results/fold_data'+ve+'.csv')
index_pd.to_csv('results/fold_index'+ve+'.csv')
jxs_pd.to_csv('results/fold_jxs'+ve+'.csv')
ts_pd.to_csv('results/fold_ts'+ve+'.csv')
tm_pd.to_csv('results/fold_tm'+ve+'.csv')


################################################################
###  (6) Real and Imaginary Parts of DEJ                     ###
################################################################
print(jxs[:time_tp].shape)
print(max_evals[:time_tp].shape)

ls = 25
fig = plt.figure(figsize=(12,13))
ax1 = fig.add_subplot(3,1,1)
ax1.plot((ts/dt)[50:],X_ori[50:,0])
ax1.set_xlim(0,20000)
ax1.tick_params(labelsize=ls)
ax1.set_ylabel(r"$s$",size=ls)

ax2 = fig.add_subplot(3,1,2)
ax2.plot(tm[:time_tp]/dt-0.5*window, jxs[:time_tp],'ko',markersize=8, label='Ground truth')
ax2.plot(tm[:time_tp]/dt-0.5*window, max_evals[:time_tp,0],'rx',markersize=15, label='Prediction')
ax2.set_xlim(0,20000)
ax2.tick_params(labelsize=ls)
ax2.set_ylabel(r"Real part",size=ls)
plt.legend(fontsize=ls)

ax3 = fig.add_subplot(3,1,3)
ax3.plot(tm[:time_tp]/dt-0.5*window, np.zeros(len(jxs[:time_tp])),'ko',markersize=8, label='Ground truth')
ax3.plot(tm[:time_tp]/dt-0.5*window, max_evals[:time_tp,1],'rx',markersize=15, label='Prediction')
ax3.set_xlim(0,20000)
ax3.tick_params(labelsize=ls)
ax3.set_ylabel(r"Imaginary part",size=ls)
ax3.set_xlabel(r"Time",size=ls)

plt.savefig("results/type_fold.pdf")



