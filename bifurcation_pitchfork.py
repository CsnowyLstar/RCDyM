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
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--connectivity', type=float, default=0.05)
    parser.add_argument('--spectral_radius', type=float, default=0.85)
    parser.add_argument('--input_scaling', type=float, default=1.0)
    parser.add_argument('--leak', type=float, default=0.0)
    parser.add_argument('--b', type=float, default=3.0)
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
def gen_data(dt=0.1, F=1, t_start=1, t_end=10000, x0=-1, sigma=0, noise_on=0):
    if len(F)==1:
        F = np.repeat(F,t_end)
    
    def diff(ut, x):
        out = 0.5 + ut*x -x**3
        return(out)
    
    L = np.zeros(t_end+1-t_start)
    
    #h = 0.001
    h = dt
    nlp = int(dt//h)
    L[0] = x0 
    for i in range(L.shape[0]-1):
        for j in range(nlp):
            x0 = x0 + diff(F[i], L[i])*h
        L_noise_add = sigma*L[i]*np.random.normal() if noise_on!=0 else 0
        x0 = x0 + L_noise_add
        L[i+1] = x0
    
    return(L)

dt = 0.1
tpoints = 20000
ts = np.arange(tpoints)*dt
#F_bifurcation = np.linspace(2.0,1.10,tpoints)
F_bifurcation = np.linspace(1.65,1.15,tpoints)
sigma = 0.01
ts_pitchfork = gen_data(F=F_bifurcation, sigma=sigma, noise_on=1, t_end=tpoints)
X = ts_pitchfork[:,None]
print("Data successfully generated!")

X_ori = X.copy()
for ti in range(tpoints):
    if X_ori[-ti-1, 0] < 0:
        tp = tpoints - ti - 1
        break


################################################################
###  (3) Calculate the RC EWS                                ###
################################################################
window = 1000; step = 300

index = 'max_eigenvalue' # select from ['max_eigenvalue','max_floquet','max_lyapunov']
continuous = True
RCDyM = RC_EWM(X, ts, window, step, args, continuous)
max_evals, tm = RCDyM.calculate(index)

# Calculate ground truth
def true_jac(s,p):
    Jx = p-3*s**2
    return(Jx)

jxs = np.zeros_like(tm)
for i in range(len(tm)):
    j = int(tm[i]/dt-0.5*window)
    jxs[i] = true_jac(X_ori[j,0],F_bifurcation[j])



################################################################
###  (4) Refined low-order polynomial                        ###
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
adv = 20
obt = time_tp - adv
cst = obt

tm_ind = (tm[:obt]-ts[0])/dt-0.5*window
x_data = tm_ind[obt-cst:]
y_data = max_evals[obt-cst:obt, 0]

degree = 3; alpha = 0.01
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
plt.ylim(-2.05,0.05)
ax12.tick_params(labelsize=ls)
ax12.tick_params(axis='y', colors='red'); ax12.set_ylabel("RCDyM (GT)",size=ls,color='red')

ax3 = fig.add_subplot(2,1,2)
ax3.plot(ts,F_bifurcation,'k-',linewidth=2.0)
ax3.tick_params(labelsize=ls)
ax3.set_xlabel(r"$t$",size=ls)
ax3.set_ylabel(r"$p$",size=ls)

plt.savefig("results/bifurcation_pitchfork.pdf")

# save
X_pd = pd.DataFrame(X)
index_pd = pd.DataFrame(max_evals)
jxs_pd = pd.DataFrame(jxs)
ts_pd = pd.DataFrame(ts)
tm_pd = pd.DataFrame(tm)
ve = 'c' if continuous else 'd'
X_pd.to_csv('results/pitchfork_data'+ve+'.csv')
index_pd.to_csv('results/pitchfork_index'+ve+'.csv')
jxs_pd.to_csv('results/pitchfork_jxs'+ve+'.csv')
ts_pd.to_csv('results/pitchfork_ts'+ve+'.csv')
tm_pd.to_csv('results/pitchfork_tm'+ve+'.csv')



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
ax2.set_ylim(-2.5,0.2)
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

plt.savefig("results/type_pitchfork.pdf")
