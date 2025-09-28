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

################################################################
###  (1) Hyperparameter setting                              ###
################################################################
def args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--max_eigenvalue', type=str, default='True') 
    parser.add_argument('--n', type=int, default=300)
    parser.add_argument('--connectivity', type=float, default=0.05)
    parser.add_argument('--spectral_radius', type=float, default=0.95)
    parser.add_argument('--input_scaling', type=float, default=0.1)
    parser.add_argument('--leak', type=float, default=0.0)
    parser.add_argument('--b', type=float, default=2.0)
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
    x0 = np.ones(2)*0.5
    
    def equ(x,rh):
        g = np.zeros(2)
        g[0] = 1 - rh*x[0]**2 + x[1]
        g[1] = 0.3*x[0]
        return g
    
    L = np.zeros((len(rho),2))
    L[0] = x0
    for i in range(len(rho)-1):
        noise_add = sigma*L[i]*np.random.normal(0,1,2)
        #noise_add = sigma*np.random.normal(0,1,2)
        L[i+1] = equ(L[i],rho[i]) + noise_add
        
    return(L)

dt = 0.01
tpoints = 20000
ts = np.arange(tpoints)*dt
F_bifurcation = np.linspace(0.15,0.4,tpoints)
sigma = 0.01
ts_hopf = gen_data(F_bifurcation, dt, sigma)
X = ts_hopf
print("Data successfully generated!")

# detrend
tp = 17000
ifdetrend = True
X_ori = X.copy()
if ifdetrend:
    fig = plt.figure(figsize=(12,5))
    plt.plot(ts/dt,X_ori[:,:],'b')
    step_de = 100
    X_detrend = X_ori.copy()
    for i in range(step_de,len(ts)-step_de):
        X_detrend[i,0] = X_ori[i,0] - np.mean(X_ori[i-step_de:i+step_de,0])
        X_detrend[i,1] = X_ori[i,1] - np.mean(X_ori[i-step_de:i+step_de,1])
    for i in range(step_de):
        X_detrend[i,0] = X_ori[i,0] - np.mean(X_ori[0:step_de,0])
        X_detrend[i,1] = X_ori[i,1] - np.mean(X_ori[0:step_de,1])
    j = tp - step_de
    for i in range(j,len(ts)):
        X_detrend[i,0] = X_ori[i,0] - np.mean(X_ori[j-step_de:j+step_de,0])
        X_detrend[i,1] = X_ori[i,1] - np.mean(X_ori[j-step_de:j+step_de,1])
    plt.plot(ts/dt,X_detrend[:,0],'r')
    plt.scatter([tp], [-0.5], marker='^')
    X = X_detrend


################################################################
###  (3) Calculate the RC EWS                                ###
################################################################
window = 2000; step = 300
index = 'max_eigenvalue' # select from ['max_eigenvalue','max_floquet','max_lyapunov']
continuous = False
RCDyM = RC_EWM(X, ts, window, step, args, continuous)
max_evals, tm = RCDyM.calculate(index)

# Calculate ground truth
def true_jac(s,p):
    Jx = np.array([[-2*p*s,1],[0.3,0]])
    return(Jx)

jxs = np.zeros_like(tm); jxs_c = np.zeros_like(tm, dtype=complex) 
for i in range(len(tm)):
    j = int(tm[i]/dt-0.5*window)
    Jx = true_jac(X_ori[j,0],F_bifurcation[j])
    evals_predm, evecs_predm = eig(Jx)
    mi = np.argmax(np.abs(evals_predm))
    jxs[i] = evals_predm[mi].real
    jxs_c[i] = evals_predm[mi]



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
    bounds.extend([(None, 0) for _ in range(degree)])

    result = minimize(objective, coeffs_init, args=(x_data, y_data, degree, alpha), 
                      bounds=bounds, method='SLSQP', options={'maxiter': 1000, 'ftol': 1e-8})
    if not result.success:
        print("warning:", result.message)
        return coeffs_init

    return result.x
    

################################################################
###  (5) Draw and save                                       ###
################################################################
adv = 25
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
#plt.ylim(-2.05,0.05)
ax12.tick_params(labelsize=ls)
ax12.tick_params(axis='y', colors='red'); ax12.set_ylabel("RCDyM (GT)",size=ls,color='red')

ax3 = fig.add_subplot(2,1,2)
ax3.plot(ts,F_bifurcation,'k-',linewidth=2.0)
ax3.tick_params(labelsize=ls)
ax3.set_xlabel(r"$t$",size=ls)
ax3.set_ylabel(r"$p$",size=ls)

plt.savefig("results/bifurcation_period-doubling.pdf")

# save
X_pd = pd.DataFrame(X)
index_pd = pd.DataFrame(max_evals)
jxs_pd = pd.DataFrame(jxs)
ts_pd = pd.DataFrame(ts)
tm_pd = pd.DataFrame(tm)
ve = 'c' if continuous else 'd'
X_pd.to_csv('results/period-doubling_data'+ve+'.csv')
index_pd.to_csv('results/period-doubling_index'+ve+'.csv')
jxs_pd.to_csv('results/period-doubling_jxs'+ve+'.csv')
ts_pd.to_csv('results/period-doubling_ts'+ve+'.csv')
tm_pd.to_csv('results/period-doubling_tm'+ve+'.csv')



################################################################
###  (6) Real and Imaginary Parts of DEJ                     ###
################################################################
print(jxs_c[:time_tp].shape)
print(max_evals[:time_tp].shape)

ls = 25
fig = plt.figure(figsize=(12,13))
ax1 = fig.add_subplot(3,1,1)
ax1.plot((ts/dt)[50:],X_ori[50:,0])
ax1.set_xlim(0,20000)
ax1.tick_params(labelsize=ls)
ax1.set_ylabel(r"$s$",size=ls)

ax2 = fig.add_subplot(3,1,2)
ax2.plot(tm[:time_tp]/dt-0.5*window, jxs_c[:time_tp].real,'ko',markersize=8, label='Ground truth')
ax2.plot(tm[:time_tp]/dt-0.5*window, max_evals[:time_tp,0],'rx',markersize=15, label='Prediction')
ax2.set_xlim(0,20000)
ax2.tick_params(labelsize=ls)
ax2.set_ylabel(r"Real part",size=ls)
plt.legend(fontsize=ls)

ax3 = fig.add_subplot(3,1,3)
ax3.plot(tm[:time_tp]/dt-0.5*window, jxs_c[:time_tp].imag,'ko',markersize=8, label='Ground truth')
ax3.plot(tm[:time_tp]/dt-0.5*window, max_evals[:time_tp,1],'rx',markersize=15, label='Prediction')
ax3.set_xlim(0,20000)
ax3.tick_params(labelsize=ls)
ax3.set_ylabel(r"Imaginary part",size=ls)
ax3.set_xlabel(r"Time",size=ls)

plt.savefig("results/type_PD.pdf")

