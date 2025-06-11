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
from sklearn.linear_model import Ridge
import joblib 
from scipy import sparse
from utils.cubic_spline import CSpline
import argparse
from scipy.optimize import fsolve
from scipy.linalg import inv
from scipy.linalg import qr
from utils.reservoir_model_RC import Reservoir

seed = 0
np.random.seed(seed)
random.seed(seed)


def equ(x, rh):
    g = np.zeros(3)
    g[0] = 10*(x[1]-x[0])
    g[1] = x[0]*(rh-x[2])-x[1]
    g[2] = x[0]*x[1]-8/3*x[2]
    return g

def disLorenz(rho, dt, sigma):
    x0 = np.ones(3)*7 
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


dt = 0.02
tpoints = 20000
ts = torch.arange(tpoints) * dt
# bifurcation rh 24.74
#rho = 10 + 20* torch.exp((ts-ts[-1])/10)
rho = 28*torch.ones(tpoints)
#rho = np.linspace(50,20,tpoints)
method = "euler"
sigma = 0.001 * 0

'''
func = ode1(sigma=0.1)
func.u  = CSpline(ts, rho) 
s0 = torch.rand(3)*10
ts_lorenz = ode.odeint(func, s0, ts, rtol=1e-6, atol=1e-8, method="euler")
X = ts_lorenz[None,:]
'''
ts_lorenz = disLorenz(rho, dt, sigma)
X = ts_lorenz

print("Data successfully generated!")


# ================ RC 超参数设置 ====================
parser = argparse.ArgumentParser() 
parser.add_argument('--n', type=int, default=500)
parser.add_argument('--connectivity', type=float, default=0.05)
parser.add_argument('--spectral_radius', type=float, default=0.85)
parser.add_argument('--input_scaling', type=float, default=0.1)
parser.add_argument('--leak', type=float, default=0.0)
parser.add_argument('--b', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=1e-5)
parser.add_argument('--warm_up', type=int, default=600)
parser.add_argument('--method', type=str, default='euler') 
args = parser.parse_args(args=[])

reservoir = Reservoir(n_internal_units=args.n,
                      spectral_radius=args.spectral_radius,
                      leak=args.leak,
                      connectivity=args.connectivity,
                      input_scaling=args.input_scaling,
                      noise_level=0.0,
                      sigma_b = args.b)
readout = Ridge(alpha=1e-8, fit_intercept=True)
b = reservoir.sigma_b
n = reservoir._n_internal_units
l = reservoir.leak


# ================== RC 多步预测 =====================
ri = reservoir.get_states(X, n_drop=0)

def system_dis(r, tilde_A, tilde_B):
    return (l-1)*r + (1-l) * np.tanh(np.dot(tilde_A,r) + tilde_B[:,0])

def jac_sys_dis(r, tilde_A, tilde_B):
    diag = np.diag((1 - np.tanh(np.dot(tilde_A,r) + tilde_B[:,0])**2))
    return (l-1) * np.eye(len(r)) + (1-l) * np.dot(diag,tilde_A)

def jac_dis(r, tilde_A, tilde_B):
    diag = np.diag((1 - np.tanh(np.dot(tilde_A,r) + tilde_B[:,0])**2))
    return l * np.eye(len(r)) + (1-l) * np.dot(diag,tilde_A)

ds = [200,500,1000,2000,3000,4000]
num = 5
MLEs = np.zeros((len(ds), num))
for dsi in range(len(ds)):
    d = ds[dsi]
    for mlei in range(num):
        inir = np.random.rand()*(1-d/tpoints)
        # train
        nst = int(X.shape[0]*inir)
        ned = int(X.shape[0]*(inir+d/tpoints))
        X_b = ri[nst:ned,:] 
        Y_b = X[nst:ned,:]
        readout.fit(X_b,Y_b)
        #print("Training complete!")
        
        # test
        #steps = 1000
        steps = 2000
        start = ned - steps
        t_pred = ts[start:start+steps]
        rt = np.zeros((steps,args.n)) 
        preds = np.zeros((steps,X.shape[-1]))
        previous_state = ri[start,:][None,:]
        current_input = X[start:start+1]
        for j in range(steps):
            rt[j] = previous_state[0]
            preds[j,:] = current_input[0]
            previous_state = reservoir._compute_next_state(previous_state,current_input)
            current_input = readout.predict(previous_state)
            
        # Calculate the Lyapunov exponent
        Win = reservoir._input_weights
        A = reservoir._internal_weights
        Wout = readout.coef_
        bias = readout.intercept_[:,None]
        tilde_A = A + np.dot(Win,Wout)
        tilde_B = np.dot(Win,bias) + args.b
        
        pl = steps
        #num_lyaps = X.shape[1]
        num_lyaps = 1
        delta, RR = np.linalg.qr(np.random.rand(args.n, num_lyaps)) 
        norm_time = 10
        R_ii = np.zeros((num_lyaps, pl // norm_time)) + 0j
        for i in range(pl):
            delta = jac_dis(rt[i],tilde_A,tilde_B) @ delta
            if i % norm_time == 0:
                QQ,RR = qr(delta, mode='economic')
                delta = QQ
                R_ii[:,i//norm_time] = np.log(np.diag(RR)+0j)
        ler = np.sum(R_ii, axis=1).real / (pl * dt)
        
        print('d =', d, mlei, "Lyapunov exponent:", ler)
        MLEs[dsi,mlei] = ler
    print('Median value', np.median(MLEs[dsi,:]))

# 开始画箱线图
plt.figure(figsize=(12, 6))  # 设置图像大小
plt.boxplot(MLEs.T, labels=ds)  # 创建箱线图，.T 表示矩阵的转置以匹配箱线图的输入格式
plt.ylim(0.2,1.5)

# 设置图表标题和坐标轴标签
plt.xlabel("Window length")
plt.ylabel("MLE")

# 显示图像
plt.show()

MLEs_pd = pd.DataFrame(MLEs)
MLEs_pd.to_csv('results/roubust_lorenz_d_MLE.csv')
plt.savefig("results/roubust_lorenz_d_MLE.pdf")


