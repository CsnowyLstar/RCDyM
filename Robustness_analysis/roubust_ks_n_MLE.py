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

def gen_ks(h,T,tpoints,ds,Nx,deltas,sigma,u,x,N):
    xs = x[::int(N//Nx)]
    tmax = T
    nmax = round(tmax/h)
     
    Xs = np.zeros((tpoints,Nx))#Used to record the status at all times
    nplt = int((tmax/tpoints)/h)
    ux = u[np.arange(0,len(u),int(N/Nx))]
    v = np.fft.fft(u)
    uu = np.array([ux])
    tt = 0
    
    for n in range(1, nmax): 
        d = ds[n-1]
        k = np.transpose(np.conj(np.concatenate((np.arange(0, N/2), np.array([0]), np.arange(-N/2+1, 0)))))*(2*np.pi/d) 
        
        g = -0.5j*k
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
            #noise = (np.random.rand(N)-0.5)*2*sigma
            noise = np.random.randn(N)*sigma
            u = u + noise
            v = np.fft.fft(u)
            ux = u[np.arange(0,len(u),int(N/Nx))]
            uu = np.append(uu, np.array([ux]), axis=0)
            tt = np.hstack((tt, n))
    if True not in np.isnan(uu):
        Xs[:,:] = uu
    return(Xs,xs,u)

'''
dt = 0.25
h = 0.05
tpoints = 20000
T = int(dt*tpoints)
ts = np.arange(tpoints) * dt
#deltas = np.linspace(1.1,0.94,int(T/h))
deltas = np.ones(int(T/h))

F_bifurcation = deltas[::int(T/dt/tpoints)]
method = "euler"
sigma = 0.00

ds = np.ones(int(T/h)) * 33.9
#ds = np.linspace(50,40,int(T/h))
    
Nx = 64; N = 64
x = ds[0]*np.transpose(np.conj(np.arange(-N/2+1, N/2+1))) / N
u = np.sin(x/4)*10
Nx = 64; N = 64
x = ds[0]*np.transpose(np.conj(np.arange(-N/2+1, N/2+1))) / N
u = np.sin(x/4)*10

nu = 500
X_ini,xs_ini,u = gen_ks(h,h*nu,nu,np.ones(nu)*ds[0],Nx,np.ones(nu)*deltas[0],sigma,u,x,N)
X_ini,xs_ini,u = gen_ks(h,T,tpoints,ds,Nx,deltas,sigma,u,x,N)

X = X_ini[:,::1].copy()
xs = xs_ini[::1].copy()


'''
dt =  0.25
h = 0.25
tpoints = 20000
T = int(dt*tpoints)
ts = np.arange(tpoints) * dt
deltas = np.ones(int(T/h))

d = 22
Nx = 64; N = 64
x = d*np.transpose(np.conj(np.arange(-N/2+1, N/2+1))) / N
X = pd.read_csv('dataset/KS.csv', header=None).values[:tpoints]
xs = x.copy()


print("Data successfully generated!")


# ================ RC 超参数设置 ====================
def system_dis(r, tilde_A, tilde_B):
    return (l-1)*r + (1-l) * np.tanh(np.dot(tilde_A,r) + tilde_B[:,0])

def jac_sys_dis(r, tilde_A, tilde_B):
    diag = np.diag((1 - np.tanh(np.dot(tilde_A,r) + tilde_B[:,0])**2))
    return (l-1) * np.eye(len(r)) + (1-l) * np.dot(diag,tilde_A)

def jac_dis(r, tilde_A, tilde_B):
    diag = np.diag((1 - np.tanh(np.dot(tilde_A,r) + tilde_B[:,0])**2))
    return l * np.eye(len(r)) + (1-l) * np.dot(diag,tilde_A)

parser = argparse.ArgumentParser() 
parser.add_argument('--n', type=int, default=2000)
parser.add_argument('--connectivity', type=float, default=0.001)
parser.add_argument('--spectral_radius', type=float, default=0.6)
parser.add_argument('--input_scaling', type=float, default=0.5)
parser.add_argument('--leak', type=float, default=0.0)
parser.add_argument('--b', type=float, default=2.0)
parser.add_argument('--alpha', type=float, default=1e-4)
parser.add_argument('--warm_up', type=int, default=600)
parser.add_argument('--method', type=str, default='euler') 
args = parser.parse_args(args=[])

b = args.b
l = args.leak

#ns = [100, 200, 400, 800, 1600, 3200]
#cs = [0.01, 0.01, 0.01, 0.002, 0.001, 0.001]
ns = [50]
cs = [0.05]
num = 50
MLEs = np.zeros((len(ns), num))
d = 10000

for nsi in range(len(ns)):
    #nsi = 3
    n = ns[nsi]
    args.n = n
    args.connectivity = cs[nsi]
    reservoir = Reservoir(n_internal_units=args.n,
                          spectral_radius=args.spectral_radius,
                          leak=args.leak,
                          connectivity=args.connectivity,
                          input_scaling=args.input_scaling,
                          noise_level=0.0,
                          sigma_b = args.b,
                          isks = True)
    readout = Ridge(alpha=args.alpha, fit_intercept=True)
    ri = reservoir.get_states(X, n_drop=0)
    
    for mlei in range(num):
        inir = int(np.random.rand()*(1-d/tpoints-args.warm_up)) + args.warm_up
        # train
        nst = inir
        ned = inir + d
        X_b = ri[nst:ned,:] 
        Y_b = X[nst:ned,:]
        readout.fit(X_b,Y_b)
        #print("Training complete!")
        
        # test
        steps = 500
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
        
        print('n =', n, mlei, "Lyapunov exponent:", ler)
        MLEs[nsi,mlei] = ler
    print('Median value', np.median(MLEs[nsi,:]))



# 开始画箱线图
plt.figure(figsize=(12, 6))  # 设置图像大小
boxplot = plt.boxplot(MLEs.T, labels=ns)  # 创建箱线图，并获取返回值
medians = [median.get_ydata()[0] for median in boxplot['medians']]
# 在每个箱线图上方标注中位数
for index, median in enumerate(medians):
    plt.text(index + 1, median, f'{median:.3f}', # index + 1因为箱线图从1开始计数
             verticalalignment='center', # 垂直对齐方式
             horizontalalignment='center', # 水平对齐方式
             fontsize=15, # 字体大小
             color='red') # 文本颜色

plt.xlabel("Window length")
plt.ylabel("MLE")
plt.show()

MLEs_pd = pd.DataFrame(MLEs)
MLEs_pd.to_csv('results/roubust_ks_n_MLE.csv')
plt.savefig("results/roubust_ks_n_MLE.pdf")


