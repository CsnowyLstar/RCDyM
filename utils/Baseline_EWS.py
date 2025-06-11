import numpy as np
import torch
import torch.nn as nn 
from sklearn.linear_model import Ridge
from scipy import sparse
import argparse
from utils.cubic_spline import CSpline
from utils.reservoir_model_RC import Reservoir
import torchdiffeq as ode
from scipy.optimize import fsolve
from tqdm import tqdm
from scipy.linalg import eig, qr, expm
from scipy.signal import find_peaks
from scipy import signal,stats
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import load_model
import warnings


def s_map(Xi, E=1, tau=1, theta=1.0, K=None):

    def embed(x, E=E, tau=tau):
        N = x.shape[0]
        if N < (E - 1) * tau + 1:
            raise ValueError("Time series is too short for the given embedding parameters.")
        emb_matrix = x[:N-E*tau]
        for i in range(E-1):
            emb_matrix = np.concatenate((emb_matrix,x[(i+1)*tau:N-E*tau+(i+1)*tau]),axis=-1)
        return emb_matrix

    def weights(d, theta):
        return np.exp(-theta * d)
    
    X_embedded = embed(Xi, E, tau)
    Y_target = Xi[E*tau:]          

    if K is None:
        K = len(X_embedded) - 1
    
    knn = NearestNeighbors(n_neighbors=K+1).fit(X_embedded)
    distances, indices = knn.kneighbors(X_embedded)

    i = X_embedded.shape[0]-1
    nbrs_inds = indices[i, :]
    nbrs_weights = weights(distances[i, :], theta)
    W = np.diag(nbrs_weights)
    C = np.linalg.lstsq(
        np.dot(W, X_embedded[nbrs_inds]), 
        np.dot(W, Y_target[nbrs_inds]), rcond=None)[0]
    
    J = np.copy(C.transpose())
    e = np.identity(C.shape[0])
    J = np.concatenate((J, e[:C.shape[0]-C.shape[1]]),axis=0)
    return(J)
    
class BL_EWS:
    def __init__(self, X, ts, window=2000, step=100, args=None, iscontinuous=True, isdetrend=False):
        print("Initialize the model...")
        warnings.filterwarnings('ignore', category=DeprecationWarning)

        self.window = window; self.step = step
        self.iscontinuous = iscontinuous
        self.isdetrend = isdetrend
        V = X.shape[-1]
        dt = (ts[-1]-ts[0])/(len(ts)-1)
        self.dt = dt
        self.V = V
        self.X = X
        self.ts = ts
        self.args = args
    
    def get_variance(self):
        window = self.window; step = self.step
        X = self.X; ts = self.ts; dt = self.dt
        args = self.args
        nwin = (len(ts)-args.warm_up-window)//step
        tm = (np.arange(nwin)*step + window + args.warm_up)*dt + ts[0]
        print("Get_variance...")
        
        if nwin<=0:
            return(np.zeros((0)), np.zeros(0))
        else:
            metrics = torch.zeros(nwin,1)
            vari = np.zeros((nwin,X.shape[-1]))
            for i in tqdm(range(nwin)):
                for j in range(X.shape[-1]):
                    Xij = X[i*step+args.warm_up:i*step+window+args.warm_up,j]
                    vari[i,j] = np.var(Xij)
            metrics = np.mean(vari, axis=-1)
                
            return(metrics, tm)
    
    def get_autocorrelation(self):
        window = self.window; step = self.step
        X = self.X; ts = self.ts; dt = self.dt
        args = self.args
        nwin = (len(ts)-args.warm_up-window)//step
        tm = (np.arange(nwin)*step + window + args.warm_up)*dt + ts[0]
        print("Get_autocorrelation...")
        
        if nwin<=0:
            return(np.zeros((0)), np.zeros(0))
        else:
            metrics = torch.zeros(nwin,1)
            auto = np.zeros((nwin,X.shape[-1]))
            for i in tqdm(range(nwin)):
                for j in range(X.shape[-1]):
                    Xij = X[i*step+args.warm_up:i*step+window+args.warm_up,j]
                    data = pd.Series(Xij)
                    auto[i,j] = data.autocorr(lag=1)
            metrics = np.mean(auto, axis=-1)
            return(metrics, tm)
    
    def get_skewness(self):
        window = self.window; step = self.step
        X = self.X; ts = self.ts; dt = self.dt
        args = self.args
        nwin = (len(ts)-args.warm_up-window)//step
        tm = (np.arange(nwin)*step + window + args.warm_up)*dt + ts[0]
        print("Get_skewness...")
        
        if nwin<=0:
            return(np.zeros((0)), np.zeros(0))
        else:
            metrics = torch.zeros(nwin,1)
            skew = np.zeros((nwin,X.shape[-1]))
            for i in tqdm(range(nwin)):
                for j in range(X.shape[-1]):
                    Xij = X[i*step+args.warm_up:i*step+window+args.warm_up,j]
                    skew[i,j] = stats.skew(Xij)
            metrics = np.abs(np.mean(skew, axis=-1))
            return(metrics, tm)
    
    def get_deep_learning(self):
        window = self.window
        step = self.step
        X = self.X; ts = self.ts; dt = self.dt
        args = self.args
        nwin = (len(ts)-args.warm_up-window)//step
        tm = (np.arange(nwin)*step + window + args.warm_up)*dt + ts[0]
        print("Get_deep_learning...")
        
        if nwin<=0:
            return(np.zeros((0,4)), np.zeros(0))
        else:
            # Load in specific DL classifier
            metrics = torch.zeros(nwin,1)
            kk = 1; model_type = 2
            ts_len = 1500 if window>1500 else 500
            metrics_total = np.zeros((10,nwin,4))                             
            for kk in np.arange(1,11):
                model_name = 'utils/best_models/best_model_{}_{}_len{}.pkl'.format(kk,model_type,ts_len)
                model = load_model(model_name)
                deep = np.zeros((nwin,X.shape[-1],4))
                for i in tqdm(range(nwin)):
                    for j in range(X.shape[-1]):
                        Xij = X[i*step+args.warm_up:i*step+window+args.warm_up,j] 
                        Xij_norm = (Xij/np.mean(np.abs(Xij)))[None,:,None]
                        if window>ts_len:
                            Xij_norm = Xij_norm[:,-ts_len:,:]
                        elif window<ts_len:
                            xadd = np.zeros((1,ts_len-window,1))
                            Xij_norm = np.concatenate((Xij_norm,xadd),axis=1)
                        y_pred = model.predict(Xij_norm)
                        deep[i,j] = y_pred[0]
                metrics = np.mean(deep, axis=1)
                metrics_total[kk-1] = metrics
            metrics = np.mean(metrics_total, axis=0)
            return(metrics, tm)
    
    def get_DEV(self, E=2, tau=1, theta=0.5, isabs=True):
        window = self.window; step = self.step
        X = self.X; ts = self.ts; dt = self.dt
        args = self.args
        nwin = (len(ts)-args.warm_up-window)//step
        tm = (np.arange(nwin)*step + window + args.warm_up)*dt + ts[0]
        print("Get_DEV...") 
        
        if nwin<=0:
            return(np.zeros((0)), np.zeros(0))
        else:
            metrics = np.zeros(nwin)
            for i in tqdm(range(nwin)):
                Xi = X[i*step+args.warm_up:i*step+window+args.warm_up,:] 
                Xi_norm = (Xi - np.mean(Xi,axis=0))/np.std(Xi,axis=0)
                #J = s_map(Xi_norm, E=E, tau=tau, theta=theta)
                J = s_map(Xi_norm[:,:1], E=E, tau=tau, theta=theta)
                evals_pred, evecs_pred = eig(J)
                ind = np.argmax(np.abs(evals_pred))
                if isabs:
                    metrics[i] = np.abs(evals_pred[ind])
                else:
                    metrics[i] = np.real(evals_pred[ind])
                
            return(metrics, tm)
    
    def calculate(self, index='Variance'):
        
        if index == 'Variance':
            metrics, tm = self.get_variance()
        elif index == 'autocorrelation':
            metrics, tm = self.get_autocorrelation()
        elif index == 'skewness':
            metrics, tm = self.get_skewness()
        elif index == 'deep_learning':
            metrics, tm = self.get_deep_learning()
        elif index == 'DEV':
            metrics, tm = self.get_DEV()
        else:
            print("Set the index to the default value of 'max_eigenvalue'.")      
                
        return(metrics, tm)
    
    
    
    