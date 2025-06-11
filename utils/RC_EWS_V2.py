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
from scipy import signal

########################################################
# Important Note:
# To significantly reduce the computation time running the analysis including 50 replicates, all results presented in this python-file are based on one time series replicate per mathematical model. 
# Due to process noise (as well as observation error), the RCDI results produced from analyzing one replicate look more variable and change less smoothly. 
# However, the findings based on analyzing one replicate still reveal the same dynamical behaviors qualitatively (e.g., the trend of RCDI).
#########################################################
    
class RC_control(nn.Module):  # myModel
    def __init__(self, Win, A, leak, b, dt):
        super(RC_control, self).__init__()
        self.csp = None
        self.Win = Win
        self.A = A
        self.leak = leak
        self.b = b
        self.gamma = dt
    
    def forward(self, t, r):
        leak = self.leak; Win = self.Win; A = self.A; b = self.b; csp=self.csp; gamma=self.gamma
        p = csp[0].fit(t)
        for i in range(len(csp)-1):
            p = torch.cat((p,csp[i+1].fit(t)),axis=0)
        guy = (leak-1)/gamma * r + (1-leak)/gamma * \
            torch.tanh(torch.mm(Win,p) + torch.mm(A,r) + b)
        return(guy)
    
class RC_autonomy(nn.Module):  # myModel
    def __init__(self, Win, A, leak, b, dt):
        super(RC_autonomy, self).__init__()
        self.readout = None
        self.Win = Win
        self.A = A
        self.leak = leak
        self.b = b
        self.gamma = dt
        
    def forward(self, t, r):
        leak = self.leak; Win = self.Win; A = self.A; b = self.b; readout = self.readout; gamma=self.gamma
        x = torch.tensor(readout.predict(r.t())).t()
        guy = (leak-1)/gamma * r + (1-leak)/gamma * \
            torch.tanh(torch.mm(Win,x) + torch.mm(A,r) + b)
        return(guy)

class RC_EWM:
    def __init__(self, X, ts, window=2000, step=100, args=None, iscontinuous=True, isdetrend=False):
        print("Initialize the model...")

        self.window = window; self.step = step
        self.iscontinuous = iscontinuous
        self.isdetrend = isdetrend
        V = X.shape[-1]
        dt = (ts[-1]-ts[0])/(len(ts)-1)
        self.dt = dt
        self.V = V
        self.X = X
        self.ts = ts
        
        if args==None:
            args = self.Args_default
        self.args = args
        
        if iscontinuous:
            internal_weights = self.initialize_internal_weights(args)
            A = torch.tensor(internal_weights)
            Win = torch.tensor((2.0*np.random.random(size=(args.n, V)) - 1.0)*args.input_scaling)
            self.RCcont = RC_control(Win, A, args.leak, args.b, dt)
            self.RCauto = RC_autonomy(Win, A, args.leak, args.b, dt)
        else:
            reservoir = Reservoir(n_internal_units=args.n,
                                  spectral_radius=args.spectral_radius,
                                  leak=args.leak,
                                  connectivity=args.connectivity,
                                  input_scaling=args.input_scaling,
                                  noise_level=0.0,
                                  sigma_b = args.b)
            self.reservoir = reservoir
        
    def Args_default(self): 
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
        return(args)
        
    def initialize_internal_weights(self, args):
        
        # Generate sparse, uniformly distributed weights.
        internal_weights = sparse.rand(args.n, args.n, density=args.connectivity).todense()

        # Ensure that the nonzero values are uniformly distributed in [-0.5, 0.5]
        internal_weights[np.where(internal_weights > 0)] -= 0.5
        
        # Adjust the spectral radius.
        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        internal_weights /= np.abs(e_max)/args.spectral_radius       
        
        return internal_weights
    
    def find_period_autocorr(self,flo_x,Tmin=100,threshold=0.92):
        V = flo_x.shape[1]
        #fflo_x = flo_x[::-1]
        fflo_x = flo_x
        T = Tmin
        corrs = np.zeros(flo_x.shape[0])
        corrs_error = np.zeros(flo_x.shape[0])+100
        for i in range((fflo_x.shape[0]-2*T)//2):
            corv = []; corv_error = []
            for j in range(V):
                a = fflo_x[:(T+i),j]
                b = fflo_x[(T+i):2*(T+i),j]
                corr = np.corrcoef((a - np.mean(a)) / np.std(a), (b - np.mean(b)) / np.std(b))[0, 1]
                corv.append(corr)
                corr_error = np.sum(np.abs(a-b))
                corv_error.append(corr_error)
            corrs[T+i] = np.mean(corv)
            corrs_error[T+i] = np.mean(corv_error)
        
        sorted_c = np.sort(corrs)
        sorted_e = np.sort(corrs_error)
        peaks, _ = find_peaks(corrs)
        
        ind = 1
        while(ind):
            for peak in peaks:
                p_c = np.searchsorted(sorted_c, peak, side='right') / len(sorted_c)
                p_e = np.searchsorted(sorted_e, corrs_error[peak], side='right') / len(sorted_e)
                if p_c>=threshold and p_e<=1-threshold:
                    period = peak 
                    ind = 0
                    break
            if ind==1:
                period = peaks[0]
                ind = 0
                print("no period!")
                
        return period
    
    
    # continuous version
    def system_con(self, r, tilde_A, tilde_B):
        args = self.args
        dt = self.dt
        return (args.leak-1)/dt * r + (1-args.leak)/dt * np.tanh(np.dot(tilde_A,r) + tilde_B[:,0])

    def jac_con(self, r, tilde_A, tilde_B):
        args = self.args
        dt = self.dt
        diag = np.diag((1 - np.tanh(np.dot(tilde_A,r[:,None]) + tilde_B)**2)[:,0])
        return (args.leak-1)/dt * np.eye(len(r)) + (1-args.leak)/dt * np.dot(diag,tilde_A)
    
    def Ri_conRC(self,tsj,Xsj):
        RCcont = self.RCcont; args = self.args; isdetrend=self.isdetrend
        if isdetrend:
            Xm = signal.detrend(Xsj, axis=0)
            Xj = Xm - Xm[0] + Xsj[0]
            #Xj = (Xj - np.mean(Xj,axis=0))/np.std(Xj,axis=0)
        else:
            Xj = Xsj
        V = Xj.shape[-1]
        csp = []
        for i in range(V):
            csp.append(CSpline(torch.tensor(tsj), torch.tensor(Xj[:,i])))
        RCcont.csp = csp
        r0 = torch.zeros(args.n,1)
        rij = ode.odeint(RCcont, r0, torch.tensor(tsj).double(), rtol=1e-6, atol=1e-8, method=args.method)
        #ri = ode.odeint(RCcont, r0, torch.tensor(ts).double(),rtol=1e-6, atol=1e-8, method=args.method, options={'step_size': 0.001})
        return(rij, Xj)
    
    def get_max_eigenvalue_con(self):
        window = self.window; step = self.step
        X = self.X; ts = self.ts; args = self.args; dt = self.dt
        if not self.isdetrend:
            ri, Xj = self.Ri_conRC(ts,X)
        RCcont = self.RCcont
        readout = Ridge(alpha = args.alpha, fit_intercept=True)
        
        nwin = (len(ts)-args.warm_up-window)//step
        if nwin<=0:
            return(np.zeros((0,2)), np.zeros(0))
        else:
            tm = (np.arange(nwin)*step + window + args.warm_up)*dt + ts[0]
            Evalues_predr = torch.zeros(nwin,args.n,2)
            print("Get_max_eigenvalues...")
            for j in tqdm(range(nwin)):
                if self.isdetrend:
                    tsj = ts[j*step:args.warm_up+j*step+window]
                    Xsj = X[j*step:args.warm_up+j*step+window,:]
                    rij, Xj = self.Ri_conRC(tsj,Xsj)
                    R_b = rij[args.warm_up:,:,0] 
                    Y_b = Xj[args.warm_up:,:]
                else:
                    R_b = ri[args.warm_up+j*step:args.warm_up+j*step+window,:,0] 
                    Y_b = X[args.warm_up+j*step:args.warm_up+j*step+window,:]
                readout.fit(R_b,Y_b)
                Wout = readout.coef_
                bias = readout.intercept_[:,None]
                
                # Calculate the Jacobian matrix of r
                tilde_A = RCcont.A + np.dot(RCcont.Win,Wout)
                tilde_B = np.dot(RCcont.Win,bias) + RCcont.b
                r_ini = R_b[-1]
                r_eq = fsolve(self.system_con, r_ini, fprime=self.jac_con, args=(tilde_A, tilde_B))
                #r_eq = R_b[-1] 
                Jr = torch.tensor(self.jac_con(r_eq, tilde_A, tilde_B))
                evals_predr, evecs_predr = torch.linalg.eig(Jr)
                Evalues_predr[j,:,0] = evals_predr.real
                Evalues_predr[j,:,1] = evals_predr.imag
            
            ind = torch.argmax(Evalues_predr[...,0],axis=-1)
            max_eigenvalues = torch.zeros(nwin,2)
            for j in range(nwin):
                max_eigenvalues[j] = Evalues_predr[j,ind[j]]
                
            return(max_eigenvalues, tm)
    
    def get_max_floquet_con(self):
        window = self.window; step = self.step
        X = self.X; ts = self.ts; args = self.args; dt = self.dt
        if not self.isdetrend:
            ri, Xj = self.Ri_conRC(ts,X)
        RCcont = self.RCcont; RCauto = self.RCauto
        readout = Ridge(alpha = args.alpha, fit_intercept=True) 
        
        nwin = (len(ts)-args.warm_up-window)//step
        tm = (np.arange(nwin)*step + window + args.warm_up)*dt + ts[0]
        max_floquets = np.zeros((nwin,2))
        
        print("Get_max_floquets...")
        for j in tqdm(range(nwin)):
            if self.isdetrend:
                tsj = ts[j*step:args.warm_up+j*step+window]
                Xsj = X[j*step:args.warm_up+j*step+window,:]
                rij, Xj = self.Ri_conRC(tsj,Xsj)
                R_b = rij[args.warm_up:,:,0] 
                Y_b = Xj[args.warm_up:,:]
            else:
                R_b = ri[args.warm_up+j*step:args.warm_up+j*step+window,:,0] 
                Y_b = X[args.warm_up+j*step:args.warm_up+j*step+window,:]
            readout.fit(R_b,Y_b)
            Wout = readout.coef_
            bias = readout.intercept_[:,None]
            tilde_A = RCcont.A + np.dot(RCcont.Win,Wout)
            tilde_B = np.dot(RCcont.Win,bias) + RCcont.b
            
            # Calculate the floquent multiplier
            inte = 2; isaut = True
            if isaut:
                RCauto.readout = readout 
                r0j = R_b[0].unsqueeze(-1)
                tsj = torch.tensor(ts[args.warm_up+j*step:args.warm_up+j*step+window])
                #r0j = R_b[-1].unsqueeze(-1)
                #tsj = torch.tensor(ts[args.warm_up+j*step:args.warm_up+j*step+window]+(window-1)*dt)
                rt = ode.odeint(RCauto, r0j, tsj, rtol=1e-6, atol=1e-8, method=args.method)[...,0]
                flo_r = rt[::inte]
            else:
                flo_r = R_b[::inte]; 
            flo_x = Y_b[::inte]; flo_dt = inte*dt
            period = self.find_period_autocorr(flo_x)
            M = np.identity(args.n)
            for i in range(period):
                r_i = flo_r[-period+i,:]
                #r_i = flo_r[i,:]
                Jr = torch.tensor(self.jac_con(r_i, tilde_A, tilde_B))
                #M = Jr @ M * flo_dt + M
                M = expm(Jr*flo_dt) @ M
            
            evals_predm, evecs_predm = eig(M)
            ind = np.argmax(np.abs(evals_predm))
            val = evals_predm[ind]
            print("period:", period*flo_dt, period)
            print("floquet:", val)
            max_floquets[j,0] = val.real
            max_floquets[j,1] = val.imag
        return(max_floquets, tm)
    
    def get_max_lyapunov_con(self):
        window = self.window; step = self.step
        X = self.X; ts = self.ts; args = self.args; dt = self.dt
        if not self.isdetrend:
            ri, Xj = self.Ri_conRC(ts,X)
        RCcont = self.RCcont; RCauto = self.RCauto
        readout = Ridge(alpha = args.alpha, fit_intercept=True) 
        
        nwin = (len(ts)-args.warm_up-window)//step
        tm = (np.arange(nwin)*step + window + args.warm_up)*dt + ts[0]
        max_lyapunovs = np.zeros(nwin)
        
        print("Get_max_lyapunovs...")
        for j in tqdm(range(nwin)):
            if self.isdetrend:
                tsj = ts[j*step:args.warm_up+j*step+window]
                Xsj = X[j*step:args.warm_up+j*step+window,:]
                rij, Xj = self.Ri_conRC(tsj,Xsj)
                R_b = rij[args.warm_up:,:,0] 
                Y_b = Xj[args.warm_up:,:]
            else:
                R_b = ri[args.warm_up+j*step:args.warm_up+j*step+window,:,0] 
                Y_b = X[args.warm_up+j*step:args.warm_up+j*step+window,:]
            readout.fit(R_b,Y_b)
            Wout = readout.coef_
            bias = readout.intercept_[:,None]
            tilde_A = RCcont.A + np.dot(RCcont.Win,Wout)
            tilde_B = np.dot(RCcont.Win,bias) + RCcont.b
            
            # Calculate the Lyapunov exponent
            pl = window
            RCauto.readout = readout 
            r0j = R_b[0].unsqueeze(-1)
            tsj = torch.tensor(ts[args.warm_up+j*step:args.warm_up+j*step+pl])
            rt = ode.odeint(RCauto, r0j, tsj, rtol=1e-6, atol=1e-8, method=args.method)[...,0]
            num_lyaps = X.shape[1]
            #num_lyaps = 1
            delta, RR = np.linalg.qr(np.random.rand(args.n, num_lyaps)) 
            norm_time = 10
            R_ii = np.zeros((num_lyaps, pl // norm_time)) + 0j
            for i in range(pl):
                delta = self.jac_con(rt[i],tilde_A,tilde_B) @ delta * dt + delta
                if i % norm_time == 0:
                    QQ,RR = qr(delta, mode='economic')
                    delta = QQ
                    R_ii[:,i//norm_time] = np.log(np.diag(RR)+0j)
            ler = np.sum(R_ii, axis=1).real / (pl * dt)
            max_lyapunovs[j] = np.max(ler)
        return(max_lyapunovs, tm)
    
    
    # discrete version
    def system_dis(self, r, tilde_A, tilde_B):
        l = self.args.leak
        return (l-1)*r + (1-l) * np.tanh(np.dot(tilde_A,r) + tilde_B[:,0])
    
    def jac_sys_dis(self, r, tilde_A, tilde_B):
        l = self.args.leak
        diag = np.diag((1 - np.tanh(np.dot(tilde_A,r) + tilde_B[:,0])**2))
        return (l-1) * np.eye(len(r)) + (1-l) * np.dot(diag,tilde_A)

    def jac_dis(self, r, tilde_A, tilde_B):
        l = self.args.leak
        diag = np.diag((1 - np.tanh(np.dot(tilde_A,r) + tilde_B[:,0])**2))
        return l * np.eye(len(r)) + (1-l) * np.dot(diag,tilde_A)
    
    '''
    def Ri_disRC(self):
        X = self.X
        ri = self.reservoir.get_states(X, n_drop=0)
        return(ri)
    '''
    
    def Ri_disRC(self,Xsj):
        isdetrend = self.isdetrend
        if isdetrend:
            Xm = signal.detrend(Xsj, axis=0)
            Xj = Xm - Xm[0] + Xsj[0]
            #Xj = (Xj - np.mean(Xj,axis=0))/np.std(Xj,axis=0)
        else:
            Xj = Xsj
        ri = self.reservoir.get_states(Xj, n_drop=0)
        return(ri,Xj)
    
    def get_max_eigenvalue_dis(self):
        window = self.window; step = self.step
        isdetrend = self.isdetrend; Ri_disRC = self.Ri_disRC
        X = self.X; ts = self.ts; args = self.args; dt = self.dt
        ri, Xj = Ri_disRC(X)
        Win = self.reservoir._input_weights
        A = self.reservoir._internal_weights
        
        readout = Ridge(alpha = args.alpha, fit_intercept=True)
        nwin = (len(ts)-args.warm_up-window)//step
        if nwin<=0:
            return(np.zeros((0,2)), np.zeros(0))
        else:
            tm = (np.arange(nwin)*step + window + args.warm_up)*dt + ts[0]
            Evalues_predr = torch.zeros(nwin,args.n,2)
            print("Get_max_eigenvalues...")
            for j in tqdm(range(nwin)):
                if isdetrend:
                    Xsj = X[j*step:args.warm_up+j*step+window,:]
                    rij, Xj = Ri_disRC(Xsj)
                    R_b = rij[args.warm_up:-1,:] 
                    Y_b = Xj[args.warm_up:,:]
                else:
                    R_b = ri[args.warm_up+j*step:args.warm_up+j*step+window,:] 
                    Y_b = X[args.warm_up+j*step:args.warm_up+j*step+window,:]
                    
                readout.fit(R_b,Y_b)
                
                Wout = readout.coef_
                bias = readout.intercept_[:,None]
                
                # Calculate the Jacobian matrix of r
                tilde_A = torch.tensor(A + np.dot(Win,Wout))
                tilde_B = np.dot(Win,bias)+args.b
                r_ini = R_b[-1]
                r_eq = fsolve(self.system_dis, r_ini, fprime=self.jac_sys_dis, args=(tilde_A, tilde_B))
                #r_eq = R_b[-1]
                Jr = torch.tensor(self.jac_dis(r_eq, tilde_A, tilde_B))
                evals_predr, evecs_predr = torch.linalg.eig(Jr)
                Evalues_predr[j,:,0] = evals_predr.real
                Evalues_predr[j,:,1] = evals_predr.imag
    
            ind = torch.argmax((Evalues_predr[...,0]**2+Evalues_predr[...,1]**2),axis=-1)
            max_eigenvalues = torch.zeros(nwin,2)
            for j in range(nwin):
                max_eigenvalues[j] = Evalues_predr[j,ind[j]]
                
            return(max_eigenvalues, tm)
    
    def get_max_floquet_dis(self, Tmin=100, threshold=0.95):
        window = self.window; step = self.step
        isdetrend = self.isdetrend; Ri_disRC = self.Ri_disRC
        X = self.X; ts = self.ts; args = self.args; dt = self.dt
        ri, Xj = self.Ri_disRC(X)
        reservoir = self.reservoir
        Win = reservoir._input_weights
        A = reservoir._internal_weights
        
        readout = Ridge(alpha = args.alpha, fit_intercept=True) 
        nwin = (len(ts)-args.warm_up-window)//step
        tm = (np.arange(nwin)*step + window + args.warm_up)*dt + ts[0]
        max_floquets = np.zeros((nwin,2))
        print("Get_max_floquets...")
        for j in tqdm(range(nwin)):
            if isdetrend:
                Xsj = X[j*step:args.warm_up+j*step+window,:]
                rij, Xj = Ri_disRC(Xsj)
                R_b = rij[args.warm_up:-1,:] 
                Y_b = Xj[args.warm_up:,:]
            else:
                R_b = ri[args.warm_up+j*step:args.warm_up+j*step+window,:] 
                Y_b = X[args.warm_up+j*step:args.warm_up+j*step+window,:]
            readout.fit(R_b,Y_b)
            
            Wout = readout.coef_
            bias = readout.intercept_[:,None]
            tilde_A = A + np.dot(Win,Wout)
            tilde_B = np.dot(Win,bias) + args.b
            
            # Calculate the floquent multiplier            
            inte = 1; isaut = False
            if isaut:
                rt = np.zeros((window,args.n)) 
                previous_state = ri[args.warm_up+j*step,:][None,:]
                #previous_state = ri[args.warm_up+j*step+window,:][None,:]
                for jw in range(window):
                    rt[jw] = previous_state[0]
                    current_input = readout.predict(previous_state)
                    previous_state = reservoir._compute_next_state(previous_state,current_input)
                flo_r = rt[::inte]
            else:
                flo_r = R_b[::inte]
            #flo_x = Y_b[::inte]
            flo_x = readout.predict(flo_r)
            flo_dt = inte*dt
            period = self.find_period_autocorr(flo_x, Tmin=Tmin, threshold=threshold)
            #period = self.find_period_error(flo_x, Tmin=Tmin, threthod=threthod)
            M = np.identity(args.n)
            for i in range(period):
                #r_i = flo_r[-period+i,:]
                #r_i = flo_r[-i-1,:]
                r_i = flo_r[i,:]
                Jr = torch.tensor(self.jac_dis(r_i, tilde_A, tilde_B))
                M = Jr @ M 
            
            evals_predm, evecs_predm = eig(M)
            ind = np.argmax(np.abs(evals_predm))
            val = evals_predm[ind]
            print("period:", period*flo_dt, period)
            print("floquet:", val)
            max_floquets[j,0] = np.abs(val)
            max_floquets[j,1] = val.imag
        return(max_floquets, tm)

    def get_max_lyapunov_dis(self):
        window = self.window; step = self.step
        X = self.X; ts = self.ts; args = self.args; dt = self.dt
        ri, Xj = self.Ri_disRC(X)
        Win = self.reservoir._input_weights
        A = self.reservoir._internal_weights
        
        readout = Ridge(alpha = args.alpha, fit_intercept=True) 
        nwin = (len(ts)-args.warm_up-window)//step
        tm = (np.arange(nwin)*step + window + args.warm_up)*dt + ts[0]
        max_lyapunovs = np.zeros(nwin)
        print("Get_max_lyapunovs...")
        for j in tqdm(range(nwin)):
            R_b = ri[args.warm_up+j*step:args.warm_up+j*step+window,:] 
            Y_b = X[args.warm_up+j*step:args.warm_up+j*step+window,:]
            readout.fit(R_b,Y_b)
            
            Wout = readout.coef_
            bias = readout.intercept_[:,None]
            tilde_A = A + np.dot(Win,Wout)
            tilde_B = np.dot(Win,bias) + args.b
            
            # Calculate the Lyapunov exponent
            pl = window
            rt = np.zeros((pl,args.n)) 
            previous_state = ri[args.warm_up+j*step,:][None,:]
            for jpl in range(pl):
                rt[jpl] = previous_state[0]
                current_input = readout.predict(previous_state)
                previous_state = self.reservoir._compute_next_state(previous_state,current_input)
            
            num_lyaps = X.shape[1]
            #num_lyaps = 1
            delta, RR = np.linalg.qr(np.random.rand(args.n, num_lyaps)) 
            norm_time = 10
            R_ii = np.zeros((num_lyaps, pl // norm_time)) + 0j
            for i in range(pl):
                delta = self.jac_dis(rt[i],tilde_A,tilde_B) @ delta
                if i % norm_time == 0:
                    QQ,RR = qr(delta, mode='economic')
                    delta = QQ
                    R_ii[:,i//norm_time] = np.log(np.diag(RR)+0j)
            ler = np.sum(R_ii, axis=1).real / (pl * dt)
            max_lyapunovs[j] = np.max(ler)
        return(max_lyapunovs, tm)
    
    def calculate(self, index='max_eigenvalue'):
        
        if self.iscontinuous:
            if index == 'max_eigenvalue':
                metrics, tm = self.get_max_eigenvalue_con()
            elif index == 'max_floquet':
                metrics, tm = self.get_max_floquet_con()
            elif index == 'max_lyapunov':
                metrics, tm = self.get_max_lyapunov_con()
            else:
                print("Set the index to the default value of 'max_eigenvalue'.")
        else:
            if index == 'max_eigenvalue':
                metrics, tm = self.get_max_eigenvalue_dis()
            elif index == 'max_floquet':
                metrics, tm = self.get_max_floquet_dis()
            elif index == 'max_lyapunov':
                metrics, tm = self.get_max_lyapunov_dis()
            else:
                print("Set the index to the default value of 'max_eigenvalue'.")            
    
        return(metrics, tm)
