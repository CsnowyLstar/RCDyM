
################################################################
###  (4) Refined low-order polynomial                        ###
################################################################
time = (tm_dis-ts[0])/dt-0.5*window
time_tp = len(time) - 1
for ti in range(len(time)):
    if time[ti]>int(tp/dt):
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
fig = plt.figure(figsize=(20,18))
ls = 40; ms = 15
ax1 = fig.add_subplot(3,1,1)
ax1.plot(ts,X,'b')
ax1.tick_params(labelsize=ls)
ax1.tick_params(axis='y', colors='b')
ax1.set_ylabel(r"$s$",size=ls,color='b')
ax1.set_xticks([])
ax1.set_xlim(ts[0],ts[-1])

ax3 = fig.add_subplot(3,1,2)
ind = 0
isrealpart = False
for ni in range(num):
    tm_dis = tm_diss[ni]; DEJ_dis = DEJ_diss[ni]
    i = next((i for i in range(len(tm_dis)) if tm_dis[i] > tp), None)
    DEJ_di = DEJ_dis[:i,0] if isrealpart else np.sqrt(DEJ_dis[:i,0]**2 + DEJ_dis[:i,1]**2)
    ax3.plot(tm_dis[:i],DEJ_di,'m*',markersize=ms)
    if ind == 0:
        xs = tm_dis[:i]; ys=DEJ_di; ind = 1
    else:
        xs = np.concatenate((xs,tm_dis[:i]),axis=0)
        ys = np.concatenate((ys,DEJ_di),axis=0)

degree = 1; alpha = 0.01
coefficients = monotonic_polyfit_nonnegative(xs, ys, degree, alpha)
poly_func = np.poly1d(coefficients[::-1])
adv = 15
obt = time_tp - adv
cst = obt
tm_fine = np.linspace(min(tm_dis[obt-cst:obt]), max(tm_dis[obt-cst:obt]), 500)
max_evals_fitted = poly_func(tm_fine)
tm_extra = np.linspace(max(tm_dis[obt-cst:obt]), tp+0, 500)
extrapolation_pre = poly_func(tm_extra)
ax3.plot(tm_fine, max_evals_fitted, 'b-', linewidth=10, alpha=0.6)
ax3.plot(tm_extra, extrapolation_pre, 'r-', linewidth=10, alpha=0.6)
ax3.tick_params(labelsize=ls)
ax3.tick_params(axis='y', colors='m')
ax3.set_ylabel("RCDyM_dis",size=ls,color='m')
ax3.set_xlim(ts[0],ts[-1])
ax3.set_xlabel('Time',size=ls)

moln = 'real_data1'
X_pd = pd.DataFrame(X)
index_pd = pd.DataFrame(np.sqrt(DEJ_dis[:,0]**2 + DEJ_dis[:,1]**2))
ts_pd = pd.DataFrame(ts)
tm_pd = pd.DataFrame(tm_dis)
X_pd.to_csv('results/X_'+moln+'_RCDyM_.csv')
index_pd.to_csv('results/index_'+moln+'_RCDyM_.csv')
ts_pd.to_csv('results/ts_'+moln+'_RCDyM_.csv')
tm_pd.to_csv('results/tm_'+moln+'_RCDyM_.csv')

plt.savefig("results/r1.png")


