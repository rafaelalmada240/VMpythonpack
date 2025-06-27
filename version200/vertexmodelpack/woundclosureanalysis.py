############################################################# Closure analysis #########################################################################
import numpy as np
from vertexmodelpack import dynamicalanalysis as dyan
from sklearn.linear_model import LinearRegression

#Global variables
LOG_EPS = 1e-16

def ensemble_avg_per_norm(L_list, LW_list, Tsim, p_array, tissues_list):
    """
    Calculates normalized perimeter averages and standard deviations across tissues.
    Inputs:
        L_list: List of length parameters
        LW_list: List of width parameters  
        Tsim: Simulation time length
        p_array: Perimeter array (shape: tissues×L×LW×time)
        tissues_list: List of tissue identifiers
    Outputs:
        avg_p: Average normalized perimeter (shape: L×LW×time)
        std_p: Standard deviation of normalized perimeter
        p_norm: Normalized perimeter values (per tissue)
    """
    avg_p = np.zeros((len(L_list),len(LW_list),Tsim))
    p_norm = np.empty_like(p_array)
    std_p = np.zeros((len(L_list),len(LW_list),Tsim))
    for i in range(len(L_list)):
        for j in range(len(LW_list)):
            
            #for l in range(len(tissues_list)):
                #p_norm[l,i,j] = p_array[l,i,j]/p_array[l,i,j,0]
            p_norm[:,i,j] = p_array[:,i,j]/p_array[:,i,j,0:1]
            avg_p[i,j] = np.mean(p_norm[:,i,j],0); std_p[i,j] = np.std(p_norm[:,i,j],0)
    return avg_p, std_p, p_norm

def ensemble_avg_t(L_list,LW_list, tissues_list,t_array, wound_sizes):
    """
    Computes average closure times normalized by wound size.
    Inputs:
        L_list: List of length parameters
        LW_list: List of width parameters
        tissues_list: List of tissue identifiers  
        t_array: Time array (shape: tissues×L×LW)
        wound_sizes: List of wound sizes per tissue
    Output:
        avg_t: Average normalized closure times (shape: L×LW)
    """
    avg_t = np.empty_like(t_array[0])
    w_list = [float(w) for w in wound_sizes]
    for i in range(len(L_list)):
        for j in range(len(LW_list)):
            avg_t[i,j] = int(np.mean(t_array[:,i,j]/(w_list[:len(tissues_list)])))
    return avg_t
        
def ensemble_avg_rho(L_list,LW_list, Tsim, p_array, a_array, tissues_list):
    """
    Calculates density-like metric (p/sqrt(a)) averages.
    Inputs:
        L_list: List of length parameters
        LW_list: List of width parameters
        Tsim: Simulation time length  
        p_array: Perimeter array
        a_array: Area array
        tissues_list: List of tissue identifiers
    Outputs:
        avg_r: Average rho values (shape: L×LW×time)
        r_norm: Normalized rho values (per tissue)
    """
    avg_r = np.zeros((len(L_list),len(LW_list),Tsim))
    r_norm = np.empty_like(p_array)
    for i in range(len(L_list)):
        for j in range(len(LW_list)):
            for l in range(len(tissues_list)):
                r_norm[l,i,j] = p_array[l,i,j]/a_array[l,i,j]**0.5
            avg_r[i,j] = np.mean(r_norm[:,i,j],0); 
    return avg_r, r_norm

def recoil_line_par_space(opt_array, Ntissues, k, p_norm, dt_list):
    """
    Identifies recoil events in parameter space along specific lines.
    Inputs:
        opt_array: Array of optimal times
        Ntissues: Number of tissues
        k: Parameter space line index
        p_norm: Normalized perimeter array
        dt_list: Time step list
    Outputs:
        opt_list1: Recoil times meeting criteria
        popl1: Corresponding perimeter values at recoil
    """
    opt_list1 = []
    popl1 = []
    for tissue in range(Ntissues):
        list_op = opt_array[tissue][[k-i for i in range(k+1)],[i for i in range(k+1)]]
        loc1  = list(set(np.where(np.array(list_op)<0.5)[0]).intersection(np.where(np.array(list_op)>0.)[0]))
        opt_list1.extend(list_op[loc1])
        p_list = p_norm[tissue][[k-i for i in range(k+1)],[i for i in range(k+1)],[int(list_op[i]/dt_list) for i in range(k+1)]]
        popl1.extend(p_list[loc1])
    return opt_list1, popl1


def nsides_par_space(avg_r):
    """
    Calculates number of sides based on final rho values.
    Input:
        avg_r: Average rho values array
    Output:
        n_array: Array of predicted side numbers
    """
    p0_array = (avg_r[:,:,-1])
    n_array = np.zeros(p0_array.shape)
    for i in range(p0_array.shape[0]):
        for j in range(p0_array.shape[1]):
            n_array[i,j] = dyan.n_shape(p0_array[i,j])
    return n_array


def timescales_per_rho_par_space(avg_r,avg_p, L_list,LW_list, Nsim,dt_list,Ncutoff):
    """
    Computes timescales from logarithmic fits of rho and perimeter.
    Inputs:
        avg_r: Average rho values
        avg_p: Average perimeter values  
        L_list: List of length parameters
        LW_list: List of width parameters
        Nsim: Number of simulation steps
        dt_list: Time steps
        Ncutoff: Cutoff index for fitting
    Outputs:
        4 arrays of fitted timescales and intercepts
    """
    y_array = np.nan_to_num(1/avg_r)
    tr1_scale = np.zeros((len(LW_list),len(L_list)))
    tr2_scale = np.zeros((len(LW_list),len(L_list)))

    cr1_scale = np.zeros((len(LW_list),len(L_list)))
    cr2_scale = np.zeros((len(LW_list),len(L_list)))

    x_array = np.linspace(0,Nsim*dt_list,Nsim)
    x1 = x_array[Ncutoff:]
    x2 = x_array[Ncutoff:]
    for i in range(len(L_list)):
        for j in range(len(LW_list)):
            y1 = np.log(y_array[i,j,Ncutoff:])
            
            y2 = np.log(avg_p[i,j,Ncutoff:])
            
            reg1 = LinearRegression().fit(x1.reshape(-1, 1),y1)
            tr1_scale[j,i] = (reg1.coef_)[0]
            cr1_scale[j,i] = (reg1.intercept_).round(5)
            
            reg2 = LinearRegression().fit(x2.reshape(-1, 1),y2)
            tr2_scale[j,i] = (reg2.coef_)[0]
            cr2_scale[j,i] = (reg2.intercept_).round(5)
    return tr1_scale, tr2_scale, cr1_scale, cr2_scale


def eigenvalues_1(L_list, LW_list, avg_p,Ncutoff,dt_list):
    """
    Computes early-time dynamics eigenvalues through logarithmic derivative analysis.
    Characterizes initial contraction rates by fitting the exponential decay constant.

    Key Mathematical Operation:
        Fits: log|dP/dt| vs time → λ = slope
        Where P is normalized perimeter [0,1] and λ is the decay rate

    Inputs:
        L_list    : Tissue length parameters (1D array)
        LW_list   : Tissue width parameters (1D array) 
        avg_p     : Perimeter array (shape: L×LW×time)
        Ncutoff   : Early-time cutoff index (int)
        dt_list   : Time steps (1D array)

    Outputs:
        t_scale   : Decay rates λ (LW×L array)
        c_scale   : Log-amplitudes (LW×L array)

    Note:
        - Uses logarithmic derivatives (dlogP/dt) for exponential regime analysis
        - LOG_EPS (1e-16) prevents log(0) instability
        - Normalization: (P-P_min)/(P_max-P_min) per parameter set
    """
    t_scale = np.zeros((len(LW_list),len(L_list)))
    c_scale = np.zeros((len(LW_list),len(L_list)))
    for i in range(len(L_list)):
        for j in range(len(LW_list)):
            
            x_normalized = (avg_p[i,j,:Ncutoff-1]-np.min(avg_p[i,j,:Ncutoff-1]))/(np.max(avg_p[i,j,:Ncutoff-1])-np.min(avg_p[i,j,:Ncutoff-1]))
            y = np.log(np.abs(np.diff(x_normalized))+LOG_EPS)
            x = (np.arange(Ncutoff)*dt_list)[:-2]
            reg4 = LinearRegression().fit(x.reshape(-1, 1),y)
            t_scale[j,i] = (reg4.coef_)[0].round(5)
            c_scale[j,i] = (reg4.intercept_).round(5)
    return t_scale, c_scale

def eigenvalues_2(L_list, LW_list, avg_p, Ninit,Nfin):
    """
    Computes late-time stability eigenvalues through linear autoregression.
    Quantifies long-term behavior by analyzing P(t+Δt) vs P(t) relationships.

    Key Mathematical Operation: 
        Fits: P(t+Δt) = βP(t) → eigenvalue = (β-1)
        Where β is the autoregression coefficient

    Inputs:
        L_list    : Tissue length parameters (1D array)
        LW_list   : Tissue width parameters (1D array)
        avg_p     : Perimeter array (shape: L×LW×time) 
        Ninit     : Start index for late-time window (int)
        Nfin      : End index for late-time window (int)

    Output:
        t0_scale  : Stability eigenvalues (LW×L array)
                    λ < 0: Stable contraction
                    λ ≈ 0: Critical slowing
                    λ > 0: Unstable growth

    Note:
        - Uses direct linear regression (no log transform)
        - Forced zero intercept (fit_intercept=False)
        - Same normalization as eigenvalues_1 but different analysis window
    """
    t0_scale = np.zeros((len(LW_list),len(L_list)))
    for i in range(len(L_list)):
        for j in range(len(LW_list)):

            x_normalized = (avg_p[i,j,Ninit:Nfin+1]-np.min(avg_p[i,j,Ninit:Nfin+1]))/(np.max(avg_p[i,j,Ninit:Nfin+1])-np.min(avg_p[i,j,Ninit:Nfin+1]))
            y = x_normalized[1:]
            x = x_normalized[:-1]
            reg4 = LinearRegression(fit_intercept = False).fit(x.reshape(-1, 1),y)
            t0_scale[j,i] = (((reg4.coef_-1)))[0].round(5)
    return t0_scale

def classify_helper1(L, class_bin,thr, p0_vec, lambda_vec):
    """Helper function for classify_par_space
    Extracts (p0, lambda) pairs where class_bin == thr for all tissue samples.
    Parameters:
        L (int): Number of tissue samples
        class_bin (ndarray): 3D binary classification array (tissues×L×LW)
        thr (int/float): Exact threshold value for classification
        p0_vec (ndarray): Array of p0 parameter values
        lambda_vec (ndarray): Array of lambda parameter values

    Returns:
        tuple: (p0_loc, lw_loc) where:
               p0_loc - List of p0 values meeting criteria
               lw_loc - List of lambda values meeting criteria
    """
    lw_loc = []
    p0_loc = []
    for i in range(L):
        loc1, loc2 = np.where(class_bin[i]==thr)
        p0_loc.extend(p0_vec[loc1].tolist())
        lw_loc.extend(lambda_vec[loc2].tolist())
    return p0_loc, lw_loc

def classify_helper2(L, class_bin,thr, p0_vec, lambda_vec):
    """Helper function for classify_par_space
    Extracts (p0, lambda) pairs where class_bin > thr for all tissue samples
    Parameters:
        L (int): Number of tissue samples
        class_bin (ndarray): 3D binary classification array (tissues×L×LW)
        thr (int/float): Exact threshold value for classification
        p0_vec (ndarray): Array of p0 parameter values
        lambda_vec (ndarray): Array of lambda parameter values

    Returns:
        tuple: (p0_loc, lw_loc) where:
               p0_loc - List of p0 values meeting criteria
               lw_loc - List of lambda values meeting criteria
    """
    lw_loc = []
    p0_loc = []
    for i in range(L):
        loc1, loc2 = np.where(class_bin[i]>thr)
        p0_loc.extend(p0_vec[loc1].tolist())
        lw_loc.extend(lambda_vec[loc2].tolist())
    return p0_loc, lw_loc

def classify_par_space(a_array, opt_array, tissues_list, aw_sizes, dt_list, beta_lin, lambda_lin):
    """
    Classifies parameter space regions by wound behavior.
    Inputs:
        a_array: Area array
        opt_array: Optimal times array
        tissues_list: List of tissues  
        aw_sizes: Wound size thresholds
        dt_list: Time steps
        beta_lin: Beta parameters
        lambda_lin: Lambda parameters
    Outputs:
        10 lists classifying parameter combinations by behavior
    """
    abin = np.zeros((a_array.shape[0],a_array.shape[1],a_array.shape[2]))
    bbin = np.zeros((a_array.shape[0],a_array.shape[1],a_array.shape[2]))
    cbin = np.zeros((opt_array.shape[0],opt_array.shape[1],opt_array.shape[2]))
    dbin = np.zeros((a_array.shape[0],a_array.shape[1],a_array.shape[2]))
    for i in range(len(tissues_list)):
        abin[i,:,:] = (a_array[i,:,:,-1]>=aw_sizes[i])
        bbin[i,:,:] = (a_array[i,:,:,-1]<aw_sizes[i])*(a_array[i,:,:,-1]>0.9*aw_sizes[i])
        cbin[i,:,:] = (opt_array[i,:,:]<=0.35)*(opt_array[i,:,:]>=dt_list)
        dbin[i] = dyan.GradArray(abin[i])>0.2
           
            
    # Closure Boundary
    bd_p0, bd_lw = classify_helper2(len(tissues_list),dbin,0,beta_lin,lambda_lin)   
            
    #Openings
    op_p0, op_lw = classify_helper1(len(tissues_list),abin,1,beta_lin,lambda_lin)
    
    #Closures
    cl_p0, cl_lw = classify_helper1(len(tissues_list),abin,0,beta_lin,lambda_lin)
    
    #Long Time Closures
    long_p0, long_lw = classify_helper2(len(tissues_list),bbin,0,beta_lin,lambda_lin)   
    
    #Recoils    
    rec_p0, rec_lw = classify_helper2(len(tissues_list),cbin,0,beta_lin,lambda_lin)
            

            
    return bd_p0,bd_lw,op_p0,op_lw, cl_p0, cl_lw, long_p0, long_lw, rec_p0, rec_lw

def closure_time(Npar, avg_t, dt_list):
    """
    Computes mean and std of closure times along parameter space diagonals.
    Inputs:
        Npar: Number of parameter combinations
        avg_t: Average time array
        dt_list: Time steps
    Outputs:
        mean_t: Mean closure times
        std_t: Std of closure times
    """
    mean_t = np.zeros(Npar)
    std_t = np.zeros(Npar)
    # mean_t2[:5] = np.mean(mean_t1[:,:5],0)
    # std_t2[:5] = np.std(mean_t1[:,:5],0)
    for k in range(0,Npar):
        mean_t[k] = np.mean(np.abs(avg_t[[k-i for i in range(k+1)],[i for i in range(k+1)]]),0)*dt_list
        std_t[k] = np.std(np.abs(avg_t[[k-i for i in range(k+1)],[i for i in range(k+1)]]),0)*dt_list
    return mean_t, std_t


# # Long time closures        
# long_lw = []
# long_p0 = []
# for i in range(len(tissues_list)):
#     loc1, loc2 = np.where(bbin[i]>0)
#     lp0 = list(beta_lin[loc1])
#     llw = list(lambda_lin[loc2])

#     for p0 in lp0:
#         long_p0.append(p0)
#     for lw in llw:
#         long_lw.append(lw)
    
# #Recoils
# rec_lw = []
# rec_p0 = []
# for i in range(len(tissues_list)):
#     loc1, loc2 = np.where(cbin[i]>0)
#     lp0 = list(beta_lin[loc1])
#     llw = list(lambda_lin[loc2])

#     for p0 in lp0:
#         rec_p0.append(p0)
#     for lw in llw:
#         rec_lw.append(lw)

    # op_lw = []
# op_p0 = []
# for i in range(len(tissues_list)):
#     loc1, loc2 = np.where(abin[i]==1)
#     lp0 = list(beta_lin[loc1])
#     llw = list(lambda_lin[loc2])
#     for p0 in lp0:
#         op_p0.append(p0)
#     for lw in llw:
#         op_lw.append(lw)

# cl_lw = []
# cl_p0 = []
# for i in range(len(tissues_list)):
#     loc1, loc2 = np.where(abin[i]==0)
#     lp0 = list(beta_lin[loc1])
#     llw = list(lambda_lin[loc2])
#     for p0 in lp0:
#         cl_p0.append(p0)
#     for lw in llw:
#         cl_lw.append(lw)

# bd_lw = []
# bd_p0 = []
# for i in range(len(tissues_list)):
#     dbin[i] = dyan.GradArray(abin[i])>0.2
#     loc1, loc2 = np.where(dbin[i]>0)
#     lp0 = list(beta_lin[loc1])
#     llw = list(lambda_lin[loc2])
#     for p0 in lp0:
#         bd_p0.append(p0)
#     for lw in llw:
#         bd_lw.append(lw)