############################################################# Closure analysis #########################################################################
import numpy as np
from vertexmodelpack import dynamicalanalysis as dyan
from sklearn.linear_model import LinearRegression

def ensemble_avg_per_norm(L_list, LW_list, Tsim, p_array, tissues_list):
    avg_p = np.zeros((len(L_list),len(LW_list),Tsim))
    p_norm = 0*p_array
    std_p = np.zeros((len(L_list),len(LW_list),Tsim))
    for i in range(len(L_list)):
        for j in range(len(LW_list)):
            for l in range(len(tissues_list)):
                p_norm[l,i,j] = p_array[l,i,j]/p_array[l,i,j,0]
            avg_p[i,j] = np.mean(p_norm[:,i,j],0); std_p[i,j] = np.std(p_norm[:,i,j],0)
    return avg_p, std_p, p_norm

def ensemble_avg_t(L_list,LW_list, tissues_list,t_array, wound_sizes):
    avg_t = 0*t_array[0]
    w_list = [float(w) for w in wound_sizes]
    for i in range(len(L_list)):
        for j in range(len(LW_list)):
            avg_t[i,j] = int(np.mean(t_array[:,i,j]/(w_list[:len(tissues_list)])))
    return avg_t
        
def ensemble_avg_rho(L_list,LW_list, Tsim, p_array, a_array, tissues_list):
    avg_r = np.zeros((len(L_list),len(LW_list),Tsim))
    r_norm = 0*p_array
    for i in range(len(L_list)):
        for j in range(len(LW_list)):

            for l in range(len(tissues_list)):
                r_norm[l,i,j] = p_array[l,i,j]/a_array[l,i,j]**0.5
            avg_r[i,j] = np.mean(r_norm[:,i,j],0); 
    return avg_r, r_norm

def recoil_line_par_space(opt_array, Ntissues, k, p_norm, dt_list):
    '''Returns a list of recoil times and perimeters in parameter space at max recoil for line lw = k - gammma.p0'''
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
    p0_array = (avg_r[:,:,-1])
    n_array = np.zeros(p0_array.shape)
    for i in range(p0_array.shape[0]):
        for j in range(p0_array.shape[1]):
            n_array[i,j] = dyan.n_shape(p0_array[i,j])
    return n_array


def timescales_per_rho_par_space(avg_r,avg_p, L_list,LW_list, Nsim,dt_list,Ncutoff):
    y_array = np.nan_to_num(1/avg_r)
    tr1_scale = np.zeros((len(LW_list),len(L_list)))
    tr2_scale = np.zeros((len(LW_list),len(L_list)))

    cr1_scale = np.zeros((len(LW_list),len(L_list)))
    cr2_scale = np.zeros((len(LW_list),len(L_list)))

    x_array = np.linspace(0,Nsim*dt_list,Nsim)
    for i in range(len(L_list)):
        for j in range(len(LW_list)):
            y1 = np.log(y_array[i,j,Ncutoff:])
            x1 = x_array[Ncutoff:]
            
            y2 = np.log(avg_p[i,j,Ncutoff:])
            x2 = x_array[Ncutoff:]
            
            
            reg1 = LinearRegression().fit(x1.reshape(-1, 1),y1)
            tr1_scale[j,i] = (reg1.coef_)[0]
            cr1_scale[j,i] = (reg1.intercept_).round(5)
            
            reg2 = LinearRegression().fit(x2.reshape(-1, 1),y2)
            tr2_scale[j,i] = (reg2.coef_)[0]
            cr2_scale[j,i] = (reg2.intercept_).round(5)
    return tr1_scale, tr2_scale, cr1_scale, cr2_scale


def eigenvalues_1(L_list, LW_list, avg_p,Ncutoff,dt_list):
    t_scale = np.zeros((len(LW_list),len(L_list)))
    c_scale = np.zeros((len(LW_list),len(L_list)))
    for i in range(len(L_list)):
        for j in range(len(LW_list)):
            
            x_normalized = (avg_p[i,j,:Ncutoff-1]-np.min(avg_p[i,j,:Ncutoff-1]))/(np.max(avg_p[i,j,:Ncutoff-1])-np.min(avg_p[i,j,:Ncutoff-1]))
            y = np.log(np.abs(np.diff(x_normalized))+10**(-16))
            x = (np.arange(Ncutoff)*dt_list)[:-2]
            reg4 = LinearRegression().fit(x.reshape(-1, 1),y)
            t_scale[j,i] = (reg4.coef_)[0].round(5)
            c_scale[j,i] = (reg4.intercept_).round(5)
    return t_scale, c_scale

def eigenvalues_2(L_list, LW_list, avg_p, Ninit,Nfin):
    t0_scale = np.zeros((len(LW_list),len(L_list)))
    for i in range(len(L_list)):
        for j in range(len(LW_list)):

            x_normalized = (avg_p[i,j,Ninit:Nfin+1]-np.min(avg_p[i,j,Ninit:Nfin+1]))/(np.max(avg_p[i,j,Ninit:Nfin+1])-np.min(avg_p[i,j,Ninit:Nfin+1]))
            y = x_normalized[1:]
            x = x_normalized[:-1]
            reg4 = LinearRegression(fit_intercept = False).fit(x.reshape(-1, 1),y)
            t0_scale[j,i] = (((reg4.coef_-1)))[0].round(5)
    return t0_scale

def classify_par_space(a_array, opt_array, tissues_list, aw_sizes, dt_list, beta_lin, lambda_lin):
    '''Classify regions in the parameter space by whether the wound opens, closes for a long time or not, and whether it has recoil'''
    abin = np.zeros((a_array.shape[0],a_array.shape[1],a_array.shape[2]))
    bbin = np.zeros((a_array.shape[0],a_array.shape[1],a_array.shape[2]))
    cbin = np.zeros((opt_array.shape[0],opt_array.shape[1],opt_array.shape[2]))
    for i in range(len(tissues_list)):
        abin[i,:,:] = (a_array[i,:,:,-1]>=aw_sizes[i])
        bbin[i,:,:] = (a_array[i,:,:,-1]<aw_sizes[i])*(a_array[i,:,:,-1]>0.9*aw_sizes[i])
        cbin[i,:,:] = (opt_array[i,:,:]<=0.35)*(opt_array[i,:,:]>=dt_list)
        
    dbin = np.zeros((a_array.shape[0],a_array.shape[1],a_array.shape[2]))

    bd_lw = []
    bd_p0 = []
    for i in range(len(tissues_list)):
        dbin[i] = dyan.GradArray(abin[i])>0.2
        loc1, loc2 = np.where(dbin[i]>0)
        lp0 = list(beta_lin[loc1])
        llw = list(lambda_lin[loc2])
        for p0 in lp0:
            bd_p0.append(p0)
        for lw in llw:
            bd_lw.append(lw)
    
    

    op_lw = []
    op_p0 = []
    for i in range(len(tissues_list)):
        loc1, loc2 = np.where(abin[i]==1)
        lp0 = list(beta_lin[loc1])
        llw = list(lambda_lin[loc2])
        for p0 in lp0:
            op_p0.append(p0)
        for lw in llw:
            op_lw.append(lw)
            
    cl_lw = []
    cl_p0 = []
    for i in range(len(tissues_list)):
        loc1, loc2 = np.where(abin[i]==0)
        lp0 = list(beta_lin[loc1])
        llw = list(lambda_lin[loc2])
        for p0 in lp0:
            cl_p0.append(p0)
        for lw in llw:
            cl_lw.append(lw)
            
    # Long time closures        
    long_lw = []
    long_p0 = []
    for i in range(len(tissues_list)):
        loc1, loc2 = np.where(bbin[i]>0)
        lp0 = list(beta_lin[loc1])
        llw = list(lambda_lin[loc2])

        for p0 in lp0:
            long_p0.append(p0)
        for lw in llw:
            long_lw.append(lw)
        
    #Recoils
    rec_lw = []
    rec_p0 = []
    for i in range(len(tissues_list)):
        loc1, loc2 = np.where(cbin[i]>0)
        lp0 = list(beta_lin[loc1])
        llw = list(lambda_lin[loc2])

        for p0 in lp0:
            rec_p0.append(p0)
        for lw in llw:
            rec_lw.append(lw)
            
    return bd_p0,bd_lw,op_p0,op_lw, cl_p0, cl_lw, long_p0, long_lw, rec_p0, rec_lw

def closure_time(Npar, avg_t, dt_list):
    mean_t = np.zeros(Npar)
    std_t = np.zeros(Npar)
    # mean_t2[:5] = np.mean(mean_t1[:,:5],0)
    # std_t2[:5] = np.std(mean_t1[:,:5],0)
    for k in range(0,Npar):
        mean_t[k] = np.mean(np.abs(avg_t[[k-i for i in range(k+1)],[i for i in range(k+1)]]),0)*dt_list
        std_t[k] = np.std(np.abs(avg_t[[k-i for i in range(k+1)],[i for i in range(k+1)]]),0)*dt_list
    return mean_t, std_t