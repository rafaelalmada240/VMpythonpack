import numpy as np
from vertexmodelpack import sppvertex as spv
#This file is for calcium dynamics in the tissue, controlling calcium concentration and gamma activation (define refractory period)

def delta_f(x,thresh,err):
    '''Approximation of Kronecker Delta Function'''
    return 1 if abs(x - thresh) <= err else 0

def step_f(x,thresh):
    '''Step function'''
    return np.ones(x.shape)*(x >= thresh)

def step_f1(x,thresh):
    '''Step function'''
    return 1.0*(x >= thresh)


def c_update(c,a,par):
    '''Change concentration on cell i'''
    c0 = par[0]
    s0 = par[1]
    clim = par[2]
    
    h1 = step_f(a,s0)
    h2 = step_f(c,0.01)*step_f(a,2*s0)+step_f(c,c0*3)-step_f(c,c0*3)*step_f(c,0.01)*step_f(a,2*s0)
    
    return (40*h1*(1-h2))*(clim - c) - 75*h2 

# def c_up_tissue(vertices, regions, p_regions, edges, c_total,a_0, par,wloc):
#     a_total =  spv.areas_vor(p_regions,regions,vertices,edges,p_regions)
#     dc = []
#     for i in range(len(p_regions)):
#         if i != wloc:
#             dc.append(c_update(c_total[p_regions[i]],np.array(a_total[p_regions[i]])-np.array(a_0[p_regions[i]]),par))
#         else:
#             dc.append(0)
#     return np.array(dc)

def c_up_tissue(c_total,s_0, par,wloc):
    dc = c_update(c_total,s_0,par)
    dc[wloc] = 0
    return np.array(dc)

# def gamma_update(c,hs, par, gamma_vec):
#     '''Change gammas'''
#     c0 = par[0]
#     # s0 = par[1]
    
#     # g0 = par[2]
#     g0 = par[3]
#     t0 = par[4]
    
#     h1 = step_f(c,3*c0)
    
#     h2 = step_f(hs,0)
    
#     gamma = gamma_vec[0]
#     gamma_state = gamma_vec[1]
#     gamma_time = gamma_vec[2]
    
#     # Activate g
#     if (h1*h2 == 1) and ((gamma_state == 0) and (gamma_time == 0)):
#         gamma += g0
#         gamma_state = 1
#         gamma_time = 1
#         # print(sg)
    
#     if (gamma_state == 1) and(gamma_time < t0):
#         gamma_time += 1
#         # print(sg)
        
#     #Refractory period
#     if ((gamma_time >= t0) and (gamma_time < 2*t0)):
#         gamma = 0
#         gamma_state = 2
#         gamma_time += 1
#         # print(sg)
        
#     if (gamma_time >= 2*t0) and (gamma_state == 2):
#         gamma_time = 0
#         gamma_state = 0
#         # print(sg)
        
#     return np.array([gamma,gamma_state,gamma_time])


# def g_up_tissue(lapA, p_regions, gamma_vec, c_total, par,wloc):
#     # a_total =  spv.areas_vor(p_regions,regions,vertices,edges,p_regions)
#     dg = []
#     for i in range(len(p_regions)):
#         if i != wloc:
#             dg.append(gamma_update(c_total[p_regions[i]],lapA[p_regions[i]],par, gamma_vec[i]))
#         else:
#             dg.append(np.array([0,0,0]))
#     return np.array(dg)

def gamma_update1(c,hs, par, gamma_vec):
    '''Change gammas'''
    c0 = par[0]
    s0 = par[1]
    g0 = par[3]
    
    h1 = step_f1(c,c0)
    h2 = step_f1(1.5*s0,hs)
    
    gamma_state = (h1*h2-gamma_vec[1])
    gamma = gamma_state*g0
    
    gamma_time = gamma_vec[2]
        
    return np.array([gamma,gamma_state,gamma_time])

def gamma_update2(hs, par, gamma_vec):
    '''Change gammas'''
    #c0 = par[0]
    s0 = par[1]
    g0 = par[3]
    
    #h1 = step_f1(c,c0)
    h2 = step_f1(hs,1.*s0)
    
    gamma_state = (h2-(h2+1-1*h2)*gamma_vec[1])
    gamma = gamma_state*g0
    gamma_time = 1
        
    return np.array([gamma,gamma_state,gamma_time])

def g_up_tissue(strain, p_regions, gamma_vec, c_total, par,wloc):
    
    dg = []
    for i in range(len(p_regions)):
        if i != wloc:
            dg.append(gamma_update1(c_total[p_regions[i]],strain[p_regions[i]],par, gamma_vec[i]))
        else:
            dg.append(np.array([0,0,0]))
    return np.array(dg)


def g_up_tissue2(strain, p_regions, gamma_vec, par,wloc):
    
    dg = []
    for i in range(len(p_regions)):
        if i != wloc:
            dg.append(gamma_update2(strain[p_regions[i]],par, gamma_vec[i]))
        else:
            dg.append(np.array([0,0,0]))
    return np.array(dg)

# def gamma_switch_list(gamma,state,time,net,par,r0):
#     l0 = (net[1:]-net[:-1]-r0)/r0
#     for i in range(len(gamma)):
#         gamma[i], state[i], time[i] = gamma_switch(gamma[i], state[i], time[i], l0[i],par)
        
#     return gamma, state, time
