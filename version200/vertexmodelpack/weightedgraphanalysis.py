import numpy as np
import scipy as scp
# from vertexmodelpack import sppvertex as spv
from vertexmodelpack import connections as fc
# from vertexmodelpack import topologicaltransitions as tpt

def adjacency_matrix(n_vertices, ridges):
    '''
    Calculates the adjacency matrix of the network
    '''
    
    ridges = list(ridges)
    A_ij = np.zeros((n_vertices,n_vertices))
    for ridge in ridges:
        if (ridge[0]!= -1) and (ridge[1]!= -1):
            A_ij[int(ridge[0]),int(ridge[1])] = 1
            A_ij[int(ridge[1]),int(ridge[0])] = 1
        
    return A_ij

def weight_matrix(vertices, ridges):
    '''
    Calculates the weight matrices of the network
    '''
    n_vertices = len(vertices)
    
    ridges = list(ridges)
    W_ij = np.zeros((n_vertices,n_vertices))
    for ridge in ridges:
        if (ridge[0]!= -1) and (ridge[1]!= -1):
            W_ij[int(ridge[0]),int(ridge[1])] = fc.norm(vertices[ridge[1]]-vertices[ridge[0]])
            W_ij[int(ridge[1]),int(ridge[0])] = fc.norm(vertices[ridge[0]]-vertices[ridge[1]])
        
    return W_ij

def graph_energy(Amatrix):
    eig_spectrum = np.linalg.eig(Amatrix)[0]
    return np.sum(np.abs(eig_spectrum))

def generalized_net_der_center(pregions, regions,A):
    dA = []
    for i in range(len(A)):
        fA = 0
        Ni = fc.find_center_neighbour_center(regions,pregions,i)
        for c in Ni:
            fA = fA + A[i]-A[c]
        dA.append(fA)
    return np.array(dA)

def generalized_net_lap_center(pregions, regions,A,wloc):
    dA = []
    for i in range(len(A)):
        fA = 0
        Ni = fc.find_center_neighbour_center(regions,pregions,i)
        for c in Ni:
            if (c != wloc and i != wloc):
                fA = fA + A[i]-A[c]
        dA.append(fA)
            
    return np.array(dA)

def generalized_net_grad_center(pregions, regions,A,wloc):
    dA = []
    for i in range(len(A)):
        fA = 0
        Ni = fc.find_center_neighbour_center(regions,pregions,i)
        for c in Ni:
            if (c != wloc and i != wloc):
                fA = fA + (A[i]-A[c])**2
        dA.append(fA**0.5)
            
    return np.array(dA)