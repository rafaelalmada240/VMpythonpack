import numpy as np
import scipy as scp
# from vertexmodelpack import sppvertex as spv
from vertexmodelpack import connections as fc
# from vertexmodelpack import topologicaltransitions as tpt

def adjacency_matrix(n_vertices, ridges):
    '''
    Constructs adjacency matrix from ridge connections.
    
    Inputs:
        n_vertices: Number of vertices in network
        ridges: List of ridge pairs connecting vertices
        
    Output:
        A_ij: Symmetric adjacency matrix (1 if connected, 0 otherwise)
    '''
    
    ridges = list(ridges)
    ridges_arr=np.array([r for r in ridges if -1 not in r], dtype=int)
    A_ij = np.zeros((n_vertices,n_vertices))
    A_ij[ridges_arr[:,0],ridges_arr[:,1]] = 1
    A_ij[ridges_arr[:,1],ridges_arr[:,0]] = 1
    # for ridge in ridges:
    #     if (ridge[0]!= -1) and (ridge[1]!= -1):
    #         A_ij[int(ridge[0]),int(ridge[1])] = 1
    #         A_ij[int(ridge[1]),int(ridge[0])] = 1
        
    return A_ij

def weight_matrix(vertices, ridges):
    '''
    Constructs weighted adjacency matrix with Euclidean distances.
    
    Inputs:
        vertices: Array of vertex coordinates  
        ridges: List of ridge pairs connecting vertices
        
    Output:
        W_ij: Symmetric weight matrix (distance if connected, 0 otherwise)
    '''
    n_vertices = len(vertices)
    
    ridges = list(ridges)
    valid_ridges = [r for r in ridges if -1 not in r]
    ridges_arr = np.array(valid_ridges, dtype=int)
    dists = np.linalg.norm(vertices[ridges_arr[:,1]]-vertices[ridges_arr[:,0]], axis=1)
    W_ij = np.zeros((n_vertices,n_vertices))
    
    W_ij[ridges_arr[:,0],ridges_arr[:,1]] = dists
    W_ij[ridges_arr[:,1],ridges_arr[:,0]] = dists
    # for ridge in ridges:
    #     if (ridge[0]!= -1) and (ridge[1]!= -1):
    #         W_ij[int(ridge[0]),int(ridge[1])] = fc.norm(vertices[ridge[1]]-vertices[ridge[0]])
    #         W_ij[int(ridge[1]),int(ridge[0])] = fc.norm(vertices[ridge[0]]-vertices[ridge[1]])
        
    return W_ij

def graph_energy(Amatrix):
    '''
    Calculates graph energy as sum of absolute eigenvalues.
    
    Input:
        Amatrix: Adjacency matrix
        
    Output:
        Energy value (float)
    '''
    eig_spectrum = np.linalg.eig(Amatrix)[0]
    return np.sum(np.abs(eig_spectrum))

def generalized_net_der_center(pregions, regions,A):
    '''
    Computes generalized network derivative at each center.
    
    Inputs:
        pregions: Point-to-region mapping
        regions: Voronoi regions
        A: Attribute values per center
        
    Output:
        dA: Array of derivative values
    '''
    dA = []
    Neigh = [fc.find_center_neighbour_center(regions,pregions,i) for i in range(len(A))]
    for i in range(len(A)):
        fA = 0
        #Ni = fc.find_center_neighbour_center(regions,pregions,i)
        for c in Neigh[i]:
            fA = fA + A[i]-A[c]
        dA.append(fA)
    return np.array(dA)

def generalized_net_lap_center(pregions, regions,A,wloc):
    '''
    Computes generalized Laplacian at each center (excluding wound location).
    
    Inputs:
        pregions: Point-to-region mapping  
        regions: Voronoi regions
        A: Attribute values
        wloc: Wound location index to exclude
        
    Output:
        dA: Array of Laplacian values
    '''
    dA = []
    Neigh = [fc.find_center_neighbour_center(regions,pregions,i) for i in range(len(A))]
    for i in range(len(A)):
        fA = 0
        #Ni = fc.find_center_neighbour_center(regions,pregions,i)
        for c in Neigh[i]:
            if (c != wloc and i != wloc):
                fA = fA + A[i]-A[c]
        dA.append(fA)
            
    return np.array(dA)

def generalized_net_grad_center(pregions, regions,A,wloc):
    '''
    Computes generalized gradient magnitude at each center (excluding wound).
    
    Inputs same as generalized_net_lap_center
    
    Output:
        dA: Array of gradient magnitudes
    '''
    dA = []
    Neigh = [fc.find_center_neighbour_center(regions,pregions,i) for i in range(len(A))]
    for i in range(len(A)):
        fA = 0
        #Ni = fc.find_center_neighbour_center(regions,pregions,i)
        for c in Neigh[i]:
            if (c != wloc and i != wloc):
                fA = fA + (A[i]-A[c])**2
        dA.append(fA**0.5)
            
    return np.array(dA)