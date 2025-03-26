import numpy as np
from vertexmodelpack import connections as fc
from vertexmodelpack import geomproperties as gmp


from multiprocessing import Pool
from functools import partial


'''
    regions - (list) set of all the vertices that compose the different regions of the network
    Some regions are empty, but it is best not to remove them for consistency
    point_region - (list) set of the different regions of the network
    ridges - (list) set of all edges in the network
    cells - array of coordinates for all cell centers of the network
    vertices - array of coordinate positions for each vertex of the network

'''


#Maybe needs to declare jit before implementation
## Functions to calculate perimeter and area terms for a given vertex


def energy_vtx_boundary(point_region,regions,ridges, vertices,vertex, K,G,A0):
    # Decreases the size of cells generated at the boundary (sometimes the vertices are wild) by displacing the boundary vertices
    R,N_c = fc.find_vertex_neighbour_centers(regions, point_region,vertex)
    N_v = fc.find_vertex_neighbour_vertices(ridges,vertex)
    E = 0
        
    Ncw = list(N_c)
    P = gmp.perimeters_vor(point_region,regions,vertices,ridges, Ncw)
    A = gmp.areas_vor(point_region,regions,vertices, ridges, Ncw)
    A0c = np.median(A0)#[A0[i] for i in Ncw]
    ESum = np.array([K[i]/2*(A[i]-A0c)**2 + G[i]/2*P[i]**2  for i in range(len(A))])
    
    
    E = E + np.sum(ESum)    
    return E


def energy_vtx_v2(point_region,regions,ridges, vertices,vertex, K,A0,G,L,boundary_tissue,Lw,wloc, bound_wound):
    
    
    R,N_c = fc.find_vertex_neighbour_centers(regions, point_region,vertex)
    N_v = fc.find_vertex_neighbour_vertices(ridges,vertex)
    
    
    Intersect_nv = list(set(N_v).intersection(bound_wound))
    E = 0
        
    Ncw = list(N_c)
    if wloc in Ncw:
        Ncw.remove(wloc)
    
    Gp = np.array([G[i] for i in Ncw])   
        
    P = gmp.perimeters_vor(point_region,regions,vertices,ridges, Ncw)
    A = gmp.areas_vor(point_region,regions,vertices, ridges, Ncw)
    A0c = [A0[i] for i in Ncw]
    
    ESum = np.array([K[i]/2*(A[i]-A0c[i])**2 + Gp[i]/2*P[i]**2 for i in range(len(A))])
    #print(np.sum([G[i] for i in Ncw]))
    E = E + np.sum(ESum)
    for j in N_v:
        if (vertex not in boundary_tissue):
            v = vertices[j]        
            edgeV = vertices[vertex] - v
            lj = fc.norm(edgeV)    
            if (j not in Intersect_nv) or (vertex not in bound_wound) :    
                E += -L*lj
            else:
                E += Lw*lj
    
    
    return E


def energy_vtx_v1(point_region,regions,ridges, vertices,vertex, K,A0,G,L,boundary_tissue):
    
    #Vertex model energy in the absence of a wound
    
    R,N_c = fc.find_vertex_neighbour_centers(regions, point_region,vertex)
    N_v = fc.find_vertex_neighbour_vertices(ridges,vertex)
    
    E = 0
        
    Ncw = list(N_c)
       
        
    P = gmp.perimeters_vor(point_region,regions,vertices,ridges, Ncw)
    A = gmp.areas_vor(point_region,regions,vertices, ridges, Ncw)
    
    
    A0c = [A0[i] for i in Ncw]
    ESum = np.array([K[i]/2*(A[i]-A0c[i])**2 + G[i]/2*P[i]**2 for i in range(len(A))])
    
    
    E = E + np.sum(ESum)
    for j in N_v:
        if (vertex not in boundary_tissue):
            v = vertices[j]        
            edgeV = vertices[vertex] - v
            lj = fc.norm(edgeV)    
            E += -L*lj
    
    return E

def energy_vtx_total(point_region,regions,ridges, vertices, K,A0,G,L, boundary_tissue):
    
    #Total tissue energy
    
    P = gmp.perimeters_vor(point_region,regions,vertices,ridges, point_region)
    A = gmp.areas_vor(point_region,regions,vertices, ridges, point_region)

    EA = np.array([K[i]/2*(A[i]-A0[i])**2 for i in range(len(A))])
    EP1=np.array([G[i]/2*P[i]**2 for i in range(len(A))])
    EP2=np.array([L/2*P[i] for i in range(len(A))])
    
    E = np.sum(EA)+np.sum(EP1)+np.sum(EP2)
    
    for vertex in range(len(vertices)):
        N_v = fc.find_vertex_neighbour_vertices(ridges,vertex)
        for j in N_v:
            if (vertex in boundary_tissue):
                v = vertices[j]        
                edgeV = vertices[vertex] - v
                lj = fc.norm(edgeV)    
                E += -L*lj
    
    return E

def energy_vtx_vector(point_region,regions,ridges, vertices, K,A0,G,L, boundary_tissue):
    
    #Total tissue energy
    lv = len(vertices)
    E = []
    for v in range(lv):
        E.append(energy_vtx_v1(point_region,regions,ridges, vertices,v, K,A0,G,L, boundary_tissue)) 
    
    return np.array(E)

def displacement_vertex(vertices, vertex, h,dir,pos):
    vertices[vertex] = vertices[vertex]+h*dir*pos
    return vertices


def force_vtx_finite_gradv2(point_region, regions, ridges, vertices, vertex, K, A0, G, L,h,boundary_tissue, Lw,wloc, bound_wound):
        
    f_v = 0.0*np.array([1.,1.])
        
    n1 = np.array([1,0])
    n2 = np.array([0,1])
    
    new_vertices1x = displacement_vertex(np.array(vertices),vertex,h,n1,-1)
    new_vertices2x = displacement_vertex(np.array(vertices),vertex,h,n1,1)
    new_vertices1y = displacement_vertex(np.array(vertices),vertex,h,n2,-1)
    new_vertices2y = displacement_vertex(np.array(vertices),vertex,h,n2,1)
        
    Ev1x = energy_vtx_v2(point_region,regions,ridges, new_vertices1x,vertex, K,A0,G,L,boundary_tissue,Lw,wloc, bound_wound)
    Ev2x = energy_vtx_v2(point_region,regions,ridges, new_vertices2x,vertex, K,A0,G,L,boundary_tissue,Lw,wloc, bound_wound)
    Ev1y = energy_vtx_v2(point_region,regions,ridges, new_vertices1y,vertex, K,A0,G,L,boundary_tissue,Lw,wloc, bound_wound)
    Ev2y = energy_vtx_v2(point_region,regions,ridges, new_vertices2y,vertex, K,A0,G,L,boundary_tissue,Lw,wloc, bound_wound)
        
    dEdx = 0.5*(Ev2x-Ev1x)/h
    dEdy = 0.5*(Ev2y-Ev1y)/h
        
    f_v = -(dEdx*n1 + dEdy*n2)
    
    return f_v


def force_vtx_finite_gradv1(point_region, regions, ridges, vertices, vertex, K, A0, G, L,h,boundary_tissue):
        
    f_v = 0.0*np.array([1.,1.])
        
    n1 = np.array([1,0])
    n2 = np.array([0,1])
    
    new_vertices1x = displacement_vertex(np.array(vertices),vertex,h,n1,-1)
    new_vertices2x = displacement_vertex(np.array(vertices),vertex,h,n1,1)
    new_vertices1y = displacement_vertex(np.array(vertices),vertex,h,n2,-1)
    new_vertices2y = displacement_vertex(np.array(vertices),vertex,h,n2,1)
        
    Ev1x = energy_vtx_v1(point_region,regions,ridges, new_vertices1x,vertex, K,A0,G,L,boundary_tissue)
    Ev2x = energy_vtx_v1(point_region,regions,ridges, new_vertices2x,vertex, K,A0,G,L,boundary_tissue)
    Ev1y = energy_vtx_v1(point_region,regions,ridges, new_vertices1y,vertex, K,A0,G,L,boundary_tissue)
    Ev2y = energy_vtx_v1(point_region,regions,ridges, new_vertices2y,vertex, K,A0,G,L,boundary_tissue)
        
    dEdx = 0.5*(Ev2x-Ev1x)/h
    dEdy = 0.5*(Ev2y-Ev1y)/h
        
    f_v = -(dEdx*n1 + dEdy*n2)
    
    return f_v

def force_vtx_boundary(point_region, regions, ridges, vertices, vertex, K, A0, G,h):
        
    f_v = 0.0*np.array([1.,1.])
        
    n1 = np.array([1,0])
    n2 = np.array([0,1])
    
    new_vertices1x = displacement_vertex(np.array(vertices),vertex,h,n1,-1)
    new_vertices2x = displacement_vertex(np.array(vertices),vertex,h,n1,1)
    new_vertices1y = displacement_vertex(np.array(vertices),vertex,h,n2,-1)
    new_vertices2y = displacement_vertex(np.array(vertices),vertex,h,n2,1)
        
    Ev1x = energy_vtx_boundary(point_region,regions,ridges, new_vertices1x,vertex, K,A0,G)
    Ev2x = energy_vtx_boundary(point_region,regions,ridges, new_vertices2x,vertex, K,A0,G)
    Ev1y = energy_vtx_boundary(point_region,regions,ridges, new_vertices1y,vertex, K,A0,G)
    Ev2y = energy_vtx_boundary(point_region,regions,ridges, new_vertices2y,vertex, K,A0,G)
        
    dEdx = 0.5*(Ev2x-Ev1x)/h
    dEdy = 0.5*(Ev2y-Ev1y)/h
        
    f_v = -(dEdx*n1 + dEdy*n2)
    
    return f_v

def force_vtx_elastic_wound(regions,point_region, ridges, K,A0,G,L,Lw,vertices,centers, wloc,h, boundary_tissue):
    
    '''
    
    Calculates the force in all of the vertices according to the vertex model (energy gradient descent method)
    Accounting for a wound healing model
    
    Variables:
    
    regions - (list) set of all the vertices that compose the different regions of the network
    point_region - (list) set of the different regions of the network
    ridges - (list) set of all edges in the network
    K, A0, G, L - (float) model parameters 
    vertices - array of coordinate positions for each vertex of the network
    h - step of gradient descent
    
    Output:

    F_V - array of all forces acting on the vertices

    
    '''
    
    LV = len(vertices) #Number of vertices

    F_V = [] # Vector of resulting forces acting on the vertices
    
    
    r0 = np.sqrt(np.mean(A0)/np.pi)
    
    #For vertices, well, this may be a huge mess
    
    #First find the boundary
    boundary_wound = fc.find_wound_boundary(regions,point_region,wloc)

    for v in range(LV):
        if v not in boundary_tissue:
            f_v = force_vtx_finite_gradv2(point_region, regions, ridges, vertices, v, K, A0, G, L,h,boundary_tissue,Lw,wloc, boundary_wound)
            #else:
            #    f_v = force_vtx_finite_grad_wound(point_region, regions, ridges, vertices, v, K, A0, G, L,)
                #print(fc.norm(f_v))
            NeighR, NeighC = fc.find_vertex_neighbour_centers(regions,point_region,v)
            for i in range(len(NeighC)):
                for j in range(i+1,len(NeighC)):
                    ci = centers[NeighC[i]]
                    cj = centers[NeighC[j]]
                    rij = fc.norm(cj-ci)
                    nij = (cj-ci)/(rij+h)
                    if rij <= r0:
                        f_v += 0.1*(rij-r0)*nij
                    else:
                        continue
        else:
            f_v = 0*force_vtx_boundary(point_region, regions, ridges, vertices, v, K, A0, G,h) 
                
        #Maybe include a regularizing force acting on the cells
        F_V.append(f_v)
        
    return np.array(F_V)



def force_vtx_elastic(regions,point_region, ridges, K,A0,G,L,vertices,centers,h, boundary_tissue):
    
    '''
    
    Calculates the force in all of the vertices according to the vertex model (energy gradient descent method)
    
    Variables:
    
    regions - (list) set of all the vertices that compose the different regions of the network
    point_region - (list) set of the different regions of the network
    ridges - (list) set of all edges in the network
    K, A0, G, L - (float) model parameters 
    vertices - array of coordinate positions for each vertex of the network
    h - step of gradient descent
    
    Output:

    F_V - array of all forces acting on the vertices

    
    '''
    
    LV = len(vertices) #Number of vertices

    F_V = [] # Vector of resulting forces acting on the vertices
    
    
    r0 = np.sqrt(np.mean(A0)/np.pi)
    
    #For vertices, well, this may be a huge mess
    
    #First find the boundary

    for v in range(LV):
        if v not in boundary_tissue:
            f_v = force_vtx_finite_gradv1(point_region, regions, ridges, vertices, v, K, A0, G, L,h,boundary_tissue)
            NeighR, NeighC = fc.find_vertex_neighbour_centers(regions,point_region,v)
            for i in range(len(NeighC)):
                for j in range(i+1,len(NeighC)):
                    ci = centers[NeighC[i]]
                    cj = centers[NeighC[j]]
                    rij = fc.norm(cj-ci)
                    nij = (cj-ci)/(rij+h)
                    if rij <= r0:
                        f_v += 0.1*(rij-r0)*nij
                    else:
                        continue
        else:
            f_v = 0*force_vtx_boundary(point_region, regions, ridges, vertices, v, K, A0, G,h) 
                
        #Maybe include a regularizing force acting on the cells
        F_V.append(f_v)
        
    return np.array(F_V)

###Parallel implementation of the force calculation (Seems to be working for now)

def calculate_force_wound_single(v, point_region, regions, ridges, vertices, K, A0, G, L, centers, h, boundary_tissue, Lw, wloc, bound_wound):
    if v in boundary_tissue:
        return np.zeros(2)
    


    f_v = force_vtx_finite_gradv2(point_region, regions, ridges, vertices, v, K, A0, G, L, h, boundary_tissue, Lw, wloc, bound_wound)
    NeighR, NeighC = fc.find_vertex_neighbour_centers(regions, point_region, v)
    r0 = np.sqrt(np.median(A0)/np.pi)
    for i in range(len(NeighC)):
        for j in range(i + 1, len(NeighC)):
            ci = centers[NeighC[i]]
            cj = centers[NeighC[j]]
            rij = fc.norm(cj - ci)
            nij = (cj - ci) / (rij + h)
            if rij <= r0:
                f_v += 0.1 * (rij - r0) * nij
    return f_v

def force_vtx_elastic_wound_parallel(regions,point_region, ridges, K,A0,G,L,Lw,vertices,centers,wloc,h, boundary_tissue):
    
    LV = len(vertices)
    r0 = np.sqrt(np.mean(A0) / np.pi)
    boundary_tissue = set(boundary_tissue)
    bound_wound = fc.find_wound_boundary(regions, point_region, wloc)
    #print(np.sum(G))
    calculate_force_partial = partial(calculate_force_wound_single, 
                                      point_region=point_region, regions=regions, ridges=ridges, 
                                      vertices=vertices, K=K, A0=A0, G=G, L=L, centers=centers, 
                                      h=h, boundary_tissue=boundary_tissue, 
                                      Lw=Lw, wloc=wloc, bound_wound=bound_wound)

    with Pool() as pool:
        F_V = pool.map(calculate_force_partial, range(LV))

    return np.array(F_V)

def calculate_force(v, point_region, regions, ridges, vertices, K, A0, G, L, centers, h, boundary_tissue, r0):
    if v in boundary_tissue:
        return np.zeros(2)

    f_v = force_vtx_finite_gradv1(point_region, regions, ridges, vertices, v, K, A0, G, L, h, boundary_tissue)
    NeighR, NeighC = fc.find_vertex_neighbour_centers(regions, point_region, v)
    for i in range(len(NeighC)):
        for j in range(i + 1, len(NeighC)):
            ci = centers[NeighC[i]]
            cj = centers[NeighC[j]]
            rij = fc.norm(cj - ci)
            nij = (cj - ci) / (rij + h)
            if rij <= r0:
                f_v += 0.1 * (rij - r0) * nij
    return f_v
    
def force_vtx_elastic_parallel(regions,point_region, ridges, K,A0,G,L,vertices,centers,h, boundary_tissue):

    LV = len(vertices)
    r0 = np.sqrt(np.mean(A0) / np.pi)
    boundary_tissue = set(boundary_tissue)
    
    calculate_force_partial = partial(calculate_force, 
                                      point_region=point_region, regions=regions, ridges=ridges, 
                                      vertices=vertices, K=K, A0=A0, G=G, L=L, centers=centers, 
                                      h=h, boundary_tissue=boundary_tissue, 
                                      r0=r0)

    with Pool() as pool:
        F_V = pool.map(calculate_force_partial, range(LV))

    return np.array(F_V)

def stress_cell(regions, point_region, vertices,centers,F):
    S = []
    for alpha in point_region:
        Nv = len(regions[alpha])
        list_vertices = vertices[regions[alpha]]
        list_forces = F[regions[alpha]]
        loc_cell = centers[alpha]
        Sa = 0
        for i in range(Nv):
            ria = list_vertices[i]-loc_cell
            nia = ria/fc.norm(ria)
            Sa += np.dot(list_forces[i],nia)
        S.append(Sa)
    return S


def cells_avg_vtx(regions,point_region,cells,vertices):
    for i in range(len(cells)):
        Neigh_c = fc.find_center_region(regions,point_region, i)
        avg_vc = np.mean(vertices[Neigh_c],0)
        cells[i] = avg_vc
        
    return cells
