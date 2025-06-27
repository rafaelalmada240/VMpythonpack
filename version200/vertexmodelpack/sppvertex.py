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
    """Calculates energy for boundary vertices with simplified target area (median A0)
    
    Args:
        point_region: Mapping from cells to Voronoi regions
        regions: List of Voronoi regions (vertex indices)
        ridges: Edge connections between vertices  
        vertices: Vertex coordinates
        vertex: Current vertex index
        K: Area elasticity modulus
        G: Perimeter contractility
        A0: Target areas for cells
        
    Returns:
        Energy contribution from neighboring cells to this vertex
    """
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

def force_vtx_boundary(point_region, regions, ridges, vertices, vertex, K, A0, G,h):
    
    ''' Calculates the force acting on the vertices at the boundary of the tissue.
    point_region - (list) set of the different regions of the network
    regions - (list) set of all the vertices that compose the different regions of the network
    ridges - (list) set of all edges in the network
    vertices - array of coordinate positions for each vertex of the network
    vertex - (int) index of the vertex at the boundary
    K, A0, G - (float) model parameters
    h - (float) step of gradient descent
    '''
    
        
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

############################################################# Wound Healing ########################################################################

def energy_vtx_v2(point_region,regions,ridges, vertices,vertex, K,A0,G,L,boundary_tissue,Lw,wloc, bound_wound):
    """Vertex energy with wound-specific modifications
    
    Key differences from v1:
    - Excludes wound cell (wloc) from calculations
    - Adds wound-specific line tension (Lw*lj) for wound boundary edges
    """
    
    R,N_c = fc.find_vertex_neighbour_centers(regions, point_region,vertex)
    N_v = fc.find_vertex_neighbour_vertices(ridges,vertex)
    
    
    Intersect_nv = list(set(N_v).intersection(bound_wound))
    E = 0
        
    Ncw = [i for i in N_c if i != wloc]
    
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


def force_vtx_finite_gradv2(point_region, regions, ridges, vertices, vertex, K, A0, G, L,h,boundary_tissue, Lw,wloc, bound_wound):
    """
    Computes vertex force via central differences using the wound-modified energy.
    
    Args:
        Lw: Wound-specific line tension coefficient
        wloc: Index of wound center cell
        bound_wound: List of wound boundary vertices
        
    Returns:
        Force vector [Fx, Fy] calculated as:
        F = -∇E ≈ -(E(x+h) - E(x-h))/(2h)
    """
        
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

def force_vtx_elastic_wound(regions,point_region, ridges, K,A0,G,L,Lw,vertices,centers, wloc,h, boundary_tissue, kreg=0.1):
    
    """Calculates forces with wound-specific modifications
    
    Differences from standard version:
    1. Uses energy_vtx_v2 which handles wound boundaries
    2. Applies wound-specific line tension Lw
    3. Special treatment for vertices near wound
    """
    
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
                    dij = max(rij,h)
                    nij = (cj-ci)/dij
                    if rij <= r0:
                        f_v += kreg*(rij-r0)*nij
                    else:
                        continue
        else:
            f_v = 0*force_vtx_boundary(point_region, regions, ridges, vertices, v, K, A0, G,h) 
                
        #Maybe include a regularizing force acting on the cells
        F_V.append(f_v)
        
    return np.array(F_V)

def calculate_force_wound_single(v, point_region, regions, ridges, vertices, K, A0, G, L, centers, h, boundary_tissue, Lw, wloc, bound_wound):
    """Single-vertex force calculation for parallel processing"""
    if v in set(boundary_tissue):
        return np.zeros(2)

    f_v = force_vtx_finite_gradv2(point_region, regions, ridges, vertices, v, K, A0, G, L, h, boundary_tissue, Lw, wloc, bound_wound)
    NeighR, NeighC = fc.find_vertex_neighbour_centers(regions, point_region, v)
    r0 = np.sqrt(np.median(A0)/np.pi)
    for i in range(len(NeighC)):
        for j in range(i + 1, len(NeighC)):
            ci = centers[NeighC[i]]
            cj = centers[NeighC[j]]
            rij = fc.norm(cj - ci)
            dij = max(rij, h)
            nij = (cj - ci) /dij
            if rij <= r0:
                f_v += 0.1 * (rij - r0) * nij
    return f_v


# def force_vtx_elastic_wound_parallel(regions,point_region, ridges, K,A0,G,L,Lw,vertices,centers,wloc,h, boundary_tissue):
    
#     LV = len(vertices)
#     r0 = np.sqrt(np.mean(A0) / np.pi)
#     boundary_tissue = set(boundary_tissue)
#     bound_wound = fc.find_wound_boundary(regions, point_region, wloc)
#     #print(np.sum(G))
#     calculate_force_partial = partial(calculate_force_wound_single, 
#                                       point_region=point_region, regions=regions, ridges=ridges, 
#                                       vertices=vertices, K=K, A0=A0, G=G, L=L, centers=centers, 
#                                       h=h, boundary_tissue=boundary_tissue, 
#                                       Lw=Lw, wloc=wloc, bound_wound=bound_wound)

#     with Pool() as pool:
#         F_V = pool.map(calculate_force_partial, range(LV))

#     return np.array(F_V)


def force_vtx_elastic_wound_parallel(regions,point_region, ridges, K,A0,G,L,Lw,vertices,centers,wloc,h, boundary_tissue, pool):
    
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

    #with Pool() as pool:
    F_V = pool.map(calculate_force_partial, range(LV))

    return np.array(F_V)


############################################## Standard Vertex Model #########################################################

def energy_vtx_v1(point_region,regions,ridges, vertices,vertex, K,A0,G,L,boundary_tissue):
    
    """Standard vertex model energy without wound
    
    Adds line tension term (-L*lj) for non-boundary vertices
    """
    
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
    """
    Computes the total mechanical energy of the entire tissue system.
    
    Args:
        point_region: Mapping from cells to Voronoi regions
        regions: List of vertex indices for each Voronoi region
        ridges: Edge connections between vertices
        vertices: Array of vertex positions
        K: Array of area elasticities for each cell
        A0: Array of target areas for each cell
        G: Array of perimeter contractilities
        L: Line tension coefficient
        boundary_tissue: List of boundary vertex indices
        
    Returns:
        Total energy (float) summing:
        1. Area elasticity: K/2*(A-A0)^2
        2. Perimeter contractility: G/2*P^2 
        3. Line tension: L/2*P (global)
        4. Boundary-specific line tension: -L*lj
    """
    
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
    """
    Computes energy contributions per vertex (vectorized).
    
    Returns:
        Array of energies where each element represents the 
        energy contribution from a single vertex's neighborhood.
        
    Note: 
        Useful for identifying local energy hotspots or
        visualizing energy distributions across the tissue.
    """
    
    #Total tissue energy
    lv = len(vertices)
    E = []
    for v in range(lv):
        E.append(energy_vtx_v1(point_region,regions,ridges, vertices,v, K,A0,G,L, boundary_tissue)) 
    
    return np.array(E)



def force_vtx_finite_gradv1(point_region, regions, ridges, vertices, vertex, K, A0, G, L,h,boundary_tissue):
    """Calculates force via finite differences on energy_vtx_v1
    
    Args:
        h: Finite difference step size
        Other params match energy_vtx_v1
        
    Returns:
        Force vector [Fx, Fy] from energy gradient
    """
        
    f_v = 0.0*np.array([1.,1.])
        
    # Unit vectors for finite differences
    n1 = np.array([1,0])
    n2 = np.array([0,1])
    
    # Perturbed positions
    new_vertices1x = displacement_vertex(np.array(vertices),vertex,h,n1,-1)
    new_vertices2x = displacement_vertex(np.array(vertices),vertex,h,n1,1)
    
    #Perturbed positions in y-direction
    new_vertices1y = displacement_vertex(np.array(vertices),vertex,h,n2,-1)
    new_vertices2y = displacement_vertex(np.array(vertices),vertex,h,n2,1)
       
    # Energy for different perturbations 
    Ev1x = energy_vtx_v1(point_region,regions,ridges, new_vertices1x,vertex, K,A0,G,L,boundary_tissue)
    Ev2x = energy_vtx_v1(point_region,regions,ridges, new_vertices2x,vertex, K,A0,G,L,boundary_tissue)
    Ev1y = energy_vtx_v1(point_region,regions,ridges, new_vertices1y,vertex, K,A0,G,L,boundary_tissue)
    Ev2y = energy_vtx_v1(point_region,regions,ridges, new_vertices2y,vertex, K,A0,G,L,boundary_tissue)
    
    #Finite difference gradients    
    dEdx = 0.5*(Ev2x-Ev1x)/h
    dEdy = 0.5*(Ev2y-Ev1y)/h
        
    f_v = -(dEdx*n1 + dEdy*n2) # F = -∇E
    
    return f_v



def force_vtx_elastic(regions,point_region, ridges, K,A0,G,L,vertices,centers,h, boundary_tissue):
    
    """Calculates forces on all vertices using standard vertex model
    
    Adds additional repulsion between nearby cell centers (r < r0)
    """
    
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
                    dij = max(rij, h)
                    nij = (cj-ci)/dij
                    if rij <= r0:
                        f_v += 0.1*(rij-r0)*nij
                    else:
                        continue
        else:
            f_v = 0*force_vtx_boundary(point_region, regions, ridges, vertices, v, K, A0, G,h) 
                
        #Maybe include a regularizing force acting on the cells
        F_V.append(f_v)
        
    return np.array(F_V)

def calculate_force(v, point_region, regions, ridges, vertices, K, A0, G, L, centers, h, boundary_tissue, r0):
    """
    Single-vertex elastic force calculation for parallel processing.
    
    Combines:
    1. Standard vertex model forces (finite diff)
    2. Additional center-center repulsion (r < r0)
    
    Args:
        r0: Interaction cutoff radius
        
    Returns:
        Total elastic force vector for vertex v
    """
    if v in boundary_tissue:
        return np.zeros(2)

    f_v = force_vtx_finite_gradv1(point_region, regions, ridges, vertices, v, K, A0, G, L, h, boundary_tissue)
    NeighR, NeighC = fc.find_vertex_neighbour_centers(regions, point_region, v)
    for i in range(len(NeighC)):
        for j in range(i + 1, len(NeighC)):
            ci = centers[NeighC[i]]
            cj = centers[NeighC[j]]
            rij = fc.norm(cj - ci)
            dij = max(rij, h)
            nij = (cj - ci) / dij
            if rij <= r0:
                f_v += 0.1 * (rij - r0) * nij
    return f_v
    
def force_vtx_elastic_parallel(regions,point_region, ridges, K,A0,G,L,vertices,centers,h, boundary_tissue):
    """
    Parallel elastic force computation.
    
    Features:
    1. Automatic r0 calculation from mean cell area
    2. Thread-safe neighborhood queries
    3. Balanced workload distribution
    
    Returns:
        Array of elastic forces for all vertices
    """

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

#####################################################Activity ####################################################################

def force_cells_active(point_region, regions, J, f_centers,boundary_tissue, wloc, coords):
    """Calculates active cell forces with alignment coupling
    
    Implements:
    - Force alignment between neighbors (magnitude J)
    - Special boundary conditions:
      * Zero force on boundary/wound cells
      * Tangential forces near wound
    """
    
    dEalign = np.zeros_like(f_centers)

    Boundary_cells = []
    Wound_cells = []
    LC = len(f_centers)  # Number of centers
    
    for c in range(LC):
        vertices_c = regions[point_region[c]]
        vertices_c_boundary = list(set(vertices_c).intersection(boundary_tissue))
        vertices_c_wound = list(set(vertices_c).intersection(regions[wloc]))
        if len(vertices_c_boundary) > 0:
            Boundary_cells.append(c)
        if len(vertices_c_wound) > 0:
            Wound_cells.append(c)
            
    for c in range(LC):
        
        f_c = f_centers[c]
        if c in Boundary_cells or c in Wound_cells:
            f_c = 0.0 * np.array([1., 1.])
            continue
        else:
            NeighC = fc.find_center_neighbour_center(regions, point_region, c)
            # for n in NeighC:  
            #     f_c += (f_centers[n])/np.linalg.norm(f_centers[n])*J# Project the force onto the direction of the alignment
                    
            for n in NeighC:  # Project the force onto the direction of the alignment
                if (n in Boundary_cells) or (n in Wound_cells):
                    coords_n = coords[n]
                    c_norm = max(np.linalg.norm(coords_n),1e-6)
                    c_perp = np.array([-coords_n[1]/c_norm, coords_n[0]/c_norm])
                    f_c += np.dot(f_centers[c], c_perp) * c_perp*J  # Project the force onto the perpendicular direction, Fn = 0 at the wound boundary
        dEalign[c] = f_c
    return dEalign

def calculate_force_cells_active_single(point_region, regions, J, f_centers,boundary_tissue, wloc,c, coords):
    """
    Computes active force for a single cell accounting for:
    1. Neighbor alignment (magnitude J)
    2. Boundary conditions:
       - Zero force on boundary/wound cells
       - Tangential forces near wound
    
    Args:
        c: Index of current cell
        coords: Cell center positions
        
    Returns:
        Active force vector for cell c
    """
    
        # Find the boundary and wound cells
    Boundary_cells = []
    Wound_cells = []
    LC = len(f_centers)  # Number of centers
    
    for c in range(LC):
        vertices_c = regions[point_region[c]]
        vertices_c_boundary = list(set(vertices_c).intersection(boundary_tissue))
        vertices_c_wound = list(set(vertices_c).intersection(regions[wloc]))
        if len(vertices_c_boundary) > 0:
            Boundary_cells.append(c)
        if len(vertices_c_wound) > 0:
            Wound_cells.append(c)

    NeighC = fc.find_center_neighbour_center(regions, point_region, c)
    f_c = 0.0 * np.array([1., 1.])
    
    for n in NeighC:  
        f_norm = max(np.linalg.norm(f_centers[n]),1e-10)
        if n == c:
            continue
        if (n in Boundary_cells) or (n in Wound_cells):
            coords_n = coords[n]
            c_dist = max(np.linalg.norm(coords_n),1e-10)
            
            c_perp = np.array([-coords_n[1]/c_dist, coords_n[0]/c_dist])
            f_c += np.dot((f_centers[n])/f_norm, c_perp) * c_perp*J  # Project the force onto the perpendicular direction, Fn = 0 at the wound boundary
        f_c += (f_centers[n])/f_norm*J# Project the force onto the direction of the alignment
    
    return fc 

def force_cells_active_parallel(regions, point_region, J, f_centers):
    """
    Parallel implementation of active force calculation.
    
    Uses multiprocessing.Pool to distribute:
    1. Cell-wise force calculations
    2. Neighborhood queries
    3. Boundary condition checks
    
    Returns:
        Array of active forces for all cells
    """
    
    LC = len(f_centers)  # Number of centers
    
    calculate_force_partial = partial(calculate_force_cells_active_single, 
                                      point_region=point_region, regions=regions, 
                                      J=J, f_centers=f_centers)
    
    with Pool() as pool:
        dEalign = pool.map(calculate_force_partial, range(LC))

    return np.array(dEalign)
            
def force_vtx_active(point_region, regions, vertices, centers, boundary_tissue, wloc, Fcenters):
    """Distributes active cell forces to their vertices
    
    Key logic:
    - Each vertex gets vector sum of forces from its adjacent cells
    - Special handling at wound boundary:
      * Forces projected to tangential direction
      * Zero force at exact wound location
    """
    F_V = np.zeros_like(vertices)
    LV = len(vertices) # Number of vertices
    LC = len(centers)
    
    Boundary_cells = []
    Wound_cells = []
    for c in range(LC):
        vertices_c = regions[point_region[c]]
        vertices_c_boundary = list(set(vertices_c).intersection(boundary_tissue))
        vertices_c_wound = list(set(vertices_c).intersection(regions[wloc]))
        if len(vertices_c_boundary) > 0:
            Boundary_cells.append(c)
        if len(vertices_c_wound) > 0:
            Wound_cells.append(c)
    
    for v in range(LV):
        if v in boundary_tissue:
            F_V[v]=0.0*np.array([1.,1.])
        else:
            NeighR, NeighC = fc.find_vertex_neighbour_centers(regions, point_region, v)
            f_v = 0.0*np.array([1.,1.])
            for c in NeighC:
                if c in Boundary_cells:
                    continue
                if c in Wound_cells:
                    if c == wloc:
                        f_v += 0
                    else:
                        v_dist = (centers[wloc] - vertices[v])/fc.norm(centers[wloc] - vertices[v])
                        v_perp = np.array([-v_dist[1], v_dist[0]])
                        f_v += np.dot(Fcenters[c],v_perp)*v_perp # project the force onto the perpendicular direction, Fn = 0 at the wound boundary
                else:
                    f_v += Fcenters[c]
            F_V[v] = f_v        
    return F_V


def calculate_force_active_single(v, point_region, regions, vertices, centers, wloc, boundary_tissue, Boundary_cells, Wound_cells, Fcenters):
    """
    Computes active force contribution at a single vertex.
    
    Special handling:
    - Boundary vertices get zero force
    - Wound neighbors get tangential forces
    - Normal cells get direct force summation
    
    Args:
        v: Vertex index
        Fcenters: Active forces from cell centers
        
    Returns:
        Resultant active force vector at vertex v
    """
    
    
    f_v = 0.0*np.array([1.,1.])
    
    if v not in boundary_tissue:
        NeighR, NeighC = fc.find_vertex_neighbour_centers(regions, point_region, v)
        for c in NeighC:
            if c in Boundary_cells:
                continue
            if c in Wound_cells:
                if c == wloc:
                    f_v += 0
                else:
                    v_dist = (centers[wloc] - vertices[v])/fc.norm(centers[wloc] - vertices[v])
                    v_perp = np.array([-v_dist[1], v_dist[0]])
                    f_v += np.dot(Fcenters[c],v_perp)*v_perp*(np.linalg.norm(centers[c])<4) # project the force onto the perpendicular direction, Fn = 0 at the wound boundary
            else:
                f_v += Fcenters[c]*(np.linalg.norm(centers[c])<4)
                    
    return f_v

def force_vtx_active_parallel(regions, point_region, vertices, centers, boundary_tissue, wloc, Fcenters):
    """
    Parallelized vertex active forces.
    
    Optimization:
    1. Pre-computes boundary/wound cell lists
    2. Uses partial function for parameter binding
    3. Distributes vertex loop via multiprocessing
    
    Returns:
        Array of active forces at all vertices
    """
    LV = len(vertices)
    #r0 = np.sqrt(np.mean(A0) / np.pi)
    boundary_tissue = set(boundary_tissue)
    #bound_wound = fc.find_wound_boundary(regions, point_region, wloc)
    
    LC = len(centers)
    
    # Find the boundary and wound cells
    Boundary_cells = []
    Wound_cells = []
    for c in range(LC):
        vertices_c = regions[point_region[c]]
        vertices_c_boundary = list(set(vertices_c).intersection(boundary_tissue))
        vertices_c_wound = list(set(vertices_c).intersection(regions[wloc]))
        if len(vertices_c_boundary) > 0:
            Boundary_cells.append(c)
        if len(vertices_c_wound) > 0:
            Wound_cells.append(c)
            
    #print(np.sum(G))
    calculate_force_partial = partial(calculate_force_active_single, 
                                      point_region=point_region, regions=regions, 
                                      vertices=vertices, centers=centers, 
                                      wloc=wloc, boundary_tissue=boundary_tissue, 
                                      Boundary_cells=Boundary_cells, Wound_cells=Wound_cells,
                                      Fcenters=Fcenters)
    # Use multiprocessing to calculate forces in parallel

    with Pool() as pool:
        F_V = pool.map(calculate_force_partial, range(LV))

    return np.array(F_V)


##################################################### Auxiliary code #############################################################
def displacement_vertex(vertices, vertex, h,dir,pos):
    """Displaces a vertex position by ±h in given direction
    
    Args:
        vertices: Array of all vertex positions
        vertex: Index of vertex to displace
        h: Step size
        dir: Direction vector [x,y]
        pos: Multiplier (+1/-1) for forward/backward step
        
    Returns:
        Updated vertices array with displaced vertex
    """
    vertices[vertex] = vertices[vertex]+h*dir*pos
    return vertices


###Parallel implementation of the force calculation (Seems to be working for now)


def stress_cell(regions, point_region, vertices,centers,F):
    """Calculates cellular stress as force dipole moment
    
    For each cell α:
    S_α = Σ_i (r_iα · F_i) where:
    - r_iα = vertex i position relative to cell α center
    - F_i = force on vertex i
    
    Returns list of stresses for each cell
    """
    S = []
    for alpha in point_region:
        Nv = len(regions[alpha])
        list_vertices = vertices[regions[alpha]]
        list_forces = F[regions[alpha]]
        loc_cell = centers[alpha]
        Sa = 0
        for i in range(Nv):
            ria = list_vertices[i]-loc_cell
            nia = ria/(fc.norm(ria)+1e-4)
            Sa += np.dot(list_forces[i],nia)
        S.append(Sa)
    return S


def cells_avg_vtx(regions,point_region,cells,vertices):
    """Updates cell positions as centroids of their vertices
    
    This is a relaxation step that helps maintain 
    reasonable cell center positions during dynamics
    """
    for i in range(len(cells)):
        Neigh_c = fc.find_center_region(regions,point_region, i)
        avg_vc = np.mean(vertices[Neigh_c],0)
        cells[i] = avg_vc
        
    return cells
