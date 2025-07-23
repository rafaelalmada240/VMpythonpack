import numpy as np
from vmpack import connections as fc
from vmpack import geomproperties as gmp


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

############################################## Standard Vertex Model + Wound Healing #########################################################

def energy_vtx(vertex, tissue_var, par_vtx, if_wound=False, par_wound=None):
    
    """
    Unified vertex model energy function that handles both cases:
    - Standard vertex model (if_wound=False)
    - Vertex model with wound modifications (if_wound=True)
    
    Args:
        vertex: Vertex index to calculate energy for
        tissue_var: Tuple containing tissue variables (point_region, regions, ridges, vertices, boundary_tissue)
        par_vtx: Tuple containing vertex parameters (K, A0, G, L)
        if_wound: Boolean flag indicating whether wound modifications should be applied
        par_wound: Tuple containing wound parameters (Lw, wloc, bound_wound) (required if if_wound=True)
        
    Returns:
        Energy contribution for the specified vertex
    """
    
    K, A0, G, L = par_vtx
    point_region, regions, ridges, vertices, boundary_tissue = tissue_var
    
    # Get neighbour cells and vertices
    R,N_c = fc.find_vertex_neighbour_centers(regions, point_region,vertex)
    N_v = fc.find_vertex_neighbour_vertices(ridges,vertex)
    
    # Initialize energy
    E = 0
    
    # Handle wound specific modifications if needed
    if if_wound:
        Lw,wloc, bound_wound = par_wound
        Ncw = [i for i in N_c if i != wloc]
        wound_mask = np.isin(N_v,bound_wound) & (vertex in bound_wound)
        
        #E = energy_vtx_v1(point_region, regions, ridges, vertices, vertex, K, A0, G, L, boundary_tissue)
    else:
        Ncw = list(N_c)
        
    # Perimeter and area terms 
    P = gmp.perimeters_vor(point_region,regions,vertices,ridges, Ncw)
    A = gmp.areas_vor(point_region,regions,vertices, ridges, Ncw)
    A0c = [A0[i] for i in Ncw]
    
    # Energy from area and perimeter terms
    dA = np.array(A)-np.array(A0c)
    ESum = 0.5*(np.array(K)[Ncw]*dA**2 + np.array(G)[Ncw]*np.array(P)**2)
    E = np.sum(ESum)
    
    # Calculate line tension terms (vectorized)
    if vertex not in boundary_tissue:
        edgeVs = vertices[vertex]-vertices[N_v]
        ljs = np.linalg.norm(edgeVs,axis=1)
        if if_wound:
            
            E += np.sum(np.where(wound_mask,Lw*ljs,-L*ljs))
        else:
            E += -L*np.sum(ljs)
    
    return E   

def force_vtx_finite_grad(vertex, vertices, point_region, regions, ridges, boundary_tissue,par_list, h, if_wound=False, par_wound= None):
    """
    Unified force calculation via finite differences that handles both:
    - Standard vertex model (if_wound=False)
    - Wound-modified vertex model (if_wound=True)
    
    Args:
        h: Finite difference step size
        if_wound: Boolean flag for wound modifications
        par_wound: Tuple of (Lw, wloc, bound_wound) if if_wound=True
        Other params match energy_vtx_v1/v2
        
    Returns:
        Force vector [Fx, Fy] calculated via central differences
    """
    #Initialize force vector
    f_v = 0.0*np.array([1.,1.])
        
    # Unit vectors for finite differences
    n1 = np.array([1,0])
    n2 = np.array([0,1])
    
    # Perturbed positions in x and y
    new_vertices = {'1x':displacement_vertex(np.array(vertices),vertex,h,n1,-1),
                    '2x':displacement_vertex(np.array(vertices),vertex,h,n1,1), 
                    '1y':displacement_vertex(np.array(vertices),vertex,h,n2,-1), 
                    '2y':displacement_vertex(np.array(vertices),vertex,h,n2,1)
                    }
    # Calculate energies for all perturbations
    energies = {}
    for direction, verts in new_vertices.items():
        tissue_var = (point_region, regions, ridges, verts, boundary_tissue)
        energies[direction] = energy_vtx(vertex, tissue_var,par_list, if_wound, par_wound)

    
    # Compute finite difference gradients
    dEdx = 0.5*(energies['2x']-energies['1x'])/h
    dEdy = 0.5*(energies['2y']-energies['1y'])/h
    
    f_v = -(dEdx*n1 + dEdy*n2) # F = -∇E
    return f_v

def calculate_vtx_force(v, tissue_list, par_list, h, r0, if_wound=False, par_wound=None):
    
    """
    Common force calculation logic used by both serial and parallel versions
    
    Args:
        wound_params: Tuple of (Lw, wloc, bound_wound) if if_wound=True
        r0: Interaction cutoff radius
    """
    
    point_region, regions, ridges, vertices, centers, boundary_tissue = tissue_list
    if v in boundary_tissue:
        return np.zeros(2)
    
    # Calculate vertex model force
    f_v = force_vtx_finite_grad(v, vertices, point_region, regions, ridges,boundary_tissue, par_list, h, if_wound, par_wound)
        
    # Regularizing cell-cell repulsion force
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

def force_vtx_serial(tissue_list, par_list,h,if_wound=False, par_wound = None):
    
    """
    Combined serial computation of elastic forces for both standard and wound cases
    
    Args:
        tissue_list: Tuple of (point_region, regions, ridges, vertices, centers, boundary_tissue)
        par_list: Tuple of (K, A0, G, L) vertex parameters
        h: Finite difference step size
        if_wound: Boolean flag for wound modifications
        par_wound: Tuple of (Lw, wloc, bound_wound) if if_wound=True
        
    Returns:
        Array of force vectors for all vertices
    """
    LV = len(tissue_list[3]) #Number of vertices
    F_V = [] # Vector of resulting forces acting on the vertices
    r0 = np.sqrt(np.mean(par_list[1])/np.pi)
    for v in range(LV):
        f_v = calculate_vtx_force(v,tissue_list, par_list,h,r0,if_wound, par_wound)
        F_V.append(f_v)
        
    return np.array(F_V)
    
def force_vtx_parallel(tissue_list, par_list,h,pool, if_wound=False, par_wound=None):
    """
    Combined parallel computation of elastic forces
    
    Args:
        pool: Initialized multiprocessing Pool instance
        ... (other params same as serial version)
    """
    LV = len(tissue_list[3])
    r0 = np.sqrt(np.mean(par_list[1]) / np.pi)
    calculate_force_partial = partial(calculate_vtx_force, tissue_list= tissue_list, par_list=par_list, h=h, r0=r0, if_wound=if_wound, par_wound=par_wound)    
    F_V = pool.map(calculate_force_partial, range(LV))

    return np.array(F_V)

def energy_vtx_total(tissue_list, par_list):
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
    point_region, regions, ridges, vertices, boundary_tissue = tissue_list
    K, A0, G, L = par_list
    #Total tissue energy
    
    P = gmp.perimeters_vor(point_region,regions,vertices,ridges, point_region)
    A = gmp.areas_vor(point_region,regions,vertices, ridges, point_region)
    
    EA = 0.5*np.array(K)*(np.array(A)-np.array(A0))**2
    EP1 = 0.5*np.array(G)*np.array(P)**2
    EP2 = -0.5*L*np.array(P)
    
    E = np.sum(EA+EP1+EP2)+len(G)*L**2/(8*np.mean(G)) #Sets minimum energy to 0
    
    EP3 = 0
    
    for vertex in range(len(vertices)):
        if vertex in set(boundary_tissue):
            N_v = fc.find_vertex_neighbour_vertices(ridges,vertex)
            
            # Only use boundary edges
            NB_v = [iv for iv in N_v if iv in boundary_tissue]
            edgeVs = vertices[vertex]-vertices[NB_v]
            ljs = np.linalg.norm(edgeVs,axis=1)
            
            #Add a half of the energy because the edges are counted twice
            E -= 0.25*L*np.sum(ljs)
            EP3 -= 0.25*L*np.sum(ljs)
            
            
    return (E, np.sum(EA), np.sum(EP1)+len(G)*L**2/(8*np.mean(G)),np.sum(EP2)+EP3)

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
            Sa += np.outer(list_forces[i],ria)

        S.append(Sa)
    return S

def single_cell_avg_vtx(i, regions, point_region, vertices):
    Neigh_c = fc.find_center_region(regions,point_region, i)
    avg_vc = np.mean([vertices[c] for c in Neigh_c],0)
    return avg_vc
    
def cells_avg_vtx(regions,point_region,cells,vertices, pool):
    """Updates cell positions as centroids of their vertices
    Parallel version using multiprocessing
    
    This is a relaxation step that helps maintain 
    reasonable cell center positions during dynamics
    """
    new_cells = np.empty_like(cells)
    
    partial_cell_avg = partial(single_cell_avg_vtx, 
                               regions=regions, 
                               point_region=point_region, 
                               vertices=vertices)
    LC = len(cells)
    
    new_cells = np.array(pool.map(partial_cell_avg, range(LC)))

        
    return new_cells
