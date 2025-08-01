import numpy as np
from vertexmodelpack import connections as fc
from vertexmodelpack import geomproperties as gmp
from scipy.spatial import Voronoi


# from multiprocessing import Pool
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

def energy_vor(center, tissue_var, par_vtx):
    
    """
    Unified voronoi model energy function:
    
    Args:
        center: center index to calculate energy for
        tissue_var: Tuple containing tissue variables (point_region, regions, ridges, vertices)
        par_vtx: Tuple containing Voronoi model parameters (K, A0, G, PO)
        
    Returns:
        Energy contribution for the specified center
    """
    
    K, A0, G, P0 = par_vtx
    point_region, regions, ridges, vertices = tissue_var
    ridges = fc.remove_minus(ridges)
    
    regions2 = []
    point_region2 = []

    for i in range(len(point_region)):
        point_region2.append(i)
        regions2.append(regions[point_region[i]])
    
    # Get neighbour cells and vertices
    N_c = fc.find_center_neighbour_center(regions2, point_region2,center)
    
    # Initialize energy
    E = 0
    
    Ncw = list(N_c)
        
    # Perimeter and area terms 
    P = gmp.perimeters_vor(point_region2,regions2,vertices,ridges, Ncw)
    A = gmp.areas_vor(point_region2,regions2,vertices, ridges, Ncw)
    A0c = [A0[i] for i in Ncw]
    
    # Energy from area and perimeter terms
    dA = np.array(A)-np.array(A0c)
    dP = np.array(P)-P0
    # if np.max(dP) > 5:
    #     print('Debug this shit')
    #     print(P0)
    #     print(P)
    #     print(np.max(dP))
    #     print(center)
    #     print(vertices[regions[point_region[center]]])

    ESum = 0.5*(np.array(K)[Ncw]*dA**2 + np.array(G)[Ncw]*dP**2)
    E = np.sum(ESum)
    
    return E   

def force_vor_finite_grad(center, centers,par_list, h):
    """
    Unified force calculation via finite differences that handles the standard voronoi model 
    
    Args:
        h: Finite difference step size
        Other params match energy_vor
        
    Returns:
        Force vector [Fx, Fy] calculated via central differences
    """
    #Initialize force vector
    f_v = 0.0*np.array([1.,1.])
        
    # Unit vectors for finite differences
    n1 = np.array([1,0])
    n2 = np.array([0,1])
    
    # Perturbed positions in x and y
    
    new_vor = {'1x':displacement_center(np.array(centers),center,h,n1,-1),
                    '2x':displacement_center(np.array(centers),center,h,n1,1), 
                    '1y':displacement_center(np.array(centers),center,h,n2,-1), 
                    '2y':displacement_center(np.array(centers),center,h,n2,1)
                    }
    
    # Calculate energies for all perturbations
    energies = {}
    for direction in new_vor.keys():
        pregions = new_vor[direction][1]
        regions = new_vor[direction][2]
        ridges = new_vor[direction][3]
        vertices = new_vor[direction][0]
        
        tissue_var = (pregions,regions ,ridges ,vertices )
        energies[direction] = energy_vor(center, tissue_var,par_list)

    
    # Compute finite difference gradients
    dEdx = 0.5*(energies['2x']-energies['1x'])/h
    dEdy = 0.5*(energies['2y']-energies['1y'])/h
    
    f_c = -(dEdx*n1 + dEdy*n2) # F = -∇E
    return f_c

def calculate_vor_force(c, centers, par_list, h, r0):
    
    """
    Common force calculation logic used by both serial and parallel versions
    
    Args:
        r0: Interaction cutoff radius
    """
    
    vor = Voronoi(centers)
    point_region = vor.point_region
    regions = vor.regions
    
    # Calculate vertex model force
    f_c = centers[c]*0
    if np.linalg.norm(centers[c]) <= 3.5:
        f_c += force_vor_finite_grad(c, centers, par_list, h)
        
    # Regularizing cell-cell repulsion force
    NeighC = fc.find_center_neighbour_center(regions, point_region, c)
    for i in range(len(NeighC)):
        
        ci = centers[i]
        cj = centers[c]
        rij = fc.norm(cj - ci)
        dij = max(rij, h)
        nij = (cj - ci) / dij
        if rij <= r0:
            f_c += 0.1 * (rij - r0) * nij
            

    return f_c

def force_vor_serial(centers, par_list,h):
    
    """
    Combined serial computation of elastic forces for standard Voronoi model
    
    Args:
        tissue_list: Tuple of (point_region, regions, ridges, vertices, centers, boundary_tissue)
        par_list: Tuple of (K, A0, G, L) vertex parameters
        h: Finite difference step size
        
    Returns:
        Array of force vectors for all centers
    """
    LC = len(centers) #Number of vertices
    F_C = [] # Vector of resulting forces acting on the vertices
    r0 = np.sqrt(25/(np.pi*LC))#np.sqrt(np.mean(par_list[1])/np.pi)
    for c in range(LC):
        f_c = calculate_vor_force(c,centers, par_list,h,r0)
        F_C.append(f_c)
        
    return np.array(F_C)
    
def force_vor_parallel(centers, par_list,h,pool):
    """
    Combined parallel computation of elastic forces
    
    Args:
        pool: Initialized multiprocessing Pool instance
        ... (other params same as serial version)
    """
    LC = len(centers)
    r0 = np.sqrt(25/(np.pi*LC))#np.sqrt(np.mean(par_list[1]) / np.pi)
    calculate_force_partial = partial(calculate_vor_force, centers= centers, par_list=par_list, h=h, r0=r0)    
    F_C = pool.map(calculate_force_partial, range(LC))

    return np.array(F_C)

def energy_vor_vector(centers, K,A0,G,L):
    """
    Computes energy contributions per center (vectorized).
    
    Returns:
        Array of energies where each element represents the 
        energy contribution from a single center's neighborhood.
        
    Note: 
        Useful for identifying local energy hotspots or
        visualizing energy distributions across the tissue.
    """
    
    #Total tissue energy
    LC = len(centers)
    E = []
    vor = Voronoi(centers)
    
    tissue_list = (vor.point_region, vor.regions, vor.ridge_vertices, vor.vertices)
    par_list = (K, A0, G, L)
    for c in range(LC):
        E.append(energy_vor(c, tissue_list, par_list)) 
    
    return np.array(E)



def energy_vor_total(centers, par_list):
    """
    Computes the total mechanical energy of the entire tissue system.
    
    Args:
        centers: Array of vertex positions
        K: Array of area elasticities for each cell
        A0: Array of target areas for each cell
        G: Array of perimeter elasticities
        P0: Array of target perimeters

        
    Returns:
        Total energy (float) summing:
        1. Area elasticity: K/2*(A-A0)^2
        2. Perimeter elasticity: G/2*(P-P0)^2 
    """
    vor = Voronoi(centers)
    #point_region, regions, ridges, vertices, boundary_tissue = tissue_list
    K, A0, G, P0 = par_list
    #Total tissue energy
    
    P = gmp.perimeters_vor(vor.point_region,vor.regions,vor.vertices,vor.ridge_vertices, np.arange(len(centers)))
    A = gmp.areas_vor(vor.point_region,vor.regions,vor.vertices,vor.ridge_vertices,  np.arange(len(centers)))
    
    EA = 0.5*np.array(K)*(np.array(A)-np.array(A0))**2
    EP = 0.5*np.array(G)*(np.array(P)-P0)**2
    #EP2 = -0.5*L*np.array(P)
    
    E = np.sum(EA+EP) #Sets minimum energy to 0

    
            
            
    return (E, np.sum(EA), np.sum(EP))


#####################################################Activity ####################################################################

def get_boundary_cells(centers, num_cells,OuterBoundary,InnerBoundary):
    Boundary_cells = []
    Wound_cells = []
    
    
    for c in range(num_cells):
        if np.linalg.norm(centers[c]) >= OuterBoundary:
            Boundary_cells.append(c)
        if np.linalg.norm(centers[c]) <= InnerBoundary:
            Wound_cells.append(c)
    return Boundary_cells, Wound_cells


def calculate_cell_active_force(c, J, f_centers,OuterBoundary, InnerBoundary, coords):
    """
    Computes active force for a single cell accounting for:
    1. Neighbor alignment (magnitude J)
    2. Boundary conditions:
       - Zero force on boundary/wound cells
       - Tangential forces near wound/boundary cells
    
    Args:
        c: Index of current cell
        coords: Cell center positions
        
    Returns:
        Active force vector for cell c
    """
    
        # Find the boundary and wound cells
    Boundary_cells, Wound_cells = get_boundary_cells(coords,len(f_centers), OuterBoundary, InnerBoundary)
    
    if c in Boundary_cells or c in Wound_cells:
        return np.zeros(2)
    
    f_c = np.zeros(2)
    vor = Voronoi(coords)
    NeighC = fc.find_center_neighbour_center(vor.regions, vor.point_region, c)
    
    
    for n in NeighC:  
        
        if n == c:
            continue
        f_n = f_centers[n]
        f_norm = max(np.linalg.norm(f_n),1e-10)
        if (n in Boundary_cells) or (n in Wound_cells):
            # Tangential projection near boundaries
            r_n = coords[n]
            r_norm = max(np.linalg.norm(r_n),1e-10)
            
            t_n = np.array([-r_n[1]/r_norm, r_n[0]/r_norm])
            f_c += np.dot((f_centers[n])/f_norm, t_n) * t_n*J  # Project the force onto the perpendicular direction, Fn = 0 at the wound boundary
        else:
            f_c += (f_n/f_norm)*J# Project the force onto the direction of the alignment
    
    return f_c 



def force_cell_active_gen(J, f_centers,OuterBoundary, InnerBoundary, coords, parallel=False, pool=None):
    """Calculates active cell forces with alignment coupling
    
    Serial and Parallel implementation of active force calculation.
    
    Uses multiprocessing.Pool to distribute:
    1. Cell-wise force calculations
    2. Neighborhood queries
    3. Boundary condition checks
    
    Implements:
    - Force alignment between neighbors (magnitude J)
    - Special boundary conditions:
      * Zero force on boundary/wound cells
      * Tangential forces near wound
    """
    
    LC = len(f_centers)
    if parallel:
        calculate_force_partial = partial(calculate_cell_active_force, 
                                      J=J, f_centers=f_centers, OuterBoundary=OuterBoundary, InnerBoundary=InnerBoundary, coords=coords)
    
    
        dEalign = np.array(pool.map(calculate_force_partial, range(LC)))
    else:
        dEalign = np.array([calculate_cell_active_force(c, J, 
                                                  f_centers, OuterBoundary, InnerBoundary, coords) for c in range(LC)])
        
    return dEalign
            

##################################################### Auxiliary code #############################################################
def displacement_center(centers, center, h,dir,pos):
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
    centers[center] = centers[center]+h*dir*pos
    vor = Voronoi(centers)
    vertices = vor.vertices
    
    return vertices, vor.point_region, vor.regions, vor.ridge_vertices


###Parallel implementation of the force calculation (Seems to be working for now)


def stress_cell(centers,F):
    """Calculates cellular stress as force dipole moment
    
    For each cell α:
    S_α = Σ_i (r_iα · F_i) where:
    - r_iα = vertex i position relative to cell α center
    - F_i = force on vertex i
    
    Returns list of stresses for each cell
    """
    S = []
    vor = Voronoi(centers)
    
    for alpha in len(centers):
        Nc = fc.find_center_neighbour_center(vor.regions,vor.point_region, alpha)
        list_centers = centers[Nc]
        list_forces = F[Nc]
        loc_cell = centers[alpha]
        Sa = 0
        for i in range(Nc):
            ria = list_centers[i]-loc_cell        
            Sa += np.outer(list_forces[i]-F[alpha],ria)

        S.append(Sa)
    return S

# def single_cell_avg_vtx(i, regions, point_region, vertices):
#     Neigh_c = fc.find_center_region(regions,point_region, i)
#     avg_vc = np.mean([vertices[c] for c in Neigh_c],0)
#     return avg_vc
    
# def cells_avg_vtx(regions,point_region,cells,vertices, pool):
#     """Updates cell positions as centroids of their vertices
#     Parallel version using multiprocessing
    
#     This is a relaxation step that helps maintain 
#     reasonable cell center positions during dynamics
#     """
#     new_cells = np.empty_like(cells)
    
#     partial_cell_avg = partial(single_cell_avg_vtx, 
#                                regions=regions, 
#                                point_region=point_region, 
#                                vertices=vertices)
#     LC = len(cells)
    
#     new_cells = np.array(pool.map(partial_cell_avg, range(LC)))

        
#     return new_cells
