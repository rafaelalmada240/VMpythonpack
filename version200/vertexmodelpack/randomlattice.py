import numpy as np
from scipy.spatial import Voronoi
from vertexmodelpack import sppvertex as sppv
import matplotlib.pyplot as plt
from scipy import ndimage as nd 
from vertexmodelpack import connections as fc

def newMod(a,b):
    """Applies a symmetric periodic boundary condition to position coordinates.
    
    This function ensures positions stay within [-b, b) by:
    1. Wrapping values > b to [-b, 0)
    2. Wrapping values < -b to [0, b)
    3. Leaving values within [-b, b] unchanged
    
    Args:
        a (np.ndarray): N x 2 array of 2D positions
        b (float): Boundary size (periodicity length)
    
    Returns:
        np.ndarray: Positions mapped to [-b, b) interval
    
    Mathematical Formulation:
        For each coordinate x:
        - If x > b: (x % b) - b → [-b, 0)
        - If x < -b: -((-x) % b) + b → [0, b)
        - Else: x remains unchanged
    
    Example:
        >>> a = np.array([[3.5, -4.2], [1.0, -1.9]])
        >>> newMod(a, 2.0)
        array([[-0.5,  1.8],  # 3.5→-0.5, -4.2→1.8
               [ 1.0, -1.9]]) # unchanged
    """
    res = np.zeros(a.shape)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            res[i,j] = (a[i,j]%b-b)*(a[i,j]>=b)+(-((-a[i,j])%b)+b)*(a[i,j]<=-b) + (a[i,j])*(np.abs(a[i,j])< b)
    return res 

def newWhere(a,b):
    """Generates a sign mask for reflexive boundary conditions based on whether positions lie inside/outside a radius `b`.
    
    Args:
        a (np.ndarray): N x 2 array of 2D positions.
        b (float): Boundary radius threshold.
    
    Returns:
        np.ndarray: Mask where:
            - `+1` if position `a[i]` has norm < `b` (inside boundary),
            - `-1` if position `a[i]` has norm ≥ `b` (outside boundary).
    
    Example:
        If `a = [[1, 0], [3, 4]]` and `b = 5`:
        - Norms: [1.0, 5.0]
        - Output: [[1], [-1]].
    """
    newA = np.zeros(a.shape)
    for i in range(a.shape[0]):
        newA[i] = 1*(np.linalg.norm(a[i])<b)-1*(np.linalg.norm(a[i])>=b)

    return newA 


def thermalize(regions, point_region,cells,vel,r0):
    """Computes forces between cell centers to simulate thermalization (energy minimization).
    
    Args:
        regions (list): Voronoi regions from `scipy.spatial.Voronoi`.
        point_region (list): Maps cell indices to Voronoi regions.
        cells (np.ndarray): Cell coordinates (N x 2).
        vel (np.ndarray): Current velocities (unused in this function).
        r0 (float): Equilibrium distance scaling factor.
    
    Returns:
        np.ndarray: Force vectors (N x 2) acting on each cell.
    
    Forces are derived from a modified Lennard-Jones potential:
        - Attractive/repulsive term: `-3*(ρ² - ρ)`, where `ρ = r0 / distance`.
        - Elastic term: `-(distance - 1.25*r0)`.
        The force model combines short-range repulsion and longer-range attraction.
    """
    
    LC = len(cells)
    FC = []
    
    for c in range(LC):
        xy_c = cells[c]
        neigh_c = fc.find_center_neighbour_center(regions,point_region,c)
        f_c = 0.0*np.array([1.,1.])#+(np.random.rand(2)-0.5)*2
        for n_c in neigh_c:
            xy_n = cells[n_c]
            v_nc = xy_c-xy_n
            r_nc = fc.norm(v_nc)
            l_nc =v_nc/(r_nc+1e-2)
            
            #Modified Lennard-Jones Potential
            rho = r0/(r_nc+1e-2)
            f_c += -3*(rho**2-rho)*l_nc-(r_nc-1.25*r0)*l_nc
                
        FC.append(f_c)         
        
    return np.array(FC)


def newcoords(N):
    """Generates and thermalizes initial cell coordinates within a circular domain.
    
    Args:
        N (int): Number of cells.
    
    Returns:
        np.ndarray: Thermalized cell coordinates (N x 2).
    
    Steps:
        1. Initializes random coordinates in a circle.
        2. Iteratively applies forces (`thermalize`) to minimize energy.
        3. Enforces reflexive boundary conditions (`newWhere`).
        4. Stops when average force drops below `thresh_f` or max steps reached.
    """
    L_max = 4.5
    L_min = -4.5
    
    R = L_max
    r = R*np.sqrt(np.random.rand(N))
    th = 2*np.pi*np.random.rand(N)

    x_coord = r*np.cos(th)
    y_coord = r*np.sin(th)

    coords = np.array((x_coord,y_coord)).T

    vel_array = np.zeros((coords.shape))

    #Prior to simulation run a thermalization process

    thresh_f = 1e-3

    avg_f = 1

    dt = 1*1e-3
    DeltaL = L_max - L_min
    r0 = min(DeltaL/(3*np.sqrt(N)),0.5)

    steps = 0
    
    n_steps = int(input('Max steps for thermalization: '))
    
    print(coords.shape)
    
    fft_array = np.zeros((100,n_steps))
    avg_f0 = 1


    while (avg_f > thresh_f) and steps < n_steps:
        
        vor = Voronoi(coords)
        vorPointRegion = vor.point_region
        vorRegions = vor.regions
    
        F_center = thermalize(vorRegions,vorPointRegion,coords,vel_array,r0)
        vel_array = F_center
    
        #Periodic boundary conditions
    
        #coords = newMod(coords + vel_array*dt,5)
    
        #Reflexive boundary conditions
    
        A = newWhere(coords + vel_array*dt,5)
        coords = coords + vel_array*dt*A
        

        avg_f = np.mean(F_center**2)**0.5
        
        if steps == 0:
            avg_f0 = 1*avg_f
            
        if steps%50 == 0:
            print('steps: ')
            print(avg_f/avg_f0)
        
        steps += 1
        
    print(steps)
    print(avg_f/avg_f0)
    print("done")
    return coords