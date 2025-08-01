import numpy as np
# from vertexmodelpack import sppvertex as sppv
from scipy.spatial import Voronoi
from vertexmodelpack import connections as fc
import random

## Topological properties

"""
T1 Transition Functions for Vertex Model Simulations

This module implements topological T1 transitions (neighbor swaps) in 2D vertex models, 
commonly used in biological tissue modeling and foam simulations.

Key Concepts:
- T1 transitions occur when an edge shrinks below critical length (thresh_len)
- Involves rewiring of neighborhood connections between four cells
- Maintains network topology while allowing tissue rearrangement
"""

# Global variables 
theta = np.pi/3
R60p = np.array([[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta), np.cos(theta)]])
R60m = np.array([[np.cos(-theta), -np.sin(-theta)],
                 [np.sin(-theta), np.cos(-theta)]])
## T1 Transitions

# def line(a,b,x):
#     """Linear function evaluation"""
#     return a*x+b

# def line_coeffs(p1,p2):
#     """
#     Calculate coefficients (slope, intercept) for line through two points
    
#     Args:
#         p1, p2: 2D coordinate points
        
#     Returns:
#         (a, b): slope (a) and y-intercept (b) of line
#     """
#     a = (p2[1]-p1[1])/(p2[0]-p1[0])
#     b = p1[1]-a*p1[0]
#     return a,b

def T1_rotations(vertices, v_orig,v_min, n_orig, n_min):
    """
    Rotates edge during T1 transition based on energy minimization
    
    Args:
        vertices: Array of all vertex positions
        v_orig, v_min: Endpoints of shrinking edge
        n_orig, n_min: Neighbor vertices of endpoints
        
    Returns:
        Tuple containing:
        - New positions for v_orig and v_min
    """
    if len(n_orig) == 0 or len(n_min) == 0:
        return v_orig, v_min #No neighbours to update
    
    old_vorig = vertices[v_orig]
    old_vmin = vertices[v_min]

    # Calculate possible new positions                
    edge_vector = old_vorig-old_vmin
    rotated_pos1 = old_vmin + R60p @ edge_vector
    rotated_pos2 = old_vmin + R60m @ edge_vector
            
    # Assign a vertex to each of the ends of the rotated vector
     
    r_vorig = old_vorig
    r_vmin = old_vmin 
    
    # Energy calculation to determine optimal rotation
    E1 = np.sum(np.linalg.norm(vertices[n_orig]-rotated_pos1, axis=1)**2)+np.sum(np.linalg.norm(vertices[n_min]-rotated_pos2, axis=1)**2)
    E2 = np.sum(np.linalg.norm(vertices[n_orig]-rotated_pos2, axis=1)**2)+np.sum(np.linalg.norm(vertices[n_min]-rotated_pos1, axis=1)**2)
    
    # Return lowest energy configuration 
    if E1 <= E2:
        r_vorig, r_vmin = rotated_pos1, rotated_pos2
    else:
        r_vorig, r_vmin = rotated_pos2, rotated_pos1
        
        
    return r_vorig, r_vmin

def T1_change_edges(ridges,vertices_neigh, vertices_neigh_min,i,v_min):
    
    """
    Updates edge connections after T1 transition
    
    Args:
        ridges: Current edge list
        vertices_neigh: Original neighbors of vertex i
        vertices_neigh_min: Original neighbors of vertex v_min
        i, v_min: Endpoints of transitioning edge
        
    Returns:
        Updated edge list with new connections
    """
    
    #print(vertices_neigh), print(vertices_neigh_min)
    loc_ridges = np.where(ridges == i)[0]

    loc_neigh_not_vm = np.where(np.array(vertices_neigh)!=v_min)[0]

    skip_parameter = int(0)
    for j in range(len(loc_ridges)):      
        if v_min in ridges[loc_ridges[j]]:
            skip_parameter += int(1)
            continue
        else:
            js = int(j-skip_parameter)
            ridges[loc_ridges[j]]= [vertices_neigh[loc_neigh_not_vm[js]],i]          
                     
    loc_ridges = np.where(ridges == v_min)[0]        
    loc_neigh_not_i = np.where(np.array(vertices_neigh_min)!= i)[0]
    #print((len(loc_ridges),len(loc_neigh_not_i),len(vertices_neigh_min)))
            
    skip_parameter = int(0)
                        
    for j in range(len(loc_ridges)):
        if i in ridges[loc_ridges[j]]:
            skip_parameter+=int(1)
            continue
        else:
            js = int(j-skip_parameter)
            ridges[loc_ridges[j]]= [vertices_neigh_min[loc_neigh_not_i[js]],v_min]
            
    

                                
    return ridges

def compute_distance_matrix(vertex_list,vertices, ridges):
    """
    Computes pairwise distance matrix for vertices
    
    Args:
        vertex_list: List of vertex indices
        vertices: Position array
        ridges: Edge list
        
    Returns:
        Distance matrix where D[i,j] = distance between i and j
    """
    
    #Compute distance matrix
    n = len(vertex_list)
    dist_matrix = np.zeros((n,n))
    
    for i in vertex_list:
        vertices_neigh = np.array(fc.find_vertex_neighbour_vertices(ridges,i), dtype = int)
        if (vertices_neigh.size)>0:
            diff = vertices[i]-vertices[vertices_neigh]
            dist_matrix[i,vertices_neigh] = np.sqrt(np.einsum('ij,ij->i', diff, diff))#np.linalg.norm(diff,axis=1)
    return dist_matrix
#To Change (switch cells of new configuration before switching vertices)


def T1transition2(vertices, ridges, regions, point_region,thresh_len):
    
    ''' This function runs through all the interior vertices on the network and does topological rearrangements (T1 transitions) 
    Using a set operation approach (change regions before doing the edge swap)
    
    Variables:
    
    vertices - list of coordinate positions for each vertex of the network
    ridges - set of all edges in the network
    regions - set of all the vertices that compose the different regions of the network
    point_region - set of the different regions of the network
    thresh_len - parameter that determines the transition length
    
    Output
    
    Updated versions of vertices, ridges and regions
    transition_counter = number of T1 transitions at this iteration of the implementation
    
    '''
    
    #make a list of vertices to run through in the next cycle
    
    transition_counter = 0
    
    # vertex_list = []
    
    # for k in range(len(vertices)):
    #     vertex_list.append(k)
    
    vertex_list = [v for v in range(len(vertices))]
    # random.shuffle(vertex_list)
    
        
    DMatrixVertices = compute_distance_matrix(vertex_list,vertices,ridges)
    
    for i in vertex_list:
        
        #First check if vertex is not empty or blown up, otherwise skip
        if np.isnan(np.sum(vertices[i])):
            continue
    
        #Find all neighbouring vertices of vertex i
        # list of neighbouring vertices lengths
        # With batch processing:
        vertices_neigh = fc.find_vertex_neighbour_vertices(ridges,i)
        list_neigh_v_not_excluded = [int(v) for v in vertices_neigh if v in vertex_list] #List of the neighbours of v that have not been excluded yet
        if len(list_neigh_v_not_excluded) <= 2:
            ridgesl =[list(l) for l in list(ridges)] 
            continue
    
        # Find closest neighbouring vertices
        #print(DMatrixVertices[i,list_neigh_v_not_excluded])
        loc_v_min = np.argmin(DMatrixVertices[i,list_neigh_v_not_excluded])
        lv_min = np.min(DMatrixVertices[i, list_neigh_v_not_excluded])
        v_min = int(list_neigh_v_not_excluded[loc_v_min])
        
        # if lv_min < thresh_len:
        #     print(list_neigh_v_not_excluded)
        #     print(DMatrixVertices[i,list_neigh_v_not_excluded])
        #     print(loc_v_min)
        #     print(v_min)

        # Find neighbours of closest vertex to vertex i
        
        vertices_neigh_min = fc.find_vertex_neighbour_vertices(ridges,v_min)
        
        #Only do this for vertices with 3 neighbours, and also avoid triangles
        if (len(vertices_neigh_min) <= 2) or (len(list(set(list_neigh_v_not_excluded).intersection(vertices_neigh_min)))>1): continue
        
        #For vertex i and neighbouring vertex v_min
        regions_neigh_v, _ = fc.find_vertex_neighbour_centers(regions,point_region,i)
        regions_neigh_vmin, _ = fc.find_vertex_neighbour_centers(regions,point_region,v_min)

        # Two cells adjacent to (i,v_min) - Discrete Poincaré dual operation on a cellular complex (Correspondence between primal graph (Vertices) and dual graph (Cells) - Finding the dual edge)
        region_common = list(set(regions_neigh_v).intersection(regions_neigh_vmin))
        if len(region_common)<= 1: continue
            
        #Region partitioning - Isolates the exclusive cell for each vertex 
        region_exc_v = list(set(regions_neigh_v).difference(region_common))
        region_exc_vmin = list(set(regions_neigh_vmin).difference(region_common))
        if (len(region_exc_v)==0) or (len(region_exc_vmin)==0): continue
        
        if lv_min <= thresh_len: 
            
        
            #Topological rearrangement is equivalent to change of neighbours.
            new_region_common = list(set(region_exc_v).union(region_exc_vmin))

            i_v, i_min = (0,1) if np.random.rand() >= 0.5 else (1,0)
        
            #New exclusive regions for i and v_min
            new_region_exc_v = region_common[i_v]   
            new_region_exc_min = region_common[i_min]

            #Region reassignment - Pachner (2-2) move on the dual complex - swaps cells in a quadrilateral formed in the dual (Delaunay) triangulation
            regions[region_exc_vmin[0]].append(i)
            regions[region_exc_v[0]].append(v_min)
            regions[new_region_exc_v].remove(v_min)
            regions[new_region_exc_min].remove(i)
                                        
            # Neighbor Set reconstruction
            edge_common_v = list(set(regions[new_region_common[0]]).intersection(regions[new_region_common[1]])) #new edge vertices between swapped regions
            new_neighv = list(set(regions[new_region_exc_v]).intersection(regions[region_exc_vmin[0]]).difference(set(edge_common_v).intersection(regions[new_region_exc_v]))) # Finds Steiner points - vertices where three cells meet after changes
            new_neighvm = list(set(regions[region_exc_v[0]]).intersection(regions[new_region_exc_min]).difference(set(edge_common_v).intersection(regions[new_region_exc_min]))) 
            #solves vertex-to-edge incidence problem in the updated dual lattice -  which edges are incident to a vertex after topological changes
            #list of neighbouring vertice lengths in the closest neighbouring vertex for rotated vector only assign vertices in vertex list  

            # Neighborhood update - combinatorial Gauss-Bonnet update, ensuring degree preservation
            vertices_neigh_min = list(set(vertices_neigh_min).difference(new_neighv).union(new_neighvm))
            vertices_neigh = list(set(vertices_neigh).difference(new_neighvm).union(new_neighv))
            
            #Maintains CW-Complex Structure, preserves Euler characteristic - [combinatorial Poincaré isomorphism]

            #Wait, you actually need to change the ridges as well, hopefully this works                        
            ridges1 = T1_change_edges(ridges,vertices_neigh,vertices_neigh_min,i,v_min) 
                                        
            # Doing the rotations before the edge change is actually harder, so instead, we will make the rotation not random after changing the edges
            vertices[i], vertices[v_min] = T1_rotations(vertices,i,v_min, vertices_neigh, vertices_neigh_min)
            
        
            transition_counter += 1
        
            vertex_list.remove(i)
            vertex_list.remove(v_min)
            list_ridges1 = list(ridges1)
            ridgesl = [list(l) for l in list_ridges1]

    return ridgesl, vertices, regions, transition_counter

