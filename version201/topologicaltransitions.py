import numpy as np
from vertexmodelpack import sppvertex as sppv
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


## T1 Transitions

def line(a,b,x):
    """Linear function evaluation"""
    return a*x+b

def line_coeffs(p1,p2):
    """
    Calculate coefficients (slope, intercept) for line through two points
    
    Args:
        p1, p2: 2D coordinate points
        
    Returns:
        (a, b): slope (a) and y-intercept (b) of line
    """
    a = (p2[1]-p1[1])/(p2[0]-p1[0])
    b = p1[1]-a*p1[0]
    return a,b

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
        - Line coefficients before/after rotation  
        - Original positions
    """
    
    old_vorig = vertices[v_orig]
    old_vmin = vertices[v_min]
    
    # Calculate rotation matrices
    a1, b1 = line_coeffs(old_vorig,old_vmin)
    R_p = np.array([[0,-1],[1,0]]) # 90 degree rotation
    R_m = np.array([[0,1],[-1,0]]) # -90 degree rotation
    
    # Calculate possible new positions                
    diff_v_vm = old_vorig-old_vmin
    new_diff_v_vm_coord_1 = old_vmin + np.sqrt(3)/2*R_p.dot(diff_v_vm)+1/2*diff_v_vm
    new_diff_v_vm_coord_2 = old_vmin + np.sqrt(3)/2*R_m.dot(diff_v_vm)+1/2*diff_v_vm
            
        
    # Assign a vertex to each of the ends of the rotated vector
     
    r_vorig = old_vorig
    r_vmin = old_vmin 
    
    # Energy calculation to determine optimal rotation
    E1 = 0
    for i in n_orig:
        E1 += np.linalg.norm(vertices[i]-new_diff_v_vm_coord_1)**2
    for j in n_min:
        E1 += np.linalg.norm(vertices[j]-new_diff_v_vm_coord_2)**2
    E2 = 0
    for k in n_orig:
        E2 += np.linalg.norm(vertices[k]-new_diff_v_vm_coord_2)**2
    for l in n_min:
        E2 += np.linalg.norm(vertices[l]-new_diff_v_vm_coord_1)**2
    
    # Return lowest energy configuration 
    if E1 <= E2:
        r_vorig = new_diff_v_vm_coord_1
        r_vmin = new_diff_v_vm_coord_2
    else:
        r_vorig = new_diff_v_vm_coord_2
        r_vmin = new_diff_v_vm_coord_1
        
    a2, b2 = line_coeffs(r_vorig,r_vmin)
        
    return r_vorig, r_vmin, (a1,b1),(a2,b2), old_vorig, old_vmin

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


def T1transitionvertex(ridges, vertices, regions, point_region,thresh_len, i, v_min, lv_min, list_neigh_v_not_excluded):
    """
    Performs T1 transition for a single vertex-edge pair
    
    Args:
        All network components + threshold length
        i: Central vertex
        v_min: Neighbor vertex forming shrinking edge
        
    Returns:
        Updated vertices, regions after transition
    """
    vertices_neigh = fc.find_vertex_neighbour_vertices(ridges,i)
    # Find neighbours of closest vertex to vertex i

    vertices_neigh_min = fc.find_vertex_neighbour_vertices(ridges,v_min)
    
    #Only do this for vertices with 3 neighbours, and also avoid triangles

    if (len(vertices_neigh_min) > 2) and (len(list(set(list_neigh_v_not_excluded).intersection(vertices_neigh_min)))<1):
        
        #For vertex i
        regions_neigh_v, __ = fc.find_vertex_neighbour_centers(regions,point_region,i)
                    
        #For neighbouring vertex v_min
        regions_neigh_vmin, __ = fc.find_vertex_neighbour_centers(regions,point_region,v_min)

        #Find neighbouring regions prior to transition
            
        region_common = list(set(regions_neigh_v).intersection(regions_neigh_vmin))
            
        region_exc_v = list(set(regions_neigh_v).difference(region_common))
    
        region_exc_vmin = list(set(regions_neigh_vmin).difference(region_common))
        
        if ((len(region_exc_v)>0) and (len(region_exc_vmin)>0)) and (len(region_common)>1): 
            if lv_min < thresh_len: 
                                    #Oh god, I forgot about the connections between the vertices and the centers which also change      

            
                #Topological rearrangement is equivalent to change of neighbours.
                new_region_common = list(set(region_exc_v).union(region_exc_vmin))
                
                

                p_1 = np.random.rand()
                i_v = 0
                i_min = 1
                if p_1 < 0.5:
                    i_v = 1
                    i_min = 0
            
                #New exclusive regions for i and v_min

                new_region_exc_v = region_common[i_v]   
                new_region_exc_min = region_common[i_min]

                #new_region_v = list(set(new_region_common).union([new_region_exc_v]))      
                #new_region_vmin = list(set(new_region_common).union([new_region_exc_min]))
            
                #Removing and adding elements in new regions list
                regions[region_exc_vmin[0]].append(i)
                regions[region_exc_v[0]].append(v_min)
            
                regions[new_region_exc_v].remove(v_min)
                regions[new_region_exc_min].remove(i)
                
                edge_common_v = list(set(regions[new_region_common[0]]).intersection(regions[new_region_common[1]]))
                
                new_neighv = list(set(regions[new_region_exc_v]).intersection(regions[region_exc_vmin[0]]).difference(set(edge_common_v).intersection(regions[new_region_exc_v])))
                new_neighvm = list(set(regions[region_exc_v[0]]).intersection(regions[new_region_exc_min]).difference(set(edge_common_v).intersection(regions[new_region_exc_min])))
    
                #new_neighvm = list(set(vertices_neigh).intersection(regions[region_exc_vmin[0]]))
                #new_neighv = list(set(vertices_neigh_min).intersection(regions[region_exc_v[0]]))

                #list of neighbouring vertice lengths in the closest neighbouring vertex for rotated vector only assign vertices in vertex list  
                vertices_neigh_min = list(set(vertices_neigh_min).difference(new_neighv).union(new_neighvm))
                vertices_neigh = list(set(vertices_neigh).difference(new_neighvm).union(new_neighv))
                
                
                
                # Doing the rotations before the edge change is actually harder, so instead, we will make the rotation not random after changing the edges
                vertices[i], vertices[v_min] = T1_rotations(vertices,i,v_min, vertices_neigh, vertices_neigh_min)[:2]
                
    return vertices, regions, point_region

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
    dist_matrix = np.zeros((len(vertex_list),len(vertex_list)))
    
    for i in vertex_list:
        vertices_neigh = np.array(fc.find_vertex_neighbour_vertices(ridges,i), dtype = int)
        if len(vertices_neigh)>0:
            dist_matrix[i,vertices_neigh] = np.linalg.norm(vertices[i]-vertices[vertices_neigh],axis=1)
    return dist_matrix
#To Change (switch cells of new configuration before switching vertices)

def T1transition3(vertices, ridges, regions, point_region,thresh_len):
    
    """
    Main T1 transition function - processes all eligible edges
    
    Args:
        Network components + threshold length
        
    Returns:
        Updated network + count of transitions performed
    """
    
    #make a list of vertices to run through in the next cycle
    
    transition_counter = 0
    
    vertex_list = [v for v in range(len(vertices))]
    
    # for k in range(len(vertices)):
    #     vertex_list.append(k)
        
    # random.shuffle(vertex_list)
    
        
    DMatrixVertices = compute_distance_matrix(vertex_list,vertices,ridges)
    for i in vertex_list:
        
        #First check if vertex is not empty or blown up, otherwise skip
        if np.isnan(np.sum(vertices[i])):
            continue
        else:
        
            #Find all neighbouring vertices of vertex i
        
            vertices_neigh = fc.find_vertex_neighbour_vertices(ridges,i)

            list_neigh_v_not_excluded = [] #List of the neighbours of v that have not been excluded yet
            for v in vertices_neigh:
                if v in vertex_list:
                    list_neigh_v_not_excluded.append(int(v))
        
            if len(list_neigh_v_not_excluded) > 2:
            
                # Find closest neighbouring vertices
        
                loc_v_min = np.argmin(DMatrixVertices[i,list_neigh_v_not_excluded])
                lv_min = DMatrixVertices[i,loc_v_min]
                v_min = int(list_neigh_v_not_excluded[loc_v_min])
                vertices_neigh_min = fc.find_vertex_neighbour_vertices(ridges,v_min)
        
                # Do the T1 transition
                vertices, regions, point_region = T1transitionvertex(ridges, vertices, regions, point_region,thresh_len, i, v_min, lv_min, list_neigh_v_not_excluded)    
                transition_counter += 1
                vertex_list.remove(i)
                
                #Wait, you actually need to change the ridges as well, hopefully this works                        
                ridges1 = T1_change_edges(ridges,vertices_neigh,vertices_neigh_min,i,v_min) 
                list_ridges1 =  list(ridges1)
                
                ridgesl = [list(l) for l in list_ridges1]
                    
            elif len(list_neigh_v_not_excluded) <= 2:
                ridgesl =[list(l) for l in list(ridges)] 
                continue

    return ridgesl, vertices, regions, transition_counter


# #To Change (switch cells of new configuration before switching vertices)

# def T1transition2(vertices, ridges, regions, point_region,thresh_len):
    
#     ''' This function runs through all the interior vertices on the network and does topological rearrangements (T1 transitions) 
#     Using a set operation approach (change regions before doing the edge swap)
    
#     Variables:
    
#     vertices - list of coordinate positions for each vertex of the network
#     ridges - set of all edges in the network
#     regions - set of all the vertices that compose the different regions of the network
#     point_region - set of the different regions of the network
#     thresh_len - parameter that determines the transition length
    
#     Output
    
#     Updated versions of vertices, ridges and regions
#     transition_counter = number of T1 transitions at this iteration of the implementation
    
#     '''
    
#     #make a list of vertices to run through in the next cycle
    
#     transition_counter = 0
    
#     # vertex_list = []
    
#     # for k in range(len(vertices)):
#     #     vertex_list.append(k)
    
#     vertex_list = [v for v in range(len(vertices))]
        
#     # random.shuffle(vertex_list)
    
        
#     DMatrixVertices = compute_distance_matrix(vertex_list,vertices,ridges)
#     for i in vertex_list:
        
#         #First check if vertex is not empty or blown up, otherwise skip
#         if np.isnan(np.sum(vertices[i])):
#             continue
#         else:
        
#             #Find all neighbouring vertices of vertex i
        
#             vertices_neigh = fc.find_vertex_neighbour_vertices(ridges,i)
        
#             # list of neighbouring vertices lengths
#             # With batch processing:
#             #neighbors = np.array(list(vertices_neigh), dtype=int)
#             #deltas = vertices[i] - vertices[neighbors]
#             #list_len = np.sqrt(np.sum(deltas**2, axis=1))
#             #list_len = []
#             list_neigh_v_not_excluded = [int(v) for v in vertices_neigh if v in vertex_list] #List of the neighbours of v that have not been excluded yet
#             # for v in vertices_neigh:
#             #     #Only include the vertices that have not been in cycle yet
#             #     if v in vertex_list:
#             #         list_neigh_v_not_excluded.append(int(v))
#             #         #deltax = vertices[i]-vertices[int(v)]
#             #         #list_len.append(np.sqrt(np.sum(deltax**2)))
        
#             if len(list_neigh_v_not_excluded) > 2:
            
#                 # Find closest neighbouring vertices
        
#                 loc_v_min = np.argmin(DMatrixVertices[i,list_neigh_v_not_excluded])
#                 lv_min = np.min(DMatrixVertices[i, list_neigh_v_not_excluded])
#                 v_min = int(list_neigh_v_not_excluded[loc_v_min])
        
#                 # Find neighbours of closest vertex to vertex i
        
#                 vertices_neigh_min = fc.find_vertex_neighbour_vertices(ridges,v_min)
                
#                 #Only do this for vertices with 3 neighbours, and also avoid triangles
            
#                 if (len(vertices_neigh_min) > 2) and (len(list(set(list_neigh_v_not_excluded).intersection(vertices_neigh_min)))<1):
                    
#                     #For vertex i
#                     regions_neigh_v, center = fc.find_vertex_neighbour_centers(regions,point_region,i)
                                
#                     #For neighbouring vertex v_min
#                     regions_neigh_vmin, center = fc.find_vertex_neighbour_centers(regions,point_region,v_min)

#                     #Find neighbouring regions prior to transition
                        
#                     region_common = list(set(regions_neigh_v).intersection(regions_neigh_vmin))
                        
#                     region_exc_v = list(set(regions_neigh_v).difference(region_common))
                
#                     region_exc_vmin = list(set(regions_neigh_vmin).difference(region_common))
                    
#                     if ((len(region_exc_v)>0) and (len(region_exc_vmin)>0)) and (len(region_common)>1): 
#                         if lv_min < thresh_len: 
#                                                 #Oh god, I forgot about the connections between the vertices and the centers which also change      
        
                        
#                             #Topological rearrangement is equivalent to change of neighbours.
#                             new_region_common = list(set(region_exc_v).union(region_exc_vmin))
                            
                            

#                             p_1 = np.random.rand()
#                             i_v = 0
#                             i_min = 1
#                             if p_1 < 0.5:
#                                 i_v = 1
#                                 i_min = 0
                        
#                             #New exclusive regions for i and v_min

                            
#                             new_region_exc_v = region_common[i_v]   
#                             new_region_exc_min = region_common[i_min]

                        
#                             new_region_v = list(set(new_region_common).union([new_region_exc_v]))      
#                             new_region_vmin = list(set(new_region_common).union([new_region_exc_min]))
                        
#                             #Removing and adding elements in new regions list

#                             regions[region_exc_vmin[0]].append(i)
#                             regions[region_exc_v[0]].append(v_min)
                        
                            
                        
#                             regions[new_region_exc_v].remove(v_min)
#                             regions[new_region_exc_min].remove(i)
                            

                            
                            
#                             edge_common_v = list(set(regions[new_region_common[0]]).intersection(regions[new_region_common[1]]))
#                             #print(edge_common_v)
                        
#                             new_neighv = list(set(regions[new_region_exc_v]).intersection(regions[region_exc_vmin[0]]).difference(set(edge_common_v).intersection(regions[new_region_exc_v])))

#                             new_neighvm = list(set(regions[region_exc_v[0]]).intersection(regions[new_region_exc_min]).difference(set(edge_common_v).intersection(regions[new_region_exc_min])))
                            

                        
                        
#                             #new_neighvm = list(set(vertices_neigh).intersection(regions[region_exc_vmin[0]]))
#                             #new_neighv = list(set(vertices_neigh_min).intersection(regions[region_exc_v[0]]))
                        
                        
#                             #list of neighbouring vertice lengths in the closest neighbouring vertex for rotated vector only assign vertices in vertex list  

#                             vertices_neigh_min = list(set(vertices_neigh_min).difference(new_neighv).union(new_neighvm))
#                             vertices_neigh = list(set(vertices_neigh).difference(new_neighvm).union(new_neighv))
                            


#                             #Wait, you actually need to change the ridges as well, hopefully this works                        
#                             ridges1 = T1_change_edges(ridges,vertices_neigh,vertices_neigh_min,i,v_min) 
                            
#                             # Doing the rotations before the edge change is actually harder, so instead, we will make the rotation not random after changing the edges
#                             vertices[i], vertices[v_min] = T1_rotations(vertices,i,v_min, vertices_neigh, vertices_neigh_min)[:2]
                            
                        
#                             transition_counter += 1
                        
#                             vertex_list.remove(i)
#                             list_ridges1 = list(ridges1)
#                             ridgesl = [list(l) for l in list_ridges1]
                    
#             elif len(list_neigh_v_not_excluded) <= 2:
#                 ridgesl =[list(l) for l in list(ridges)] 
#                 continue

#     return ridgesl, vertices, regions, transition_counter
