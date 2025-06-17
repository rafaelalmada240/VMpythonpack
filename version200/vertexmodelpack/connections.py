import numpy as np
import networkx as nx

def norm(vec):
    return np.linalg.norm(vec)

def vertices_in_bound(vertices,L):
    "Ensures all vertices are within bounds"
    
    for vertex in vertices:
        for i in range(len(vertex)):
            if vertex[i] > L:
                vertex[i] = L
            if vertex[i] < -L:
                vertex[i] = -L
                
    return vertices

def remove_minus(ridges):
    if isinstance(ridges,np.ndarray):
        ridges = ridges.tolist()
        
    index_to_remove = []
    for ridge in ridges:
        for elem in ridge:
            if elem == -1:
                index_to_remove.append(ridge)
    
    for j in index_to_remove:
        ridges.remove(j)
    return ridges

def adj_mat(R,ridges):
    arrayR = np.array(R)
    bin_mat = np.zeros((len(R),len(R)))
    for vi in R:
        N_v = find_vertex_neighbour_vertices(ridges,vi)
        loc_i = np.argwhere(arrayR==vi)[0][0]
        for vj in N_v:
            if vj in R:
                loc_j = np.argwhere(arrayR==vj)[0][0]
                bin_mat[loc_i,loc_j] = 1
    
    return bin_mat

def flatten(l):
    return [item for sublist in l for item in sublist]

def rearrange(n, bin_mat,to_print = False):
        
    G = nx.from_numpy_array(bin_mat)
    cyclesG = nx.cycle_basis(G,0)

    #print("sorted"+str(sorted(cyclesG)))
    if len(cyclesG)>0:
        arr_new = sorted(cyclesG)[0]
        if to_print == True:
            print(arr_new)
    else:
        arr_new = list(np.arange(n))

      
    return arr_new

def rearrange_regions(regions,point_region,ridges):
    new_regions = []
    for c in point_region:
        R = find_center_region(regions,point_region,c)
        rearrange_loc = rearrange(len(R),adj_mat(R,ridges))
        R = [R[i] for i in rearrange_loc]
        new_regions.append(R)
    return new_regions


 
def nsides_vor(point_region,regions,i):

    R = find_center_region(regions,point_region,i)
    nsides=len(R)
    return nsides


############################ Find elements in neighbourhoods #################################################
                

#Auxiliary functions to find locations of vertices and centers

def find_center_region(regions,point_region,center):
    '''Gives all the neighbours of a center in a voronoi tesselation (removes all -1 vertices)'''
    point = point_region[center]
    R = regions[point]
    
    for e in R:
        if e==-1:
            R.remove(e)
    return R

def find_vertex_neighbour_vertices(ridges,vertex):
    
    '''
    The function receives as input the list of ridges (equivalent to the set of edges E) and 
    an integer corresponding to the vertex v_i, and return a list of vertices v_j, such that (v_i, v_j) is in E
    '''

    list_vertex_neigh = []
       
    for ridge in ridges:
        
        if vertex in list(ridge):  
            for elem in list(ridge):
                if (elem != vertex) and (elem != -1):#This condition ensures that we don't include either the vertex v_i or the vertex at infinity
                    list_vertex_neigh.append(int(elem))
    
    return np.sort(list_vertex_neigh)

def find_vertex_neighbour_centers(regions, point_region,vertex):
    
    '''
    Inputs: 
    regions - (list) set of all the vertices that compose the different regions of the network
    Some regions are empty, but it is best not to remove them for consistency
    point_region - (list) set of the different regions of the network
    vertex - (int) a specific vertex
    
    Outputs:
    list_regions, list_centers - list of regions and corresponding centers that are neighbouring a vertex.
    
    '''
    list_regions = []
    list_centers = []
    i = 0

        
    for i in range(len(regions)):
    #Only consider neighbouring regions that form polygons
        if vertex in regions[i]:
            list_regions.append(i) 
            #print(regions[i])
            loc_points = np.where(np.array(point_region)==i)
            #print(loc_points)
            list_centers.append(loc_points[0][0])
             
    return list_regions, np.array(list_centers)

def find_vertex_neighbour(regions, point_region, ridges,vertex):
    
    '''
    Inputs: 
    regions - (list) set of all the vertices that compose the different regions of the network
    Some regions are empty, but it is best not to remove them for consistency
    point_region - (list) set of the different regions of the network
    ridges - (list) set of all edges in the network
    vertex - (int) a specific vertex
    
    Outputs:
    list_vertex neigh, list_centers - list of centers and vertices that are neighbouring a vertex.
    
    '''
    
    #Gives all the neighbours of a vertex in a voronoi tesselation (removes all -1 vertices)
    
    ff = find_vertex_neighbour_centers(regions, point_region,vertex)
    if ff is None:
        list_centers = []
    else:
        list_regions, list_centers = ff 
    list_vertex_neigh = find_vertex_neighbour_vertices(ridges,vertex)
    
    
    
    return np.sort(list_centers), np.sort(list_vertex_neigh)    

def find_center_neighbour_center(regions,point_region,center):
    
    # Find all neighbouring cells for a given cell i
    List_centers = []

    R = find_center_region(regions, point_region, center)
    for v in R:
        A, L_c = find_vertex_neighbour_centers(regions,point_region,v)
        List_centers = list(set(List_centers).union(L_c))
    List_centers.remove(center)
    return List_centers

def find_boundary_vertices(n_vertices,ridges):
    '''
    This function finds all vertices in the boundary of the network, under the initial assumption that vertices have 3 connections in planar graphs if they are in a bulk and 2 edges or less if they are in the boundary
    
    Variables:
    
    n_vertices - is the number of vertices in the network (the cardinality of the vertex set)
    
    ridges - is the set of all edges in the network
    
    Output:
    
    Bound_set - the set containing all vertices that are in the boundary, this list doesn't contain all vertices in the boundary, but it ensures all vertices in the list are in the boundary.
    '''
    vertex_list = []
    
    ridges = remove_minus(ridges)
    
    for k in range(n_vertices):
        vertex_list.append(k)
        
    #Add all vertices that have less than 2 neighbours
    Bound_set = []
    Bound_set1 = []
    Bound_set_neighbours = []
    
    for v in vertex_list:
        Neigh_V = find_vertex_neighbour_vertices(ridges,v)
        

        if len(Neigh_V) < 3:
            Bound_set.append(v)
            Bound_set1.append(v)
            Bound_set_neighbours.append(Neigh_V)

    
    #Add all vertices that are neighbouring the previous vertices
    Bound_set2 = []
    Bound_set_neighbours_2 = []
    for i in range(len(Bound_set1)):
        neigh1 = Bound_set_neighbours[i]
        for b in neigh1:
            Neigh_B = find_vertex_neighbour_vertices(ridges,b)
            if b not in Bound_set:
                Bound_set.append(b)
                Bound_set2.append(b)
                Bound_set_neighbours_2.append(Neigh_B)
    
    #Add all vertices neighbouring the vertices that were neighbours the vertices that have less than 2 neighbours           
    for j in range(len(Bound_set2)):
        for k in range(len(Bound_set2)):
            neigh2 = Bound_set_neighbours_2[j]
            neigh3 = Bound_set_neighbours_2[k]
            if j != k:
                list_c = list(set(neigh2).intersection(neigh3))
                if len(list_c)>0:
                    c = list_c[0] 
                    if c not in Bound_set:
                        Bound_set.append(c)
        
                    
    return Bound_set

def find_boundary_vertices_square(n_vertices,ridges):
    '''
    This function finds all vertices in the boundary of the network, under the initial assumption
    that vertices have 4 connections in square graphs if they are in a bulk and 3 edges or less if they are in the boundary
    This is an adaptation of find_boundary vertices for a square lattice.
    
    Variables:
    
    n_vertices - is the number of vertices in the network (the cardinality of the vertex set)
    
    ridges - is the set of all edges in the network
    
    Output:
    
    Bound_set - the set containing all vertices that are in the boundary, this list doesn't contain all vertices in the boundary, but it ensures all vertices in the list are in the boundary.
    '''
    vertex_list = []
    
    ridges = remove_minus(ridges)
    
    for k in range(n_vertices):
        vertex_list.append(k)
        
    #Add all vertices that have less than 4 neighbours
    Bound_set = []

    for v in vertex_list:
        Neigh_V = find_vertex_neighbour_vertices(ridges,v)
        
        if len(Neigh_V) < 4:
            Bound_set.append(v)
                    
    return Bound_set

def find_wound_boundary(regions, point_region, wound_loc):
    return find_center_region(regions,point_region,wound_loc)


