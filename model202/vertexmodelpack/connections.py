import numpy as np
import networkx as nx


##  A utility module for analyzing and manipulating Voronoi tessellations, with applications in computational geometry, physics simulations, or biological modeling (e.g., cell networks).

################################################################ Geometric Operations #################################################################
def norm(vec):
    return np.linalg.norm(vec)

def vertices_in_bound(vertices,L):
    """Clips vertex coordinates to stay within a bounding box of size [-L, L] in all dimensions.
    
    Args:
        vertices (list or array): List of vertices (each vertex is a list/array of coordinates).
        L (float): Half the side length of the bounding box.
    
    Returns:
        list: Vertices with coordinates clipped to [-L, L].
        
    Useful for enforcing periodic or constrained boundary conditions.
    """
    
    for vertex in vertices:
        for i in range(len(vertex)):
            if vertex[i] > L:
                vertex[i] = L
            if vertex[i] < -L:
                vertex[i] = -L
                
    return vertices



############################ Find elements in neighbourhoods #################################################
                

def find_center_region(regions,point_region,center):
    """Retrieves the vertices of a Voronoi region associated with a center point, excluding invalid vertices (-1).

    Args:
        regions (list): List of all Voronoi regions (each region is a list of vertex indices).
        point_region (list): Maps each center point to its Voronoi region index.
        center (int): Index of the center point.

    Returns:
        list: Vertex indices of the region, excluding -1 (invalid vertices, often used for infinite vertices in Voronoi tessellations).
    """
    #point = point_region[center]
    #R = regions[point]
    
    #for e in R:
    #    if e == -1:
    #        R.remove(e)
    R = [v for v in regions[point_region[center]] if v != -1]
    return R

def find_vertex_neighbour_vertices(ridges,vertex):
    
    """Finds all neighboring vertices of a given vertex based on ridge (edge) connections.

    Args:
        ridges (list): List of ridges (edges) as pairs/tuples of vertex indices.
        vertex (int): Target vertex index.

    Returns:
        np.ndarray: Sorted array of adjacent vertex indices (excluding -1 and the input vertex).
    """

    list_vertex_neigh = []
       
    for ridge in ridges:
        
        if vertex in list(ridge):  
            for elem in list(ridge):
                if (elem != vertex) and (elem != -1):#This condition ensures that we don't include either the vertex v_i or the vertex at infinity
                    list_vertex_neigh.append(int(elem))
    
    return np.sort(list_vertex_neigh)

def find_vertex_neighbour_centers(regions, point_region,vertex):
    
    """Finds all Voronoi regions (and their centers) that share a given vertex.

    Args:
        regions (list): List of all Voronoi regions.
        point_region (list): Maps center points to region indices.
        vertex (int): Vertex index to query.

    Returns:
        tuple: (list_regions, list_centers) where:
            - list_regions: Indices of regions containing the vertex.
            - list_centers: Corresponding center points of those regions.
    """
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
    
    """Combines vertex and center neighbors for a given vertex.

    Args:
        regions (list): List of all Voronoi regions.
        point_region (list): Maps centers to region indices.
        ridges (list): List of ridges (edges).
        vertex (int): Target vertex index.

    Returns:
        tuple: (list_centers, list_vertex_neigh) where:
            - list_centers: Sorted array of neighboring center indices.
            - list_vertex_neigh: Sorted array of neighboring vertex indices.
    """
    
    #Gives all the neighbours of a vertex in a voronoi tesselation (removes all -1 vertices)
    
    ff = find_vertex_neighbour_centers(regions, point_region,vertex)
    if ff is None:
        list_centers = []
    else:
        list_regions, list_centers = ff 
    list_vertex_neigh = find_vertex_neighbour_vertices(ridges,vertex)
    
    
    
    return np.sort(list_centers), np.sort(list_vertex_neigh)    

def find_center_neighbour_center(regions,point_region,center):
    """Finds all neighboring Voronoi centers for a given center.

    Args:
        regions (list): List of all Voronoi regions.
        point_region (list): Maps centers to region indices.
        center (int): Target center index.

    Returns:
        list: Indices of adjacent Voronoi centers (excluding the input center).
    """
    
    # Find all neighbouring cells for a given cell i
    List_centers = []

    R = find_center_region(regions, point_region, center)
    for v in R:
        A, L_c = find_vertex_neighbour_centers(regions,point_region,v)
        List_centers = list(set(List_centers).union(L_c))
    if center in List_centers:
        List_centers.remove(center)
    return List_centers

############################################################ Boundary Detection ###########################################################

def find_boundary_vertices(n_vertices,ridges):
    """Identifies boundary vertices in a planar graph (degree < 3).

    Args:
        n_vertices (int): Total number of vertices.
        ridges (list): List of ridges (edges).

    Returns:
        list: Vertex indices on the boundary (degree ≤ 2 or adjacent to low-degree vertices).
        
    Assumption: Boundary vertices have fewer than 3 edges (bulk vertices have ≥ 3)
    """
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
    """Identifies boundary vertices in a square lattice (degree < 4).

    Args:
        n_vertices (int): Total number of vertices.
        ridges (list): List of ridges (edges).

    Returns:
        list: Vertex indices on the boundary (degree ≤ 3).
        
    Adapts find_boundary_vertices for square lattices (bulk degree = 4)
    """
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
    """Retrieves vertices of a Voronoi region representing a wound boundary.

    Args:
        regions (list): List of all Voronoi regions.
        point_region (list): Maps centers to region indices.
        wound_loc (int): Center index of the wound region.

    Returns:
        list: Vertex indices of the wound boundary (equivalent to `find_center_region`).
    
     Labels wound boundaries in biological or physical simulations.
    """
    return find_center_region(regions,point_region,wound_loc)


################################## Adjacency matrix and Graph Operation functions ###########################################

def adj_mat(R,ridges):
    """Constructs an adjacency matrix for a subset of vertices `R` based on ridge connections.
    
    Args:
        R (list): List of vertex indices.
        ridges (list): List of ridges defining connections between vertices.
    
    Returns:
        np.ndarray: Binary adjacency matrix where `1` indicates connected vertices.
    """
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



def rearrange(n, bin_mat,to_print = False):
    
    """Reorders indices based on the shortest cycle in the adjacency graph.
    
    Args:
        n (int): Number of vertices.
        bin_mat (np.ndarray): Adjacency matrix.
        to_print (bool): If True, prints the reordered cycle.
    
    Returns:
        list: Reordered indices (prioritizing cycles if they exist).
    """
        
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
    """Reorders vertices in each region based on cyclic adjacency.
    
    Args:
        regions (list): List of regions (each region is a list of vertex indices).
        point_region (list): Maps points to their respective regions.
        ridges (list): Ridge definitions for adjacency.
    
    Returns:
        list: Regions with vertices reordered cyclically.
    """
    new_regions = []
    for c in point_region:
        R = find_center_region(regions,point_region,c)
        rearrange_loc = rearrange(len(R),adj_mat(R,ridges))
        R = [R[i] for i in rearrange_loc]
        new_regions.append(R)
    return new_regions

########################################################################## Auxiliary Utility functions ##########################################################################
def flatten(l):
    """Flattens a nested list into a 1D list.
    
    Args:
        l (list): Nested list (e.g., [[a, b], [c]]).
    
    Returns:
        list: Flattened list (e.g., [a, b, c]).
    """
    return [item for sublist in l for item in sublist]
 
def nsides_vor(point_region,regions,i):
    """Returns the number of sides (vertices) of the Voronoi region for point `i`.
    
    Args:
        point_region (list): Maps points to region indices.
        regions (list): List of all regions.
        i (int): Index of the point of interest.
    
    Returns:
        int: Number of sides of the Voronoi region.
    
    A Voronoi region with n vertices is an n-sided polygon.
    """

    R = find_center_region(regions,point_region,i)
    nsides=len(R)
    return nsides

def remove_minus(ridges):
    """Removes ridges (edges) containing the placeholder value -1.
    
    Args:
        ridges (list or np.ndarray): List of ridges (each ridge is a list of vertex indices).
    
    Returns:
        list: Ridges with entries containing -1 removed.
        
        Converts NumPy arrays to lists for compatibility with list operations.
    """
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




