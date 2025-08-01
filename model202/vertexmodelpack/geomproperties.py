import numpy as np
from vertexmodelpack import connections as fc

#Calculates geometrical properties of the tissue network, which may be relevant for simulations.

def perimeter_vor(point_region,regions,vertices,ridges,i):
    """
    Calculates perimeter of a single Voronoi cell.
    
    Inputs:
        point_region: Mapping from points to regions
        regions: List of Voronoi regions
        vertices: Vertex coordinates
        ridges: Voronoi ridge information
        i: Index of target cell
        
    Output:
        P: Perimeter length (float)
    """

    R = fc.find_center_region(regions,point_region,i)
    
    #Calculate the perimeter term
    if len(R)>2:
        rearrange_loc = fc.rearrange(len(R),fc.adj_mat(R,ridges))
        if len(rearrange_loc)==len(R):
            V = vertices[R][rearrange_loc]
        else:
            V = vertices[R]
        #P = 0
        V1 = np.array([v for v in V if np.linalg.norm(v) <= 10.5])
        if len(V1) <= 2:
            P = 0
        else:
            diffs = np.roll(V1,-1, axis=0)-V1
            
            P = np.sqrt(np.einsum('ij,ij->i',diffs,diffs, optimize='optimal')).sum() #np.sum(np.linalg.norm(np.roll(V,-1,axis=0)-V,axis=1))
            # for i in range(len(V)):
            #     P += fc.norm(V[(i+1)%len(V)]-V[i])
    else:
        P = 0
    return P

def area_vor(point_region,regions,vertices,ridges,i,to_print = False):
    """
    Calculates area of a single Voronoi cell.
    
    Inputs:
        point_region: Mapping from points to regions
        regions: List of Voronoi regions
        vertices: Vertex coordinates  
        ridges: Voronoi ridge information
        i: Index of target cell
        to_print: Debug flag
        
    Output:
        Area (float)
    """

    R = fc.find_center_region(regions,point_region,i)

    
    #Calculate the area term
    if len(R)>2:
        rearrange_loc = fc.rearrange(len(R),fc.adj_mat(R,ridges),to_print)
    
        if len(rearrange_loc)==len(R):
            V = vertices[R][rearrange_loc]
        else:
            V = vertices[R]
        #A1 = 0
        V1 = np.array([v for v in V if np.linalg.norm(v) <= 10.5])
        if len(V1) <= 2:
            A1 = 0
        else:
            rolled = np.roll(V1,-1,axis=0)
            x,y = V1[:,0], V1[:,1]
            x_r,y_r = rolled[:,0], rolled[:,1]
            A1 = np.sum(x*y_r-y*x_r)#np.sum(np.cross(V, rolled))
            #for i in range(len(V)):
            #    A1 += np.cross(V[i],V[(i+1)%len(V)])  
    else:
        A1 = 0
                
    return 1/2*np.abs(A1)


def perimeters_vor(point_region,regions,vertices,ridges,list_i):
    """
    Calculates perimeters for multiple Voronoi cells.
    
    Inputs:
        point_region: Mapping from points to regions  
        regions: List of Voronoi regions
        vertices: Vertex coordinates
        ridges: Voronoi ridge information
        list_i: Indices of target cells
        
    Output:
        Plist: List of perimeters
    """

    Rlist = [fc.find_center_region(regions,point_region,i) for i in list_i]
    Plist = []
    for R in Rlist:
    #Calculate the perimeter term
        if len(R)>2:
            rearrange_loc = fc.rearrange(len(R),fc.adj_mat(R,ridges))
            if len(rearrange_loc)==len(R):
                V = vertices[R][rearrange_loc]
            else:
                V = vertices[R]
                
            V1 = np.array([v for v in V if np.linalg.norm(v) <= 10.5])
            if len(V1) <= 2:
                P = 0
            else:
                diffs = np.roll(V1,-1, axis=0)-V1
                P = np.sqrt(np.einsum('ij,ij->i',diffs,diffs, optimize='optimal')).sum()
        else:
            P = 0
        Plist.append(P)
    return Plist


def areas_vor(point_region,regions,vertices,ridges,list_i,to_print = False):
    """
    Calculates areas for multiple Voronoi cells.
    
    Inputs/Outputs same as perimeters_vor but returns areas
    """
    Rlist = [fc.find_center_region(regions,point_region,i) for i in list_i]
    Alist = []
    for R in Rlist:
        #Calculate the area term
        if len(R)>2:
            rearrange_loc = fc.rearrange(len(R),fc.adj_mat(R,ridges),to_print)
        
            if len(rearrange_loc)==len(R):
                V = vertices[R][rearrange_loc]
            else:
                V = vertices[R]
            V1 = np.array([v for v in V if np.linalg.norm(v) <= 10.5])

            if len(V1) <= 2:
                A1 = 0
            else:
                rolled = np.roll(V1,-1,axis=0)
                x,y = V1[:,0], V1[:,1]
                x_r,y_r = rolled[:,0], rolled[:,1]
                A1 = np.sum(x*y_r-y*x_r)#np.sum(np.cross(V, rolled))
            #A1 = np.sum(np.cross(V,np.roll(V,-1,axis=0)))
            #A1 = np.sum(np.array([np.cross(V[i],V[(i+1)%len(V)])  for i in range(len(V))]))
        else:
            A1 = 0
        Alist.append(1/2*np.abs(A1))
    return Alist
            

def area_time(PointRegion,Regions,Vertices,Ridges,wloc,NIter):
    """
    Tracks area over time for a specific location.
    
    Inputs:
        PointRegion: Time series of point_region
        Regions: Time series of regions
        Vertices: Time series of vertices  
        Ridges: Time series of ridges
        wloc: Target location index
        NIter: Number of time steps
        
    Output:
        Array of areas over time
    """
    area_list = np.empty(NIter)#[]
    for i in range(NIter):
        #areaWound = area_vor(PointRegion[i],Regions[i],Vertices[i],Ridges[i],wloc)
        #area_list.append(areaWound)
        area_list[i] = area_vor(PointRegion[i],Regions[i],Vertices[i],Ridges[i],wloc)
        
    return np.array(area_list)

def perimeter_time(PointRegion,Regions,Vertices,Ridges,wloc,NIter):
    """
    Tracks perimeter over time for a specific location.
    
    Inputs:
        PointRegion: Time series of point_region
        Regions: Time series of regions
        Vertices: Time series of vertices  
        Ridges: Time series of ridges
        wloc: Target location index
        NIter: Number of time steps
        
    Output:
        Array of perimeters over time
    """
    perimeter_list = np.empty(NIter)#[]
    for i in range(NIter):
        #perimeterWound = perimeter_vor(PointRegion[i],Regions[i],Vertices[i],Ridges[i],wloc)
        #perimeter_list.append(perimeterWound)
        perimeter_list[i] = perimeter_vor(PointRegion[i],Regions[i],Vertices[i],Ridges[i],wloc)
        
    return np.array(perimeter_list)

def shape_neighbour(regions, pregions):
    """
    Calculates region sizes and average neighbor sizes.
    
    Inputs:
        regions: Voronoi regions
        pregions: Point-to-region mapping
        
    Output:
        (lreg, lneigh): Tuple of region sizes and average neighbor sizes
    """
    lreg = []
    lneigh = []
    for c in pregions:
        lreg.append(len(fc.find_center_region(regions,pregions,c)))
        Nc = fc.find_center_neighbour_center(regions,pregions,c)
        lneigh.append(np.mean([len(fc.find_center_region(regions,pregions,k)) for k in Nc]))
        
    return lreg, lneigh

def perimeter_strain_over_time(locCells,Regions,Vertices):
    """
    Calculates perimeter strain over time for multiple cells.
    
    Inputs:
        locCells: Cell locations to track
        Regions: Time series of regions
        Vertices: Time series of vertices
        
    Output: 
        Array of strain values (time × cells)
    """
    
    cellstrain1= []
    for i in range(len(Regions)):
        straini = []
        for loc in locCells:
            regloc = Regions[i][loc]
            vtxloc =Vertices[i][regloc]
            
            regloc0 = Regions[0][loc]
            vtxloc0 =Vertices[0][regloc0]
            
            #l0 = []
            #l1 = []
            
            
            #for c in range(len(vtxloc0)-1):
            #    lc = np.sum((vtxloc0[c+1]-vtxloc0[c])**2)**0.5
            #    l0.append(lc)
            #l0.append(np.sum((vtxloc0[0]-vtxloc0[-1])**2)**0.5)
            
            # for c in range(len(vtxloc)-1):
            #     lc = np.sum((vtxloc[c+1]-vtxloc[c])**2)**0.5
            #     l1.append(lc)
            # l1.append(np.sum((vtxloc[0]-vtxloc[-1])**2)**0.5)
            
            l0 = np.linalg.norm(np.roll(vtxloc0,-1,axis=0)-vtxloc0,axis=1)
            l1 = np.linalg.norm(np.roll(vtxloc,-1,axis=0)-vtxloc,axis=1)
            st = np.sum((l0-l1)/l0)
            # st = 0
            # for l in range(len(l1)):
            #     st += (l1[l]-l0[l])/l0[l]
            straini.append(st)
        cellstrain1.append(straini)
    return np.array(cellstrain1)

def shape_tensor(regions, point_region, vertices,centers):
    """
    Calculates shape tensor for each cell.
    
    Inputs:
        regions: Voronoi regions
        point_region: Point-to-region mapping  
        vertices: Vertex coordinates
        centers: Cell center coordinates
        
    Output:
        List of shape tensors
    """
    S = []
    for alpha in point_region:
        Nv = len(regions[alpha])
        list_vertices = vertices[regions[alpha]]
        loc_cell = centers[alpha]
        # Sa = [[0,0],[0,0]]
        
        ri = list_vertices-loc_cell
        Sa = ri.T @ ri #Does tensor product for each vertex in the region and adds them all up
        # for i in range(Nv):
        #     ria = list_vertices[i]-loc_cell

        #     Sa[0][0] += ria[0]**2
        #     # print(ria[0])
        #     Sa[1][1] += ria[1]**2
        #     Sa[0][1] += ria[0]*ria[1]
        #     Sa[1][0] += ria[1]*ria[0]
        
        S.append(Sa)
    return S

def shape_anisotropy(regions,pregions,vertices, coords):
    """
    Calculates shape anisotropy index for cells.
    
    Inputs same as shape_tensor
    
    Output:
        Array of anisotropy values
    """
    Sa1 = np.array(shape_tensor(regions,pregions,vertices,coords))
    shape_anisotropy = []
    for i in range(len(Sa1)):
        eigshape = np.linalg.eig(Sa1[i])[0]
        # print(eigshape)
        eigvals = eigshape[np.argsort(np.abs(eigshape))]
        aniso = (eigvals[-1]-eigvals[0])/eigvals.sum()
        # if np.abs(eigshape[0]) >= np.abs(eigshape[1]):
        #     aniso = (eigshape[0]-eigshape[1])/(eigshape[0]+eigshape[1])
        # if np.abs(eigshape[1]) > np.abs(eigshape[0]):
        #     aniso = (eigshape[1]-eigshape[0])/(eigshape[0]+eigshape[1])
        shape_anisotropy.append(aniso)
    return np.array(shape_anisotropy)


def cell_polarity(regions,pregions,vertices, coords):
    Sa1 = np.array(shape_tensor(regions,pregions,vertices,coords))
    polar = []
    """
    Calculates cell polarity vectors.
    
    Inputs same as shape_tensor
    
    Output:
        Array of polarity vectors
    """
    for i in range(len(Sa1)):
        eigshape = np.linalg.eig(Sa1[i])
        eigval = eigshape[0]
        # print(eigshape)

        p = eigshape[1][np.argmax(np.abs(eigval))]
        # if np.abs(eigval[0]) > np.abs(eigval[1]):
        #     p = eigshape[1][0]
        # if np.abs(eigval[1]) > np.abs(eigval[0]):
        #     p = eigshape[1][1]
        polar.append(p)
    return np.array(polar)