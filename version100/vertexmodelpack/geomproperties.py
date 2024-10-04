import numpy as np
from vertexmodelpack import connections as fc

#Calculates geometrical properties of the tissue network, which may be relevant for simulations.

def perimeter_vor(point_region,regions,vertices,ridges,i):

    R = fc.find_center_region(regions,point_region,i)
    
    #Calculate the perimeter term
    if len(R)>2:
        rearrange_loc = fc.rearrange(len(R),fc.adj_mat(R,ridges))
        if len(rearrange_loc)==len(R):
            V = vertices[R][rearrange_loc]
        else:
            V = vertices[R]
        P = 0
        for i in range(len(V)):
            P += fc.norm(V[(i+1)%len(V)]-V[i])
    else:
        P = 0
    return P

def area_vor(point_region,regions,vertices,ridges,i,to_print = False):

    R = fc.find_center_region(regions,point_region,i)

    
    #Calculate the area term
    if len(R)>2:
        rearrange_loc = fc.rearrange(len(R),fc.adj_mat(R,ridges),to_print)
    
        if len(rearrange_loc)==len(R):
            V = vertices[R][rearrange_loc]
        else:
            V = vertices[R]
        A1 = 0
        for i in range(len(V)):
            A1 += np.cross(V[i],V[(i+1)%len(V)])  
    else:
        A1 = 0
                
    return 1/2*fc.norm(A1)


def perimeters_vor(point_region,regions,vertices,ridges,list_i):

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
            P = np.sum(np.array([fc.norm(V[(i+1)%len(V)]-V[i]) for i in range(len(V))]))
        else:
            P = 0
        Plist.append(P)
    return Plist


def areas_vor(point_region,regions,vertices,ridges,list_i,to_print = False):
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
            A1 = np.sum(np.array([np.cross(V[i],V[(i+1)%len(V)])  for i in range(len(V))]))
        else:
            A1 = 0
        Alist.append(1/2*fc.norm(A1))
    return Alist
            

def area_time(PointRegion,Regions,Vertices,Ridges,wloc,NIter):
    perimeter_list = []
    for i in range(NIter):
        perimeterWound = area_vor(PointRegion[i],Regions[i],Vertices[i],Ridges[i],wloc)
        perimeter_list.append(perimeterWound)
        
    return np.array(perimeter_list)

def perimeter_time(PointRegion,Regions,Vertices,Ridges,wloc,NIter):
    perimeter_list = []
    for i in range(NIter):
        perimeterWound = perimeter_vor(PointRegion[i],Regions[i],Vertices[i],Ridges[i],wloc)
        perimeter_list.append(perimeterWound)
        
    return np.array(perimeter_list)

def shape_neighbour(regions, pregions):
    lreg = []
    lneigh = []
    for c in pregions:
        lreg.append(len(fc.find_center_region(regions,pregions,c)))
        Nc = fc.find_center_neighbour_center(regions,pregions,c)
        lneigh.append(np.mean([len(fc.find_center_region(regions,pregions,k)) for k in Nc]))
        
    return lreg, lneigh

def perimeter_strain_over_time(locCells,Regions,Vertices):
    cellstrain1= []
    for i in range(len(Regions)):
        straini = []
        for loc in locCells:
            regloc = Regions[i][loc]
            vtxloc =Vertices[i][regloc]
            
            regloc0 = Regions[0][loc]
            vtxloc0 =Vertices[0][regloc0]
            
            l0 = []
            l1 = []
            for c in range(len(vtxloc0)-1):
                lc = np.sum((vtxloc0[c+1]-vtxloc0[c])**2)**0.5
                l0.append(lc)
            l0.append(np.sum((vtxloc0[0]-vtxloc0[-1])**2)**0.5)
            
            for c in range(len(vtxloc)-1):
                lc = np.sum((vtxloc[c+1]-vtxloc[c])**2)**0.5
                l1.append(lc)
            l1.append(np.sum((vtxloc[0]-vtxloc[-1])**2)**0.5)
            
            st = 0
            for l in range(len(l1)):
                st += (l1[l]-l0[l])/l0[l]
            straini.append(st)
        cellstrain1.append(straini)
    return np.array(cellstrain1)

def shape_tensor(regions, point_region, vertices,centers):
    S = []
    for alpha in point_region:
        Nv = len(regions[alpha])
        list_vertices = vertices[regions[alpha]]
        loc_cell = centers[alpha]
        Sa = [[0,0],[0,0]]
        for i in range(Nv):
            ria = list_vertices[i]-loc_cell

            Sa[0][0] += ria[0]**2
            # print(ria[0])
            Sa[1][1] += ria[1]**2
            Sa[0][1] += ria[0]*ria[1]
            Sa[1][0] += ria[1]*ria[0]
        
        S.append(Sa)
    return S

def shape_anisotropy(regions,pregions,vertices, coords):
    Sa1 = np.array(shape_tensor(regions,pregions,vertices,coords))
    shape_anisotropy = []
    for i in range(len(Sa1)):
        eigshape = np.linalg.eig(Sa1[i])[0]
        # print(eigshape)
        aniso = 1
        if eigshape[0] > eigshape[1]:
            aniso = eigshape[1]/eigshape[0]
        if eigshape[1] > eigshape[0]:
            aniso = eigshape[0]/eigshape[1]
        shape_anisotropy.append(aniso)
    return np.array(shape_anisotropy)

def cell_polarity(regions,pregions,vertices, coords):
    Sa1 = np.array(shape_tensor(regions,pregions,vertices,coords))
    polar = []
    for i in range(len(Sa1)):
        eigshape = np.linalg.eig(Sa1[i])
        # print(eigshape)
        p = 1
        if eigshape[0][0] > eigshape[0][1]:
            p = eigshape[1][0]
        if eigshape[0][1] > eigshape[0][0]:
            p = eigshape[1][1]
        polar.append(p)
    return np.array(polar)