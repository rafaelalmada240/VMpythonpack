import numpy as np
from vmpack import randomlattice as rlt
from vmpack import geomproperties as gmp
from scipy.spatial import Voronoi
from vmpack import connections as fc
import itertools
from vmpack import readTissueFiles as rTF
import time

ti = time.time()
def newAbs(a,b):
    res = np.zeros(a.shape)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if np.abs(a[i,j])<b:
                res[i,j] = a[i,j]
            else:
                res[i,j] = np.sign(a[i,j])*b
    return res 

option_1 = int(input(" Do you want to generate a random lattice: y(1) or n(0): "))

if option_1 == 0:
    # Define a regular lattice
    option_hex = int(input("Do you want a square (0) or hexagonal (1) lattice?: "))
    if option_hex == 0:
        Nvert = int(input("Number of vertices?: "))
        Nrows = int(input("Number of rows?: "))

        Nvert = 100
        Nrows = 10
        N = (Nvert//Nrows-1)*(Nrows-1)

        vorVertex = np.zeros((Nvert,2))
        coords = np.zeros((N,2))
        for i in range(Nvert):
            for j in range(Nrows):
                if i%Nrows == j:
                    vorVertex[i] = np.array([i//Nrows,j])
            # if i%3 == 1:
            #     V_list[i] = np.array([i//3,1])
            # if i%3 == 2:
            #     V_list[i] = np.array([i//3,2])
                
        for k in range(N):
            for l in range(Nrows-1):
                if k%(Nrows-1) == l:
                    coords[k] = np.array([0.5+l,k//(Nrows-1)+0.5])
                    
        vorVertex = vorVertex-(Nvert//Nrows)/2
        coords = coords -(Nvert//Nrows)/2

        dist_mat = np.zeros((Nvert,Nvert))

        for i in range(Nvert):
            for j in range(Nvert):
                dist_mat[i,j] = np.sqrt(np.sum((vorVertex[j]-vorVertex[i])**2))<=1
                
        AdjMat = dist_mat - np.eye(Nvert)


        distcv_mat = np.zeros((N,Nvert))

        for i in range(N):
            for j in range(Nvert):
                distcv_mat[i,j] = np.sqrt(np.sum((vorVertex[j]-coords[i])**2))<=1
                
        RegMat = distcv_mat

        vorRidges = [ ]

        row = np.where(AdjMat == 1)[0]
        col = np.where(AdjMat == 1)[1]

        for i in range(len(row)):
            edge1 = [row[i],col[i]]
            if sorted(edge1) not in vorRidges:
                vorRidges.append(edge1)
            
        vorRegions1 = []
        for i in range(N):
            vorRegions1.append(list(np.where(RegMat[i]==1)[0]))
            
        vorPointRegion1 = list(np.arange(N))

    if option_hex == 1:
        option_2 = int(input("Use which method to generated regular lattice: "))
        N_m = int(input("Square root of number of points you want to add onto the network"))
        
        if option_2 == 0:
        
            L_max = 5
            L_min = -5  
            dL = (L_max-L_min)/N_m
            A_0 = np.array([[1,0],[-1/2,np.sqrt(3)/2]])
            A = 1/(2*np.linalg.det(A_0))*(A_0.T+A_0)
            A_inv = np.linalg.inv(A)

            LxyM = A_inv.dot(np.array([L_max,L_max]))
            Lxym = A_inv.dot(np.array([L_min,L_min]))

            u = np.linspace(Lxym[0],LxyM[0],N_m)
            v = np.linspace(Lxym[1],LxyM[1],N_m)

            coord_list = []

            for j in range(N_m):
                for i in range(N_m):
                    coord_list.append(A.dot(np.array([u[j],v[i]])))
                
            coords = np.array(coord_list)
            N = N_m**2
        # if option_2 == 1:
        #     # Generate a regular lattice in a square domain
        #     L_max = 5
        #     L_min = -5  
        #     dL = (L_max-L_min)/N_m
        #     A_0 = np.array([[1,np.sqrt(1)/2],[0,1]])
        #     u = np.linspace(L_min,L_max,N_m)
        #     v = np.linspace(L_min,L_max,N_m)

        #     coord_list = []

        #     for j in range(N_m):
        #         for i in range(N_m):
        #             coord_list.append(A_0.dot(np.array([u[j],v[i]])))
                
        #     coords = np.array(coord_list)
        #     N = N_m**2
            
        #     for i in range(N):
        #         if coords[i,0]<-5:
        #             coords[i,0] = coords[i,0]+dL*N_m+dL+np.sqrt(1)/(2*N_m)
        #         if coords[i,0] > 5:
        #             coords[i,0] = coords[i,0]-dL*N_m-dL-np.sqrt(1)/(2*N_m)
        else:
            x1 = []
            for i in range(N_m):
                for j in range(N_m):
                    if i%2==0:
                        x1.append([i*np.sqrt(3)/2,j])
                    else:
                        x1.append([i*np.sqrt(3)/2,j+0.5])
                        
            coords = np.array(x1)-np.array([(N_m-1)*np.sqrt(3)/4,(N_m)/2])
            coords = coords/(np.max(coords[:,0])-np.min(coords[:,0]))*10
            N = N_m**2
                


if option_1 == 1:
    option_hex = 1
    # Define a random lattice

    N = int(input("Square root of the number of points you want to add onto the network "))**2
    coords = newAbs(rlt.newcoords(N),5)
    

  
# For wound modelling, find a cell region to classify as a wound
#An initial approach is to choose the region closest to the center

L1_array = np.zeros(N)
for i in range(N):
    L1_array[i] = np.linalg.norm(coords[i])
    
wound_loc = np.argmin(L1_array)
coords = coords-coords[wound_loc]  # Center the coordinates around the wound cell
L_max = 5
L_min = -5  
dL = (L_max-L_min)/N**0.5
size_of_wound = int(input("Number of cells to replace with wound: "))

if option_hex == 1:
    vor = Voronoi(coords)
    vorPointRegion1 = vor.point_region
    vorRegions1 = vor.regions
    vorVertex = np.array(fc.vertices_in_bound(list(vor.vertices),2*L_max))
    vorRidges = fc.remove_minus(vor.ridge_vertices)

for e in range(len(vorRidges)):
    vorRidges[e] = sorted(vorRidges[e])
    
vorRegions = []
vorPointRegion2 = []

for i in range(len(vorPointRegion1)):
    vorPointRegion2.append(i)
    vorRegions.append(vorRegions1[vorPointRegion1[i]])
    
if (size_of_wound == 1) or (size_of_wound == 0):
    print('Single cell wound')

else: 
    L_array = np.zeros(N)
    for i in range(N):
        L_array[i] = np.linalg.norm(coords[i]-coords[wound_loc])
    wound_cells_loc = np.argsort(L_array)[:size_of_wound]
    print('Multiple cell wounds at: ')

    #Removing the edge associated with the intersection between wound elements.
    
    UnionIntersect = []
    List_pair = itertools.permutations(wound_cells_loc,2)
    for listit in List_pair:
        Ri = vorRegions[listit[0]]           
        Rj = vorRegions[listit[1]]
        eij = sorted(list(set(Ri).intersection(Rj)))
        if (eij not in UnionIntersect) and (len(eij)>0):
            UnionIntersect.append(eij)
    print("Edge shared by the regions, to be removed: ")

    for edge in UnionIntersect:
        if len(edge)>1:
            vorRidges.remove(sorted(edge))

    #Removing the vertices associated with the intersection between 3 wound elements
    if size_of_wound >= 3:
        List_iter = itertools.permutations(wound_cells_loc, 3)
        TripleIntersect = []
        for listiter in List_iter:
            Ri = vorRegions[listiter[0]]          
            Rj = vorRegions[listiter[1]]
            Rk = vorRegions[listiter[2]]
            vijk = list(set(Ri).intersection(Rj).intersection(Rk))
            if (vijk not in TripleIntersect) and (len(vijk)>0):
                TripleIntersect.append(vijk)
        flat_list = []
        for row in TripleIntersect:
            flat_list += row
        TripleIntersect = flat_list
        print("Vertex shared by multiple regions, to be removed: ")
        
    #Merge the regions of neighbouring cells together and remove them from the list
    
    wound_cells_loc= np.delete(wound_cells_loc,0)
    print("Merging the following regions")


    for i in range(size_of_wound-1):
        for v in vorRegions[wound_cells_loc[i]]:
            if v not in vorRegions[wound_loc]:
                vorRegions[wound_loc].append(v)
    
                
    if size_of_wound >= 3:           
        for vr in TripleIntersect:
            for r in range(len(vorRegions)):
                if vr in vorRegions[r]:
                    vorRegions[r].remove(vr)
        # Removing isolated vertices in an edge
        
        
        # for v in vorRegions[wound_loc]:
        #     neighv = fc.find_vertex_neighbour_vertices(vorRidges,v)
        #     if len(neighv)<3:
        #         for vi in neighv:
                
                
        # The removal of vertices throws some labels out of whack. Maybe this will solve some of the issues 
    # TpInter = list(np.sort(TripleIntersect)[::-1])
    # print(TpInter)
    # nvorRegions = list(vorRegions)
    # nvorRidges = list(vorRidges)
    # for v in TpInter:
    #     print(v)
    #     for r in range(len(vorRegions)):
    #         for vr in range(len(vorRegions[r])):
    #             if (nvorRegions[r][vr] >= v) and (nvorRegions[r][vr] not in TpInter) :
    #                 vorRegions[r][vr] = vorRegions[r][vr] -1
    #     for e in range(len(vorRidges)):
    #         for ve in range(len(vorRidges[e])):
    #             if (nvorRidges[e][ve] >= v) and (nvorRidges[e][ve] not in TpInter):
    #                 vorRidges[e][ve] = vorRidges[e][ve] -1
                

    print("Removing intersect vertices")       

    nvorRegions = list(vorRegions)
    for i in range(size_of_wound-1):
        vorRegions.remove(nvorRegions[wound_cells_loc[i]]) #remove elements from location
    coords = np.delete(coords,np.array(wound_cells_loc),0)
    

    
# print(N)
N_new = len(coords)
L_array = np.zeros(N_new)
for i in range(N_new):
    L_array[i] = np.linalg.norm(coords[i])
    
wound_loc = np.argmin(L_array)
N = N_new

# Define arrays to insert numerical results

vorPointRegion = []

for i in range(N):
    vorPointRegion.append(i)
av = gmp.areas_vor(vorPointRegion,vorRegions,vorVertex,vorRidges,vorPointRegion)

#Maybe modify things a bit
L_max = 5
L_min = -5  
DeltaL = L_max - L_min

# Find 2-degree vertices
wound_vertices = vorRegions[wound_loc]
loosey = []
edges_loosey = []
for v in wound_vertices:
    if len(fc.find_vertex_neighbour_vertices(vorRidges,v))<3:
        loosey.append(v)
cell_loosey = []
for v in loosey:
    #print(fc.find_vertex_neighbour_vertices(vorRidges,v))
    edges_loosey.append(fc.find_vertex_neighbour_vertices(vorRidges,v))
    nv = fc.find_vertex_neighbour_centers(vorRegions,vorPointRegion,v)[0]
    for c in nv:
        if c != wound_loc:
            if c not in cell_loosey:
                cell_loosey.append(c)
                
loc_loosey_edges = []
loc_loosey_edges_ind = []
for ind_e in range(len(vorRidges)):
    edge = vorRidges[ind_e]
    timesx = 0
    for elem in edge:
        if elem in loosey:
            timesx+=1
    if timesx!=0:
        loc_loosey_edges.append(ind_e)
        loc_loosey_edges_ind.append(timesx)
        
sorted_edges = np.argsort(loc_loosey_edges)

print('Cells with 2-neighbour vertices')
print(np.sort(cell_loosey))

print('Vertices with 2-neighbour vertices')
print(np.sort(loosey))

print('Edges with 2-neighbour vertices')
print(np.array(loc_loosey_edges)[sorted_edges])

print('Number of 2-neighbour vertices in each edge')
print(np.array(loc_loosey_edges_ind)[sorted_edges])

#Remove array elements from list function

def removearrayfromlist(arr, thelist):
    return [x for x in thelist if not (x==arr).all()]

# To fix these 2-neighbour inner vertices, we first remove the edges in loc_loosey edges from the list vorRidges (can't leave blank edges)
edges_removed = []
for elem in range(len(loc_loosey_edges)):
    edges_removed.append(vorRidges[loc_loosey_edges[elem]])

print(len(edges_removed))
print(len(vorRidges))   
for edge in edges_removed:
    vorRidges = removearrayfromlist(np.array(edge), vorRidges)
print(len(vorRidges))
# Then we remove the vertices from the list of cell regions vorRegions (we know the location of them from cell_loosey) and from the wound boundary in boundaries[1]
for region in vorRegions:
    for elem in loosey:
        if elem in region:
            region.remove(elem)
# for elem in loosey:
#     vorRegions[wound_loc].remove(elem)
# Then we leave the vertices blank (can't delete them from the list) 
for elem in loosey:
    vorVertex[elem] = np.array([0,0])
# and we add new edges from the edges loosey list to the list of vorRidges
for edge in edges_loosey:
    vorRidges = np.vstack((vorRidges,np.array(edge)))
    
Ridges = []
for edge in vorRidges:
    Ridges.append(list(edge))


foldername=input('Folder name for tissue: ')
rTF.saveGeneratedNetwork(coords, size_of_wound,vorPointRegion,vorRegions,vorVertex,Ridges,wound_loc,foldername,option_hex)

tf = time.time() - ti
with open(foldername+'/log_tissue_generation'+str(size_of_wound)+'.txt','a') as log:
    
    log.write('Random Lattice? - ')
    log.write(' ')
    log.write(str(option_1)+'\n')
    
    if size_of_wound<= 1:
        log.write('Single cell wound at - ')
        log.write(' ')
        log.write(str(wound_loc)+'\n')
    else:    
        log.write('Multiple cells wound at - ')
        log.write(' ')
        log.write(str(wound_loc)+'\n')
        
        log.write('Edge shared by the regions, to be removed - ')
        log.write(' ')
        log.write(str(UnionIntersect)+'\n')
        
        log.write('Regions removed - ')
        log.write(' ')
        log.write(str(wound_cells_loc)+'\n')
    
    if size_of_wound>=3:
        log.write('Vertex shared by multiple regions, to be removed - ')
        log.write(' ')
        log.write(str(TripleIntersect)+'\n')
    
    
    
    log.write('Number of remaining regions - ')
    log.write(' ')
    log.write(str(N_new)+'\n')
    
    log.write('Simulation time (s) - ')
    log.write(' ')
    log.write(str(tf)+'\n')
