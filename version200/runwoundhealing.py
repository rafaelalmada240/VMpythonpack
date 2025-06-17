#import sys
#sys.path.insert(1, '/vertexmodelpack')
import numpy as np
from vertexmodelpack import sppvertex as sppv
from vertexmodelpack import randomlattice as rlt
from vertexmodelpack import topologicaltransitions as tpt
from vertexmodelpack import readTissueFiles as rTF
from vertexmodelpack import geomproperties as gmp
import time


''' 
Run simulations in this module for a prebuilt tissue network 
(that is we generate the tissue in a separate code, and we run the simulations in this code)
'''

########################################################################################################################
#User inputs


list_inputs = []
with open('/home/rafael/Documents/2nd Year/backup5git/inputrwh.txt','r') as text:
    for line in text:
        list_inputs.append(line.replace('\n','').split('-')[-1])

input_tissues = list_inputs[0]
tissues_list = input_tissues.split()
tissues = [int(num) for num in tissues_list]#[1,2,3,4,5,6,7,8,9,10]


epsilonx = float(list_inputs[1])
epsilont = float(list_inputs[2])

K_run = float(list_inputs[3])
G_run = float(list_inputs[4])

list_plw = bool(int(list_inputs[5]))
if list_plw:
    p0_min = float(list_inputs[13])
    p0_max = float(list_inputs[14])
    p0_Nbin = int(list_inputs[15])

    lw_min = float(list_inputs[16])
    lw_max = float(list_inputs[17])
    lw_Nbin = int(list_inputs[18])

    #To make a list of Ls and Lws (dimensionless)
    L_List =  list(np.linspace(p0_min,p0_max,p0_Nbin))
    print("bin resolution p0")
    print(float((p0_max-p0_min)/(p0_Nbin-1)))


    Lw_List = list(np.linspace(lw_min,lw_max,lw_Nbin))
    print("bin resolution lw")
    print(float((lw_max-lw_min)/(lw_Nbin-1)))
    
else:
    p0_min = float(list_inputs[6])
    p0_max = p0_min
    p0_Nbin = 2

    lw_min = float(list_inputs[7])
    lw_max = lw_min
    lw_Nbin = 2

    #To make a list of Ls and Lws (dimensionless)
    L_List =  [p0_min]
    print("bin resolution p0")
    print(float((p0_max-p0_min)/(p0_Nbin-1)))


    Lw_List = [lw_min]
    print("bin resolution lw")
    print(float((lw_max-lw_min)/(lw_Nbin-1)))
    

L_max = 5
L_min = -5  
DeltaL = L_max - L_min

T = float(list_inputs[8])

UseT1 = bool(int(list_inputs[9]))

simple_output = bool(int(list_inputs[10]))

woundsize = int(list_inputs[11]) 

# Opening files in respective folders - probably use a function that turns this into a dictionary:
##########################################################################################################
bigfoldername = list_inputs[12].strip()

for tissue_n in tissues:
    
    tme = time.time()
    
    print("tissue number: "+str(tissue_n))
    
    

    foldername = bigfoldername+'/tissue'+str(tissue_n)+'/size'+str(woundsize)
    
    dataset = rTF.open_tissuefile(foldername,0)
    
    coords = dataset['centers']
    vtxPRegions = dataset['point regions']
    vtxRegions = dataset['regions']
    vertices = dataset['vertices']
    vtxEdges = dataset['Edge connections']
    Boundaries = dataset['boundaries']
    wloc = dataset['WoundLoc']
    
    coords_array = np.zeros(())
    
    N = len(coords[:,0])
    print(N)

#################################################################################################################
    
    
    # Simulation parameters    
    av = []
    
    av = gmp.areas_vor(vtxPRegions,vtxRegions,vertices,vtxEdges,vtxPRegions)
    print("Median cell area")
    print(np.median(av))
    # print(np.std(av))
   
    r0 = np.sqrt(np.median(av)/np.pi)
    h = epsilonx*DeltaL/(2*np.sqrt(N)) #size step
    print("Simulation spatial resolution")
    print(h)
    

    A0_run = av# or np.median(av), maybe it depends on the model

    K_vec = K_run*np.ones(len(vtxPRegions))
    G_vec = G_run*np.median(av)*np.ones(len(vtxPRegions))

    mu = 1 #1/720 1/s
    dt = epsilont/((K_run*np.median(av))/mu) #time step normalized by mu/(K A0)
    print("Simulation temporal resolution")
    print(dt)

    #for PC
    
    #for cluster
    #M = 50000 
    

    M = int(T/epsilont)
    print("Simulation Max number of iterations")
    print(M)
    

    
    
    
    areaWound0 = gmp.area_vor(vtxPRegions,vtxRegions,vertices,vtxEdges,wloc)

#####################################################################################################################
    i_final = 0
    
    
    # Set up boundary for initial relaxation
    BoundariesPrior = []
    for v in Boundaries[0]:
        BoundariesPrior.append(v)
    for v in Boundaries[1]:
        BoundariesPrior.append(v)
    
    # Loop over P0
    for lr in L_List:
        
        # Define copy of tissue network for initial relaxation
        
        coords_array = np.zeros((coords.shape[0],coords.shape[1],M))
        pregions_array = np.zeros((len(vtxPRegions),M))
        regions_array = np.zeros((len(vtxRegions),M))
        vertices_array = np.zeros((vertices.shape[0],vertices.shape[1],M))
        edges_array = np.zeros((len(vtxEdges),M))
        #boundaries_array = np.zeros((len(Boundaries),M))
        #wloc_array = np.zeros((len(wloc),M))
    
        coords_array[:,:,0] = np.array(coords)
        pregions_array[:,0] = list(vtxPRegions)
        regions_array[:,0] = list(vtxRegions)
        vertices_array[:,:,0] = np.array(vertices)
        edges_array[:,0] = list(vtxEdges)
        #boundaries_array[:,0] = list(Boundaries)
        #wloc_array[:,0] = wloc
        # coords_1 = coords.copy()
        # coords_1_vertex = vertices.copy()
        Lr = (lr*G_run*2)*K_run*np.median(av)**(3/2)
        #Initial relaxation of the tissue
        for t in range(M//5):
            F_vertex = sppv.force_vtx_elastic_wound_parallel(regions_array[:,0], pregions_array[:,0], 
                                                             edges_array[:,0], K_vec,A0_run,G_vec,Lr,0,
                                                             vertices_array[:,:,t],coords_array[:,:,t],wloc,h,
                                                             BoundariesPrior)   
            vertices_array[:,:,t+1] = vertices_array[:,:,t] + mu*(F_vertex)*dt
            
            coords_array[:,:,t+1] = sppv.cells_avg_vtx(regions_array[:,0],vtxPRegions[:,0],
                                          np.array(coords_array[:,:,t]),np.array(vertices_array[:,:,t]))
                
        for lw in Lw_List:
            Lr = (lr*G_run*2)*K_run*np.median(av)**(3/2)
            Lw = (lw-G_run*lr)*K_run*np.median(av)**(3/2)
            
            i = 0
            transitionsList = []
            periWList = []
            areaWList= []
            total_transitions = 0
            
            areaWound0 = gmp.area_vor(pregions_array[:,0],regions_array[:,0],vertices_array[:,:,0],edges_array[:,0],wloc)
            areaWound = areaWound0*1
            
            coords_array[:,:,0] = coords_array[:,:,t]
            vertices_array[:,:,0] = vertices_array[:,:,t]
            #regions_evo = vtxRegions.copy()
            #edges_vertex_evo = vtxEdges.copy()
            
            list_coords = []
            list_vertex = []
            list_points = []
            list_regions = []
            list_edges = []
            list_boundaries = []
            list_wloc = []
            
            list_coords.append(coords_array[:,:,0])
            list_vertex.append(vertices_array[:,:,0])
            list_edges.append(edges_array[:,0])
            list_points.append(pregions_array[:,0])
            list_regions.append(regions_array[:,0])
            list_boundaries.append(Boundaries)
            list_wloc.append(wloc)

            while (i < M) and ((areaWound>= areaWound0/16) and (areaWound<= 16*areaWound0)):
                
                #Compute the area and perimeter of the wound
                perimeterWound = gmp.perimeter_vor(vtxPRegions,regions_evo,coords_evo_vertex,
                                                   edges_vertex_evo,wloc)
                areaWound = gmp.area_vor(vtxPRegions,regions_evo,coords_evo_vertex,
                                         edges_vertex_evo,wloc)
                
                #Compute forces
                J = 0.0
                Rand_vertex = J*np.random.rand(vertices_array[:,:,0].shape[0],2)
                Rand_vertex[Boundaries[0]] = 0
                F_vertex = sppv.force_vtx_elastic_wound_parallel(regions_array[:,i], pregions_array[:,i], edges_array[:,i], 
                                                                 K_vec,A0_run,G_vec,Lr,Lw,vertices__array[:,:,i],
                                                                 coords_array,wloc,h,Boundaries[0])
                # print(np.mean(F_vertex,0))    
                #Reflexive boundary conditions
                A_vertex = rlt.newWhere(coords_evo_vertex + mu*F_vertex*dt,16)
                coords_evo_vertex = coords_evo_vertex + mu*A_vertex*(F_vertex+Rand_vertex)*dt
                
                #Cell center positions are the average of the cell vertex positions
                coords_evo = sppv.cells_avg_vtx(regions_evo,vtxPRegions,np.array(coords_evo),np.array(coords_evo_vertex))
                
                #Do topological rearrangements
                transition_counter = 0
                if UseT1:
                    edges_vertex_evo, coords_evo_vertex, regions_evo, transition_counter =  tpt.T1transition2(np.array(coords_evo_vertex),np.array(edges_vertex_evo),regions_evo, vtxPRegions,0.01*r0)
                    
                i = i + 1
                #print(areaWound/areaWound0)
                total_transitions += transition_counter 
                
                #Store values in list to be saved later
                periWList.append(perimeterWound)
                areaWList.append(areaWound)
                transitionsList.append(transition_counter)
                list_coords.append(coords_evo)
                list_vertex.append(coords_evo_vertex)
                list_edges.append(edges_vertex_evo)
                list_points.append(vtxPRegions)
                list_regions.append(regions_evo)
                list_boundaries.append(Boundaries)
                list_wloc.append(wloc)
            
            
            #Output of simulations for analysis
            
            if simple_output: 
                rTF.simpleOutputTissues(foldername,[areaWound0,G_run,lr,lw,N],[periWList,areaWList,transitionsList])
            else:
                rTF.movieOutputTissues(foldername,[len(list_wloc),lr,lw],[list_coords,list_points,list_regions,list_vertex,list_edges,list_boundaries,list_wloc])
                
            i_final += i
    #################################################################################################################################################################   
            
    #Log of simulations

    tf = time.time()-tme

    with open(bigfoldername+'/log'+str(tissue_n)+'.txt','a') as log:
        log.write('Median cell area - ')
        log.write(' ')
        log.write(str(np.median(av).round(6))+'\n')
        log.write('Initial Wound Area - ')
        log.write(' ')
        log.write(str(areaWound0.round(6))+'\n')
        log.write('Spatial resolution of the tissue - ')
        log.write(' ')
        log.write(str(h.round(6))+'\n')
        log.write('Temporal resolution of the tissue - ')
        log.write(' ')
        log.write(str(dt.round(6))+'\n')
        log.write('Simulation time (s) - ')
        log.write(' ')
        log.write(str(tf)+'\n')
        log.write('Total number of iterations - ')
        log.write(' ')
        log.write(str(i_final)+'\n')
        log.write('Bin resolution p0 - ')
        log.write(' ')
        log.write(str(float((p0_max-p0_min)/(p0_Nbin-1)))+'\n')
        # log.write('Bin resolution lw - ')
        # log.write(' ')
        # log.write(str(float((lw_max-lw_min)/(lw_Nbin-1)))+'\n')
        log.write('Used T1? - ')
        log.write(' ')
        log.write(str(UseT1)+'\n')
