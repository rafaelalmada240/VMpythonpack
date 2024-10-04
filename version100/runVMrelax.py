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

input_tissues = input('Enter list of tissues as individual elements separated by space: ')
tissues_list = input_tissues.split()
tissues = [int(num) for num in tissues_list]

epsilonx = float(input("Enter the spatial resolution you want (float): "))
epsilont = float(input("Enter the temporal resolution you want (float): "))

K_run = float(input("Enter K: "))
G_run = float(input("Enter G: "))

p0_min = float(input("Lower bound p0: "))
p0_max = float(input("Upper bound p0: "))
p0_Nbin = int(input("p0 Number of bins: "))

#To make a list of Ls and Lws (dimensionless)
L_List =  list(np.linspace(p0_min,p0_max,p0_Nbin))
print("bin resolution p0")
print(float((p0_max-p0_min)/(p0_Nbin-1)))

L_max = 5
L_min = -5  
DeltaL = L_max - L_min

T = float(input("How many time units do we want the simulation to go to in Tmax?: "))
UseT1 = bool(int(input("Use topological rearrangements? (y - 1, n - 0): ")))
simple_output = bool(int(input('Do you want a simple output? (y-1/n-0): ')))
woundsize = int(input('Enter wound size: ')) 

# Opening files in respective folders - probably use a function that turns this into a dictionary:
##########################################################################################################
bigfoldername = input('Enter name of big folder where the tissue folders are: ')
for tissue_n in tissues:
    
    tme = time.time()
    
    print("tissue number: "+str(tissue_n))

    foldername = bigfoldername+'/tissue'+str(tissue_n)+'/size'+str(woundsize)
    dataset = rTF.open_tissuefile(foldername,0)
    
    coords = dataset['centers']
    vorPointRegion = dataset['point regions']
    vorRegions = dataset['regions']
    vertices = dataset['vertices']
    vorRidges = dataset['Edge connections']
    Boundaries = dataset['boundaries']
    wloc = dataset['WoundLoc']
    
    N = len(coords[:,0])
    print(N)

################################################################################################################# 
    # Simulation parameters    
    av = []
    
    av = gmp.areas_vor(vorPointRegion,vorRegions,vertices,vorRidges,vorPointRegion)
    print("Median cell area")
    print(np.median(av))
    # print(np.std(av))
   
    r0 = np.sqrt(np.median(av)/np.pi)
    h = epsilonx*DeltaL/(2*np.sqrt(N)) #size step
    print("Simulation spatial resolution")
    print(h)
    

    A0_run = av# or np.median(av), maybe it depends on the model
    mu = 1
    dt = (K_run*np.median(av))/mu*epsilont #time step normalized by K_run A0_run/mu
    print("Simulation temporal resolution")
    print(dt)

    #for PC
    
    #for cluster
    #M = 50000 
    

    M = int(T/dt)
    print("Simulation Max number of iterations")
    print(M)
    
    areaWound0 = gmp.area_vor(vorPointRegion,vorRegions,vertices,vorRidges,wloc)

#####################################################################################################################
    i_final = 0
    for lr in L_List:
        # Run simulations
        
        Lr = lr*G_run*2*K_run*np.mean(av)**(1/2)

        
        i = 0
        
        K_vec = K_run*np.ones(N)
        G_vec = G_run*np.ones(N)
        
        transitionsList = []
        periWList = []
        areaWList= []
        EVMList= []
        
        total_transitions = 0
        areaWound = areaWound0
        coords_evo = np.array(coords)
        coords_evo_vertex = np.array(vertices)
        
        list_coords = []
        list_vertex = []
        list_points = []
        list_regions = []
        list_edges = []
        list_boundaries = []
        list_wloc = []

        while (i < M):
            
            #Compute the area and perimeter of the wound
            perimeterAvg = np.mean(gmp.perimeters_vor(vorPointRegion,vorRegions,coords_evo_vertex,vorRidges,vorPointRegion))
            areaAvg = np.mean(gmp.areas_vor(vorPointRegion,vorRegions,coords_evo_vertex,vorRidges,vorPointRegion))
            E_vm = sppv.energy_vor_total(vorPointRegion,vorRegions,  vorRidges,coords_evo_vertex, K_vec,A0_run,G_vec,Lr)
            #Compute forces
            
            J = 0.0
            Rand_vertex = J*(np.random.rand(coords_evo_vertex.shape[0],2)-0.5)
            Rand_vertex[Boundaries[0]] = 0
            
            F_vertex = sppv.force_vtx_elastic(vorRegions, vorPointRegion, vorRidges, K_vec,A0_run,G_vec,Lr,coords_evo_vertex,coords_evo,h,Boundaries[0])
   
            #Reflexive boundary conditions
            A_vertex = rlt.newWhere(coords_evo_vertex + mu*F_vertex*dt,15)
            coords_evo_vertex = coords_evo_vertex + mu*A_vertex*(F_vertex+Rand_vertex)*dt
            #Cell center positions are the average of the cell vertex positions
            coords_evo = sppv.cells_avg_vtx(vorRegions,vorPointRegion,np.array(coords_evo),np.array(coords_evo_vertex))
            
            #Do topological rearrangements
            transition_counter = 0
            if UseT1:
                vorRidges, coords_evo_vertex, vorRegions, transition_counter =  tpt.T1transition2(np.array(coords_evo_vertex),np.array(vorRidges),vorRegions, vorPointRegion,0.01*r0)
                
            i = i + 1
            total_transitions += transition_counter 
            
            #Store values in list to be saved later
            periWList.append(perimeterAvg)
            areaWList.append(areaAvg)
            transitionsList.append(transition_counter)
            EVMList.append(E_vm)
            list_coords.append(coords_evo)
            list_vertex.append(coords_evo_vertex)
            list_edges.append(vorRidges)
            list_points.append(vorPointRegion)
            list_regions.append(vorRegions)
            list_boundaries.append(Boundaries)
            list_wloc.append(wloc)
        
            
            #Output of simulations for analysis
            
        if simple_output: 
            rTF.simpleOutputTissues(foldername,[areaWound0,G_run,lr,0,N],[periWList,areaWList,transitionsList])
        else:
            rTF.simpleOutputTissues(foldername,[areaWound0,G_run,lr,0,N],[periWList,areaWList,EVMList])
            rTF.movieOutputTissues(foldername,[len(list_wloc),lr,0],[list_coords,list_points,list_regions,list_vertex,list_edges,list_boundaries,list_wloc])
#################################################################################################################################################################   
        i_final += i
    #Log of simulations

    tf = time.time()-tme

    with open(bigfoldername+'/log'+str(tissue_n)+'.txt','a') as log:
        log.write('Median cell area - ')
        log.write(' ')
        log.write(str(np.median(av).round(6))+'\n')
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
        log.write('Used T1? - ')
        log.write(' ')
        log.write(str(UseT1)+'\n')
