#import sys
#sys.path.insert(1, '/vmpack')
import numpy as np
from vmpack import sppvertex as sppv
from vmpack import randomlattice as rlt
from vmpack import topologicaltransitions as tpt
from vmpack import readTissueFiles as rTF
from vmpack import geomproperties as gmp

from vmpack import cytoF as cyF

import os
import time
import copy

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from functools import partial


''' 
Run simulations in this module for a prebuilt tissue network 
(that is, we generate the tissue in a separate code, and we run the simulations in this code)
'''

########################################################################################################################
#User inputs


list_inputs = []
with open('/home/rafael/Documents/3rdYear/codesInternship/inputrwh.txt','r') as text:
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
    #M <= 1000
    
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
    current_directory = os.getcwd()
    with Pool(16) as pool:
        for lr in L_List:
            
            # Define copy of tissue network for initial relaxation
            
            coords_array = np.zeros((coords.shape[0],coords.shape[1],M+1))
            
            vertices_array = np.zeros((vertices.shape[0],vertices.shape[1],M+1))
  
            #Makes deep copies to avoid changes in the original arrays and lists
        
            coords_array[:,:,0] = copy.deepcopy(coords)
            pregions1 = copy.deepcopy(vtxPRegions)
            regions1 = copy.deepcopy(vtxRegions)
            vertices_array[:,:,0] = copy.deepcopy(vertices)
            edges1 = copy.deepcopy(vtxEdges)
            
            Lr = (lr*G_run*2)*K_run*np.median(av)**(3/2)
            #Initial relaxation of the tissue
            print("Relaxation of the tissue network before the wound healing simulation starts.")
            trelax = time.time()
            par_vec = (K_vec, A0_run, G_vec, Lr)
            Mrel = 50
            for t in range(Mrel):
                t_iter = time.time()
                tissue_vec = (pregions1,regions1,edges1, vertices_array[:,:,t], coords_array[:,:,t],BoundariesPrior)
                F_vertex = sppv.force_vtx_parallel(tissue_vec, par_vec, h, pool)  
                vertices_array[:,:,t+1] = vertices_array[:,:,t] + mu*(F_vertex)*dt
                coords_array[:,:,t+1] = sppv.cells_avg_vtx(regions1,pregions1,
                                            np.array(coords_array[:,:,t]),np.array(vertices_array[:,:,t]), pool)
                en_tissue_vec = (pregions1, regions1, edges1, vertices_array[:,:,t+1], BoundariesPrior)
                
                
                if t%(Mrel//10) ==0:
                    Et, E_ar, E_per, E_lin = sppv.energy_vtx_total(en_tissue_vec, par_vec)
                    print("Total energy - "+str(Et.round(3))+", area energy - "+str(E_ar.round(3))+", perimeter energy - "+str(E_per.round(3))+", line energy- "+str(E_lin.round(3)))
                #print("Relaxation iteration "+str(t)+" of "+str(M//5)+" - time taken: "+str(time.time()-t_iter))
            print("Relaxation time (s)")
            print(time.time()-trelax)
            print("Relaxation iteration rate")
            print((time.time()-trelax)/(Mrel))
            print("Relaxation finished.")   
                
            coords_array[:,:,0] = coords_array[:,:,t]
            vertices_array[:,:,0] = vertices_array[:,:,t]
                
                
            for lw in Lw_List:
                pregions2 = copy.deepcopy(pregions1)
                regions2 = copy.deepcopy(regions1)
                edges2 = copy.deepcopy(edges1) 
                
                Lr = (lr*G_run*2)*K_run*np.median(av)**(3/2)
                Lw = (np.sqrt(woundsize)*lw-G_run*lr)*K_run*np.median(av)**(3/2)
                
                i = 0
                transitionsList = []
                periWList = []
                areaWList= []
                total_transitions = 0
                
                areaWound0 = gmp.area_vor(pregions1,regions1,vertices_array[:,:,0],edges1,wloc)
                areaWound = areaWound0*1
                
                list_coords = []
                list_vertex = []
                list_points = []
                list_regions = []
                list_edges = []
                list_boundaries = []
                list_wloc = []
                
                list_coords.append(coords_array[:,:,0])
                list_vertex.append(vertices_array[:,:,0])
                list_edges.append(edges2)
                list_points.append(pregions2)
                list_regions.append(regions2)
                list_boundaries.append(Boundaries)
                list_wloc.append(wloc)
                
                list_stress = []

                while (i < M) and ((areaWound>= areaWound0/64) and (areaWound<= 2*areaWound0)):
                    
                    #Compute the area and perimeter of the wound
                    perimeterWound = gmp.perimeter_vor(pregions2,regions2,vertices_array[:,:,i],
                                                    edges2,wloc)
                    areaWound = gmp.area_vor(pregions2,regions2,vertices_array[:,:,i],
                                            edges2,wloc)
                    
                    #Compute forces
                    tissue_vec2 = (pregions2,regions2,edges2, vertices_array[:,:,i], coords_array[:,:,i],Boundaries[0])
                    par_wound = (Lw, wloc, Boundaries[1])
                    F_vertex = sppv.force_vtx_parallel(tissue_vec2,par_vec, h, pool, True, par_wound)
                    
                    #Reflexive boundary conditions
                    A_vertex = rlt.newWhere(vertices_array[:,:,i]+ mu*F_vertex*dt,1.5*L_max)
                    vertices_array[:,:,i+1] = vertices_array[:,:,i] + mu*A_vertex*(F_vertex)*dt
                   
                    #Cell center positions are the average of the cell vertex positions
                    coords_array[:,:,i+1] = sppv.cells_avg_vtx(regions2,pregions2,np.array(coords_array[:,:,i]),np.array(vertices_array[:,:,i+1]), pool)
                    
                    #Save stress tensor components
                    S = sppv.stress_cell(regions2, pregions2, vertices_array[:,:,i+1], coords_array[:,:,i+1], F_vertex)
                    final_directory = os.path.join(current_directory,foldername+'/simple_output/l'+str(int(lr*100))+'lw'+str(int(lw*100)))
                    if not os.path.exists(final_directory):
                        os.makedirs(final_directory)
                    with open(final_directory+'/stress'+str(i)+'.txt','a') as filest:
                        for j in range(len(S)):
                            filest.write(str(j)+' '+str(S[j][0][0])+' '+str(S[j][0][1])
                                         +' '+str(S[j][1][0])+' '+str(S[j][1][1])+' '+'\n')
                    #Do topological rearrangements
                    transition_counter = 0
                    if UseT1:
                        edges2, vertices_array[:,:,i+1], regions2, transition_counter =  tpt.T1transition2(np.array(vertices_array[:,:,i+1]),np.array(edges2),
                                                                                                        regions2, pregions2,0.05*r0)
                        
            
                    #print(areaWound/areaWound0)
                    total_transitions += transition_counter 
                    
                    #Store values in list to be saved later
                    periWList.append(perimeterWound)
                    areaWList.append(areaWound)
                    transitionsList.append(transition_counter)
                    list_coords.append(coords_array[:,:,i+1])
                    list_vertex.append(vertices_array[:,:,i+1])
                    list_edges.append(copy.deepcopy(edges2))
                    list_points.append(copy.deepcopy(pregions2))
                    list_regions.append(copy.deepcopy(regions2))
                    list_boundaries.append(copy.deepcopy(Boundaries))
                    list_wloc.append(wloc)
                    
                    if i%20 == 0:
                        print('step - '+str(i) + ' steps left - '+str(M-i))
                        print('Area wound - '+str(areaWound/areaWound0))
                    i = i + 1
                    
                
                
                #Output of simulations for analysis
                
                if simple_output: 
                    rTF.simpleOutputTissues(foldername,[areaWound0,G_run,lr,lw,N],[periWList,areaWList,transitionsList])
                else:
                    rTF.movieOutputTissues(foldername,[len(list_wloc),lr,lw],[list_coords,list_points,list_regions,list_vertex,list_edges,list_boundaries,list_wloc])
                    rTF.simpleOutputTissues(foldername,[areaWound0,G_run,lr,lw,N],[periWList,areaWList,transitionsList])
                    
                    
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
        log.write('p0 - ')
        log.write(' ')
        log.write(str(float(lr))+'\n')
        log.write('lw - ')
        log.write(' ')
        log.write(str(float(lw))+'\n')
        log.write('kc - ')
        log.write(' ')
        log.write(str(float(0))+'\n')
        log.write('K - ')
        log.write(' ')
        log.write(str(float(K_run))+'\n')
        log.write('Used T1? - ')
        log.write(' ')
        log.write(str(UseT1)+'\n')
# Opening files in respective folders - probably use a function that turns this into a dictionary:
