#import sys
#sys.path.insert(1, '/vertexmodelpack')
import numpy as np
from vertexmodelpack import sppvertex as sppv
from vertexmodelpack import randomlattice as rlt
from vertexmodelpack import topologicaltransitions as tpt
from vertexmodelpack import readTissueFiles as rTF
from vertexmodelpack import geomproperties as gmp
import calciumdynamics2d as cd2
from vertexmodelpack import weightedgraphanalysis as wga
import os
import time
import copy


''' 
Run simulations in this module for a prebuilt tissue network 
(that is we generate the tissue in a separate code, and we run the simulations in this code)
'''

########################################################################################################################
#User inputs


list_inputs = []
with open('/home/rafael/Documents/2nd Year/backup5git/inputrwhg.txt','r') as text:
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

    gw_min = float(list_inputs[16])
    gw_max = float(list_inputs[17])
    gw_Nbin = int(list_inputs[18])

    #To make a list of Ls and Lws (dimensionless)
    L_List =  list(np.linspace(p0_min,p0_max,p0_Nbin))
    print("bin resolution p0")
    print(float((p0_max-p0_min)/(p0_Nbin-1)))


    Lw_List = list(np.linspace(gw_min,gw_max,gw_Nbin))
    print("bin resolution gw")
    print(float((gw_max-gw_min)/(gw_Nbin-1)))
    
else:
    p0_min = float(list_inputs[6])
    p0_max = p0_min
    p0_Nbin = 2

    gw_min = float(list_inputs[7])
    gw_max = gw_min
    gw_Nbin = 2

    #To make a list of Ls and Lws (dimensionless)
    L_List =  [p0_min]
    print("bin resolution p0")
    print(float((p0_max-p0_min)/(p0_Nbin-1)))


    Lw_List = [gw_min]
    print("bin resolution lw")
    print(float((gw_max-gw_min)/(gw_Nbin-1)))
    

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
    pointregion = dataset['point regions']
    regions = dataset['regions']
    vertices = dataset['vertices']
    edges = dataset['Edge connections']
    Boundaries = dataset['boundaries']
    wloc = dataset['WoundLoc']
    
    N = len(coords[:,0])
    print(N)

#################################################################################################################
    
    
    # Simulation parameters    
    av = []
    
    av = gmp.areas_vor(pointregion,regions,vertices,edges,pointregion)
    print("Median cell area")
    print(np.median(av))
    # print(np.std(av))
   
    r0 = np.sqrt(np.median(av)/np.pi)
    h = epsilonx*DeltaL/(2*np.sqrt(N)) #size step
    print("Simulation spatial resolution")
    print(h)
    

    A0_run = copy.deepcopy(av)# or np.median(av), maybe it depends on the model
    mu = 1
    dt = (K_run*np.median(av))/mu*epsilont #time step normalized by K_run A0_run/mu
    K_vec = np.ones(coords.shape[0])*K_run
    print("Simulation temporal resolution")
    print(dt)

    #for PC
    
    #for cluster
    #M = 50000 
    

    M = int(T/dt)
    print("Simulation Max number of iterations")
    print(M)
    
    #Define calcium concentration
    calcium_C = np.zeros(coords.shape[0])
    areaWound0 = gmp.area_vor(pointregion,regions,vertices,edges,wloc)
    

#####################################################################################################################
    i_final = 0
    current_directory = os.getcwd()
    
    for lr in L_List:
        # Run simulations
        for gw in Lw_List:
            par = [0.3,0.01,1, gw, 100]
            Lr = lr*G_run*2*K_run*np.median(av)**(3/2)
            Lw = (np.sqrt(woundsize)*0-G_run*lr)*K_run*np.median(av)**(3/2)
            
            i = 0
            transitionsList = []
            periWList = []
            areaWList= []
            total_transitions = 0
            areaWound = areaWound0
            coords_evo = coords
            vertex_evo = vertices
            calcium_new = np.zeros(calcium_C.shape)
            gamma_vec = np.zeros((coords.shape[0],3))
            
            
            
            list_coords = []
            list_vertex = []
            list_points = []
            list_regions = []
            list_edges = []
            list_boundaries = []
            list_wloc = []

            while (i < M) and ((areaWound>= areaWound0/8) and (areaWound<= 8*areaWound0)):
                
                G_vec = (G_run*(np.ones(coords.shape[0])) + (gamma_vec[:,0])*(np.linalg.norm(coords_evo,axis=1)<9))*np.median(av)
                
            
                
                #Compute the area and perimeter of the wound
                perimeterWound = gmp.perimeter_vor(pointregion,regions,vertex_evo,edges,wloc)
                areaWound = gmp.area_vor(pointregion,regions,vertex_evo,edges,wloc)
                
                #Compute forces
                
                #Noise
                J = 0.0
                Rand_vertex = J*np.random.rand(vertex_evo.shape[0],2)
                Rand_vertex[Boundaries[0]] = 0
                
                F_vertex = sppv.force_vtx_elastic_wound_parallel(regions, pointregion, edges, K_vec,A0_run,
                                                                 G_vec,Lr,Lw,vertex_evo,coords_evo,wloc,h,
                                                                 Boundaries[0])
                
                #Reflexive boundary conditions
                A_vertex = rlt.newWhere(vertex_evo + mu*F_vertex*dt,15)
                vertex_evo = vertex_evo + mu*A_vertex*(F_vertex+Rand_vertex)*dt
                
                #Cell center positions are the average of the cell vertex positions (for hyperuniformity perhaps)
                coords_evo = sppv.cells_avg_vtx(regions,pointregion,np.array(coords_evo),np.array(vertex_evo))
                
                #Do topological rearrangements
                transition_counter = 0
                if UseT1:
                    edges, vertex_evo, regions, transition_counter =  tpt.T1transition2(np.array(vertex_evo),np.array(edges),regions, pointregion,0.01*r0)
                
                #Change calcium concentration and Gamma values
                #shape_anisotropy = gmp.shape_anisotropy(regions,pointregion,vertex_evo,coords_evo)
                
                av1 = gmp.areas_vor(pointregion,regions,vertex_evo,edges,pointregion)
                
                strain = (np.array(A0_run) - np.array(av1))/np.array(A0_run)
                
                
                #difA1 = wga.generalized_net_lap_center(pointregion,regions,np.ones(len(shape_anisotropy))*(np.array(shape_anisotropy)<0.95),wloc)
                #dcdt = cd2.c_up_tissue(calcium_new,shape_anisotropy,par,wloc)
                #gamma_vec = cd2.g_up_tissue(difA1,pointregion,gamma_vec,calcium_new,par,wloc)
                
                #difA1 = wga.generalized_net_lap_center(pointregion,regions,np.ones(len(shape_anisotropy))*(np.array(shape_anisotropy)<0.95),wloc)
                dcdt = cd2.c_up_tissue(calcium_new,strain,par,wloc)
                dgdt = cd2.g_up_tissue(strain,pointregion,gamma_vec,calcium_new,par,wloc)
                
                gamma_vec = gamma_vec + dgdt*dt/par[4]
                calcium_new = calcium_new + dcdt*dt*(np.linalg.norm(coords_evo,axis=1)<9)*10**(-3)   
                if (i%(M//20) == 0) and list_plw != 1:
                    print('step - '+str(i) + ' steps left - '+str(M-i))
                    print('Area wound - '+str(areaWound/areaWound0))
                i = i + 1
                total_transitions += transition_counter 
                
                
                #Store values in list to be saved later
                periWList.append(perimeterWound)
                areaWList.append(areaWound)
                transitionsList.append(transition_counter)
                list_coords.append(coords_evo)
                list_vertex.append(vertex_evo)
                list_edges.append(copy.deepcopy(edges))
                list_points.append(pointregion)
                list_regions.append(copy.deepcopy(regions))
                list_boundaries.append(copy.deepcopy(Boundaries))
                list_wloc.append(wloc)
                final_directory = os.path.join(current_directory,foldername+'/simple_output/l'+str(int(lr*100))+'lw'+str(int(gw*100)))
                if not os.path.exists(final_directory):
                    os.makedirs(final_directory)
                with open(final_directory+'/calciumconc'+str(i)+'.txt','a') as ctxt:
                    for alpha in pointregion:
                        ctxt.write(str(alpha)+';'+str(calcium_new[alpha])+'\n')
                with open(final_directory+'/gammaarray'+str(i)+'.txt','a') as gtxt:
                    for alpha in pointregion:
                        gtxt.write(str(alpha)+';'+str(gamma_vec[alpha,0])+';'+str(gamma_vec[alpha,1])+';'+str(gamma_vec[alpha,2])+'\n')
            
            
            #Output of simulations for analysis
        
            if simple_output: 
                rTF.simpleOutputTissues(foldername,[areaWound0,G_run,lr,gw,N],[periWList,areaWList,transitionsList])
            else:
                rTF.movieOutputTissues(foldername,[len(list_wloc),lr,gw],[list_coords,list_points,list_regions,list_vertex,list_edges,list_boundaries,list_wloc])
                rTF.simpleOutputTissues(foldername,[areaWound0,G_run,lr,gw,N],[periWList,areaWList,transitionsList])
                
            i_final += i
    #################################################################################################################################################################   
            
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
        log.write('Tau - ')
        log.write(' ')
        log.write(str(par[4]))
        log.write('; Calcium parameters - ')
        log.write(' ')
        log.write(str(par[:3])+'\n')
        log.write('Bin resolution p0 - ')
        log.write(' ')
        log.write(str(float((p0_max-p0_min)/(p0_Nbin-1)))+'\n')
        log.write('Max lr - ')
        log.write(' ')
        log.write(str(float(lr))+'\n')
        log.write('Bin resolution gw - ')
        log.write(' ')
        log.write(str(float((gw_max-gw_min)/(gw_Nbin-1)))+'\n')
        log.write('Max gw - ')
        log.write(' ')
        log.write(str(float(gw))+'\n')
        log.write('Used T1? - ')
        log.write(' ')
        log.write(str(UseT1)+'\n')