import numpy as np
import os 
from vertexmodelpack import connections as fc



def open_tissuefile(filename,N=0):
    
    coords_list = []
    point_regions = []
    regions_list = []
    vertex_list = []
    boundaries_list = []
    edges_list = []
    
    if N != 0:
        for i in range(N):
            #print(i)
            coords = []
            vorPointRegion1 = []
            vorRegions = []

            with open(filename+'/centers'+str(i)+'.txt','r') as text:
                        for line in text:
                            last_line = (line.replace("\n","")).split(';')
                            coords.append([float(last_line[1]),float(last_line[2])])
                            vorPointRegion1.append(int(last_line[3]))
                            l4 = last_line[4].replace("[","").replace("]","").split(',')
                            lint4 = []
                            if all('' == s or s.isspace() for s in l4):
                                vorRegions.append(lint4)
                                continue
                            else:
                                for r in l4:
                                    lint4.append(int(r))
                            vorRegions.append(lint4)
                            
            coords = np.array(coords)

            vertices = []
            with open(filename+'/vertices'+str(i)+'.txt','r') as text:
                        for line in text:
                            last_line = (line.replace("\n","")).split(';')
                            vertices.append([float(last_line[1]),float(last_line[2])])
            vertices = np.array(vertices)

            #print(vorPointRegion)

            vorRidges = []
            with open(filename+'/edges'+str(i)+'.txt','r') as text:
                        for line in text:
                            last_line = (line.replace("\n","")).split(';')
                            l4 = last_line[1].replace("[","").replace("]","").replace(","," ").split(' ')
                            # print(l4)
                            lint4 = []
                            if all('' == s or s.isspace() for s in l4):
                                vorRidges.append(lint4)
                                continue
                            else:
                                for r in l4:
                                    if r != '':
                                        lint4.append(int(r))
                            vorRidges.append(np.array(lint4))
                            

            wloc = 0

            with open(filename+'/woundloc'+str(i)+'.txt','r') as text:
                for line in text:
                    wloc = int(line.replace("\n",""))

            vorPointRegion= []
            for k in range(len(coords)):
                vorPointRegion.append(k)

            Boundaries = []                
            with open(filename+'/boundaries'+str(i)+'.txt','r') as text:
                for line in text:
                    last_line = line.replace("\n","")
                    l4 = line.replace("[","").replace("]","").split(',')
                    lint4 = []
                    if all('' == s or s.isspace() for s in l4):
                        Boundaries.append(lint4)
                        continue
                    else:
                        for r in l4:
                            lint4.append(int(r))
                    Boundaries.append(lint4)
                    
            coords_list.append(coords)
            point_regions.append(vorPointRegion)
            regions_list.append(vorRegions)
            vertex_list.append(vertices)
            boundaries_list.append(Boundaries)
            edges_list.append(vorRidges)
        dataset = {"centers": coords_list, "vertices": vertex_list, "Edge connections":edges_list
                , "WoundLoc":wloc,"boundaries":boundaries_list,"regions":regions_list,"point regions": point_regions}
    else:
            #print(i)
            coords = []
            vorPointRegion1 = []
            vorRegions = []

            with open(filename+'/centers.txt','r') as text:
                        for line in text:
                            last_line = (line.replace("\n","")).split(';')
                            coords.append([float(last_line[1]),float(last_line[2])])
                            vorPointRegion1.append(int(last_line[3]))
                            l4 = last_line[4].replace("[","").replace("]","").split(',')
                            lint4 = []
                            if all('' == s or s.isspace() for s in l4):
                                vorRegions.append(lint4)
                                continue
                            else:
                                for r in l4:
                                    lint4.append(int(r))
                            vorRegions.append(lint4)
                            
            coords = np.array(coords)

            vertices = []
            with open(filename+'/vertices.txt','r') as text:
                        for line in text:
                            last_line = (line.replace("\n","")).split(';')
                            vertices.append([float(last_line[1]),float(last_line[2])])
            vertices = np.array(vertices)

            #print(vorPointRegion)

            vorRidges = []
            with open(filename+'/edges.txt','r') as text:
                        for line in text:
                            last_line = (line.replace("\n","")).split(';')
                            l4 = last_line[1].replace("[","").replace("]","").split(',')
                            lint4 = []
                            # print(l4)
                            if all('' == s or s.isspace() for s in l4):
                                vorRidges.append(lint4)
                                continue
                            else:
                                for r in l4:
                                    lint4.append(int(r))
                            vorRidges.append(np.array(lint4))
                            

            wloc = 0

            with open(filename+'/woundloc.txt','r') as text:
                for line in text:
                    wloc = int(line.replace("\n",""))

            vorPointRegion= []
            for k in range(len(coords)):
                vorPointRegion.append(k)

            Boundaries = []                
            with open(filename+'/boundaries.txt','r') as text:
                for line in text:
                    last_line = line.replace("\n","")
                    l4 = line.replace("[","").replace("]","").split(',')
                    lint4 = []
                    if all('' == s or s.isspace() for s in l4):
                        Boundaries.append(lint4)
                        continue
                    else:
                        for r in l4:
                            lint4.append(int(r))
                    Boundaries.append(lint4)
                    
            coords_list.append(coords)
            point_regions.append(vorPointRegion)
            regions_list.append(vorRegions)
            vertex_list.append(vertices)
            boundaries_list.append(Boundaries)
            edges_list.append(vorRidges)
            dataset = {"centers": coords, "vertices": vertices, "Edge connections":vorRidges
                , "WoundLoc":wloc,"boundaries":Boundaries,"regions":vorRegions,"point regions": vorPointRegion}        
    return dataset


def simpleOutputTissues(foldername,par,Lists,filename="woundinfo"):
    Aw0 = par[0]
    G = par[1]
    lr = par[2]
    lw = par[3]
    Nc = par[4]
    
    perimeterL = Lists[0]
    areaL = Lists[1]
    transitionsL = Lists[2]
    
    #Create folder 
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory,foldername+'/simple_output')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    
    with open(foldername+'/simple_output'+'/'+filename+'A'+str(round(Aw0,3))+'G'+str(G)+'L-'+str(round(lr,2))+'Lw-'+str(round(lw,2))+'Ncells'+str(Nc)+'.txt','a') as text:
        for i in range(len(perimeterL)):
                    
            text.write(str(i))
            text.write(' ')
            text.write(str(transitionsL[i]))
            text.write(' ')
            text.write(str(perimeterL[i]))
            text.write(' ')
            text.write(str(areaL[i])+ '\n')
    return
    
    
def movieOutputTissues(foldername,par,Lists):
    #Need to change this 
    centers = Lists[0]
    vorPointRegion = Lists[1]
    vorRegions = Lists[2]
    
    vertices = Lists[3]
    vorRidges = Lists[4]
    
    Boundaries = Lists[5]
    wloc = Lists[6]
    
    
    Niter = par[0]
    lr = par[1]
    lw = par[2]
    
    #Create folder 
    current_directory = os.getcwd()
    
    moviefolder = foldername+'/movie_output/l'+str(int(round(lr,4)*100))+'lw'+str(int(round(lw,4)*100))
    final_directory = os.path.join(current_directory,moviefolder)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    
    for i in range(Niter):       
        
        for c in range(len(centers[i])):
            with open(moviefolder+'/centers'+str(i)+'.txt','a') as text:
                text.write(str(c)+";"+str(centers[i][c,0])+";"+str(centers[i][c,1])+";"+str(vorPointRegion[i][c])+";"+str(vorRegions[i][vorPointRegion[i][c]])+"\n")

        for v in range(len(vertices[i])):
            #print(v)
            NeighR, Neigh_C = fc.find_vertex_neighbour_centers(vorRegions[i],vorPointRegion[i],v)
            Neigh_V = fc.find_vertex_neighbour_vertices(vorRidges[i],v)
            with open(moviefolder+'/vertices'+str(i)+'.txt','a') as text:
                text.write(str(v)+";"+str(vertices[i][v,0])+";"+str(vertices[i][v,1])+";"+str(Neigh_C)+";"+str(Neigh_V)+"\n")
                
        for e in range(len(vorRidges[i])):
            with open(moviefolder+'/edges'+str(i)+'.txt','a') as text:
                text.write(str(e)+";"+str(vorRidges[i][e])+"\n")

        #First find the boundary
        boundary_tissue = Boundaries[i][0]#fc.find_boundary_vertices(len(vertices),vorRidges)
        #truebound = np.where(vertices[boundary_tissue,0]**2+vertices[boundary_tissue,1]**2>9)
        #boundary_tissue= list(np.array(boundary_tissue)[truebound])
        boundary_wound = vorRegions[i][wloc[i]]

        with open(moviefolder+'/boundaries'+str(i)+'.txt','a') as text:
            text.write(str(boundary_tissue)+"\n")
            text.write(str(boundary_wound)+"\n")
            
        with open(moviefolder+'/woundloc'+str(i)+'.txt','a') as text:
            text.write(str(wloc[i])+"\n")
    return
            
            
def saveGeneratedNetwork(coords, size_of_wound,vorPointRegion,vorRegions,vorVertex,vorRidges,wound_loc,foldername,isnotsquare):
    N = len(coords)
    filename = foldername+'/size'+str(size_of_wound)
    
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory,filename)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
        
    for i in range(N):
        with open(filename+'/centers.txt','a') as text:
            text.write(str(i)+";"+str(coords[i,0])+";"+str(coords[i,1])+";"+str(vorPointRegion[i])+";"+str(vorRegions[vorPointRegion[i]])+"\n")

    for v in range(len(vorVertex)):
        #print(v)
        NeighR, Neigh_C = fc.find_vertex_neighbour_centers(vorRegions,vorPointRegion,v)
        Neigh_V = fc.find_vertex_neighbour_vertices(vorRidges,v)
        with open(filename+'/vertices.txt','a') as text:
            text.write(str(v)+";"+str(vorVertex[v,0])+";"+str(vorVertex[v,1])+";"+str(Neigh_C)+";"+str(Neigh_V)+"\n")
            
    for e in range(len(vorRidges)):
        with open(filename+'/edges.txt','a') as text:
            text.write(str(e)+";"+str(vorRidges[e])+"\n")

    #First find the boundary
    if isnotsquare == 1:
        boundary_tissue = fc.find_boundary_vertices(len(vorVertex),vorRidges)
    else: 
        boundary_tissue = fc.find_boundary_vertices_square(len(vorVertex),vorRidges)
    truebound = np.where(vorVertex[boundary_tissue,0]**2+vorVertex[boundary_tissue,1]**2>9)
    boundary_tissue= list(np.array(boundary_tissue)[truebound])
    boundary_wound = vorRegions[wound_loc]

    with open(filename+'/boundaries.txt','a') as text:
        text.write(str(boundary_tissue)+"\n")
        text.write(str(boundary_wound)+"\n")


    with open(filename+'/woundloc.txt','a') as text:
        text.write(str(wound_loc)+"\n")

    # print("tissue boundary "+str(boundary_tissue))
    return

            

def openTissueSimpleFile(Tsim, wound_sizes,tissues_list,G,LW1_list,L1_list,Nc,foldername):

    a1_array1 = np.zeros((len(tissues_list),len(L1_list),len(LW1_list),Tsim))
    p1_array1 = np.zeros((len(tissues_list),len(L1_list),len(LW1_list),Tsim))
    t_array1 = np.zeros((len(tissues_list),len(L1_list),len(LW1_list)))
    t1_array1 = np.zeros((len(tissues_list),len(L1_list),len(LW1_list)))
    opt_array1 = np.zeros((len(tissues_list),len(L1_list),len(LW1_list)))
    
    

    for tissue in range(len(tissues_list)):
        p1_list = []
        a1_list = []
        t_list = []
        t1_list = []
        opt_list = []
        filename_str = foldername + str(tissues_list[tissue])+'/simple_output/woundinfoA'
        for lr in L1_list:
            for lw in LW1_list:
                opt_ = []
                p1 = []
                a1 = []
                with open(filename_str+wound_sizes[tissue]+'G'+str(G)+'L-'+str(lr)+'Lw-'+str(lw)+'Ncells'+str(Nc)+'.txt','r') as text:
                    for line in text:
                        # print(line)  
                        opt_.append(float(line.replace("\n","").split(' ')[2]))
                        l_line = (line.replace("\n","")).split(' ')
                        p1.append(float(l_line[2]))
                        a1.append(float(l_line[3]))
                        pass
                    
                    last_line = (line.replace("\n","")).split(' ')
                    opt_list.append([lr,lw,np.argmax(opt_)/float(last_line[0])])
                    #print(last_line)
                    t_list.append([lr,lw,float(last_line[0])*float(wound_sizes[tissue])])
                    t1_list.append([lr,lw,float(last_line[1])])
                    p1_list.append([lr,lw,p1[:-2]])
                    a1_list.append([lr,lw,a1[:-2]])
                    
        t_array = np.array(t_list)
        t1_array = np.array(t1_list)
        opt_array = np.array(opt_list)

        
        for i in range(Tsim):
            for j in range(len(a1_list)): 

                a1_array1[tissue,j//len(LW1_list),j%len(LW1_list),i] = (a1_list[j][2][min(i,len(a1_list[j][2])-1)])
            
        
        for i in range(Tsim):
            for j in range(len(p1_list)-1):
                
                p1_array1[tissue,j//len(LW1_list),j%len(LW1_list),i] = (p1_list[j][2][min(i,len(p1_list[j][2])-1)])
        
        for i in range(t_array.shape[0]):
            t_array1[tissue,i//len(LW1_list),i%len(LW1_list)] = (t_array[i,2])
            
        for i in range(t1_array.shape[0]):
            t1_array1[tissue,i//len(LW1_list),i%len(LW1_list)] = (t1_array[i,2])
            
        for i in range(opt_array.shape[0]):
            opt_array1[tissue,i//len(LW1_list),i%len(LW1_list)] = (opt_array[i,2])
        
    return a1_array1, p1_array1,t_array1,t1_array1,opt_array1

def openTissueSimpleFileEnergy(Tsim, wound_sizes,tissues_list,G,LW1_list,L1_list,Nc,foldername):

    a1_array1 = np.zeros((len(tissues_list),len(L1_list),len(LW1_list),Tsim))
    p1_array1 = np.zeros((len(tissues_list),len(L1_list),len(LW1_list),Tsim))
    t_array1 = np.zeros((len(tissues_list),len(L1_list),len(LW1_list)))
    t1_array1 = np.zeros((len(tissues_list),len(L1_list),len(LW1_list),Tsim))
    opt_array1 = np.zeros((len(tissues_list),len(L1_list),len(LW1_list)))
    
    

    for tissue in range(len(tissues_list)):
        p1_list = []
        a1_list = []
        t_list = []
        t1_list = []
        opt_list = []
        filename_str = foldername + str(tissues_list[tissue])+'/simple_output/woundinfoA'
        for lr in L1_list:
            for lw in LW1_list:
                opt_ = []
                p1 = []
                a1 = []
                e1 = []
                with open(filename_str+wound_sizes[tissue]+'G'+str(G)+'L-'+str(lr)+'Lw-'+str(lw)+'Ncells'+str(Nc)+'.txt','r') as text:
                    for line in text:
                        # print(line)  
                        opt_.append(float(line.replace("\n","").split(' ')[2]))
                        l_line = (line.replace("\n","")).split(' ')
                        e1.append(float(l_line[1]))
                        p1.append(float(l_line[2]))
                        a1.append(float(l_line[3]))
                        pass
                    
                    last_line = (line.replace("\n","")).split(' ')
                    opt_list.append([lr,lw,np.argmax(opt_)/float(last_line[0])])
                    #print(last_line)
                    t_list.append([lr,lw,float(last_line[0])*float(wound_sizes[tissue])])
                    t1_list.append([lr,lw,e1[:-2]])
                    p1_list.append([lr,lw,p1[:-2]])
                    a1_list.append([lr,lw,a1[:-2]])
                    
        t_array = np.array(t_list)
        # t1_array = np.array(t1_list)
        opt_array = np.array(opt_list)

        
        for i in range(Tsim):
            for j in range(len(a1_list)): 

                a1_array1[tissue,j//len(LW1_list),j%len(LW1_list),i] = (a1_list[j][2][min(i,len(a1_list[j][2])-1)])
            
        
        for i in range(Tsim):
            for j in range(len(p1_list)-1):
                
                p1_array1[tissue,j//len(LW1_list),j%len(LW1_list),i] = (p1_list[j][2][min(i,len(p1_list[j][2])-1)])
        
        for i in range(t_array.shape[0]):
            t_array1[tissue,i//len(LW1_list),i%len(LW1_list)] = (t_array[i,2])
            
        for i in range(Tsim):
            for j in range(len(t1_list)-1):
                t1_array1[tissue,j//len(LW1_list),j%len(LW1_list),i] = (t1_list[j][2][min(i,len(t1_list[j][2])-1)])
            
        for i in range(opt_array.shape[0]):
            opt_array1[tissue,i//len(LW1_list),i%len(LW1_list)] = (opt_array[i,2])
        
    return a1_array1, p1_array1,t_array1,t1_array1,opt_array1