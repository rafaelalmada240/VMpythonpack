import numpy as np
import re

def open_file(foldername0,N,issubfolder):
    
    coords_list = []
    point_regions = []
    regions_list = []
    vertex_list = []
    boundaries_list = []
    edges_list = []
    for i in range(N):
        #print(i)
        coords = []
        vorPointRegion1 = []
        vorRegions = []
        
        if issubfolder == 0:
            foldername = foldername0+'movie_output/'
        else:
            foldername = foldername0

        with open(foldername+'centers'+str(i)+'.txt','r') as text:
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
        with open(foldername+'vertices'+str(i)+'.txt','r') as text:
                    for line in text:
                        last_line = (line.replace("\n","")).split(';')
                        vertices.append([float(last_line[1]),float(last_line[2])])
        vertices = np.array(vertices)

        #print(vorPointRegion)

        vorRidges = []
        with open(foldername+'edges'+str(i)+'.txt','r') as text:
            for line in text:
                last_line = (line.replace("\n","")).split(';')
                l4 = re.split(' |, ',last_line[1].replace("[","").replace("]","").replace("np.int64(","").replace(")",""))#

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

        with open(foldername+'woundloc'+str(i)+'.txt','r') as text:
            for line in text:
                wloc = int(line.replace("\n",""))

        vorPointRegion= []
        for k in range(len(coords)):
            vorPointRegion.append(k)

        Boundaries = []                
        with open(foldername+'boundaries'+str(i)+'.txt','r') as text:
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
    return dataset