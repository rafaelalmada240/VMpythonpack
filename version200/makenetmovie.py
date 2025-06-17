import numpy as np
import matplotlib.pyplot as plt
from vertexmodelpack import movies_from_plots as mfp
from scipy.spatial import Voronoi, voronoi_plot_2d
from vertexmodelpack import sppvertex as sppv
from vertexmodelpack import connections as fc
from vertexmodelpack import geomproperties as gmp
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
#import pathlib
#import h5py as h5
import psutil as ps
from IPython.core.display import HTML
import time
import re

#from https://www.kaggle.com/getting-started/210022 with some changes
def restart_kernel_and_run_all_cells():
    HTML(
        '''
            <script>
                code_show = false;
                function restart_run_all(){
                    IPython.notebook.kernel.restart();
                }
                function code_toggle() {
                    if (code_show) {
                        $('div.input').hide(200);
                    } else {
                        $('div.input').show(200);
                    }
                    code_show = !code_show
                }
                code_toggle() 
                restart_run_all()
            </script>

        '''
    )
    

# Plot frames from different time steps   

def plot_voronoi_colored(vor, values, fr, xl, yl, wloc, cmap='viridis'):

    #vor = Voronoi(points)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Normalize values for color mapping
    norm = plt.Normalize(0, np.pi)
    #norm = plt.Normalize(0,np.max(values))
    # norm = plt.Normalize(0,1)
    colormap = plt.cm.get_cmap(cmap)
    
    patches = []
    colors = []
    
    for i, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if (-1 in region or len(region) == 0) or (i == wloc):
            continue  # Skip infinite regions
        
        #print(i)
        #print(len(values))
        polygon = [vor.vertices[j] for j in region]
        patches.append(Polygon(polygon, closed=True))
        colors.append(colormap(norm(values[i])))
    
    # Create collection of patches
    p = PatchCollection(patches, facecolor=colors, edgecolor='black', alpha=0.7)
    ax.add_collection(p)
    
    # Plot original Voronoi diagram for reference
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', alpha=0.3)
    
    ax.set_xlim([-xl,xl])
    ax.set_ylim([-yl,yl])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')
    
    plt.quiver(vor.points[:,0],vor.points[:,1], np.cos(values), 
               np.sin(values),scale=25, color='k', pivot = 'mid')
    plt.quiver(vor.points[:,0],vor.points[:,1], -np.cos(values), 
               -np.sin(values),scale=25, color='k', pivot = 'mid')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array(values)
    plt.colorbar(sm, ax=ax, label='Polarity')
    fig.tight_layout()
    plt.savefig('f'+str(fr)+'.png',dpi = 200, bbox_inches='tight')
    fig.clear()
    plt.close(fig)
    #plt.colorbar(sm, ax=ax, label='Shape Anisotropy')
    
    
    
    #plt.title("Voronoi Diagram Colored by Scalar Field")
    #plt.show()
    
def plot_colored_images(x_array, v_array, ridge_array_1, boundaries, N_init,N_fin,f_step,file_path,regions, wloc, xl, yl):
     
    
    for inst in range(N_init,N_fin,f_step):
        pregions = np.arange(len(x_array[inst]))
        shape_anisotropy = gmp.shape_anisotropy(regions[inst],pregions,v_array[inst],x_array[inst])
        shape_anisotropy[wloc] = 0
        polardir = gmp.cell_polarity(regions[inst],pregions,v_array[inst],x_array[inst])
        polar_shift(polardir, x_array[inst])
        ang_dir = np.arctan2(polardir[:,1],polardir[:,0])
        #difA1 = wga.generalized_net_lap_center(pregions[tnumber][inst],regions[tnumber][inst],np.array((areas_vec1[inst])),wloc)
        regions_1 = regions[inst]
        for r in range(len(regions_1)):
            regions_1[r] = list(np.array(regions_1[r])[fc.rearrange(len(regions_1[r]),fc.adj_mat(regions_1[r],ridge_array_1[inst]))])
        
        vor = Voronoi(x_array[inst])
        vor.point_region = pregions
        vor.regions = regions_1
        vor.ridge_vertices = ridge_array_1[inst]
        vor.vertices = v_array[inst]
    
        plot_voronoi_colored(vor, unwrap_atan2(ang_dir), int(inst/f_step), xl, yl,wloc,'hsv')
        #plot_voronoi_colored(vor, shape_anisotropy, int(inst/f_step), xl, yl,wloc, 'jet')

    # for i in range(len(edges[tnumber][inst])):
    #     if any(np.array(edges[tnumber][inst][i])==-1)==0:
    #         plt.plot(vertices[tnumber][inst,edges[tnumber][inst][i],0],
    #                 vertices[tnumber][inst,edges[tnumber][inst][i],1],'-',color='k',alpha=1,lw=1)

    # plt.scatter(coords[tnumber][inst][:,0],
    #             coords[tnumber][inst][:,1], c=(difA1),s=250,cmap='rainbow')
    # plt.colorbar()
    # plt.clim(0,1)
    #plt.title("Area change - f" + str(inst))
        
        
def plot_colored_images1(x_array, v_array, ridge_array_1, foldername, N_init,N_fin,f_step,file_path,regions, wloc, xl, yl):
     
    
    for inst in range(N_init,N_fin,f_step):
        pregions = np.arange(len(x_array[inst]))
        regions_1 = regions[inst]
        for r in range(len(regions_1)):
            regions_1[r] = list(np.array(regions_1[r])[fc.rearrange(len(regions_1[r]),fc.adj_mat(regions_1[r],ridge_array_1[inst]))])
        
        vor = Voronoi(x_array[inst])
        vor.point_region = pregions
        vor.regions = regions_1
        vor.ridge_vertices = ridge_array_1[inst]
        vor.vertices = v_array[inst]
    
        #plot_voronoi_colored(vor, unwrap_atan2(ang_dir),'hsv')
        
        calcium_i = []
        with open(foldername+'calciumconc'+str(inst+1)+'.txt','r') as f:
            for line in f:
                calcium_i.append(float(line.replace("\n","").split(';')[1]))
        plot_voronoi_colored(vor, 3*np.array(calcium_i), int(inst/f_step), xl, yl,wloc, 'gnuplot2')

    # for i in range(len(edges[tnumber][inst])):
    #     if any(np.array(edges[tnumber][inst][i])==-1)==0:
    #         plt.plot(vertices[tnumber][inst,edges[tnumber][inst][i],0],
    #                 vertices[tnumber][inst,edges[tnumber][inst][i],1],'-',color='k',alpha=1,lw=1)

    # plt.scatter(coords[tnumber][inst][:,0],
    #             coords[tnumber][inst][:,1], c=(difA1),s=250,cmap='rainbow')
    # plt.colorbar()
    # plt.clim(0,1)
    #plt.title("Area change - f" + str(inst))
        # plt.quiver(coords[inst][:,0],coords[inst][:,1], polardir[:,0], 
        #        polardir[:,1],scale=25, color='k', pivot = 'mid')

def plot_colored_images2(x_array, v_array, ridge_array_1, foldername, N_init,N_fin,f_step,file_path,regions, wloc, xl, yl):
     
    
    for inst in range(N_init,N_fin,f_step):
        pregions = np.arange(len(x_array[inst]))
        
        regions_1 = regions[inst]
        for r in range(len(regions_1)):
            regions_1[r] = list(np.array(regions_1[r])[fc.rearrange(len(regions_1[r]),fc.adj_mat(regions_1[r],ridge_array_1[inst]))])
        
        vor = Voronoi(x_array[inst])
        vor.point_region = pregions
        vor.regions = regions_1
        vor.ridge_vertices = ridge_array_1[inst]
        vor.vertices = v_array[inst]
    
        #plot_voronoi_colored(vor, unwrap_atan2(ang_dir),'hsv')
        
        calcium_i = []
        with open(foldername+'gammaarray'+str(inst+1)+'.txt','r') as f:
            for line in f:
                lf = line.replace("\n","").split(';')
                # print(lf)
                calcium_i.append([float(lf[1]),float(lf[2]),float(lf[3])])
        plot_voronoi_colored(vor, np.array(calcium_i)[:,1]>0, int(inst/f_step), xl, yl, wloc ,'binary')
        #plot_voronoi_colored(vor, np.array(calcium_i)[:,1], int(inst/f_step), xl, yl,wloc, 'Reds')

    # for i in range(len(edges[tnumber][inst])):
    #     if any(np.array(edges[tnumber][inst][i])==-1)==0:
    #         plt.plot(vertices[tnumber][inst,edges[tnumber][inst][i],0],
    #                 vertices[tnumber][inst,edges[tnumber][inst][i],1],'-',color='k',alpha=1,lw=1)

    # plt.scatter(coords[tnumber][inst][:,0],
    #             coords[tnumber][inst][:,1], c=(difA1),s=250,cmap='rainbow')
    # plt.colorbar()
    # plt.clim(0,1)
    #plt.title("Area change - f" + str(inst))
        # plt.quiver(coords[inst][:,0],coords[inst][:,1], polardir[:,0], 
        #        polardir[:,1],scale=25, color='k', pivot = 'mid')
    
    
    
def unwrap_atan2(atan):
    for i in range(len(atan)):
        if atan[i] < 0:
            atan[i] += np.pi
    return atan

def polar_shift(pol, coords):
    for p in range(len(pol)):
        if coords[p,1]<= -(coords[p,0]+0.5):
            pol[p] = np.array([pol[p,0], -pol[p,1]])
        else:
            pol[p] = np.array([-pol[p,0], pol[p,1]])
            
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


#Add regions and wloc to the mix:


def saveframe(x_array, v_array, ridge_array_1, boundaries, N_init,N_fin,f_step,file_path,regions, wloc, xl, yl):
    t_initial = time.time() #time where the printing of frames starts
    fig = plt.figure(figsize=(8,8))
    #ordered_boundary1 = sppv.rearrange(len(boundaries[0][1]),sppv.adj_mat(boundaries[0][1],ridge_array))
    #ordered_boundary1.append(ordered_boundary[0])
    #plt.plot(v_array[0,list(np.array(boundaries[0][1])[ordered_boundary]),0],v_array[0,list(np.array(boundaries[0][1])[ordered_boundary]),1],'--', color='k', lw = 3)  
    for k in range(N_init,N_fin,f_step):
        ridge_array = ridge_array_1[k,:,:].astype(int)
        
        
        V = Voronoi(x_array[k,:,:])
        # voronoi_plot_2d(V,show_vertices=False,show_points=False,line_alpha=0.1)
        ordered_boundary = fc.rearrange(len(regions[k][wloc]),fc.adj_mat(regions[k][wloc],ridge_array))
        ordered_boundary.append(ordered_boundary[0])
        for i in range(len(ridge_array)):
           
            if any(np.array(ridge_array[i])==-1)==0:
                plt.plot(v_array[k,ridge_array[i],0],v_array[k,ridge_array[i],1],'-',color='black',alpha=1,lw=2)
        # plt.plot(v_array[k,:,0],v_array[k,:,1],'r.',alpha=0.5)
        
        #plt.plot(v_array[k,list(np.array(regions[k][wloc])[ordered_boundary]),0],v_array[k,list(np.array(regions[k][wloc])[ordered_boundary]),1],'-', color='maroon', lw = 5)  
        #
        # polygon = [[-7,-7],[-7,7],[7,7],[7,-7]]
        # plt.fill(*zip(*polygon),'midnightblue',alpha=0.5)
        for r in range(len(regions[k])):
            if r != wloc:
                region = list(np.array(regions[k][r])[fc.rearrange(len(regions[k][r]),fc.adj_mat(regions[k][r],ridge_array))])
                region.append(region[0])
                if (not -1 in region):
                    polygon = [v_array[k,j] for j in region]
                    if len(polygon) <= 4:
                        plt.fill(*zip(*polygon),'darkviolet',alpha=1.0)
                    if len(polygon) == 5:
                        plt.fill(*zip(*polygon),'midnightblue',alpha=1.0)
                    if len(polygon) == 6:
                        plt.fill(*zip(*polygon),'deepskyblue',alpha=1.0)
                    if len(polygon) == 7:
                        plt.fill(*zip(*polygon),'green',alpha=1.0)
                    if len(polygon) >= 8:
                        plt.fill(*zip(*polygon),'darkorange',alpha=1.0)
        
        wound_poly =  [v_array[k,i] for i in list(np.array(boundaries[k][1])[ordered_boundary]) if fc.norm(v_array[k,i]) < np.sqrt(2)*5]    
        plt.fill(*zip(*wound_poly),'r',alpha=1.0)
        # plt.figtext(0.5,0.8,'$\mathregular{\lambda_W = 5.}$',fontsize = 50,color='white')
        # plt.figtext(0.5,0.7,'$\mathregular{p_0 = 4.}$',fontsize = 50,color='white')
        
        #plt.plot(x_array[k,:,0],x_array[k,:,1],'b.')
        
        

        # plt.xlabel('x')
        # plt.ylabel('y')
        #plt.title('Step = '+str(k*0.01))
        
        plt.xlim(-xl,xl)
        plt.ylim(-yl,yl)
        # plt.grid('False')
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])   
        plt.savefig(file_path+str(int(k/f_step))+'.png',dpi=150,bbox_inches='tight')
        fig.clear()
    
        
        
        
        memory_val = ps.virtual_memory()
        memory_percent=memory_val.percent

        memory_thresh = 80.0
        if memory_percent>=memory_thresh:
            print('Memory overload')
            t_final = time.time() - t_initial
            with open('log.txt','w') as log:
                log.write('Memory overloaded: Simulation stops at: '+str(t_final)+'s \n')
                log.write('frame where memory crashed: '+str(k))
            restart_kernel_and_run_all_cells()
            #HTML("<script>Jupyter.notebook.kernel.restart()</script>")
            break
    plt.close(fig)
    #plt.show()
    return

def makemovie(x_array, v_array, ridge_array,boundaries,regions,wloc, foldername):
    N_initial = int(input('Starting frame: '))
    N_final = int(input('Stopping frame: '))
    fr_step = int(input('Frame step: '))
    file_path = 'f'
    
    xl = float(input('X limit plot: '))
    yl = float(input('Y limit plot: '))
    
    option = int(input('Do you want to save the frames with color showing number of sides, anisotropy,calcium production or myosin activation [n(1),s(0), c(2), a(3)]: '))
    if option == 1:
        saveframe(x_array, v_array, ridge_array, boundaries, N_initial,N_final,fr_step,file_path,regions, wloc, xl, yl)
    if option == 2:
        plot_colored_images1(x_array, v_array, ridge_array, foldername, N_initial,N_final,fr_step,file_path,regions, wloc,xl,yl)
    if option == 3:
        plot_colored_images2(x_array, v_array, ridge_array, foldername, N_initial,N_final,fr_step,file_path,regions, wloc,xl,yl)
    if option == 0:
        plot_colored_images(x_array, v_array, ridge_array, boundaries, N_initial,N_final,fr_step,file_path,regions, wloc,xl,yl)
    return

# Load dataset to use

#main_path = pathlib.Path().absolute()
datafileloadname = input('Number of points to open: ')
#datafileloadname = datafileloadname + '.h5'
# foldername1 = input('Where is the movieoutput?: ')
foldername2 = input('Which tissue?: ')
foldername3 = input('Which wound size?: ')
issubfolder = int(input('Is it in a sub-folder (y-1,n-0): '))

if issubfolder==1:
    whichl = input('p0? - ')
    whichlw = input('lw? - ')
    foldername1 = 'l'+whichl+'lw'+whichlw+'/'
    foldername = 'tissues/tissue'+foldername2+'/size'+foldername3+'/movie_output/'+foldername1
    foldername4 = 'tissues/tissue'+foldername2+'/size'+foldername3+'/simple_output/l'+whichl+'lw'+whichlw+'/'
else:
    foldername = 'tissues/tissue'+foldername2+'/size'+foldername3+'/'

data_set = open_file(foldername,int(datafileloadname),issubfolder)

coords_evo = np.array(data_set['centers'])
print(coords_evo.shape)
coords_evo_vertex = np.array(data_set['vertices'])
print(coords_evo_vertex.shape)
ridge_vectors = np.array(data_set['Edge connections'])
print(ridge_vectors.shape)
print(ridge_vectors[0,1])
boundary = data_set['boundaries']
print(boundary[0][1])

#Additional code for polygon coloring
wloc = data_set['WoundLoc']
regions = data_set['regions']

# Make the movie
makemovie(coords_evo,coords_evo_vertex, ridge_vectors,boundary,regions, wloc, foldername4)

prompt = int(input('Do you want to make the movie now [y(1),n(0)]: '))
if prompt == 1:
        
    N_init = 0
    N_end = int(input('Stopping frame: '))
    fr_step = 1
    file_path = 'f'
    filename = input('Video file name (with .mp4 included): ')
        
    img_array,size = mfp.loadframe(N_init,N_end,fr_step,file_path)
    mfp.savevideo(img_array,filename,size,file_path)
