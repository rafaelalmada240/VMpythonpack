import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import sys
sys.path.insert(0, '/home/rafael/Documents/2nd Year/backup5git')
from vertexmodelpack import connections as fc
from vertexmodelpack import geomproperties as gmp
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import psutil as ps
import time
from vertexvisualization.utils import unwrap_atan2, polar_shift

MEMORYTHRESHOLD = 80.0
SAVE_DPI = 200

def plot_voronoi_colored(vor, values, fr, xl, yl, wloc, cmap='viridis'):

    #vor = Voronoi(points)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Normalize values for color mapping
    # print(np.max(values))
    # print(np.min(values))
    # M = np.min(np.abs(np.max(values)),np.abs(np.min(values)))
    
    norm = plt.Normalize(-1e-1,1e-1)
    
    colormap = plt.cm.get_cmap(cmap)
    
    patches = []
    colors = []
    
    for i, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if (-1 in region or len(region) == 0) or (i == wloc):
            continue  # Skip infinite regions
        
        
        polygon = [vor.vertices[j] for j in region]
        patches.append(Polygon(polygon, closed=True))
        colors.append(colormap(norm(values[i])))
    
    # Create collection of patches
    p = PatchCollection(patches, facecolor=colors, edgecolor='black', alpha=0.7)
    ax.add_collection(p)
    
    # Plot original Voronoi diagram for reference
    voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, line_colors='black', alpha=0.3)
    ax.plot(vor.points[:,0],vor.points[:,1],'k.')
    
    ax.set_xlim([-xl,xl])
    ax.set_ylim([-yl,yl])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array(values)
    plt.colorbar(sm, ax=ax)
    fig.tight_layout()
    plt.savefig('f'+str(fr)+'.png',dpi = SAVE_DPI, bbox_inches='tight')
    fig.clear()
    
    
    plt.close(fig)
    
    
    
    
    #plt.title("Voronoi Diagram Colored by Scalar Field")
    #plt.show()
    
def plot_voronoi_colored_polar(vor, values, fr, xl, yl, wloc, cmap='viridis'):

    #vor = Voronoi(points)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Normalize values for color mapping
    norm = plt.Normalize(0, np.pi)
    colormap = plt.cm.get_cmap(cmap)
    
    patches = []
    colors = []
    
    for i, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if (-1 in region or len(region) == 0) or (i == wloc):
            continue  # Skip infinite regions

        polygon = [vor.vertices[j] for j in region]
        patches.append(Polygon(polygon, closed=True))
        colors.append(colormap(norm(values[i])))
    
    # Create collection of patches
    p = PatchCollection(patches, facecolor=colors, edgecolor='black', alpha=0.7)
    ax.add_collection(p)
    
    # Plot original Voronoi diagram for reference
    voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, line_colors='black', alpha=0.3)
    ax.plot(vor.points[:,0],vor.points[:,1],'k.')
    
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
    plt.savefig('f'+str(fr)+'.png',dpi = SAVE_DPI, bbox_inches='tight')
    fig.clear()
    plt.close(fig)
    
    
    
    
    #plt.title("Voronoi Diagram Colored by Scalar Field")
    #plt.show()
    
def plot_voronoi_colored_aniso(vor, values, fr, xl, yl, wloc, cmap='viridis'):

    #vor = Voronoi(points)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Normalize values for color mapping
    norm = plt.Normalize(0,1)
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
    voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, line_colors='black', alpha=0.3)
    ax.plot(vor.points[:,0],vor.points[:,1],'k.')
    
    ax.set_xlim([-xl,xl])
    ax.set_ylim([-yl,yl])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array(values)
    plt.colorbar(sm, ax=ax, label='Shape Anisotropy')
    fig.tight_layout()
    plt.savefig('f'+str(fr)+'.png',dpi = SAVE_DPI, bbox_inches='tight')
    fig.clear()
    plt.close(fig)
    
    
    
    
    #plt.title("Voronoi Diagram Colored by Scalar Field")
    #plt.show()
    
def plot_colored_images_polar(x_array, v_array, ridge_array_1, boundaries, N_init,N_fin,f_step,file_path,regions, wloc, xl, yl):
     
    t_initial = time.time()
    for inst in range(N_init,N_fin,f_step):
        pregions = np.arange(len(x_array[inst]))
        polardir = gmp.cell_polarity(regions[inst],pregions,v_array[inst],x_array[inst])
        polar_shift(polardir, x_array[inst])
        ang_dir = np.arctan2(polardir[:,1],polardir[:,0])
        regions_1 = regions[inst]
        for r in range(len(regions_1)):
            regions_1[r] = list(np.array(regions_1[r])[fc.rearrange(len(regions_1[r]),fc.adj_mat(regions_1[r],ridge_array_1[inst]))])
        
        vor = Voronoi(x_array[inst])
        vor.point_region = pregions
        vor.regions = regions_1
        vor.ridge_vertices = ridge_array_1[inst]
        vor.vertices = v_array[inst]    
        plot_voronoi_colored_polar(vor, unwrap_atan2(ang_dir), int(inst/f_step), xl, yl,wloc,'hsv')
        memory_val = ps.virtual_memory()
        memory_percent=memory_val.percent


        if memory_percent>=MEMORYTHRESHOLD:
            print(f'Memory overload at frame {inst} - stopping')
            t_final = time.time() - t_initial
            with open('log.txt','w') as log:
                log.write('Memory overloaded: Simulation stops at: '+str(t_final)+'s \n')
                log.write('frame where memory crashed: '+str(inst))
            break
    
def plot_colored_images_aniso(x_array, v_array, ridge_array_1, boundaries, N_init,N_fin,f_step,file_path,regions, wloc, xl, yl):
     
    t_initial = time.time()
    for inst in range(N_init,N_fin,f_step):
        pregions = np.arange(len(x_array[inst]))
        shape_anisotropy = gmp.shape_anisotropy(regions[inst],pregions,v_array[inst],x_array[inst])
        shape_anisotropy[wloc] = 0
        
        regions_1 = regions[inst]
        for r in range(len(regions_1)):
            regions_1[r] = list(np.array(regions_1[r])[fc.rearrange(len(regions_1[r]),fc.adj_mat(regions_1[r],ridge_array_1[inst]))])
        
        vor = Voronoi(x_array[inst])
        vor.point_region = pregions
        vor.regions = regions_1
        vor.ridge_vertices = ridge_array_1[inst]
        vor.vertices = v_array[inst]
        plot_voronoi_colored_aniso(vor, shape_anisotropy, int(inst/f_step), xl, yl,wloc, 'jet')
        memory_val = ps.virtual_memory()
        memory_percent=memory_val.percent

        if memory_percent>=MEMORYTHRESHOLD:
            print(f'Memory overload at frame {inst} - stopping')
            t_final = time.time() - t_initial
            with open('log.txt','w') as log:
                log.write('Memory overloaded: Simulation stops at: '+str(t_final)+'s \n')
                log.write('frame where memory crashed: '+str(inst))
            break


def plot_colored_images(x_array, v_array, ridge_array_1, boundaries, N_init,N_fin,f_step,file_path,regions, wloc, xl, yl):
     
    t_initial = time.time()
    for inst in range(N_init,N_fin,f_step):
        pregions = np.arange(len(x_array[inst]))
        # shape_anisotropy = gmp.shape_anisotropy(regions[inst],pregions,v_array[inst],x_array[inst])
        # shape_anisotropy[wloc] = 0
        # polardir = gmp.cell_polarity(regions[inst],pregions,v_array[inst],x_array[inst])
        # polar_shift(polardir, x_array[inst])
        # ang_dir = np.arctan2(polardir[:,1],polardir[:,0])
        #difA1 = wga.generalized_net_lap_center(pregions[tnumber][inst],regions[tnumber][inst],np.array((areas_vec1[inst])),wloc)
        regions_1 = regions[inst]
        for r in range(len(regions_1)):
            regions_1[r] = list(np.array(regions_1[r])[fc.rearrange(len(regions_1[r]),fc.adj_mat(regions_1[r],ridge_array_1[inst]))])
        
        vor = Voronoi(x_array[inst])
        vor.point_region = pregions
        vor.regions = regions_1
        vor.ridge_vertices = ridge_array_1[inst]
        vor.vertices = v_array[inst]
    
        # plot_voronoi_colored(vor, unwrap_atan2(ang_dir), int(inst/f_step), xl, yl,wloc,'hsv')
        #plot_voronoi_colored_aniso(vor, shape_anisotropy, int(inst/f_step), xl, yl,wloc, 'jet')
        memory_val = ps.virtual_memory()
        memory_percent=memory_val.percent

  
        if memory_percent>=MEMORYTHRESHOLD:
            print(f'Memory overload at frame {inst} - stopping')
            t_final = time.time() - t_initial
            with open('log.txt','w') as log:
                log.write('Memory overloaded: Simulation stops at: '+str(t_final)+'s \n')
                log.write('frame where memory crashed: '+str(inst))
            break
    # for i in range(len(edges[tnumber][inst])):
    #     if any(np.array(edges[tnumber][inst][i])==-1)==0:
    #         plt.plot(vertices[tnumber][inst,edges[tnumber][inst][i],0],
    #                 vertices[tnumber][inst,edges[tnumber][inst][i],1],'-',color='k',alpha=1,lw=1)

    # plt.scatter(coords[tnumber][inst][:,0],
    #             coords[tnumber][inst][:,1], c=(difA1),s=250,cmap='rainbow')
    # plt.colorbar()
    # plt.clim(0,1)
    #plt.title("Area change - f" + str(inst))
            
def plot_colored_images1(x_array, v_array, ridge_array_1, foldername, N_init,N_fin,f_step,file_path,regions, wloc, xl, yl, stress_option):
     
    t_initial = time.time()
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
        with open(foldername+'stress'+str(inst)+'.txt','r') as f:
            for line in f:
                if stress_option == 0:
                    calcium_i.append(float(line.replace("\n","").split(' ')[1])+float(line.replace("\n","").split(' ')[4]))
                if stress_option == 1:
                    calcium_i.append(float(line.replace("\n","").split(' ')[2])+float(line.replace("\n","").split(' ')[3]))
                if stress_option == 2:
                    calcium_i.append(float(line.replace("\n","").split(' ')[2])-float(line.replace("\n","").split(' ')[3]))
        plot_voronoi_colored(vor, 3*np.array(calcium_i), int(inst/f_step), xl, yl,wloc, 'seismic')
        memory_val = ps.virtual_memory()
        memory_percent=memory_val.percent

        if memory_percent>=MEMORYTHRESHOLD:
            print(f'Memory overload at frame {inst} - stopping')
            t_final = time.time() - t_initial
            with open('log.txt','w') as log:
                log.write('Memory overloaded: Simulation stops at: '+str(t_final)+'s \n')
                log.write('frame where memory crashed: '+str(inst))
            break


def plot_colored_images2(x_array, v_array, ridge_array_1, foldername, N_init,N_fin,f_step,file_path,regions, wloc, xl, yl):
     
    t_initial = time.time()
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
        memory_val = ps.virtual_memory()
        memory_percent=memory_val.percent

        if memory_percent>=MEMORYTHRESHOLD:
            print(f'Memory overload at frame {inst} - stopping')
            t_final = time.time() - t_initial
            with open('log.txt','w') as log:
                log.write('Memory overloaded: Simulation stops at: '+str(t_final)+'s \n')
                log.write('frame where memory crashed: '+str(inst))
            break
        #plot_voronoi_colored(vor, np.array(calcium_i)[:,1], int(inst/f_step), xl, yl,wloc, 'Reds')

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
        plt.savefig(file_path+str(int(k/f_step))+'.png',dpi=SAVE_DPI,bbox_inches='tight')
        fig.clear()
    
        
        
        
        memory_val = ps.virtual_memory()
        memory_percent=memory_val.percent

        if memory_percent>=MEMORYTHRESHOLD:
            print(f'Memory overload at frame {k} - stopping')
            t_final = time.time() - t_initial
            with open('log.txt','w') as log:
                log.write('Memory overloaded: Simulation stops at: '+str(t_final)+'s \n')
                log.write('frame where memory crashed: '+str(k))
            break
    plt.close(fig)
    #plt.show()
    return
