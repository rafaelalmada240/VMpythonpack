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
THRESHOLD_COLORBAR = 1e-1

def _base_voronoi_plotter(vor, values, fr, xl, yl, wloc, cmap='viridis', 
                         norm_range=None, label=None, quiver_data=None):
        """
    Core function for generating colored Voronoi diagrams with optional vector fields.
    
    Parameters
    ----------
    vor : scipy.spatial.Voronoi
        Voronoi diagram object containing point and region data
    values : array-like
        Numerical values used for coloring each Voronoi cell
    fr : int
        Frame number used in output filename (f{fr}.png)
    xl : float
        Half-width of x-axis limits (sets xlim to [-xl, xl])
    yl : float
        Half-height of y-axis limits (sets ylim to [-yl, yl])
    wloc : int
        Index of special cell to exclude from plotting (typically wound location)
    cmap : str, optional
        Matplotlib colormap name (default: 'viridis')
    norm_range : tuple, optional
        (min, max) values for colormap normalization (default: autoscale to data)
    label : str, optional
        Label for the colorbar (default: no label)
    quiver_data : array-like, optional
        (N,2) array of vector components for quiver plots (default: None)
        
    Notes
    -----
    - Automatically skips infinite regions and the special region at index wloc
    - Saves plot to 'f{fr}.png' with DPI specified by SAVE_DPI
    - Clears figure memory after saving to prevent leaks
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set up colormap
    colormap = plt.cm.get_cmap(cmap)
    norm = plt.Normalize(*norm_range) if norm_range else plt.Normalize(values.min(), values.max())
    
    # Create patches with colors
    patches = []
    colors = []
    for i, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if (-1 in region or len(region) == 0) or (i == wloc):
            continue
            
        polygon = [vor.vertices[j] for j in region]
        patches.append(Polygon(polygon, closed=True))
        colors.append(colormap(norm(values[i])))
    
    # Add main collection
    p = PatchCollection(patches, facecolor=colors, edgecolor='black', alpha=0.7)
    ax.add_collection(p)
    
    # Add Voronoi skeleton
    voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, 
                   line_colors='black', alpha=0.3)
    ax.plot(vor.points[:,0], vor.points[:,1], 'k.')
    
    # Add quivers if specified
    if quiver_data:
        vectors = np.array(quiver_data)
        ax.quiver(vor.points[:,0], vor.points[:,1], 
                 vectors[:,0], vectors[:,1],
                 scale=25, color='k', pivot='mid')
    
    # Axis setup
    ax.set_xlim([-xl, xl])
    ax.set_ylim([-yl, yl])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array(values)
    cbar = plt.colorbar(sm, ax=ax)
    if label:
        cbar.set_label(label)
    
    # Save and clean up
    plt.savefig(f'f{fr}.png', dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)

def plot_voronoi_colored(vor, values, fr, xl, yl, wloc, cmap='viridis'):
    """
    Plot Voronoi diagram with cells colored by signed scalar values.
    
    Parameters
    ----------
    values : array-like
        Scalar values in range [-THRESHOLD_COLORBAR, THRESHOLD_COLORBAR]
        
    See Also
    --------
    _base_voronoi_plotter : Core plotting function
    """
    _base_voronoi_plotter(vor, values, fr, xl, yl, wloc, cmap, 
                         norm_range=(-THRESHOLD_COLORBAR, THRESHOLD_COLORBAR))

def plot_voronoi_colored_polar(vor, values, fr, xl, yl, wloc, cmap='viridis'):
    """
    Plot Voronoi diagram with polarity vectors and angle coloring (0 to π).
    
    Parameters
    ----------
    values : array-like
        Angle values in radians (0 to π)
    Notes
    -----
    - Adds bidirectional quiver arrows showing orientation
    - Colorbar represents angular values
    """
    quiver_data = zip(np.cos(values), np.sin(values))
    _base_voronoi_plotter(vor, values, fr, xl, yl, wloc, cmap,
                         norm_range=(0, np.pi), label='Polarity',
                         quiver_data=quiver_data)

def plot_voronoi_colored_aniso(vor, values, fr, xl, yl, wloc, cmap='viridis'):
    """
    Plot Voronoi diagram colored by shape anisotropy (0 to 1).
    
    Parameters
    ----------
    values : array-like
        Anisotropy values in range [0, 1]
        
    See Also
    --------
    _base_voronoi_plotter : Core plotting function
    """
    _base_voronoi_plotter(vor, values, fr, xl, yl, wloc, cmap,
                         norm_range=(0, 1), label='Shape Anisotropy')
    
def _plot_colored_images_core(x_array, v_array, ridge_array_1, regions, wloc, xl, yl,
                            N_init, N_fin, f_step, file_path=None, foldername=None,
                            value_func=None, plot_func=None, cmap='viridis', 
                            stress_option=None, gamma_plot_index=None):
    """
    Core function for plotting time-series of colored Voronoi diagrams.
    
    Parameters
    ----------
    x_array : list of arrays
        Cell positions for each frame
    v_array : list of arrays
        Voronoi vertices for each frame
    ridge_array_1 : list of arrays
        Ridge information for each frame
    regions : list of lists
        Voronoi regions for each frame
    wloc : int
        Index of wound cell to exclude
    xl, yl : float
        Plot boundaries
    N_init, N_fin, f_step : int
        Frame range and step size
    file_path : str, optional
        Output directory path
    foldername : str, optional
        Subdirectory for stress/gamma files
    value_func : function, optional
        Function to calculate cell values from frame data
    plot_func : function
        Voronoi plotting function to use
    cmap : str
        Colormap name
    stress_option : int, optional
        Stress component selection (0-2)
    gamma_plot_index : int, optional
        Gamma component index to plot
        
    Returns
    -------
    bool
        True if completed successfully, False if memory threshold exceeded
    """
    t_initial = time.time()
    
    for inst in range(N_init, N_fin, f_step):
        # Prepare Voronoi data
        pregions = np.arange(len(x_array[inst]))
        regions_1 = regions[inst]
        
        for r in range(len(regions_1)):
            regions_1[r] = list(np.array(regions_1[r])[
                fc.rearrange(len(regions_1[r]), 
                fc.adj_mat(regions_1[r], ridge_array_1[inst]))
            ])
        
        vor = Voronoi(x_array[inst])
        vor.point_region = pregions
        vor.regions = regions_1
        vor.ridge_vertices = ridge_array_1[inst]
        vor.vertices = v_array[inst]
        
        # Calculate plot values
        if value_func:
            values = value_func(inst, vor, foldername, stress_option, gamma_plot_index)
        else:
            values = np.zeros(len(x_array[inst]))
        
        # Generate plot
        plot_func(vor, values, int(inst/f_step), xl, yl, wloc, cmap)
        
        # Memory check
        memory_percent = ps.virtual_memory().percent
        if memory_percent >= MEMORYTHRESHOLD:
            print(f'Memory overload at frame {inst} - stopping')
            t_final = time.time() - t_initial
            with open('log.txt','w') as log:
                log.write(f'Memory overloaded: Simulation stops at: {t_final}s\n')
                log.write(f'frame where memory crashed: {inst}')
            return False
    
    return True

# Specialized functions with original names
def plot_colored_images_polar(x_array, v_array, ridge_array_1, boundaries, 
                            N_init, N_fin, f_step, file_path, regions, wloc, xl, yl):
    """Plot cell polarity angles over time"""
    def polarity_values(inst, vor, *_):
        polardir = gmp.cell_polarity(regions[inst], np.arange(len(x_array[inst])),
                                v_array[inst], x_array[inst])
        polar_shift(polardir, x_array[inst])
        return unwrap_atan2(np.arctan2(polardir[:,1], polardir[:,0]))
    
    return _plot_colored_images_core(
        x_array, v_array, ridge_array_1, regions, wloc, xl, yl,
        N_init, N_fin, f_step, file_path=file_path,
        value_func=polarity_values,
        plot_func=plot_voronoi_colored_polar,
        cmap='hsv'
    )

def plot_colored_images_aniso(x_array, v_array, ridge_array_1, boundaries,
                            N_init, N_fin, f_step, file_path, regions, wloc, xl, yl):
    """Plot shape anisotropy over time"""
    def anisotropy_values(inst, vor, *_):
        values = gmp.shape_anisotropy(regions[inst], np.arange(len(x_array[inst])),
                                    v_array[inst], x_array[inst])
        values[wloc] = 0
        return values
    
    return _plot_colored_images_core(
        x_array, v_array, ridge_array_1, regions, wloc, xl, yl,
        N_init, N_fin, f_step, file_path=file_path,
        value_func=anisotropy_values,
        plot_func=plot_voronoi_colored_aniso,
        cmap='jet'
    )

def plot_colored_images1(x_array, v_array, ridge_array_1, foldername,
                       N_init, N_fin, f_step, file_path, regions, wloc, xl, yl, 
                       stress_option):
    """Plot stress components over time"""
    def stress_values(inst, _, foldername, stress_option, *__):
        stress_i = []
        with open(f'{foldername}stress{inst}.txt','r') as f:
            for line in f:
                parts = line.strip().split()
                if stress_option == 0:
                    stress_i.append(float(parts[1]) + float(parts[4]))
                elif stress_option == 1:
                    stress_i.append(float(parts[2]) + float(parts[3]))
                elif stress_option == 2:
                    stress_i.append(float(parts[2]) - float(parts[3]))
        return 3 * np.array(stress_i)
    
    return _plot_colored_images_core(
        x_array, v_array, ridge_array_1, regions, wloc, xl, yl,
        N_init, N_fin, f_step, foldername=foldername,
        value_func=stress_values,
        plot_func=plot_voronoi_colored,
        cmap='seismic',
        stress_option=stress_option
    )

def plot_colored_images2(x_array, v_array, ridge_array_1, foldername,
                       N_init, N_fin, f_step, file_path, regions, wloc, xl, yl):
    """Plot gamma values over time"""
    def gamma_values(inst, _, foldername, *__):
        gamma_i = []
        with open(f'{foldername}gammaarray{inst+1}.txt','r') as f:
            for line in f:
                parts = line.strip().split(';')
                gamma_i.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return (np.array(gamma_i)[:,1] > 0).astype(float)
    
    return _plot_colored_images_core(
        x_array, v_array, ridge_array_1, regions, wloc, xl, yl,
        N_init, N_fin, f_step, foldername=foldername,
        value_func=gamma_values,
        plot_func=plot_voronoi_colored,
        cmap='binary'
    )

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
