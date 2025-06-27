import numpy as np
from scipy.signal import convolve2d as conv2
# from vertexmodelpack import connections as fc

#########################################    Analysis functions    #################################################

def loc_dist_vtx(vertices,w,size_box):
    """
    Groups vertices into spatial bins based on distance from origin.
    
    Inputs:
        vertices: Array of vertex coordinates (N x 2)
        w: Bin width
        size_box: System size
        
    Outputs:
        loc_locs: List of vertex indices in each bin
        loc_spots: Bin center positions
    """
    loc_spots = np.linspace(w,size_box,int(size_box/w))
    distances_vtx = np.linalg.norm(vertices,axis=1)
    loc_locs = []
    for l in loc_spots:
        loc_vtx = np.where(np.abs(distances_vtx - l) < w/2)[0]
        loc_locs.append(loc_vtx)
        
    return loc_locs, loc_spots

#################################################################

def avg_speed(vertices,velocity, size_box,w):
    """
    Calculates average speed in spatial bins.
    
    Inputs:
        vertices: Reference vertex positions
        velocity: Velocity array (T x N x 2)
        size_box: System size
        w: Bin width
        
    Outputs:
        vel_avg: Average speeds per bin per time
        vtx_dist: Bin centers
    """
    vtx_ind, vtx_dist = loc_dist_vtx(vertices, w, size_box)
    vel_avg = np.zeros(velocity.shape[0])#[]
    vdist = np.linalg.norm(velocity, axis=2)
    for i in range(velocity.shape[0]):
        # vl = []
        # for l in vtx_ind:
        #     vl.append(np.nanmean(vdist[i][l]))
        #vel_avg.append([np.nanmean(vdist[i][l]) for l in vtx_ind])
        vel_avg[i] = [np.nanmean(vdist[i][l]) for l in vtx_ind]
        
    return vel_avg, vtx_dist

def avg_radvel(vertices,velocity, size_box,w):
    """
    Calculates average radial velocity components in spatial bins.
    
    Inputs:
        vertices: Reference vertex positions (N x 2)
        velocity: Velocity array (T x N x 2)
        size_box: System size
        w: Bin width
        
    Outputs:
        vel_avg: Mean radial velocities per bin per time
        vel_std: Std of radial velocities per bin per time  
        vtx_dist: Bin center distances
    """
    vtx_ind, vtx_dist = loc_dist_vtx(vertices, w, size_box)
    vel_avg = []
    vel_std = []
    
    distv = np.linalg.norm(vertices,axis=1)
    dirvtx = np.array([vertices[i]/distv[i] for i in range(len(distv))]) 
    for i in range(velocity.shape[0]):
        vl = []
        vs = []
        for l in vtx_ind:
            vdotvtx = np.einsum('ij,ij->i',velocity[i][l], dirvtx[l])
            vl.append(np.nanmean(vdotvtx))
            vs.append(np.nanstd(vdotvtx))
            #vl.append(np.nanmean([np.dot(velocity[i][li],dirvtx[li]) for li in l]))
            #vs.append(np.nanstd([np.dot(velocity[i][li],dirvtx[li]) for li in l])/np.sqrt(len(l)))
        vel_avg.append(vl)
        vel_std.append(vs)
        
    return vel_avg,vel_std, vtx_dist

def avg_tgvel(vertices,velocity, size_box,w):
    """
    Calculates average tangential velocity components in spatial bins.
    
    Inputs/Outputs same as avg_radvel but for tangential components
    """
    vtx_ind, vtx_dist = loc_dist_vtx(vertices, w, size_box)
    vel_avg = []
    vel_std = []
    
    distv = np.linalg.norm(vertices,axis=1)
    dirvtx = np.array([vertices[i]/distv[i] for i in range(len(distv))]) 
    for i in range(velocity.shape[0]):
        vl = []
        vs = []
        for l in vtx_ind:
            tangent = np.array([-dirvtx[l][:,1],dirvtx[l][:,0]]).T
            vdottg = np.einsum('ij,ij->i',velocity[i][l], tangent[l])
            vl.append(np.nanmean(vdottg))
            vs.append(np.nanstd(vdottg))
            #vl.append(np.nanmean((np.array([np.dot(velocity[i][li],np.array([-dirvtx[li][1],dirvtx[li][0]])) for li in l]))))
            #vs.append(np.nanstd((np.array([np.dot(velocity[i][li],np.array([-dirvtx[li][1],dirvtx[li][0]])) for li in l]))))
        vel_avg.append(vl)
        vel_std.append(vs)
        
    return vel_avg,vel_std, vtx_dist

#########################################################################################################

def avg_kymograph(vertices,velocity, size_box,w):
    """
    Computes radial and tangential velocity kymographs.
    
    Inputs/Outputs similar to velocity functions but returns:
        velr_avg: Radial velocities
        velt_avg: Tangential velocities
    """
    
    velr_avg = []
    # velr_std = []
    
    velt_avg = []
    # velt_std = []
    
    
    for i in range(velocity.shape[0]):
        vtx_ind, vtx_dist = loc_dist_vtx(vertices[i], w, size_box)
        distv = np.linalg.norm(vertices[i],axis=1)
        dirvtx = np.array([vertices[i,j]/distv[j] for j in range(len(distv))]) 
        vlr = []
        # vsr = []
        vlt = []
        # vst = []
        for l in vtx_ind:
            vlrx = np.array(np.nanmean([np.dot(velocity[i][li],dirvtx[li]) for li in l]))
            vlrx[np.isnan(vlrx)] = 0
            vlr.append(vlrx)
            # vsr.append(np.nanstd([np.dot(velocity[i][li],dirvtx[li]) for li in l])/np.sqrt(len(l)))
            vltx = np.array(np.nanmean((np.array([np.dot(velocity[i][li],np.array([-dirvtx[li][1],dirvtx[li][0]])) for li in l]))))
            vltx[np.isnan(vltx)] = 0
            vlt.append(vltx)
            # vst.append(np.nanstd((np.array([np.dot(velocity[i][li],np.array([-dirvtx[li][1],dirvtx[li][0]])) for li in l]))))
        velr_avg.append(vlr)
        # velr_std.append(vsr)
        velt_avg.append(vlt)
        # velt_std.append(vst)
        
    return velr_avg,velt_avg

#################################################################################################

def avg_disp(vertices, size_box,w):
    
    """
    Calculates average displacement magnitudes in spatial bins.
    
    Inputs:
        vertices: Vertex positions over time (T x N x 2)
        size_box: System size
        w: Bin width
        
    Outputs:
        vel_avg: Mean displacements per bin per time
        vtx_dist: Bin center distances
    """
    
    vtx_ind, vtx_dist = loc_dist_vtx(vertices[0], w, size_box)
    vel_avg = []
    disp = np.linalg.norm(vertices-vertices[0],axis=2)
    for i in range(vertices.shape[0]):
        vl = []
        for l in vtx_ind:
            vl.append(np.nanmean(disp[i][l]))
        vel_avg.append(vl)
        
    return vel_avg, vtx_dist

def avg_raddisp(vertices, size_box,w):
    """
    Calculates average radial displacements in spatial bins.
    
    Inputs/Outputs similar to avg_disp but for radial components
    """
    vtx_ind, vtx_dist = loc_dist_vtx(vertices[0], w, size_box)
    vel_avg = []
    vel_std = []
    
    distv = np.linalg.norm(vertices[0],axis=1)
    dirvtx = np.array([vertices[0][i]/distv[i] for i in range(len(distv))]) 
    dispvec = vertices-vertices[0]
    for i in range(vertices.shape[0]):
        vl = []
        vs = []
        for l in vtx_ind:
            vdotvtx = np.einsum('ij,ij->i',dispvec[i][l], dirvtx[l])
            vl.append(np.nanmean(vdotvtx))
            vs.append(np.nanstd(vdotvtx))
            #vl.append(np.nanmean([np.dot(vertices[i][li]-vertices[0][li],dirvtx[li]) for li in l]))
            #vs.append(np.nanstd([np.dot(vertices[i][li]-vertices[0][li],dirvtx[li]) for li in l])/np.sqrt(len(l)))
        vel_avg.append(vl)
        vel_std.append(vs)
        
    return vel_avg,vel_std, vtx_dist

def avg_tgdisp(vertices, size_box,w):
    """
    Calculates average tangential displacements in spatial bins.
    
    Inputs/Outputs similar to avg_disp but for tangential components
    """
    vtx_ind, vtx_dist = loc_dist_vtx(vertices[0], w, size_box)
    vel_avg = []
    vel_std = []
    
    distv = np.linalg.norm(vertices[0],axis=1)
    dirvtx = np.array([vertices[0][i]/distv[i] for i in range(len(distv))]) 
    dispvec = vertices-vertices[0]
    for i in range(vertices.shape[0]):
        vl = []
        vs = []
        for l in vtx_ind:
            tangent = np.array([-dirvtx[l][:,1],dirvtx[l][:,0]]).T
            vdottg = np.einsum('ij,ij->i',dispvec[i][l], tangent[l])
            vl.append(np.nanmean(vdottg))
            vs.append(np.nanstd(vdottg))
            #vl.append(np.nanmean((np.array([np.dot(vertices[i][li]-vertices[0][li],np.array([-dirvtx[li][1],dirvtx[li][0]])) for li in l]))))
            #vs.append(np.nanstd((np.array([np.dot(vertices[i][li]-vertices[0][li],np.array([-dirvtx[li][1],dirvtx[li][0]])) for li in l]))))
        vel_avg.append(vl)
        vel_std.append(vs)
        
    return vel_avg,vel_std, vtx_dist

##################################################################################

def min_dist_vec(X,len_vec):
    """
    Finds representative vectors by minimum distance to window mean.
    
    Inputs:
        X: Data matrix (D x N)
        len_vec: Number of output vectors
        
    Output:
        sample_vec: Indices of representative vectors
    """
    n_real = len(X.T)/len_vec
    int_win = int(n_real)
    if n_real - int_win < 0.5:
        n_win = int_win
    else:
        n_win = int_win + 1
    sample_vec = np.zeros(len_vec)
    i_stop = 0
    for i in range(len_vec):
        mean_x = np.mean(X[:,i*n_win:(i+1)*n_win],axis=1)
        dist_x = np.sum((X[:,i*n_win:(i+1)*n_win].T-mean_x)**2,axis=1)
        if len(dist_x)==0:
            i_stop = i
            break
        else:
            sample_vec[i] = np.arange(i*n_win,(i+1)*n_win)[np.argmin(dist_x)]
            i_stop += 1
    return np.array(sample_vec[:i_stop],'int32')

def bin_xy(x_vec,y_vec,n_bins):
    """
    Bins y values based on x coordinates.
    
    Inputs:
        x_vec: x-coordinates
        y_vec: y-values
        n_bins: Number of bins
        
    Outputs:
        y_bin: Binned averages
        bins: Bin edges
    """
    min_x = -30#np.nanmin(x_vec)
    max_x = 30#np.nanmax(x_vec)#int(np.sqrt(len(x_vec)))
    if max_x == np.nan:
        max_x = np.nanquantile(x_vec,0.995)
        
    bin_size = (max_x - min_x)/n_bins

    bins = np.arange(min_x,max_x,bin_size)
    # y_bin = np.zeros((bins.shape))
    digits = np.digitize(x_vec, bins)
    y_bin = np.bincount(digits, weights=y_vec, minlength=len(bins))/np.bincount(digits, minlength=len(bins))
    # for i in range(n_bins-1):
    #     bin_vec = np.where((x_vec>= bins[i])== (x_vec < bins[i+1]))
    #     y_bin[i] = np.nanmean(y_vec[bin_vec])
    # y_bin[-1] = y_bin[-2]
    return y_bin,bins

def bin_xyz(x_vec,y_vec,z_vec,n_bins):
    """
    Bins z values into a 2D grid based on x and y coordinates.
    
    Inputs:
        x_vec, y_vec: Coordinate arrays
        z_vec: Values to bin
        n_bins: Number of bins in each dimension
        
    Outputs:
        z_bin: 2D array of binned z values
        binsx, binsy: Bin edges for each dimension
    """
    # min_x = -4.5#np.ma.min(x_vec)
    # max_x = 4.5#np.ma.max(x_vec)#int(np.sqrt(len(x_vec)))
        
        
    # min_y = -4.5#np.ma.min(y_vec)
    # max_y = 4.5#np.ma.max(y_vec)
        
    # bin_sizex = (max_x - min_x)/n_bins
    # bin_sizey = (max_y - min_y)/n_bins
    
    z_bin, binsx, binsy = np.histogram2d(x_vec, y_vec, bins=n_bins, weights=z_vec, range=[[-4.5,4.5],[-4.5,4.5]])
    count,_,_= np.histogram2d(x_vec,y_vec, bins=n_bins, range=[[-4.5,4.5],[-4.5,4.5]])[0]
    z_bin = np.divide(z_bin, count, out=np.zeros_like(z_bin),where=count!=0)

    # binsx = np.arange(min_x,max_x,bin_sizex)
    # binsy = np.arange(min_y,max_y,bin_sizey)
    
    # z_bin = np.zeros((binsx.shape[0],binsy.shape[0]))
    
    # for i in range(n_bins-1):
    #     bin_vecx = list(np.where((x_vec>= binsx[i])== (x_vec < binsx[i+1]))[0])
    #     for j in range(n_bins-1):
    #         bin_vecy = list(np.where((y_vec>= binsy[j])== (y_vec < binsy[j+1]))[0])
    #         bin_vec = list(set(bin_vecx).intersection(bin_vecy))
    #         if len(bin_vec) < 1:
    #             z_bin[j,i] = 0
    #         else:
    #             z_bin[j,i] = np.nanmean(z_vec[bin_vec])
    return z_bin,binsx,binsy


def smoothfunc(arr,window_size):
    """
    Simple moving average smoothing.
    
    Inputs:
        arr: Input array
        window_size: Smoothing window
        
    Output:
        moving_averages: Smoothed array
    """
    
    # i = 0
    # Initialize an empty list to store moving averages
    #moving_averages = []
    
    window = np.ones(window_size)/window_size
    moving_averages = np.convolve(arr,window, mode='same')
    # # Loop through the array to consider
    # # every window of size 3
    # while i < len(arr):
    
    #     # Store elements from i to i+window_size
    #     # in list to get the current window
    #     next = i + (window_size)*((i + window_size)<=len(arr))+((i + window_size)%len(arr))*((i + window_size)>len(arr))
    #     window = arr[i :next]
    
    #     # Calculate the average of current window
    #     window_average = sum(window) / len(window)
        
    #     # Store the average of current
    #     # window in moving average list
    #     moving_averages.append(window_average)
        
    #     # Shift window to right by one position
    #     i += 1
    return np.array(moving_averages)

def corrfunc(x,y,T):
    """
    Computes lagged correlation between x and y.
    
    Inputs:
        x, y: Input arrays
        T: Maximum lag
        
    Output:
        f: Correlation coefficients
    """
    f = np.zeros(T)
    for i in range(T):
        xf = len(x)-i
        f[i] = np.corrcoef(x[0:xf]*y[i:])
    return f

def GradArray(Array):
    """
    Computes gradient magnitude using Sobel-like operator.
    
    Input:
        Array: 2D input array
        
    Output:
        Gradient magnitude array
    """
    D_array = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    return np.sqrt(conv2(Array,D_array,'same','symm')**2 + conv2(Array,D_array.T,'same','symm')**2)/3

def n_shape(p0):
    
    """
    Estimates number of sides for given perimeter.
    
    Input:
        p0: Normalized perimeter
        
    Output:
        Estimated number of sides
    """
    
    x_init = 4
    epsilon = 1
    i = 0
    while (epsilon >= 0.005) and (i <= 500):
        ptrial = 2*np.sqrt(x_init*np.tan(np.pi/x_init))
        epsilon = np.abs(p0-ptrial)
        if epsilon >= 0.005:
            new_factor = (np.tan(np.pi/x_init)-np.pi/x_init*1/np.cos(np.pi/x_init)**2)/np.sqrt(x_init*np.tan(np.pi/x_init))
            # print(new_factor)
            x_init = x_init - ptrial/new_factor*0.01*np.sign(ptrial-p0)
        i = i +1
    if x_init < 3:
        x_init = 3
    if p0 <= 2*np.sqrt(np.pi):
        x_init = 50
    return x_init

def avg_scalar(vertices,velocity, size_box,w):
    """
    Calculates average scalar values in spatial bins.
    
    Inputs:
        vertices: Reference positions
        velocity: Scalar values (T x N x D)
        size_box: System size
        w: Bin width
        
    Outputs:
        vel_avg: Binned averages
        vtx_dist: Bin centers
    """
    vtx_ind, vtx_dist = loc_dist_vtx(vertices, w, size_box)
    vel_avg = []
    for i in range(velocity.shape[0]):
        vl = []
        for l in vtx_ind:
            vl.append(np.nanmean(velocity[i][l],axis=0))
        vel_avg.append(vl)
        
    return vel_avg, vtx_dist

def avg_scalar2(vertices,velocity, size_box,w):
    """
    Calculates median scalar values in spatial bins (single time point).
    
    Inputs/Outputs similar to avg_scalar but using median
    """
    vtx_ind, vtx_dist = loc_dist_vtx(vertices, w, size_box)
    vel_avg = []

    for l in vtx_ind:
        vel_avg.append(np.nanmedian(velocity[l],axis=0))
        
    return vel_avg, vtx_dist

