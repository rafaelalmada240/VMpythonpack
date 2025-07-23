import numpy as np
from scipy.signal import convolve2d as conv2


#########################################    Radial distributions    #################################################

def loc_dist_vtx(points,w,size_box):
    """
    Groups points into spatial bins based on distance from origin.
    
    Inputs:
        points: Array of center coordinates (N x 2)
        w: Bin width
        size_box: System size
        
    Outputs:
        loc_locs: List of center indices in each bin
        loc_spots: Bin center positions
    """
    loc_spots = np.linspace(w,size_box,int(size_box/w))
    distances_vtx = np.linalg.norm(points,axis=1)
    loc_locs = []
    for l in loc_spots:
        loc_vtx = np.where(np.abs(distances_vtx - l) < w/2)[0]
        loc_locs.append(loc_vtx)
        # print(np.abs(l))
        
    return loc_locs, loc_spots

###########################################################################################

def avg_magnitude(points,vectorfield, size_box,w):
    """
    Calculates average magnitude in spatial bins.
    
    Inputs:
        points: Reference vertex positions
        vectorfield: vectorfield array (T x N x 2)
        size_box: System size
        w: Bin width
        
    Outputs:
        mag_avg: Average magnitudes per bin per time
        vtx_dist: Bin centers
    """
    vtx_ind, vtx_dist = loc_dist_vtx(points, w, size_box)
    mag_avg = np.zeros(vectorfield.shape[0])#[]
    vdist = np.linalg.norm(vectorfield, axis=2)
    for i in range(vectorfield.shape[0]):
        mag_avg[i] = [np.nanmean(vdist[i][l]) for l in vtx_ind]
        
    return mag_avg, vtx_dist

def avg_radvel(points,vectorfield, size_box,w):
    """
    Calculates average radial vectorfield components in spatial bins.
    
    Inputs:
        points: Reference vertex positions (N x 2)
        vectorfield: vectorfield array (T x N x 2)
        size_box: System size
        w: Bin width
        
    Outputs:
        rad_avg: Mean radial velocities per bin per time
        vel_std: Std of radial velocities per bin per time  
        vtx_dist: Bin center distances
    """
    vtx_ind, vtx_dist = loc_dist_vtx(points, w, size_box)
    rad_avg = []
    vel_std = []
    
    distv = np.linalg.norm(points,axis=1)
    dirvtx = np.array([points[i]/distv[i] for i in range(len(distv))]) 
    for i in range(vectorfield.shape[0]):
        vl = []
        vs = []
        for l in vtx_ind:
            vdotvtx = np.einsum('ij,ij->i',vectorfield[i][l], dirvtx[l])
            vl.append(np.nanmean(vdotvtx))
            vs.append(np.nanstd(vdotvtx))
            #vl.append(np.nanmean([np.dot(vectorfield[i][li],dirvtx[li]) for li in l]))
            #vs.append(np.nanstd([np.dot(vectorfield[i][li],dirvtx[li]) for li in l])/np.sqrt(len(l)))
        rad_avg.append(vl)
        vel_std.append(vs)
        
    return rad_avg,vel_std, vtx_dist

def avg_tgvel(points,vectorfield, size_box,w):
    """
    Calculates average tangential vectorfield components in spatial bins.
    
    Inputs/Outputs same as avg_radvel but for tangential components
    """
    vtx_ind, vtx_dist = loc_dist_vtx(points, w, size_box)
    tg_avg = []
    vel_std = []
    
    distv = np.linalg.norm(points,axis=1)
    dirvtx = np.array([points[i]/distv[i] for i in range(len(distv))]) 
    for i in range(vectorfield.shape[0]):
        vl = []
        vs = []
        for l in vtx_ind:
            tangent = np.array([-dirvtx[l][:,1],dirvtx[l][:,0]]).T
            vdottg = np.einsum('ij,ij->i',vectorfield[i][l], tangent[l])
            vl.append(np.nanmean(vdottg))
            vs.append(np.nanstd(vdottg))
            #vl.append(np.nanmean((np.array([np.dot(vectorfield[i][li],np.array([-dirvtx[li][1],dirvtx[li][0]])) for li in l]))))
            #vs.append(np.nanstd((np.array([np.dot(vectorfield[i][li],np.array([-dirvtx[li][1],dirvtx[li][0]])) for li in l]))))
        tg_avg.append(vl)
        vel_std.append(vs)
        
    return tg_avg,vel_std, vtx_dist

#########################################################################################################

def avg_kymograph(points,vectorfield, size_box,w):
    """
    Computes radial and tangential vectorfield kymographs.
    
    Inputs/Outputs similar to vectorfield functions but returns:
        velr_avg: Radial velocities
        velt_avg: Tangential velocities
    """
    
    velr_avg = []
    # velr_std = []
    
    velt_avg = []
    # velt_std = []
    
    
    for i in range(vectorfield.shape[0]):
        vtx_ind, vtx_dist = loc_dist_vtx(points[i], w, size_box)
        distv = np.linalg.norm(points[i],axis=1)
        dirvtx = np.array([points[i,j]/distv[j] for j in range(len(distv))]) 
        vlr = []
        # vsr = []
        vlt = []
        # vst = []
        for l in vtx_ind:
            vlrx = np.array(np.nanmean([np.dot(vectorfield[i][li],dirvtx[li]) for li in l]))
            vlrx[np.isnan(vlrx)] = 0
            vlr.append(vlrx)
            # vsr.append(np.nanstd([np.dot(vectorfield[i][li],dirvtx[li]) for li in l])/np.sqrt(len(l)))
            vltx = np.array(np.nanmean((np.array([np.dot(vectorfield[i][li],np.array([-dirvtx[li][1],dirvtx[li][0]])) for li in l]))))
            vltx[np.isnan(vltx)] = 0
            vlt.append(vltx)
            # vst.append(np.nanstd((np.array([np.dot(vectorfield[i][li],np.array([-dirvtx[li][1],dirvtx[li][0]])) for li in l]))))
        velr_avg.append(vlr)
        # velr_std.append(vsr)
        velt_avg.append(vlt)
        # velt_std.append(vst)
        
    return velr_avg,velt_avg

#################################################################################################


def avg_scalar(points,scalarfield, size_box,w):
    """
    Calculates average scalar values in spatial bins.
    
    Inputs:
        points: Reference positions
        scalarfield: Scalar values (T x N)
        size_box: System size
        w: Bin width
        
    Outputs:
        scl_avg: Binned averages
        vtx_dist: Bin centers
    """
    _, vtx_dist = loc_dist_vtx(points[0], w, size_box)
    scl_avg = []
    for i in range(scalarfield.shape[0]):
        vl = []
        vtx_ind1, _ = loc_dist_vtx(points[i], w, size_box)
        for l in vtx_ind1:
            vl.append(np.nanmean(scalarfield[i][l],axis=0))
        scl_avg.append(vl)
        
    return scl_avg, vtx_dist

def avg_scalar2(points,scalarfield, size_box,w):
    """
    Calculates average scalar values in spatial bins.
    
    Inputs:
        points: Reference positions
        scalarfield: Scalar values (T x N)
        size_box: System size
        w: Bin width
        
    Outputs:
        scl_avg: Binned averages
        vtx_dist: Bin centers
    """
    vtx_ind, vtx_dist = loc_dist_vtx(points[0], w, size_box)
    scl_avg = []
    for i in range(scalarfield.shape[0]):
        vl = []

        for l in vtx_ind:
            vl.append(np.nanmean(scalarfield[i][l],axis=0))
        scl_avg.append(vl)
        
    return scl_avg, vtx_dist

def avg_scalar3(points,scalarfield, size_box,w):
    """
    Calculates median scalar values in spatial bins (single time point).
    
    Inputs/Outputs similar to avg_scalar but using median
    """
    vtx_ind, vtx_dist = loc_dist_vtx(points, w, size_box)
    scl_avg = []

    for l in vtx_ind:
        
        # print(l)
        scl_avg.append(np.nanmedian(scalarfield[l],axis=0))
        
    return scl_avg, vtx_dist

