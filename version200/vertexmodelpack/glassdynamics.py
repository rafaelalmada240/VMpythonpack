import numpy as np

def loc_dist_gr(vertices,w,size_box):
    """
    Calculates radial distribution function (RDF) for vertices.
    
    Inputs:
        vertices: Array of vertex coordinates (N x 2)
        w: Bin width
        size_box: System size
        
    Outputs:
        loc_locs: Normalized RDF values
        loc_spots: Bin center positions
        
    Note: The normalization (np.pi*size_box**2/len(vertices)) seems unusual for RDF.
    Standard RDF normalization would use local density rather than global average.
    """
    loc_spots = np.linspace(w,size_box,int(size_box/w))
    distances_vtx = np.linalg.norm(vertices,axis=1)
    #loc_locs = []
    counts, bin_edges = np.histogram(distances_vtx, bins=loc_spots)
    loc_locs = counts/(2*np.pi*2*bin_edges[:-1])
    # for l in loc_spots:
    #     loc_vtx = np.where(np.abs(distances_vtx - l) < w/2)[0]
    #     nr = len(loc_vtx)
    #     rgr = nr/(w*l*2*np.pi)
    #     loc_locs.append(np.pi*size_box**2*rgr/len(vertices))
        
    return loc_locs, loc_spots

def delta_center_pos2d(Ri,Nbins):
    """
    Computes 2D histograms of cell center positions over time.
    
    Inputs:
        Ri: Cell center positions (T x N x 2)
        Nbins: Number of histogram bins
        
    Outputs:
        xy_hist_array: Array of 2D histograms (T x Nbins x Nbins)
        xb, yb: Bin edges
    """
    xy_hist_list = []
    for i in range(Ri.shape[0]):
        xy_hist, xb,yb = np.histogram2d(Ri[i,:,0],Ri[i,:,1],bins=Nbins)
        xy_hist_list.append(xy_hist)
    xy_hist_array = np.array(xy_hist_list)
    return xy_hist_array, xb, yb


def pairwise_diff(Ri):
    """
    Computes pairwise differences and distances between cell centers.
    
    Inputs:
        Ri: Cell center positions (T x N x 2)
        
    Outputs:
        pairdiff: Pairwise differences (T x N x N x 2)
        pairdist: Pairwise distances (T x N x N)
        
    Note: Diagonal elements (i=j) are left as zeros, which is standard.
    """
    # pairdiff = np.zeros((Ri.shape[0],Ncells,Ncells,2))
    # pairdist = np.zeros((Ri.shape[0],Ncells,Ncells))
    
    pairdiff = Ri[:,:,None]-Ri[:,None,:]
    pairdist = np.linalg.norm(pairdiff,axis=-1)
    # for i in range(Ncells):
    #     for j in range(Ncells):
    #         if i != j:
    #             pairdiff[:,i,j] = Ri[:,i]-Ri[:,j]
    #             pairdist[:,i,j] = np.linalg.norm(Ri[:,i]-Ri[:,j],axis=1)
    return pairdiff, pairdist

def correlation_r(Ri,pairdiff, pairdist, nbins):
    """
    Computes spatial correlation functions from pairwise differences.
    
    Inputs:
        Ri: Cell positions (T x N x 2)
        pairdiff: Precomputed pairwise differences
        pairdist: Precomputed pairwise distances
        nbins: Number of bins
        
    Outputs:
        gr_array: 2D correlation functions
        rx, ry: Bin edges for differences
        nr_array: Radial distribution
        nx: Bin edges for distances
    """
    gr_list = []
    nr_list = []
    for i in range(Ri.shape[0]):
        # DeltaRi ,Rx, Ry = bin_xyz(Ri[i,:,0]-np.mean(Ri[i,:,0]),Ri[i,:,1]-np.mean(Ri[i,:,1]),np.ones(len(Ri[i])),50)
        pd0, rx, ry = np.histogram2d(pairdiff[i,:,:,0].flatten(),pairdiff[i,:,:,1].flatten(),bins=nbins,density=True)
        nr, nx = np.histogram(pairdist[i,:,:].flatten(),bins=nbins,density=True)
        # DeltaRi_list.append(DeltaRi)
        # pd0[50,50] = 0
        gr_list.append(pd0)
        nr_list.append(nr)
        
    # DeltaRi_array = np.array(DeltaRi_list)
    gr_array = np.array(gr_list)
    nr_array = np.array(nr_list)
    return gr_array, rx, ry, nr_array, nx


def structure_factor(gr_array, npoints, boxsize, nbins):
    
    """
    Computes structure factor from correlation functions.
    
    Inputs:
        gr_array: Correlation functions
        npoints: Number of points
        boxsize: System size
        nbins: Number of bins
        
    Outputs:
        fx: Frequency bins
        fi: Fourier components
        Magnitude spectrum (sqrt of power spectrum)
        
    Note: The multiplication by gr_array[0] before FFT is unusual.
    Standard practice would use the raw correlation function.
    """
    
    fx = np.fft.fftshift(np.fft.fftfreq(nbins,d = boxsize/npoints))
    fi = np.fft.fftshift(np.fft.fft2(np.array(gr_array-1),(nbins,nbins)))
    fj = np.conjugate(fi)
    
    return fx, fi, np.abs(fi*fj)

def structure_fac_r(FiFj, Nbins):
    """
    Computes radially averaged structure factor.
    
    Inputs:
        FiFj: Power spectrum
        Nbins: Number of radial bins
        
    Outputs:
        Sq: Radially averaged structure factor
    """
    
    # Sq = np.zeros((int(Nbins/2),))
    y, x = np.indices(FiFj.shape)-Nbins//2
    r = np.sqrt(x**2+y**2).astype(int)
    Sq = np.bincount(r.ravel(),weights=FiFj.ravel())/np.bincount(r.ravel())
    # for i in range(0,int(Nbins/2)):
    #     S = 0
    #     N = 0
    #     for j in range(Nbins):
    #         for k in range(Nbins):
    #             if (np.abs(j-int(Nbins/2))<=i) and (np.abs(k-int(Nbins/2))<=i):
    #                 S += FiFj[j,k]
    #                 N += 1
    #     Sq[i] = S/N
    return Sq

def ultrametricity_energy(e_extend):
    """
    Computes pairwise energy differences.
    
    Input:
        e_extend: Energy values (N,)
        
    Output:
        E_mm: Matrix of absolute differences (N x N)
    """
    
    # E_mm = np.zeros((e_extend.shape[0],e_extend.shape[0]))
    
    # Broadcasting
    
    E_mm = np.abs(e_extend[:,None]-e_extend[None,:])

    # for i in range(e_extend.shape[0]):
    #     for j in range(e_extend.shape[0]):
    #         E_mm[i,j] = np.abs(e_extend[j] -e_extend[i])
    return E_mm

def ultrametricity_dist_embed(Z1):
    """
    Computes pairwise distances in embedding space.
    
    Inputs:
        Z1: Embedded coordinates (N x D)
        
    Output:
        Z_mm: Distance matrix (N x N)
    """
    # Z_mm = np.zeros((e_extend.shape[0],e_extend.shape[0]))
    
    Z_mm = np.linalg.norm(Z1[:,None]-Z1[None,:],axis=-1)

    # for i in range(e_extend.shape[0]):
    #     for j in range(e_extend.shape[0]):
    #         Z_mm[i,j] = np.linalg.norm(Z1[j]-Z1[i])
    return Z_mm