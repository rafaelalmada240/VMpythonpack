import numpy as np

def loc_dist_gr(vertices,w,size_box):
    loc_spots = np.linspace(w,size_box,int(size_box/w))
    distances_vtx = np.linalg.norm(vertices,axis=1)
    loc_locs = []
    for l in loc_spots:
        loc_vtx = np.where(np.abs(distances_vtx - l) < w/2)[0]
        nr = len(loc_vtx)
        rgr = nr/(w*l*2*np.pi)
        loc_locs.append(np.pi*size_box**2*rgr/len(vertices))
        
    return loc_locs, loc_spots

def delta_center_pos2d(Ri,Nbins):
    xy_hist_list = []
    for i in range(Ri.shape[0]):
        xy_hist, xb,yb = np.histogram2d(Ri[i,:,0],Ri[i,:,1],bins=Nbins)
        xy_hist_list.append(xy_hist)
    xy_hist_array = np.array(xy_hist_list)
    return xy_hist_array, xb, yb


def pairwise_diff(Ri, Ncells):
    pairdiff = np.zeros((Ri.shape[0],Ncells,Ncells,2))
    pairdist = np.zeros((Ri.shape[0],Ncells,Ncells))
    for i in range(Ncells):
        for j in range(Ncells):
            if i != j:
                pairdiff[:,i,j] = Ri[:,i]-Ri[:,j]
                pairdist[:,i,j] = np.linalg.norm(Ri[:,i]-Ri[:,j],axis=1)
    return pairdiff, pairdist

def correlation_r(Ri,pairdiff, pairdist, nbins):
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
    fx = np.fft.fftshift(np.fft.fftfreq(nbins,boxsize/npoints))
    fi = np.fft.fftshift(np.fft.fft2(np.array(gr_array*gr_array[0]),(nbins,nbins)))
    fj = np.conjugate(fi)
    
    return fx, fi, np.abs(fi*fj)**0.5

def structure_fac_r(FiFj, Nbins):
    Sq = np.zeros((int(Nbins/2),))
    for i in range(0,int(Nbins/2)):
        S = 0
        N = 0
        for j in range(Nbins):
            for k in range(Nbins):
                if (np.abs(j-int(Nbins/2))<=i) and (np.abs(k-int(Nbins/2))<=i):
                    S += FiFj[j,k]
                    N += 1
        Sq[i] = S/N
    return Sq

def ultrametricity_energy(e_extend):
    E_mm = np.zeros((e_extend.shape[0],e_extend.shape[0]))

    for i in range(e_extend.shape[0]):
        for j in range(e_extend.shape[0]):
            E_mm[i,j] = np.abs(e_extend[j] -e_extend[i])
    return E_mm

def ultrametricity_dist_embed(e_extend, Z1):
    Z_mm = np.zeros((e_extend.shape[0],e_extend.shape[0]))

    for i in range(e_extend.shape[0]):
        for j in range(e_extend.shape[0]):
            Z_mm[i,j] = np.linalg.norm(Z1[j]-Z1[i])
    return Z_mm