import numpy as np
from vmvideo import movies_from_plots as mfp
from vmvideo import io
from vmvideo import core

def makemovie(x_array, v_array, ridge_array,boundaries,regions,wloc, foldername):
    N_initial = int(input('Starting frame: '))
    N_final = int(input('Stopping frame: '))
    fr_step = int(input('Frame step: '))
    file_path = 'f'
    
    xl = float(input('X limit plot: '))
    yl = float(input('Y limit plot: '))
    
    option = int(input('Do you want to save the frames with color showing number of sides, anisotropy, polarity, stress or deformation [n(0),a(1), p(2), s(3), d(4)]: '))
    if option == 0:
        core.saveframe(x_array, v_array, ridge_array, boundaries, N_initial,N_final,fr_step,file_path,regions, wloc, xl, yl)
    if option == 3:
        which_stressoption = (int(input('Which stress option do you want: trace(0), shear(1),torque(2) ')))
        core.plot_colored_images1(x_array, v_array, ridge_array, foldername, N_initial,N_final,fr_step,file_path,regions, wloc,xl,yl,which_stressoption)
    if option == 4:
        core.plot_colored_images2(x_array, v_array, ridge_array, foldername, N_initial,N_final,fr_step,file_path,regions, wloc,xl,yl)
    if option == 1:
        core.plot_colored_images_aniso(x_array, v_array, ridge_array, boundaries, N_initial,N_final,fr_step,file_path,regions, wloc,xl,yl)
    if option == 2:
        core.plot_colored_images_polar(x_array, v_array, ridge_array, boundaries, N_initial,N_final,fr_step,file_path,regions, wloc,xl,yl)
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

data_set = io.open_file(foldername,int(datafileloadname),issubfolder)

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
