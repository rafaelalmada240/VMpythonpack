import numpy as np
import matplotlib.pyplot as plt
import cv2 
from scipy.spatial import Voronoi, voronoi_plot_2d
import os

def loadframe(N_init,N_fin,f_step,file_path):
    img_array = []
    #file_path = g
    for k in range(N_init,N_fin,f_step):
        img = cv2.imread(file_path + str(k)+'.png')
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    return img_array,size

def delete_frame(k, file_path):
    os.remove(file_path + str(k)+'.png')
    return 
 
def savevideo(img_array,filename,size, file_path):
    out = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
 
    for i in range(len(img_array)):
        out.write(img_array[i])
        delete_frame(i,file_path)
    out.release()
    return