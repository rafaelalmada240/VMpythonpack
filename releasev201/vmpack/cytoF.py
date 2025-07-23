import numpy as np

def force_cytoskeleton(regions, vertices, coords, par):
    Fv = np.zeros_like(vertices)
    Fc = np.zeros_like(coords)
    return Fv, Fc


def dist_vertex_center(regions,vertices, coords):
    dist_vc = np.zeros((len(coords),len(vertices)))
    return dist_vc

def update_rest_length(r0_array, regions, vertices, coords):
    dist_vc = np.zeros((len(coords),len(vertices)))
    new_ro_array = dist_vc
    return new_ro_array