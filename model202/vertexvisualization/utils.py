import numpy as np

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