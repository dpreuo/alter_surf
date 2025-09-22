import numpy as np 

def projector2layer(layer,len_z=2):
    """Projector operator onto specific layer."""
    proj = np.zeros(len_z)
    proj[layer] = 1
    return proj