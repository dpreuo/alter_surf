import numpy as np 
import matplotlib
from matplotlib.colors import LinearSegmentedColormap


def projector2layer(layer,len_z=2):
    """Projector operator onto specific layer."""
    proj = np.zeros(len_z)
    proj[layer] = 1
    return proj


def project_doublelayer(layer,H):
    """Constructs the observable to project in layer and layer+1"""
    if layer%2==0: #for even layers
        return np.kron(projector2layer(layer//2,len_z=H.param['numb_z']),np.abs(H.suboperator.layer))
    else: #odd layers
        layer1 = np.kron(projector2layer(layer//2,len_z=H.param['numb_z']),H.suboperator.layer<0)
        layer2 = np.kron(projector2layer(layer//2+1,len_z=H.param['numb_z']),H.suboperator.layer>0)
        return layer1 + layer2


#create a custom colormap bkr
# grab the endpoints from bwr
cmap_bwr = matplotlib.colormaps["bwr"]
# define your own colormap: blue → black → red
colors = [cmap_bwr(0.0), (0, 0, 0, 1),cmap_bwr(1.0)]
cmap_bkr = LinearSegmentedColormap.from_list("bkr", colors, N=256)


def two_param_color(x, y, a=0.5):
    """
    x ∈ [-1,1]: blue → black → red
    y ∈ [0,1]: base color → white
    """
    x = (x + 1) / 2  # scale x to [0,1]

    base = np.array(cmap_bkr(x)) * np.array([1, 1, 1, 1])  # RGB only
    white = np.ones((*x.shape, 4)) * np.array([1, 1, 1, 1])

    y_scaled = 2 * (1 - y) * y * a + y * y

    y_expanded = y_scaled[..., np.newaxis] * np.array([1, 1, 1, 1])
    rgb = base * y_expanded + white * (1 - y_expanded)

    return rgb