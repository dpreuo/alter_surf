from matplotlib.colors import Normalize
import numpy as np

import blochK
from alter_surf.hamiltonian_CuMnAs import H_slab_fct
from jax.numpy import einsum

from matplotlib.colors import LinearSegmentedColormap

import matplotlib

from alter_surf.utils3D import projector2layer


bwr_map = matplotlib.colormaps["bwr"]

cmap_bwr = matplotlib.colormaps["bwr"]
colors = [cmap_bwr(0.0), (0, 0, 0, 1), cmap_bwr(1.0)]
cmap_bkr = LinearSegmentedColormap.from_list("bkr", colors, N=256)


def two_param_color(x, y, a=0.5):
    """
    x ∈ [0,1]: blue → black → red
    y ∈ [0,1]: base color → white
    """
    base = np.array(cmap_bkr(x)) * np.array([1, 1, 1, 1])  # RGB only
    white = np.ones((*x.shape, 4)) * np.array([1, 1, 1, 1])

    y_scaled = 2 * (1 - y) * y * a + y * y

    y_expanded = y_scaled[..., np.newaxis] * np.array([1, 1, 1, 1])
    rgb = base * y_expanded + white * (1 - y_expanded)

    return rgb


# slab Hamiltonian
def slab_ham(Hparam0):
    Hparam0["numb_z"] = (Hparam0["len_z"] + 1) // 2
    if Hparam0["len_z"] % 2 == 0:  # len_z even: everything fine
        H = blochK.Hamiltonian2D(H_slab_fct, param=Hparam0)
        H.add_suboperator("proj", np.eye(H.n_orbitals))
        H.add_suboperator("spin", np.array([1, 1, 1, 1, -1, -1, -1, -1]))
        H.add_suboperator("layer", np.array([1, 1, -1, -1, 1, 1, -1, -1]))
        H.add_operator(
            "spin",
            np.kron(np.ones(Hparam0["numb_z"]), np.array([1, 1, 1, 1, -1, -1, -1, -1])),
        )
    else:  # len_z odd: need to project out last layer from operator definitions
        H = blochK.Hamiltonian2D(H_slab_fct, param=Hparam0)
        # construct projector to project out the last layer, which to use with operators
        selected = np.delete(np.arange(H.param["numb_z"] * 8), [-6, -5, -2, -1])
        proj = np.eye(H.param["numb_z"] * 8)[*np.ix_(selected), ...]
        H.add_suboperator("proj", proj)
        # add the other operators. operators need be constructed with H.suboperator.proj.dot(O)
        H.add_suboperator("spin", np.array([1, 1, 1, 1, -1, -1, -1, -1]))
        H.add_suboperator("layer", np.array([1, 1, -1, -1, 1, 1, -1, -1]))
        print(proj.shape, H.n_orbitals)
        H.add_operator(
            "spin",
            proj.dot(
                np.kron(
                    np.ones(Hparam0["numb_z"]), np.array([1, 1, 1, 1, -1, -1, -1, -1])
                )
            ),
        )

    return H


def project_doublelayer(layer,H):
    """Constructs the observable to project in layer and layer+1"""
    if layer%2==0: #for even layers
        return np.kron(projector2layer(layer//2,len_z=H.param['numb_z']),np.abs(H.suboperator.layer))
    else: #odd layers
        layer1 = np.kron(projector2layer(layer//2,len_z=H.param['numb_z']),H.suboperator.layer<0)
        layer2 = np.kron(projector2layer(layer//2+1,len_z=H.param['numb_z']),H.suboperator.layer>0)
        return layer1 + layer2

def spec_func(psis, es, e_vals, eta, proj):

    greens_diag = 1 / ((es + 1j * eta)[..., None] - e_vals[None, :])

    greens = einsum(
        "j...i,j...g,j...k -> ik...g", psis, greens_diag, psis.conj()
    )  # indices are matrix(i,j), momenta, energy

    # hit it with the projector
    spec = -(1 / np.pi) * einsum("ii...g,i-> ...g", greens, proj).imag

    return spec


def make_spectral_functions(H, chosen_layers, psis, es, energy_resolution, eta):

    p_layers = np.zeros(H.param["len_z"]*4)
    for layer in chosen_layers:
        p_layers += H.suboperator.proj @ project_doublelayer(layer,H)

    print(p_layers)

    p_spin = H.operator.spin * p_layers

    spec = spec_func(psis, es, energy_resolution, eta, p_layers).T
    spec_spin = spec_func(psis, es, energy_resolution, eta, p_spin).T

    return np.squeeze(spec), np.squeeze(spec_spin)


def normalize_and_find_colours(specs_bulk, specs_boundary, blackness_parameter=0.5):

    spec_same_norm_bulk = Normalize(
        np.min(specs_bulk[0]),
        np.max(specs_bulk[0]),
    )
    spec_same_norm_boundary = Normalize(
        np.min(specs_boundary[0]),
        np.max(specs_boundary[0]),
    )

    l = np.max(np.abs([specs_bulk[1], specs_boundary[1]]))

    spec_diff_norm = Normalize(-l,l)

    colors_bulk = two_param_color(
        spec_diff_norm(specs_bulk[1]),
        spec_same_norm_bulk(specs_bulk[0]),
        blackness_parameter,
    )
    colors_boundary = two_param_color(
        spec_diff_norm(specs_boundary[1]),
        spec_same_norm_boundary(specs_boundary[0]),
        blackness_parameter,
    )

    return colors_bulk, colors_boundary
