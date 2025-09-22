import numpy as np
from alter_surf.utils3D import projector2layer


def test_projector2layer():
    len_z = 5
    layer = 2
    proj = projector2layer(layer, len_z)
    assert np.sum(proj) == 1
    assert proj[layer] == 1
    assert np.all(proj[:layer] == 0)
    assert np.all(proj[layer + 1 :] == 0)