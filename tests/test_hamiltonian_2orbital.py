
from alter_surf.hamiltonian_2orbital import H_2orb_2D, H_2orb_3D
import numpy as np

def test_H_2orb_2D():

    kx = np.array([0, 2])
    ky = np.array([1, 1])
    H = H_2orb_2D(kx, ky)

    assert H.shape == (4, 4, 2)


def test_H_2orb_2D():

    kx = np.array([0, 2])
    ky = np.array([1, 1])
    H = H_2orb_3D(kx, ky, 6)

    assert H.shape == (4*6, 4*6, 2)