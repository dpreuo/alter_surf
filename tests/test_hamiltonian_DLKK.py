
from alter_surf.hamiltonian_DLKK import H_DLKK_2D, H_DLKK_3D
import numpy as np

def test_H_DLKK_2D():
    param = dict(t=1,tp=0.5,delta=0.3,mu=0,m=1)
    kx = np.array([0, 2])
    ky = np.array([1, 1])
    H = H_DLKK_2D(kx, ky, **param)

    assert H.shape == (4, 4, 2)

def test_H_DLKK_3D():
    param = dict(len_z=3,delta=1, tz=1,m=0.9,Q_z=0,delta_Q_z=0,mu=0,PBC=False)
    kx = np.array([1, 2])
    ky = np.array([1, 1])
    H = H_DLKK_3D(kx, ky, **param)

    assert H.shape == (4*param['len_z'], 4*param['len_z'], 2)
