
from alter_surf.hamiltonian_2orbital import H_2orb_2D_fct, H_2orb_3D_fct
import numpy as np

def test_H_2orb_2D_fct():

    kx = np.array([0, 2])
    ky = np.array([1, 1])
    H = H_2orb_2D_fct(kx, ky)

    assert H.shape == (4, 4, 2)


def test_H_2orb_3D_fct():
    param = dict(len_z=6,t1=1,t2=0.5,mu=0,m=0.3,t3=0,tz=1,tzp=0,PBC=False,Q_z=np.pi)

    kx = np.array([0, 2])
    ky = np.array([1, 1])
    H = H_2orb_3D_fct(kx, ky, **param)

    assert H.shape == (4*6, 4*6, 2)