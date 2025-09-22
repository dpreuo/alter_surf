
from alter_surf.hamiltonian_DLKK import H_DLKK_2D, H_DLKK_3D, H_DLKK_3D_MF, Econst_DLKK_3D_MF, find_m_and_n_values
import numpy as np

def test_H_DLKK_2D():
    param = dict(t=1,tp=0.5,delta=0.3,mu=0,mAF=1,mF=0.1)
    kx = np.array([0, 2])
    ky = np.array([1, 1])
    H = H_DLKK_2D(kx, ky, **param)

    assert H.shape == (4, 4, 2)


def test_H_DLKK_3D():
    param = dict(len_z=3,delta=1, tz=1,mAF=0.9,mF=0.1,Q_z=0,delta_Q_z=0,mu=0,PBC=False)
    kx = np.array([1, 2])
    ky = np.array([1, 1])
    H = H_DLKK_3D(kx, ky, **param)

    assert H.shape == (4*param['len_z'], 4*param['len_z'], 2)


def test_H_DLKK_3D_MF():
    param = dict(len_z=3,delta=1, tz=1 ,delta_Q_z=0,mu=0,PBC=False, U=2, filling=0.5)
    [param['mAF'], param['mF'], param['ns']] = np.random.rand(3,param['len_z'])
    kx = np.array([1, 2])
    ky = np.array([1, 1])
    H = H_DLKK_3D_MF(kx,ky,**param)
    
    assert H.shape == (4*param['len_z'], 4*param['len_z'], 2)


def test_Econst_DLKK_3D_MF():
    param = dict(len_z=3,delta=1, tz=1 ,delta_Q_z=0,mu=0,PBC=False, U=2, filling=0.5)
    [param['mAF'], param['mF'], param['ns']] = np.random.rand(3,param['len_z'])
    E_const = Econst_DLKK_3D_MF(**param)


def test_find_m_and_n_values():

    from blochK.methods_basic import sample_reducedBZ
    from blochK.observable import eigs_H
    import matplotlib.pyplot as plt

    ks = sample_reducedBZ(10)
    es, psi = eigs_H(*ks, H_DLKK_3D, dict(len_z=10, tz=1, delta=1, mAF=0.9, Q_z=np.pi, mu=-3))
    m,n = find_m_and_n_values(es,psi,0)

    assert m.shape == (10,2)
    assert n.shape == (10,2)

    plt.plot(m)
    plt.plot(n)

