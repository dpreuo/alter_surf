from alter_surf.mean_field import hartree_fock
import numpy as np


def test_mean_field():
    
    #first test for fixed filling
    Hparam = dict(len_z=10,delta=1, tz=1, U=8, filling=0.2)
    initial_parameters = dict(initial_mAF=(-1)**np.arange(Hparam['len_z']), initial_mF=np.ones(Hparam['len_z'])/10, initial_n=Hparam['filling']*np.ones(Hparam['len_z']))

    mAFMs, mFMs, ns, fermi_energys = hartree_fock(Hparam,initial_parameters, 20, Lq=20)

    #fixed fermi energy
    Hparam = dict(len_z=10,delta=1, tz=1, U=8, filling=0.2, mu=-2)
    initial_parameters = dict(initial_mAF=(-1)**np.arange(Hparam['len_z']), initial_mF=np.ones(Hparam['len_z'])/10, initial_n=Hparam['filling']*np.ones(Hparam['len_z']))
    
    mAFMs, mFMs, ns, fermi_energys = hartree_fock(Hparam,initial_parameters, 20, Lq=20, fixed_Fermi_energy=True)
    assert np.allclose(fermi_energys,Hparam['mu'])