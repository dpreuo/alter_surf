from alter_surf.mean_field import hartree_fock
import numpy as np


def test_mean_field():
    
    Hparam = dict(len_z=10,delta=1, tz=1, U=8, filling=0.2)
    initial_parameters = dict(initial_m=(-1)**np.arange(Hparam['len_z']), initial_n=Hparam['filling']*np.ones(Hparam['len_z']))

    m_values, n_values, fermi_energys = hartree_fock(Hparam,initial_parameters, 20, Lq=20)