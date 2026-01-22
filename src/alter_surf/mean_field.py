from blochK import Hamiltonian2D

from alter_surf.hamiltonian_DLKK import find_m_and_n_values, Econst_DLKK_3D_MF

import numpy as np
from tqdm import tqdm
import scipy.optimize as optimize


def single_hartree_fock_step(
    Hamiltonian: Hamiltonian2D,
    Lq = 100,
    fixed_fermi_energy=False,
):  
    
    # make and solve the Hamiltonian
    ks = Hamiltonian.BZ.sample(Lq)
    es, psi = Hamiltonian.diagonalize(*ks)

    # find the fermi energy for the given filling
    if not fixed_fermi_energy: #standard case: filling is fixed
        normalization_factor = np.prod(es.shape)
        res_root = optimize.root_scalar(lambda mu: np.sum(es<mu)-normalization_factor*Hamiltonian.param['filling'], bracket=[es.min(), es.max()], method='bisect')
        if not res_root.converged:
            raise ValueError(
                "Root finding for the fermi energy did not converge"
            )
        fermi_energy = res_root.root
    else: 
        fermi_energy = Hamiltonian.param['mu']

    # calculate the new MF values
    m_z_sublattice, n_z_sublattice = find_m_and_n_values(es, psi, fermi_energy) #.shape = (layers, sublattice) 

    #Restrictions
    #NO CDW
    #enforce equal density
    n_z = np.sum(n_z_sublattice,axis=1)/2 #density at sites A from 0 to 2
    #AFM and FM ordering on A,B
    #enforce equal 
    mAF_z = (m_z_sublattice[:,0]-m_z_sublattice[:,1])/2 #AF magnetization at sites A from 0 to 1
    mF_z  = (m_z_sublattice[:,0]+m_z_sublattice[:,1])/2 #F  magnetization at sites A from 0 to 1
    #these definitions implies 
    #n_{i,z,up} = n_z/2 + mAF_z/2 + mF_z/2
    #n_{i,z,dw} = n_z/2 - mAF_z/2 - mF_z/2

    return mAF_z, mF_z, n_z, fermi_energy


def hartree_fock(
    Hamiltonian: Hamiltonian2D,
    initial_parameters: dict,
    n_steps=100,
    Lq=100, #linear number of k-points in the BZ
    mixing_proportion=0.2,
    verbose=True,
    tol_mdiff=1e-6,
    leave=True,
    adjust_learning_rate=False,
    fixed_Fermi_energy=False,
):

    prange = tqdm(range(n_steps), leave=leave) if verbose else range(n_steps)
    skip_counter = 0

    #the 3 mean fields are unrestricted in z-direction
    mAFMs = np.zeros((n_steps + 1, Hamiltonian.param['len_z']))
    mAFMs[0] = initial_parameters["initial_mAF"]

    mFMs = np.zeros((n_steps + 1, Hamiltonian.param['len_z']))
    mFMs[0] = initial_parameters["initial_mF"]

    ns = np.zeros((n_steps + 1, Hamiltonian.param['len_z']))
    ns[0] = initial_parameters["initial_n"]

    fermi_energys = np.zeros((n_steps + 1))
    if fixed_Fermi_energy:
        fermi_energys[0] = Hamiltonian.param['mu']
    else:
        fermi_energys[0] = 0
        Hamiltonian.param['mu'] = 0

    for n in prange:

        #update the Hamiltonian parameters 
        Hamiltonian.param['mAF'] = mAFMs[n]
        Hamiltonian.param['mF']  = mFMs[n]
        Hamiltonian.param['ns'] = ns[n]

        new_mAFM, new_mFM, new_n, fermi_energy = single_hartree_fock_step(Hamiltonian,Lq=Lq,fixed_fermi_energy=fixed_Fermi_energy)

        mAFMs[n + 1] = (1 - mixing_proportion) * new_mAFM + mixing_proportion * mAFMs[n]
        mFMs[n + 1]  = (1 - mixing_proportion) * new_mFM  + mixing_proportion * mFMs[n]
        ns[n + 1]    = (1 - mixing_proportion) * new_n + mixing_proportion * ns[n]
        fermi_energys[n + 1] = fermi_energy

        # check for convergence
        diff = np.linalg.norm(mAFMs[n + 1] - mAFMs[n])/(np.sqrt(len(mAFMs[n + 1])))
        avg_m = np.mean(mAFMs[n + 1])
        avg_m_stag = np.mean(mAFMs[n + 1]*(-1)**np.arange(Hamiltonian.param['len_z']))
        
        u = 6
        learning_condition = adjust_learning_rate and n > u
        
        if verbose and not learning_condition:
            prange.set_description(f"Avg:{avg_m:.2f}, Stag. avg:{avg_m_stag:.2f}, diff: {diff:.6f}")

        # if the difference is small enough, we can stop
        if diff < tol_mdiff:
            skip_counter += 1
            if skip_counter >= 3:
                mAFMs = mAFMs[: n + 1]
                mFMs  = mFMs[: n + 1]
                ns = ns[: n + 1]
                fermi_energys = fermi_energys[: n + 1]
                break


        if learning_condition: #learning is only implemented for AFM

            last_m_vals = mAFMs[n-u:n]
            last_m_vals = last_m_vals[::-1]

            dif = last_m_vals[:-2] - last_m_vals[1:-1]
            next_dif = last_m_vals[:-2] - last_m_vals[2:]

            dif_vals = np.abs(dif)
            double_dif_vals = np.abs(next_dif)

            zizag = np.mean(dif_vals/double_dif_vals)
            if verbose:
                prange.set_description(f"Avg:{avg_m:.2f}, diff: {diff:.6f}, zigzag: {zizag:.4f}, mix: {mixing_proportion:.4f}")
            if zizag > 1:
                # mix more
                mixing_proportion = 0.98-(0.98-mixing_proportion)*0.98 
            else:
                # mix less
                mixing_proportion = (mixing_proportion-0.05)*0.95+0.05

    return mAFMs, mFMs, ns, fermi_energys


def total_energy(Hamiltonian:Hamiltonian2D,Lq=50):
    """total energy per unit cell (sqrt(2) x sqrt(2) x 1)"""
    ks = Hamiltonian.BZ.sample(Lq)
    es,_ = Hamiltonian.diagonalize(*ks)
    total_energy = np.sum(es[es<0])/(Lq**2)/Hamiltonian.param['len_z'] + Econst_DLKK_3D_MF(**Hamiltonian.param)

    return total_energy