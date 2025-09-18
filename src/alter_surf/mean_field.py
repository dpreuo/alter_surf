from alter_surf.hamiltonian_DLKK import H_DLKK_3D_MF, Spin_operator, Sublattice_operator, find_m_and_n_values
import blochK.observable as observable
import blochK.methods_basic as methods_basic
import numpy as np
from tqdm import tqdm
import scipy.optimize as optimize


def single_hartree_fock_step(
    Hparam: dict,
    uniform_m = False,
    Lq = 100,
):  
    
    # make and solve the Hamiltonian
    ks = methods_basic.sample_reducedBZ(Lq)
    es, psi = observable.eigs_H(*ks,H_DLKK_3D_MF,Hparam)

    # find the fermi energy for the given filling
    normalization_factor = np.prod(es.shape)
    res_root = optimize.root_scalar(lambda mu: np.sum(es<mu)-normalization_factor*Hparam['filling'], bracket=[es.min(), es.max()], method='bisect')
    if not res_root.converged:
        raise ValueError(
            "Root finding for the fermi energy did not converge"
        )
    fermi_energy = res_root.root

    # calculate the new magnetization values
    m_values, n_values = find_m_and_n_values(es, psi, fermi_energy)

    # also have the option to average the magnetization values
    if uniform_m:
        m_values = np.mean(m_values) * np.ones_like(m_values)

    return m_values, n_values, fermi_energy


def hartree_fock(
    Hparam: dict,
    initial_parameters: dict,
    n_steps: int,
    Lq=100, #linear number of k-points in the BZ
    mixing_proportion=0.2,
    verbose=True,
    tol_mdiff=1e-6,
    leave=True,
    adjust_learning_rate=False,
):

    prange = tqdm(range(n_steps), leave=leave) if verbose else range(n_steps)
    skip_counter = 0

    m_values = np.zeros((n_steps + 1, Hparam['len_z']))
    m_values[0] = initial_parameters["initial_m"]

    n_values = np.zeros((n_steps + 1, Hparam['len_z']))
    n_values[0] = initial_parameters["initial_n"]

    fermi_energys = np.zeros((n_steps + 1))
    fermi_energys[0] = 0

    for n in prange:

        Hparam['m_values'] = m_values[n]
        Hparam['n_values'] = n_values[n]

        new_m, new_n, fermi_energy = single_hartree_fock_step(Hparam,Lq=Lq)

        m_values[n + 1] = (1 - mixing_proportion) * new_m + mixing_proportion * m_values[n]
        n_values[n + 1] = (1 - mixing_proportion) * new_n + mixing_proportion * n_values[n]
        fermi_energys[n + 1] = fermi_energy

        # check for convergence
        diff = np.linalg.norm(m_values[n + 1] - m_values[n])/(np.sqrt(len(m_values[n + 1])))
        avg_m = np.mean(m_values[n + 1])
        avg_m_stag = np.mean(m_values[n + 1]*(-1)**np.arange(Hparam['len_z']))
        
        u = 6
        learning_condition = adjust_learning_rate and n > u
        
        if verbose and not learning_condition:
            prange.set_description(f"Avg:{avg_m:.2f}, Stag. avg:{avg_m_stag:.2f}, diff: {diff:.6f}")

        # if the difference is small enough, we can stop
        if diff < tol_mdiff:
            skip_counter += 1
            if skip_counter >= 3:
                m_values = m_values[: n + 1]
                n_values = n_values[: n + 1]
                fermi_energys = fermi_energys[: n + 1]
                break


        if learning_condition:

            last_m_vals = m_values[n-u:n]
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

    return m_values, n_values, fermi_energys