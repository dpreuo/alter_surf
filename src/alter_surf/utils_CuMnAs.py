import numpy as np

from alter_surf.utils3D import project_doublelayer




from blochK.observable import conductivity_list_of_operators

def get_conductivity_layer_resolved(Hamiltonian, **conductivity_args):
    """Computes the layer resolved conductivity for a given DLKK Hamiltonian 

    Returns:
        layers (.shape=(N)), conductivity_total (.shape=(N,2,2)), conductivity_spin (.shape=(N,2,2))
    -----------
    the index order matters here!
    spin_cond[:,1,0] is sigma_yx, that is current in y direction when E_x applied it is different from sigma_xy = spin_cond[:,0,1]

    -----------
    Example:
    How to use the results to get the spin splitting angle (in degrees)
    There are to angles, depending on the direction of the applied electric field:
    For E_x applied:
        2*np.arctan(spin_cond[:,1,0]/cond0[:,0,0])*180/pi
    For E_y applied:
        2*np.arctan(spin_cond[:,0,1]/cond0[:,1,1])*180/pi
    -----------
    """
    assert 'len_z' in Hamiltonian.param, "Hamiltonian must have 'len_z' parameter defined."
    assert hasattr(Hamiltonian, "operator") and getattr(Hamiltonian.operator, "spin", None) is not None, "Hamiltonian must have 'spin' operator defined."
    assert hasattr(Hamiltonian, "suboperator") and hasattr(Hamiltonian.suboperator, "proj"), "Hamiltonian must have 'proj' suboperator defined."

    # Constrcut the layers. 
    layers = np.arange(0,Hamiltonian.param['len_z']-1,1)

    #construct the list of projectors
    operators = []
    for layer in layers:
        operators.append([Hamiltonian.suboperator.proj.dot(project_doublelayer(layer,Hamiltonian)),
                        Hamiltonian.suboperator.proj.dot(project_doublelayer(layer,Hamiltonian))*Hamiltonian.operator.spin])
    operators = np.swapaxes(operators,0,1) #.shape=(2, n_layers,localH,localH)
    

    cond_tensor = conductivity_list_of_operators(Hamiltonian,list_of_operators=operators,optimize=['einsum_path', (0, 6), (1, 5), (1, 5), (1, 4), (1, 2), (1, 2), (0, 1)],**conductivity_args)
    cond = cond_tensor[0]
    spin_cond = cond_tensor[1]

    return layers, cond, spin_cond


