import numpy as np
from blochK.observable import conductivity_list_of_operators
from alter_surf.utils3D import project_doublelayer


def get_conductivity_layer_resolved(Hamiltonian, **conductivity_args):
    """Computes the layer resolved conductivity for a given DLKK Hamiltonian 
    Returns:
        layers, conductivity_total, conductivity_spin
    -----------
    
    -----------
    Example:
    How to use the results to get the spin splitting angle (in degrees)
        2*np.arctan(spin_cond[:,0,0]/cond0[:,0,0])*180/pi
    """
    #indices is fixed by n_layers
    len_z = Hamiltonian.param['len_z']
    n_layers = 2 #sum always n_layers layers up
    layers = np.arange(0,len_z+1-n_layers,n_layers)

    #construct the list of projectors
    operators = []
    for layer in layers:
        operators.append([project_doublelayer(layer,Hamiltonian),
                          project_doublelayer(layer,Hamiltonian)*Hamiltonian.operator.spin])
    operators = np.swapaxes(operators,0,1) #.shape=(2, n_layers,localH,localH)
    
    #compute the conductivity
    optimize = [['einsum_path', (0, 7), (2, 6), (2, 6), (1, 5), (2, 3), (1, 3), (1, 2), (0, 1)], ['einsum_path', (0, 7), (2, 6), (2, 6), (2, 5), (2, 3), (1, 3), (1, 2), (0, 1)]]
    cond_tensor = conductivity_list_of_operators(Hamiltonian,list_of_operators=operators,optimize=optimize,**conductivity_args)
    
    #get spin and total conductivity
    cond = cond_tensor[0]
    spin_cond = cond_tensor[1]

    return layers, cond, spin_cond


