import numpy as np
from blochK.observable import conductivity_orbital_resolved


def get_conductivity_layer_resolved(Hamiltonian, **conductivity_args):
    """Computes the layer resolved conductivity for a given DLKK Hamiltonian 
    Returns:
        layers, conductivity_total, conductivity_spin
    """
    len_z = Hamiltonian.param['len_z']
    # Calculate spin conductivity
    cond_tensor = conductivity_orbital_resolved(Hamiltonian,**conductivity_args) #.shape=(layer, spin, sublattice, x, y)
    
    #1) way of dealing with this tensor: smart reshaping and then contracting all unwanted indices
    n_layers = 2 #sum always n_layers layers up
    cond_tensor2 = np.reshape(cond_tensor,(len_z,2,2,2,2)) #.shape=(layer,spin,sublattice,x,y)
    cond_tensor2 = cond_tensor2.reshape(-1,n_layers,2,2,2,2) #.shape=(layer/n_layers,n_layers,sublattice,spin,x,y)
    cond = np.sum(cond_tensor2[:,:,:,:,:,:],axis=(1,3)) #.shape=(layer/n_layers,spin,i=(x,y),j=(x,y))
    spin_cond = cond[:,0]-cond[:,1] #difference of spin indices
    cond0 = cond[:,0]+cond[:,1] #total cond: sum of spin indices
    #indices is fixed by n_layers
    layers = np.arange(0,len_z+1-n_layers,n_layers)

    # #2) way of dealing with this tensor: construct full observable and then contract (slower but does't matter)
    # #we can compute pairs (0,1), (1,2), (2,3), ...
    # spin_cond00 = []; cond00 = []
    # layers = np.arange(0,param['len_z']-1,1) #specify 

    # for layer in layers:
    #     proj_2layers = projector2layer(layer,len_z=param['len_z'])+projector2layer(layer+1,len_z=param['len_z'])
    #     Spin_op_layer = np.kron(proj_2layers,Spin_operator) #projected spin_operator
    #     spin_cond00.append(np.sum(Spin_op_layer*cond_tensor[:,0,0]))
    #     cond00.append(np.sum(np.kron(proj_2layers,np.ones(4))*cond_tensor[:,0,0]))

    return layers, cond0, spin_cond