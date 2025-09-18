import numpy as np
from numpy import pi,cos,sin,exp
import matplotlib.pyplot as plt

#operators
Spin_operator = np.array([1,1,-1,-1]) #spin up +1, spin down -1
Sublattice_operator = np.array([1,-1,1,-1]) #orbital x +1, orbital y -1


#Definitions for the Square lattice
#lattice vectors
n1 = np.array([1,0])
n2 = np.array([0,1])

# Area of unit cell (2D cross product)
A = n1[0]*n2[1] - n1[1]*n2[0]
#reciprocal lattice vectors
m1 = 2*np.pi/A * np.array([n2[1], -n2[0]])
m2 = 2*np.pi/A * np.array([-n1[1], n1[0]])

# Define High symmetry points in BZ
points_BZ = {
    "\Gamma": [0,0],
    "X": [pi,0],
    "Y": [0,pi],
    "R": [pi,pi],
    "R'": [pi,-pi],
    "-R": [-pi,pi],
    "-R'": [-pi,-pi]
}


def H_DLKK_2D(kx,ky,t=1,tp=0.5,delta=0,mu=0,m=0): 
    """
    t: NN hopping
    tp: NNN hopping
    delta: unisotropy of NNN hopping [1,1], [-1,1] direction
    mu: chemical potential
    m: AFM magnetization -m on A, +m on B
    """
    Hk = np.zeros((4,4,*kx.shape),dtype=complex) #Basis (A up, B up, A down, B down)

    # #set hamiltonian structure
    Hk[0,1] = - 2*t*cos(kx) - 2*t*cos(ky)
    Hk[0,0] = -mu  - 2*tp*cos(kx+ky)*(1+delta) - 2*tp*cos(kx-ky)*(1-delta)
    Hk[1,1] = -mu  - 2*tp*cos(kx+ky)*(1-delta) - 2*tp*cos(kx-ky)*(1+delta)

    # make hermitian
    Hk[1,0] = np.conjugate(Hk[0,1])

    # spin degenerate
    Hk[2:,2:] = Hk[:2,:2]

    # Add AFM
    AFM_AB = -Spin_operator*Sublattice_operator*m
    for j in range(Hk.shape[0]):
        Hk[j,j] += AFM_AB[j]

    return Hk


def H_DLKK_3D(kx,ky,len_z=2,t=1,tp=0.5,delta=0,tz=1,tzp=0,mu=0,m=0,Q_z=np.pi,delta_Q_z=0,PBC=False): 
    """
    len_z: number of layers in z-direction
    t: NN hopping
    tp: NNN hopping
    tz: NN hopping in [0,0,1] direction
    TODO: tzp: NNN hopping in [1,0,1],[-1,0,1],[0,1,1],[0,-1,1] direction
    delta: unisotropy of NNN hopping [1,1], [-1,1] direction
    delta_Q_z: wave vector of unisotropy -> typical values are 0 (stacked DLKK model), pi (alternating patterns of DLKK model)
    mu: chemical potential
    m (float or np.ndarray): AFM magnetization -m on A, +m on B. If float, same m for all layers. If .shape=(len_z,), m[j] is the magnetization in layer j
    """
    Hk = np.zeros((4*len_z,4*len_z,*kx.shape),dtype=complex)
    #Basis (z=0(x up, y up, x down, y down), z=1(...), ..., z=len_z-1(..))
    #sublattices A are fixed by the first layer
    #-> if site at (x,y,z) is sublattice A, then site at (x,y,z+1) is also sublattice A

    #magnetization can be layer dependent
    if isinstance(m,(int,float)): #same magnetization for all layers
        m = m*np.cos(Q_z*np.arange(len_z)) #magnetization changes sign depending Q_z and layer
    elif m.shape!=(len_z,):
        raise ValueError("m has to be a float or an array with shape (len_z,)")
    
    #chemical potential can be layer dependent
    if isinstance(mu,(int,float)): #same chemical potential for all layers
        mu = mu*np.ones(len_z) 
    elif mu.shape!=(len_z,):
        raise ValueError("mu has to be a float or an array with shape (len_z,)")

    #shift energy such that the lower end of the spectrum stays the same
    delta_energy = tz
     #set 2D structure
    for j in range(len_z):
         Hk[4*j:4*j+4,4*j:4*j+4] = H_DLKK_2D(kx,ky,t=t,tp=tp,delta=delta*np.cos(delta_Q_z*j),mu=mu[j]-delta_energy,m=m[j])

    #extend to 3D
    #z-hoppings
    Hz = np.zeros((4,4,*kx.shape),dtype=complex)
    # (z=1,x,up), (z=1,y,up), (z=1,x,down), (z=1,y,down)  -> (z=0,x,up), (z=0,y,up), (z=0,x,down), (z=0,y,down)

    Hz[0,0] = -tz
    Hz[1,1] = -tz
    #spin degenerate
    Hz[2:,2:] = Hz[:2,:2]

    #add to final Hamiltonian
    for j in range(len_z-1):
        Hk[4*j:4*j+4,4*j+4:4*j+8] = Hz #upper block
        Hk[4*j+4:4*j+8,4*j:4*j+4] = Hz #lower block

    if PBC: # Periodic boundary conditions
        Hk[4*(len_z-1):4*len_z,0:4] = Hz
        Hk[0:4,4*(len_z-1):4*len_z] = Hz

    return Hk


###############################################################################################################################################
#interacting mean field Hamiltonians
###############################################################################################################################################
def H_DLKK_3D_MF(kx,ky,len_z=2,t=1,tp=0.5,delta=0,tz=1,tzp=0,mu=0,m_values=None,n_values=None,delta_Q_z=0,PBC=False,U=0, filling=0.5):
    """
    Same as H_DLKK_3D, but with a mean field Hubbard term.
    Additional parameters:
    m_values: np.ndarray, shape=(len_z,), magnetic moments in each layer
    n_values: np.ndarray, shape=(len_z,), particle densities in each layer
    U: float, on-site interaction strength
    filling: float, filling fraction (0 to 1)
    """
    if m_values is None:
        m_values = np.zeros(len_z)
    if n_values is None:
        n_values = np.ones(len_z)

    if m_values.shape!=(len_z,) or n_values.shape!=(len_z,):
        raise ValueError("m_values and n_values have to be arrays with shape (len_z,)")

    #add MF contributions
    mu = mu - U*n_values
    m = U*m_values

    H_MF = H_DLKK_3D(kx,ky,len_z=len_z,t=t,tp=tp,delta=delta,tz=tz,tzp=tzp,mu=mu,m=m,delta_Q_z=delta_Q_z,PBC=PBC)

    return H_MF


def H_DLKK_2D_MF(kx,ky,t=1,tp=0.5,delta=0,mu=0,m_values=None,n_values=None,U=0, filling=0.5):
    """
    Same as H_DLKK_3D, but with a mean field Hubbard term.
    Additional parameters:
    m_values: np.ndarray, shape=(len_z,), magnetic moments in each layer
    n_values: np.ndarray, shape=(len_z,), particle densities in each layer
    U: float, on-site interaction strength
    filling: float, filling fraction (0 to 1)
    """

    #add MF contributions
    mu = mu - U*n_values
    m = U*m_values

    H_MF = H_DLKK_2D(kx,ky,t=t,tp=tp,delta=delta,mu=mu,m=m)

    return H_MF



###############################################################################################################################################
#specific functions for mean field calculations
###############################################################################################################################################

def find_m_and_n_values(es, psi, fermi_energy):
    """
    Given a set of eigenstates, calculates the mean field values for the magnetic moments and densities
    Args:
        psi (np.ndarray): The eigenstates .shape = (bands, momenta, momenta, localH)
        filling (float): filling fraction (0 to 1)
    Returns:
        np.ndarray: The spatially resolved mean field values
    """
    Lq = psi.shape[1]
    len_z = psi.shape[0]//4 

    fermi_occupation = es<fermi_energy #boolean array, true if state is occupied

    local_densities = np.einsum('nkqa,nkq->a',np.abs(psi)**2,fermi_occupation)/Lq**2 #sum up bands, momenta, momenta
    
    m_legend = np.kron(np.ones(len_z),Spin_operator*Sublattice_operator)


    #We enforce A up = B down = - A up = - B down
    m_values = local_densities * m_legend
    m_values = m_values.reshape(-1, 4).sum(axis=-1)/4 #this is m(z)

    n_values = local_densities.reshape(-1, 4).sum(axis=-1)/4 #this is n(z)

    # check for complex values - this should not happen
    if not np.allclose(m_values.imag, 0):
        raise ValueError(
            "Magnetization values are complex - somewhere, somehow you fucked it up"
        )
    elif not np.allclose(n_values.imag, 0):
        raise ValueError(
            "Density values are complex - somewhere, somehow you fucked it up"
        )

    return m_values.real, n_values.real
