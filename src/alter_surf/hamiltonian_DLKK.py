# here all functions are defined such that the unit cell has NN hoppings in x+y, x-y directions.
# The lattice vectors are n1=(1,0) n2=(0,1), Atoms are at (0,0) and (1/2,1/2)
# BZ is standard square BZ

import numpy as np
from numpy import pi,cos,sin,exp
import matplotlib.pyplot as plt

import blochK

#operators
Spin_operator = np.array([1,1,-1,-1]) #spin up +1, spin down -1
Sublattice_operator = np.array([1,-1,1,-1]) #orbital x +1, orbital y -1
Sz_diag = np.array([1,-1])
Sz = np.diag(Sz_diag)
Sx = np.array([[0,1],[1,0]])
Sy = np.array([[0,-1j],[1j,0]])

#Definitions for the Square lattice
#lattice vectors
n1 = np.array([1,0])
n2 = np.array([0,1])

def create_H_DLKK_3D(param=dict()):

    assert 'len_z' in param, "You need to specify 'len_z' you dumbo."
    param['numb_z'] = (param['len_z']+1)//2
    

    H_DLKK_3D = blochK.Hamiltonian2D(H_DLKK_3Dslab_fct,
                        param=param,
                         n1=n1,n2=n2,
                         basis = ['layers','spin','sublattice'],
                         basis_states=['z=0,A,up', 'z=0,B,up', 'z=0,A,down', 'z=0,B,down','z=1,A,up', 'z=1,B,up', 'z=1,A,down', 'z=1,B,down','...'],)
    
    #the Hamiltonian has len_z dependent orbitals
    #define operators for each z layer
    H_DLKK_3D.add_suboperator('spin',Spin_operator)
    H_DLKK_3D.add_suboperator('sublattice',Sublattice_operator)
    H_DLKK_3D.add_suboperator('layer',np.array([1,1,1,1,-1,-1,-1,-1]))
    H_DLKK_3D.add_suboperator('proj',np.eye(H_DLKK_3D.n_orbitals))
    #define operators acting on entire system
    H_DLKK_3D.add_operator('spin',np.kron(np.ones(H_DLKK_3D.n_orbitals//4),Spin_operator))
    H_DLKK_3D.add_operator('sublattice',np.kron(np.ones(H_DLKK_3D.n_orbitals//4),Sublattice_operator))

    return H_DLKK_3D


def create_H_DLKK_true3D(param=dict()):
    H_DLKK_true3D = blochK.Hamiltonian3D(H_DLKK_3D_fct,
                        param=param,
                         n1=np.array([1,0,0]),n2=np.array([0,1,0]),n3=np.array([0,0,2]),
                         basis = ['layers','spin','sublattice'],
                         basis_states=['z=0,A,up', 'z=0,B,up', 'z=0,A,down', 'z=0,B,down','z=1,A,up', 'z=1,B,up', 'z=1,A,down', 'z=1,B,down'],)
    
    H_DLKK_true3D.add_operator('spin',np.kron(np.ones(H_DLKK_true3D.n_orbitals//4),Spin_operator))
    H_DLKK_true3D.add_operator('sublattice',np.kron(np.ones(H_DLKK_true3D.n_orbitals//4),Sublattice_operator))

    return H_DLKK_true3D


def create_H_DLKK_true3Das2D(param=dict()):
    H_DLKK_true3D = blochK.Hamiltonian2D(lambda kx,ky,kz=0,**kwargs: H_DLKK_3D_fct(kx,ky,kz,**kwargs),
                        param=param,
                         n1=np.array([1,0,0]),n2=np.array([0,1,0]),
                         basis = ['layers','spin','sublattice'],
                         basis_states=['z=0,A,up', 'z=0,B,up', 'z=0,A,down', 'z=0,B,down','z=1,A,up', 'z=1,B,up', 'z=1,A,down', 'z=1,B,down'],)
    
    H_DLKK_true3D.add_operator('spin',np.kron(np.ones(H_DLKK_true3D.n_orbitals//4),Spin_operator))
    H_DLKK_true3D.add_operator('sublattice',np.kron(np.ones(H_DLKK_true3D.n_orbitals//4),Sublattice_operator))

    return H_DLKK_true3D


def create_H_DLKK_3D_MF(param=dict()):

    H_DLKK_3D = blochK.Hamiltonian2D(H_DLKK_3D_MF_fct,
                         param=param,
                         n1=n1,n2=n2,
                         basis = ['layers','spin','sublattice'],
                         basis_states=['z=0,A,up', 'z=0,B,up', 'z=0,A,down', 'z=0,B,down','z=1,A,up', 'z=1,B,up', 'z=1,A,down', 'z=1,B,down','...'],)
    
    #the Hamiltonian has len_z dependent orbitals
    #define operators for each z layer
    H_DLKK_3D.add_suboperator('spin',Spin_operator)
    H_DLKK_3D.add_suboperator('sublattice',Sublattice_operator)
    #define operators acting on entire system
    H_DLKK_3D.add_operator('spin',np.kron(np.ones(H_DLKK_3D.n_orbitals//4),Spin_operator))
    H_DLKK_3D.add_operator('sublattice',np.kron(np.ones(H_DLKK_3D.n_orbitals//4),Sublattice_operator))


    return H_DLKK_3D


#################################################################################################################################################
#Hamiltonian functions
#################################################################################################################################################

def H_DLKK_2D_fct(kx,ky,t=1,tp=0.5,delta=0,mu=0,mAF=0,mF=0,jpx=0,jpy=0): 
    """
    t: NN hopping
    tp: NNN hopping
    delta: unisotropy of NNN hopping [1,1], [-1,1] direction
    mu: chemical potential
    mAF: AF magnetization +m on A, -m on B
    mF: F magnetization +m on A, +m on B
    jpx: LCO NNN in x direction
    jpy: LCO NNN in y direction
    """
    Hk = np.zeros((4,4,*kx.shape),dtype=complex) #Basis (A up, B up, A down, B down)

    # #set hamiltonian structure
    Hk[0,1] = - 2*t*cos(kx/2+ky/2) - 2*t*cos(kx/2-ky/2)
    Hk[0,0] = -mu  - 2*tp*cos(kx)*(1+delta) - 2*tp*cos(ky)*(1-delta) + jpx*np.sin(kx) + jpy*np.sin(ky)
    Hk[1,1] = -mu  - 2*tp*cos(kx)*(1-delta) - 2*tp*cos(ky)*(1+delta) - jpx*np.sin(kx) - jpy*np.sin(ky)

    # make hermitian
    Hk[1,0] = np.conjugate(Hk[0,1])

    # spin degenerate
    Hk[2:,2:] = Hk[:2,:2]

    # Add AFM, FM
    AFM_AB = Spin_operator*Sublattice_operator*mAF
    FM = Spin_operator*mF
    for j in range(Hk.shape[0]):
        Hk[j,j] += AFM_AB[j]
        Hk[j,j] += FM[j]

    return Hk


def H_DLKK_3D_fct(kx,ky,kz,t=1,tp=0.5,delta=0,tz=1,tzp=0,mu=0,mAF=0,mF=0,jpx=0,jpy=0,PBC=False): 
    """
    t: NN hopping
    tp: NNN hopping
    tz: NN hopping in [0,0,1] direction
    TODO: tzp: NNN hopping in [1,0,1],[-1,0,1],[0,1,1],[0,-1,1] direction
    delta: unisotropy of NNN hopping [1,1], [-1,1] direction
    mu: chemical potential
    mAF (float or np.ndarray): AF magnetization +m on A, -m on B. If float, same m for all layers. If .shape=(len_z,), m[j] is the magnetization in layer j
    mF (float or np.ndarray):   F magnetization +m on A, +m on B. If float, same m for all layers. If .shape=(len_z,), m[j] is the magnetization in layer j
    jpx: LCO NNN in x direction
    jpy: LCO NNN in y direction
    """
    Hk = np.zeros((8,8,*kx.shape),dtype=complex)
    #Basis (z=0(x up, y up, x down, y down), z=1(...))
    #sublattices A are fixed by the first layer
    #-> if site at (x,y,z) is sublattice A, then site at (x,y,z+1) is also sublattice A

    #shift energy such that the lower end of the spectrum stays the same
    delta_energy = tz
     #set 2D structure
    Hk[:4,:4] = H_DLKK_2D_fct(kx,ky,t=t,tp=tp,delta=delta,mu=mu-delta_energy,mAF= mAF,mF=mF,jpx=jpx,jpy=jpy)
    Hk[4:,4:] = H_DLKK_2D_fct(kx,ky,t=t,tp=tp,delta=delta,mu=mu-delta_energy,mAF=-mAF,mF=mF,jpx=jpx,jpy=jpy)

    #extend to 3D
    #z-hoppings
    Hk[4,0] = Hk[0,4] = -2*tz*cos(kz/2)
    Hk[5,1] = Hk[1,5] = -2*tz*cos(kz/2)
    Hk[6,2] = Hk[2,6] = -2*tz*cos(kz/2)
    Hk[7,3] = Hk[3,7] = -2*tz*cos(kz/2)

    return Hk


def H_DLKK_3Dslab_fct(kx,ky,len_z=2,t=1,tp=0.5,delta=0,tz=1,tzp=0,mu=0,mAF=0,mF=0,Q_z=np.pi,delta_Q_z=0,PBC=False, numb_z=None,jpx=0,jpy=0): 
    """
    len_z: number of layers in z-direction
    t: NN hopping
    tp: NNN hopping
    tz: NN hopping in [0,0,1] direction
    TODO: tzp: NNN hopping in [1,0,1],[-1,0,1],[0,1,1],[0,-1,1] direction
    delta: unisotropy of NNN hopping [1,1], [-1,1] direction
    delta_Q_z: wave vector of unisotropy -> typical values are 0 (stacked DLKK model), pi (alternating patterns of DLKK model)
    mu: chemical potential
    mAF (float or np.ndarray): AF magnetization +m on A, -m on B. If float, same m for all layers. If .shape=(len_z,), m[j] is the magnetization in layer j
    mF (float or np.ndarray):   F magnetization +m on A, +m on B. If float, same m for all layers. If .shape=(len_z,), m[j] is the magnetization in layer j
    jpx: LCO NNN in x direction
    jpy: LCO NNN in y direction
    numb_z: dummy variable for constructing observables, counts the number of unit cell (rounded to next integer)
    """
    Hk = np.zeros((4*len_z,4*len_z,*kx.shape),dtype=complex)
    #Basis (z=0(x up, y up, x down, y down), z=1(...), ..., z=len_z-1(..))
    #sublattices A are fixed by the first layer
    #-> if site at (x,y,z) is sublattice A, then site at (x,y,z+1) is also sublattice A

    #we are stacking len_z/2 unit cells, each containing 2 layers
    if len_z%2==0: #if len_z is even
        numb_z = len_z//2 #everything is fine. Each unit cell has 2 layers
    else: #if len_z is odd
        numb_z = (len_z+1)//2 #we construct a larger Hamiltonian and later project out the last layer

    #AFM can be layer dependent
    if isinstance(mAF,(int,float)): #same magnetization for all layers
        mAF = mAF*np.cos(Q_z*np.arange(len_z)) #magnetization changes sign depending Q_z and layer
    elif mAF.shape!=(len_z,):
        raise ValueError("mAF has to be a float or an array with shape (len_z,)")
    #FM can be layer dependent
    if isinstance(mF,(int,float)): #same magnetization for all layers
        mF = mF*np.cos(Q_z*np.arange(len_z)) #magnetization changes sign depending Q_z and layer
    elif mF.shape!=(len_z,):
        raise ValueError("mF has to be a float or an array with shape (len_z,)")
    
    #chemical potential can be layer dependent
    if isinstance(mu,(int,float)): #same chemical potential for all layers
        mu = mu*np.ones(len_z) 
    elif mu.shape!=(len_z,):
        raise ValueError("mu has to be a float or an array with shape (len_z,)")

    #shift energy such that the lower end of the spectrum stays the same
    delta_energy = tz
     #set 2D structure
    for j in range(len_z):
         Hk[4*j:4*j+4,4*j:4*j+4] = H_DLKK_2D_fct(kx,ky,t=t,tp=tp,delta=delta*np.cos(delta_Q_z*j),mu=mu[j]-delta_energy,mAF=mAF[j],mF=mF[j],jpx=jpx,jpy=jpy)

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
def H_DLKK_3D_MF_fct(kx,ky,len_z=2,t=1,tp=0.5,delta=0,tz=1,tzp=0,mu=0,mF=None,mAF=None,ns=None,delta_Q_z=0,PBC=False,U=0, filling=0.5):
    """
    Same as H_DLKK_3D, but with a mean field Hubbard term.
    Additional parameters:
    mAF: np.ndarray, shape=(len_z,), AFmagnetic moments in each layer
    mF:  np.ndarray, shape=(len_z,),  Fmagnetic moments in each layer
    ns: np.ndarray, shape=(len_z,), particle densities in each layer
    U: float, on-site interaction strength
    filling: float, filling fraction (0 to 1)
    """
    if mAF is None:
        mAF = np.zeros(len_z)
    if mF is None:
        mF = np.zeros(len_z)
    if ns is None:
        ns = np.ones(len_z)

    if mAF.shape!=(len_z,) or ns.shape!=(len_z,) or mAF.shape!=(len_z,):
        raise ValueError("mAF, mF, and ns have to be arrays with shape (len_z,)")

    #add MF contributions
    mu = mu - U/2 * (ns-1) # this 1 needs to be removed!!
    mAF = -U/2*mAF
    mF  = -U/2*mF

    H_MF = H_DLKK_3Dslab_fct(kx,ky,len_z=len_z,t=t,tp=tp,delta=delta,tz=tz,tzp=tzp,mu=mu,mAF=mAF,mF=mF,delta_Q_z=delta_Q_z,PBC=PBC)

    return H_MF


def Econst_DLKK_3D_MF(len_z=2,U=0,mAF=None,mF=None,ns=None,**other_arguments_Hparam):
    """
    Constant energy shift due to mean field Hubbard term, per unit cell, i.e. (sqrt(2),sqrt(2),1)
    This is not included in the Hamiltonian, but has to be added to the total energy.
    """

    if mAF.shape!=(len_z,) or mF.shape!=(len_z,) or ns.shape!=(len_z,):
        raise ValueError("mAF, mF, and ns have to be arrays with shape (len_z,)")

    mA = mF+mAF
    mB = mF-mAF

    # Calculate the constant energy shift for each sublattice (normally equal)
    EA = U/4 * (np.sum(mA**2) - np.sum(ns**2))/len_z
    EB = U/4 * (np.sum(mA**2) - np.sum(ns**2))/len_z
    E_const = EA+EB

    #different calulation
    E_const = U/4 * (np.sum(mA**2) - np.sum(ns**2))/len_z

    return E_const



###############################################################################################################################################
#specific functions for mean field calculations
###############################################################################################################################################

def find_m_and_n_values(es, psi, fermi_energy):
    """
    Given a set of eigenstates, calculates the density and magnetization at each site.
    Args:
        es (np.ndarray): The eigenenergies .shape = (bands, momenta, momenta)
        psi (np.ndarray): The eigenstates .shape = (bands, momenta, momenta, localH)
        fermi_energy (float): The Fermi energy realizing the desired filling
    Returns:
        np.ndarray: The spatially resolved mean field values
    """
    Lq = psi.shape[1]
    len_z = psi.shape[0]//4 #divide by 4 because of spin and sublattice

    fermi_occupation = es<fermi_energy #boolean array, true if state is occupied

    local_densities = np.einsum('nkqa,nkq->a',np.abs(psi)**2,fermi_occupation)/Lq**2 #sum up bands, momenta, momenta
    local_densities = local_densities.reshape(len_z,2,2) #.shape = (layer, spin, sublattice)


    #We evalute m and n from the local densities. No restrictions beside unit cell.
    m_values = np.einsum('zsa,s->za',local_densities,Sz_diag) #m.shape = (layer, sublattice)
    n_values = np.einsum('zsa->za',local_densities) #n.shape = (layer, sublattice)
   
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
