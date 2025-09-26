import numpy as np
from numpy import pi,cos,sin,exp
import matplotlib.pyplot as plt
import blochK

#operators
Spin_operator = np.array([1,1,-1,-1]) #spin up +1, spin down -1
Orbital_operator = np.array([1,-1,1,-1]) #orbital x +1, orbital y -1

def create_H_2orb_3D(param=dict()):

    H_2orb_3D = blochK.Hamiltonian2D(H_2orb_3D_fct,
                        param=param,
                         n1=n1,n2=n2,
                         basis = ['layers','spin','sublattice'],
                         basis_states=['z=0,A,up', 'z=0,B,up', 'z=0,A,down', 'z=0,B,down','z=1,A,up', 'z=1,B,up', 'z=1,A,down', 'z=1,B,down','...'],)
    
    #the Hamiltonian has len_z dependent orbitals
    #define operators for each z layer
    H_2orb_3D.add_suboperator('spin',Spin_operator)
    H_2orb_3D.add_suboperator('sublattice',Orbital_operator)
    #define operators acting on entire system
    H_2orb_3D.add_operator('spin',np.kron(np.ones(H_2orb_3D.n_orbitals//4),Spin_operator))
    H_2orb_3D.add_operator('sublattice',np.kron(np.ones(H_2orb_3D.n_orbitals//4),Orbital_operator))

    return H_2orb_3D


def H_2orb_2D_fct(kx,ky,t1=1,t2=0,t3=0,mu=0,m=0): 
    """
    t1: NN hopping x orbital in x direction, y orbital in y direction
    t2: NN hopping x orbital in y direction, y orbital in x direction
    t3: NNN hopping x,y orbital in [1,1] direction, x,y orbital in [1,-1] direction
    mu: chemical potential
    m: AFM magnetization +m on x, -m on y
    """
    Hk = np.zeros((4,4,*kx.shape),dtype=complex) #Basis (x up, y up, x down, y down)

    #set hamiltonian structure
    Hk[0,0] = -mu - 2*t1*cos(kx) - 2*t2*cos(ky) - 2*t3*cos(kx+ky) - 2*t3*cos(kx-ky)
    Hk[1,1] = -mu - 2*t2*cos(kx) - 2*t1*cos(ky) - 2*t3*cos(kx+ky) - 2*t3*cos(kx-ky)
    Hk[0,1] = -2*t3*cos(kx+ky) - 2*t3*cos(kx-ky)

    #make hermitian
    Hk[1,0] = np.conjugate(Hk[0,1])

    #spin degenerate
    Hk[2:,2:] = Hk[:2,:2]

    #Add AFM
    AFM_AB = [+m, -m, -m, +m]

    for j in range(Hk.shape[0]):
        Hk[j,j] += AFM_AB[j]

    return Hk



def H_2orb_3D_fct(kx,ky,len_z=2,t1=1,t2=0,t3=0,tz=1,tzp=0,mu=0,m=0,Q_z=np.pi,PBC=False): 
    """
    len_z: number of layers in z-direction
    t1: NN hopping x orbital in x direction, y orbital in y direction
    t2: NN hopping x orbital in y direction, y orbital in x direction
    t3: NNN hopping x->x, y->y, x->y, y->x orbital in [1,1] and [1,-1] direction
    tz: NN hopping x,y orbital in [0,0,1] direction
    tzp: NNN hopping x->y, y->x orbital in [0,0,1] direction
    mu: chemical potential
    m: AFM magnetization +m
    PBC: periodic boundary conditions in z-direction
    Q_z: ordering vector in z-direction 
    """
    Hk = np.zeros((4*len_z,4*len_z,*kx.shape),dtype=complex)
    #Basis (z=0(x up, y up, x down, y down), z=1(...), ..., z=len_z-1(..))

    #set 2D structure
    #magnetization changes sign depending Q_z and layer
    #shift energy such that the lower end of the spectrum stays the same
    delta_energy = tz
    for j in range(len_z):
         Hk[4*j:4*j+4,4*j:4*j+4] = H_2orb_2D_fct(kx,ky,t1=t1,t2=t2,t3=t3,mu=mu-delta_energy,m=m*np.cos(Q_z*j))

    #extend to 3D
    #z-hoppings
    Hz = np.zeros((4,4,*kx.shape),dtype=complex)
    # (z=1,x,up), (z=1,y,up), (z=1,x,down), (z=1,y,down)  -> (z=0,x,up), (z=0,y,up), (z=0,x,down), (z=0,y,down)

    Hz[0,0] = -tz
    Hz[1,1] = -tz
    Hz[0,1] = Hz[1,0] = -tzp

    #spin degenerate
    Hz[2:,2:] = Hz[:2,:2]

    for j in range(len_z-1):
        Hk[4*j:4*j+4,4*j+4:4*j+8] = Hz #upper block
        Hk[4*j+4:4*j+8,4*j:4*j+4] = Hz #lower block


    if PBC: # Periodic boundary conditions
        Hk[4*(len_z-1):4*len_z,0:4] = Hz
        Hk[0:4,4*(len_z-1):4*len_z] = Hz

    return Hk
