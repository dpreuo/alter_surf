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
    AFM_AB = [-m, +m, +m, -m]
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
    m: AFM magnetization -m on A, +m on B
    """
    Hk = np.zeros((4*len_z,4*len_z,*kx.shape),dtype=complex)
    #Basis (z=0(x up, y up, x down, y down), z=1(...), ..., z=len_z-1(..))
    #sublattices A are fixed by the first layer
    #-> if site at (x,y,z) is sublattice A, then site at (x,y,z+1) is also sublattice A

    #set 2D structure
    #magnetization changes sign depending Q_z and layer
    #shift energy such that the lower end of the spectrum stays the same
    delta_energy = tz
    for j in range(len_z):
         Hk[4*j:4*j+4,4*j:4*j+4] = H_DLKK_2D(kx,ky,t=t,tp=tp,delta=delta*np.cos(delta_Q_z*j),mu=mu-delta_energy,m=m*np.cos(Q_z*j))

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




