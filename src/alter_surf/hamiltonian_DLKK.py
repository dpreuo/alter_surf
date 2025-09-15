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
    delta: unistropy of NNN hopping [1,1] direction, x,y orbital in [1,-1] direction
    mu: chemical potential
    m: AFM magnetization +m on A, -m on B
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



