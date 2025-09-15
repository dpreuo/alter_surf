import numpy as np
from numpy import pi,cos,sin,exp
import matplotlib.pyplot as plt

#operators
Spin_operator = np.array([1,1,-1,-1]) #spin up +1, spin down -1
Orbital_operator = np.array([1,-1,1,-1]) #orbital x +1, orbital y -1


#this needs to be implemented
def H_DLKK_2D(kx,ky,): 
    """
    t1: NN hopping x orbital in x direction, y orbital in y direction
    t2: NN hopping x orbital in y direction, y orbital in x direction
    t3: NNN hopping x,y orbital in [1,1] direction, x,y orbital in [1,-1] direction
    mu: chemical potential
    m: AFM magnetization +m on x, -m on y
    """
    Hk = np.zeros((4,4,*kx.shape),dtype=complex) #Basis (A up, B up, A down, B down)

    # #set hamiltonian structure
    # Hk[0,0] = -mu - 2*t1*cos(kx) - 2*t2*cos(ky) - 2*t3*cos(kx+ky) - 2*t3*cos(kx-ky)
    # Hk[1,1] = -mu - 2*t2*cos(kx) - 2*t1*cos(ky) - 2*t3*cos(kx+ky) - 2*t3*cos(kx-ky)
    # Hk[0,1] = -2*t3*cos(kx+ky) - 2*t3*cos(kx-ky)

    # #make hermitian
    # Hk[1,0] = np.conjugate(Hk[0,1])

    # #spin degenerate
    # Hk[2:,2:] = Hk[:2,:2]

    # #Add AFM
    # AFM_AB = [+m, -m, -m, +m]

    # for j in range(Hk.shape[0]):
    #     Hk[j,j] += AFM_AB[j]

    return Hk



