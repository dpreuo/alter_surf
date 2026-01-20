# The minimal model for CuMnAs. 
# topological surface state
# Hamiltonian function defined 

import numpy as np
from numpy import pi,cos,sin,exp
import matplotlib.pyplot as plt

import blochK
from blochK.utils.hamiltonian_fct import operator_expand_dims,s0,sx,sy,sz


def H_3D_fct(kx,ky,kz,t=0,tp=0,tz=1,t2=0,dt2=0,dtpp=0,Delta0=0,Delta1=0,Delta2=0,Delta3=0,lambda1=0,lambda2=0,lambda3=0,V_layer_z=0,mu=0): 
    """
    translation invariant model of CuMnAs in 3D
    ----------
    Parameters: 
    t: NN hopping (1/2,1/2,0) in (x,y) plane
    tp: NNN hopping (1,0,0) in (x,y) plane
    dtpp: NNN hopping (1,1,0) in anisotropy
    tz: NN hopping (0,0,1/2) in z-direction
    t2: NNN hopping (1/2,1/2,1/2) 
    dt2: NNN hopping (1/2,1/2,1/2) anisotropy
    Delta0: Spin-dependent NN hopping in (x,y) plane (magnetism)
    Delta1: Spin-dependent NN hopping in z-direction (magnetism)
    Delta2: Spin-dependent NNN hopping in (x,y,z)->(x+1/2,y+1/2,z+1); p-wave formfactor (magnetism)
    Delta3: Spin-dependent NNN hopping in (x,y,z)->(x+1/2,y+1/2,z+1); f-wave formfactor (magnetism)
    lambda1: SOC strength
    lambda2: SOC strength
    lambda3: SOC strength
    V_layer_z: local potential alternating in each layer 
    mu: Fermi energy
    kz: momentum in z direction
    """
    n_orbitals2D = 8
    Hk = np.zeros((n_orbitals2D,n_orbitals2D,*kx.shape),dtype=complex) # s, tau, sigma

    [s0_,sx_,sy_,sz_] = operator_expand_dims([s0,sx,sy,sz],kx)

    ck_p = cos((kx+ky)/2) + cos((kx-ky)/2)
    sk_p = sin((kx+ky)/2) + sin((kx-ky)/2)
    ck_m = cos((kx+ky)/2) - cos((kx-ky)/2)
    sk_m = sin((kx+ky)/2) - sin((kx-ky)/2)

    #define hamiltonian for each spin sector
    def Hk_spin(spin:int):
        Hk = np.zeros((n_orbitals2D//2,n_orbitals2D//2,*kx.shape),dtype=complex) #tau, sigma

        Hk[0,0] = -mu - 2*tp*(cos(kx)+cos(ky)) + V_layer_z + 2*dtpp*(cos(kx+ky) - cos(kx-ky))
        Hk[1,1] = -mu - 2*tp*(cos(kx)+cos(ky)) + V_layer_z - 2*dtpp*(cos(kx+ky) - cos(kx-ky))
        Hk[2,2] = -mu - 2*tp*(cos(kx)+cos(ky)) - V_layer_z - 2*dtpp*(cos(kx+ky) - cos(kx-ky))
        Hk[3,3] = -mu - 2*tp*(cos(kx)+cos(ky)) - V_layer_z + 2*dtpp*(cos(kx+ky) - cos(kx-ky))

        #NN hoppings
        Hk[0,1] += 2*t*ck_p - 1j*spin*2*Delta0*sk_m
        Hk[2,3] += 2*t*ck_p - 1j*spin*2*Delta0*sk_m
        Hk[0,2] += 2*tz*cos(kz/2)
        Hk[1,3] += 2*tz*cos(kz/2)

        #NNN hoppings
        Hk[0,2] += -2j*Delta1*spin*sin(kz/2)
        Hk[1,3] += +2j*Delta1*spin*sin(kz/2)
        Hk[0,3] += +2j*Delta2*sk_m*spin*cos(kz/2) + 2j*Delta3*ck_m*spin*sin(kz/2) + 2*t2*ck_p*cos(kz/2) - 2*dt2*sk_p*sin(kz/2)
        Hk[1,2] += -2j*Delta2*sk_m*spin*cos(kz/2) + 2j*Delta3*ck_m*spin*sin(kz/2) + 2*t2*ck_p*cos(kz/2) + 2*dt2*sk_p*sin(kz/2)

        #SOC coupling
        Hk[0,1] += - 1j*spin*lambda1*ck_p
        Hk[2,3] += - 1j*spin*lambda1*ck_p

        return Hk
    
    #coupling between spin sectors due to SOC
    Hk_SOC = np.zeros((n_orbitals2D//2,n_orbitals2D//2,*kx.shape),dtype=complex) #tau, sigma
    Hk_SOC[0,1] += +lambda2*ck_p
    Hk_SOC[1,0] += -lambda2*ck_p
    Hk_SOC[2,3] += -lambda2*ck_p
    Hk_SOC[3,2] += +lambda2*ck_p
    Hk_SOC[0,2] += -1j*lambda3*2*cos(kz/2)
    Hk_SOC[2,0] += +1j*lambda3*2*cos(kz/2)
    Hk_SOC[1,3] += -1j*lambda3*2*cos(kz/2)
    Hk_SOC[3,1] += +1j*lambda3*2*cos(kz/2)

    #set hamiltonian structure
    Hk[:4,:4] = Hk_spin(spin=1)
    Hk[4:,4:] = Hk_spin(spin=-1)
    Hk[:4,4:] = Hk_SOC
    
    # make hermitian
    Hk = blochK.hamiltonian_fct.make_hermitian(Hk)

    return Hk


def H_slab_fct(kx,ky,len_z=2,t=0,tp=0,tz=1,t2=0,dt2=0,dtpp=0,Delta0=0,Delta1=0,Delta2=0,Delta3=0,lambda1=0,lambda2=0,lambda3=0,V_layer_z=0,mu=0,PBC=False,numb_z=0): 
    """
    slab model of CuMnAs in slab geometry with len_z layers in z-direction
    ----------
    Parameters:
    len_z: number of layers in z-direction (HERE a layer = 1 atomic layers = 1/2 unit cell)
    t: NN hopping (1/2,1/2,0) in (x,y) plane
    tp: NNN hopping (1,0,0) in (x,y) plane
    dtpp: NNN hopping (1,1,0) in anisotropy
    tz: NN hopping (0,0,1/2) in z-direction
    t2: NNN hopping (1/2,1/2,1/2) 
    dt2: NNN hopping (1/2,1/2,1/2) anisotropy
    Delta0: Spin-dependent NN hopping in (x,y) plane (magnetism)
    Delta1: Spin-dependent NN hopping in z-direction (magnetism)
    Delta2: Spin-dependent NNN hopping in (x,y,z)->(x+1/2,y+1/2,z+1); p-wave formfactor (magnetism)
    Delta3: Spin-dependent NNN hopping in (x,y,z)->(x+1/2,y+1/2,z+1); f-wave formfactor (magnetism)
    lambda1: SOC strength
    lambda2: SOC strength
    lambda3: SOC strength
    V_layer_z: local potential alternating in each layer 
    mu: Fermi energy
    PBC: periodic boundary conditions in z-direction, only works for even len_z
    numb_z: dummy variable for constructing observables, counts the number of unit cell (rounded to next integer)
    """
    n_orbitals2D = 8 #number of orbitals in unit cell: spin(2) x layer(2) x sublattice(2)

    #we are stacking len_z/2 unit cells, each containing 2 layers
    if len_z%2==0: #if len_z is even
        numb_z = len_z//2 #everything is fine. Each unit cell has 2 layers
    else: #if len_z is odd
        numb_z = (len_z+1)//2 #we construct a larger Hamiltonian and later project out the last layer

    Hk = np.zeros((n_orbitals2D*numb_z,n_orbitals2D*numb_z,*kx.shape),dtype=complex)

    ck_p = cos((kx+ky)/2) + cos((kx-ky)/2)
    sk_p = sin((kx+ky)/2) + sin((kx-ky)/2)
    ck_m = cos((kx+ky)/2) - cos((kx-ky)/2)
    sk_m = sin((kx+ky)/2) - sin((kx-ky)/2)

    #define hamiltonian for each spin sector
    def H2D_spin(spin:int):
        """Intralayer hoppings"""
        Hk = np.zeros((4,4,*kx.shape),dtype=complex) #tau, sigma
        Hk[0,0] = -mu - 2*tp*(cos(kx)+cos(ky)) + V_layer_z + 2*dtpp*(cos(kx+ky) - cos(kx-ky))
        Hk[1,1] = -mu - 2*tp*(cos(kx)+cos(ky)) + V_layer_z - 2*dtpp*(cos(kx+ky) - cos(kx-ky))
        Hk[2,2] = -mu - 2*tp*(cos(kx)+cos(ky)) - V_layer_z - 2*dtpp*(cos(kx+ky) - cos(kx-ky))
        Hk[3,3] = -mu - 2*tp*(cos(kx)+cos(ky)) - V_layer_z + 2*dtpp*(cos(kx+ky) - cos(kx-ky))
        
        #NN hoppings
        Hk[0,1] += 2*t*ck_p - 1j*spin*2*Delta0*sk_m
        Hk[2,3] += 2*t*ck_p - 1j*spin*2*Delta0*sk_m
        Hk[0,2] += tz
        Hk[1,3] += tz

        #NNN hoppings
        Hk[0,2] += -Delta1*spin
        Hk[1,3] += +Delta1*spin
        Hk[0,3] +=  1j*Delta2*sk_m*spin + Delta3*ck_m*spin + t2*ck_p + 1j*dt2*sk_p
        Hk[1,2] += -1j*Delta2*sk_m*spin + Delta3*ck_m*spin + t2*ck_p - 1j*dt2*sk_p

        #SOC coupling
        Hk[0,1] += - 1j*spin*lambda1*ck_p
        Hk[2,3] += - 1j*spin*lambda1*ck_p

        return Hk
    
    def Hz_spin(spin:int):
        """Interlayer hoppings. Hk[i,j] = Hz[i,j-4]"""
        Hk = np.zeros((4,4,*kx.shape),dtype=complex)

        #NN hoppings
        Hk[3,1] += tz 
        Hk[2,0] += tz 

        #NNN hoppings
        Hk[2,0] += +Delta1*spin
        Hk[3,1] += -Delta1*spin
        Hk[2,1] +=  1j*Delta2*sk_m*spin - Delta3*ck_m*spin + t2*ck_p + 1j*dt2*sk_p 
        Hk[3,0] += -1j*Delta2*sk_m*spin - Delta3*ck_m*spin + t2*ck_p - 1j*dt2*sk_p

        return Hk
    
    def Hk_SOC_layer():
        """Intralayer SOC hoppings"""
        Hk = np.zeros((4,4,*kx.shape),dtype=complex) #tau, sigma
        Hk[0,1] += +lambda2*ck_p
        Hk[1,0] += -lambda2*ck_p
        Hk[2,3] += -lambda2*ck_p
        Hk[3,2] += +lambda2*ck_p

        Hk[0,2] += -1j*lambda3
        Hk[2,0] += +1j*lambda3
        Hk[1,3] += -1j*lambda3
        Hk[3,1] += +1j*lambda3
        return Hk
    
    def Hk_SOC_interlayer():
        """interlayer SOC hoppings"""
        Hk = np.zeros((4,4,*kx.shape),dtype=complex) #tau, sigma

        Hk[2,0] += +1j*lambda3
        Hk[3,1] += +1j*lambda3
        return Hk

    #hoppings in x,y plane
    Hk_2D = np.zeros((n_orbitals2D,n_orbitals2D,*kx.shape),dtype=complex)
    Hk_2D[:4,:4] = H2D_spin(spin=1)
    Hk_2D[4:,4:] = H2D_spin(spin=-1)
    Hk_2D[:4,4:] = Hk_SOC_layer()
    Hk_2D = blochK.hamiltonian_fct.make_hermitian(Hk_2D)
    
    #fill diagonal blocks
    for j in range(numb_z):
         Hk[n_orbitals2D*j:n_orbitals2D*j+n_orbitals2D, n_orbitals2D*j:n_orbitals2D*j+n_orbitals2D] = Hk_2D

    #hoppings in z-direction, 3D
    Hz = np.zeros((n_orbitals2D,n_orbitals2D,*kx.shape),dtype=complex)
    Hz[:4,:4] = Hz_spin(spin=1)
    Hz[4:,4:] = Hz_spin(spin=-1)
    Hz[:4,4:] = Hk_SOC_interlayer()
    Hz[4:,:4] = Hk_SOC_interlayer()
    #add NN hoppings in z-direction blocks
    for j in range(numb_z-1):
        Hk[n_orbitals2D*j:n_orbitals2D*(j+1), n_orbitals2D*(j+1):n_orbitals2D*(j+2)] = Hz #upper block

    if PBC: # Periodic boundary conditions
        Hk[0:n_orbitals2D,n_orbitals2D*(numb_z-1):n_orbitals2D*numb_z] = np.swapaxes(np.conjugate(Hz),0,1)

    # add lower block by hermiticity
    Hk = blochK.hamiltonian_fct.make_hermitian(Hk)

    #if len_z is odd, project out last layer
    if len_z%2==1:
        #select all orbitals except the ones of the last layer
        selected = np.arange(n_orbitals2D*numb_z)
        #selected = np.delete(selected,[-12,-11,-10,-9,-4,-3,-2,-1])
        selected = np.delete(selected,[-6,-5,-2,-1])
        Hk = Hk[*np.ix_(selected,selected),...] #only keep selected rows and columns
    
    return Hk
