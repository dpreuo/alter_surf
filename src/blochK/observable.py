import numpy as np
from numpy import pi,cos,sin,exp



#solve a hamiltonian
def eigs_H(kx,ky,Hamiltonian_fct,Hparam): 
    """
    Return energies 'es' and wavefunctions 'psis' in the shape:
    es.shape = band x kys (x kxs)
    psi.shape = band x kys (x kxs) x localH
    """
    Hk = Hamiltonian_fct(kx,ky,**Hparam)
    
    Hk = np.moveaxis(np.moveaxis(Hk,1,-1),0,-2) #make it ...x band x band dimensional
    es,vs = np.linalg.eigh(Hk)
    es = np.moveaxis(es,-1,0) #.shape=band x kys (x kxs)
    psi = np.moveaxis(vs,-1,0) #.shape=band x kys (x kxs) x localH
    
    return es, psi


#expectation value of operators
def exp_value_O(O,psi):
    """evalute the expectation value of an Operator O for a set of states psi (band x kys (x kxs) x localH)
    Input:
    O: ndarray, shape=(localH) or (localH x localH)
    """
    if len(O.shape)==1: # O.shape = (localH)
        return exp_value_Odiag(O,psi) #call the faster diagnoal version
    
    # O.shape = (localH1 x localH2)
    if len(psi.shape) == 3:
        res = np.einsum('akb,bc,dkc->adk',np.conjugate(psi),O,psi)
    elif len(psi.shape) == 4:
        res = np.einsum('akqb,bc,dkqc->adkq',np.conjugate(psi),O,psi)
    
    return res #.shape=band1 x band2 x kys (x kys) or band1 x kys (x kys)


def exp_value_Odiag(O,psi):
    """evalute the expectation value of an Operator O (.shape=(n)) for a set of states psi (band x kys (x kxs) x localH)"""
    return np.sum((np.abs(psi)**2*O),axis=-1)

#############################################################################################################################################
#degeneracy of eigenstates

def isDegenerate_1D(es):
    """Checks if an energie in es (.shape=band)"""
    es = np.round(es,10) #to evade numerical errors
    
    single_es, indices, counts = np.unique(es,return_counts=True,return_inverse=True)
    counts = counts>1
    
    condlist = [indices==i for i in range(0,len(counts))]
    xs = np.select(condlist,counts)
    return xs


isDegenerate_vectorized = np.vectorize(isDegenerate_1D,doc='vectorized version of isDegenerate_1D is forwarded to isDegenerate',signature='(n)->(n)') #could not find another way yet


def isDegenerate(es):
    """Checks if an energie in es (.shape=band x ...) is degenerate, i.e. appears more than once in the spectrum"""
    es = np.moveaxis(es,0,-1) #move band to last axis
    res = isDegenerate_vectorized(es) #axis needs to be last otherwise doesn't work
    return np.moveaxis(res,-1,0)


def isDegenerateIn_1D(es, observable_values, threshold=10):
    """Checks if an energie in es (.shape=band)"""
    es = np.round(es,threshold) #to evade numerical errors
    
    single_es, indices, counts = np.unique(es,return_counts=True,return_inverse=True)
    
    selected_sum = np.ones_like(es)
    for i in range(0,len(counts)): #sum over the observable values*index. degenerate states under observable values are degenerate
        selected_sum = np.where(indices==i,np.sum(observable_values*(indices+1),where=indices==i),selected_sum)
        
    return np.abs(selected_sum)<1e-10


isDegenerateIn_vectorized = np.vectorize(isDegenerateIn_1D,doc='vectorized version of isDegenerateIn_1D is forwarded to isDegenerateIn',signature='(n),(n)->(n)',excluded=['threshold']) #could not find another way yet


def isDegenerateIn(es,observable_values,threshold=10):
    """Check if an energie in es (.shape=band x ...) is degenerate in observable_values (.shape=band x ...) which is determined from exp_value_O(xxx_operator)"""
    es = np.moveaxis(es,0,-1) #move band to last axis
    observable_values = np.moveaxis(observable_values,0,-1) #move band to last axis
    res = isDegenerateIn_vectorized(es,observable_values,threshold=threshold) #axis needs to be last otherwise doesn't work
    return np.moveaxis(res,-1,0)

