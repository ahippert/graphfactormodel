import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

from scipy.linalg.lapack import dtrtri


from functools import partial



### general elliptical

def elliptical_cost(R, x, logh):
    # real case, if complex then multiply det of L by 2
    # Careful: logh needs to also be the real version
    p, n = x.shape
    L = la.cholesky(R) # spptrf might be advantageous for very large matrices (np.tril_indices very slow, this is why not always better)
    iL, _ = dtrtri(L, lower=1) # way faster than using solve_triangular
    v = iL @ x
    a = np.einsum('ij,ji->i',v.T,v)
    return np.log(np.prod(np.diag(L))) - np.sum(logh(a)) / n

def elliptical_egrad(R, x, u):
    p, n = x.shape
    L = la.cholesky(R)
    iL, _ = dtrtri(L, lower=1)
    v = iL @ x
    a = np.einsum('ij,ji->i',v.T,v)
    y = np.sqrt(u(a)) * x
    psi = y @ y.T
    iR = iL.T @ iL
    return iR @ ( R - psi / n) / 2 @ iR

def elliptical_rgrad(R, x, u):
    p, n = x.shape
    L = la.cholesky(R)
    iL, _ = dtrtri(L, lower=1)
    v = iL @ x
    a = np.einsum('ij,ji->i',v.T,v)
    y = np.sqrt(u(a)) * x
    psi = y @ y.T
    return (R - psi / n) / 2

### specific functions
# Normal
def normal_logh(t):
    return -t/2

def normal_cost(R, x):
    return elliptical_cost(R, x, normal_logh)

def normal_egrad(R,x):
    p, n = x.shape
    iR = la.inv(R)
    return iR @ (R - x @ x.T / n) / 2 @ iR

def normal_rgrad(R,x):
    p, n = x.shape
    return (R - x @ x.T / n) / 2


# Student t
def t_logh(t, df, dim):
    return -(df+dim)/2*np.log(df+t)

def t_u(t, df, dim):
    return (df+dim)/(df+t)

def t_cost(R, x, df):
    p, _ = R.shape
    return elliptical_cost(R,x,partial(t_logh,df=df,dim=p))

def t_egrad(R, x, df):
    p, _ = R.shape
    return elliptical_egrad(R,x,partial(t_u,df=df,dim=p))

def t_rgrad(R, x, df):
    p, _ = R.shape
    return elliptical_rgrad(R,x,partial(t_u,df=df,dim=p))


# Tyler
def tyler_logh(t, dim):
    return -dim/2*np.log(t)

def tyler_u(t, dim):
    return dim/t

def tyler_cost(R, x):
    p, _ = R.shape
    return elliptical_cost(R,x,partial(tyler_logh,dim=p))

def tyler_egrad(R, x):
    p, _ = R.shape
    return elliptical_egrad(R,x,partial(tyler_u,dim=p))

def tyler_rgrad(R, x):
    p, _ = R.shape
    return elliptical_rgrad(R,x,partial(tyler_u,dim=p))



