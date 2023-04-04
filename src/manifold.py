import numpy as np
import numpy.linalg as la
from scipy.linalg import sqrtm

from scipy.stats import ortho_group, norm, uniform

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold, Manifold

from pymanopt.manifolds.product import _ProductTangentVector


def sym(x):
    return 0.5 * (x + x.T)


class SPD(EuclideanEmbeddedSubmanifold):
    def __init__(self, p):
        self._p = p
        name = f"Manifold of positive definite {p}x{p} matrices"
        dimension = int(p*(p+1)/2)
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return np.sqrt(self.dim)

    #def rand(self):
    #   p = self._p
    #   U = ortho_group.rvs(p)
    #   d = np.power(norm.rvs(size=p),2)
    #   return U @ np.diag(d) @ U.T

    def rand(self, cond=100):
        p = self._p
        U = ortho_group.rvs(p)
        #
        d = np.zeros(p)
        if p>2:
            d[:p-2] = uniform.rvs(loc=1/np.sqrt(cond),scale=np.sqrt(cond)-1/np.sqrt(cond),size=p-2)
        d[p-2] = 1/np.sqrt(cond)
        d[p-1] = np.sqrt(cond)
        #
        return U @ np.diag(d) @ U.T

    def randvec(self, x):
        p = self._p
        return self.proj(x, norm.rvs(size=(p,p)))

    def zerovec(self, x):
        p = self._p
        return np.zeros((p,p))

    def inner(self,x, u, v):
        ix_u = la.solve(x, u) # better replace by Cholesky + inv_triu?
        if u is v:
            ix_v = ix_u
        else:
            ix_v = la.solve(x, v)
        return np.tensordot(ix_u, ix_v.T, axes=x.ndim)

    def norm(self, x, u):
        return self.inner(x, u, u)**0.5

    def proj(self, x, u):
        return sym(u)

    def egrad2rgrad(self, x, egrad):
        return x @ sym(egrad) @ x.T

    def retr(self, x, u):
        return sym( x + u + 0.5 * u @ la.solve(x, u))

    def transp(self, x1, x2, u):
        tmp = sqrtm(la.solve(x1, x2).T) # (x2 x1^{-1})^{1/2}
        return tmp @ u @ tmp.T

    def dist(self, x1, x2):
        c = la.cholesky(x1)
        ic = la.inv(c)
        tmp = ic @ x2 @ ic.T
        return np.sum(np.log(la.eigh(tmp)[0])**2)**0.5



class manFactorModel(EuclideanEmbeddedSubmanifold):
    def __init__(self, p, k):
        self._p = p
        self._k = k
        name = f"Manifold for the factor model: {p}x{p} symmetric positive semi-definite matrix of rank {k} and {p}x{p} diagonal positive definite matrix"
        dimension = int(p*k - k*(k-3)/2 + p)
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return np.sqrt(self.dim)

    def rand(self):
        p = self._p
        k = self._k
        U = ortho_group.rvs(p)
        U = U[:,range(k)]
        #
        V = ortho_group.rvs(k)
        d = norm.rvs(size=k)**2
        #
        l = norm.rvs(size=p)**2
        return [U, V@np.diag(d)@V.T, l]

    def randvec(self, x):
        p = self._p
        k = self._k
        return self.proj(x, [norm.rvs(size=(p,k)), norm.rvs(size=(k,k)), norm.rvs(size=p)])

    def zerovec(self, x):
        p = self._p
        k = self._k
        return _ProductTangentVector([np.zeros((p,k)), np.zeros((k,k)), np.zeros(p)])
        
    def inner(self, x, u, v):
        p = self._p
        inner_st =  np.tensordot( u[0], (np.eye(p) - 0.5 * x[0] @ x[0].T) @ v[0], axes=x[0].ndim)
        #
        ix_u = la.solve(x[1], u[1])
        if u is v:
            ix_v = ix_u
        else:
            ix_v = la.solve(x[1], v[1])
        inner_spd = np.tensordot(ix_u, ix_v.T, axes=x[1].ndim)
        inner_dpd = np.tensordot(u[2] / x[2], v[2] / x[2], axes = x[2].ndim)
        return inner_st + inner_spd + inner_dpd

    def norm(self, x, u):
        return self.inner(x, u, u)**0.5
        
    def proj(self, x, u):
        return _ProductTangentVector(self._proj_horizontal(x,self._proj_tangent(x,u)))
        
    def _proj_tangent(self, x, u):
        return _ProductTangentVector([u[0] - x[0] @ sym(x[0].T @ u[0]), sym(u[1]), u[2]])
        
    def _proj_horizontal(self, x, u):
        p = self._p
        k = self._k
        iS = la.inv(x[1])
        tmp1 = -3*np.eye(k**2) + 2*(np.kron(x[1],iS) + np.kron(iS,x[1]))
        tmp2 = x[0].T @ u[0] + 2*(u[1] @ iS - iS @ u[1])
        Omeg = la.solve(tmp1,tmp2.flatten('F'))
        Omeg = np.reshape(Omeg,(k,k),order='F')
        Omeg = Omeg - sym(Omeg)
        return _ProductTangentVector([u[0] - u[0] @ Omeg, x[1] + Omeg @ x[1] - x[1] @ Omeg, u[2]])

    def egrad2rgrad(self, x, egrad):
        return _ProductTangentVector([egrad[0] - x[0] @ egrad[0].T @ x[0], x[1] @ sym(egrad[1]) @ x[1], x[2]**2 *egrad[2]])
    
    def retr(self, x, u):
        v, _, w = la.svd(x[0] + u[0], full_matrices=False)
        return [v @ w, sym( x[1] + u[1] + 0.5 * u[1] @ la.solve(x[1], u[1])), x[2] + u[2] + 0.5 * u[2]**2 / x[2]]
        
    def transp(self, x1, x2, u):
        return self.proj(x2, u)




def mapFactor2SPD(x):
    return x[0] @ x[1] @ x[0].T + np.diag(x[2])

def egradSPD2egradFactor(x, egradSPD):
    return _ProductTangentVector([2*egradSPD @ x[0] @ x[1], x[0].T @ egradSPD @ x[0], np.diag(egradSPD)])


