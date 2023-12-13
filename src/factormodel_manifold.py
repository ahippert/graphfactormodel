from typing import Any
import numpy as np
import numpy.linalg as la
from scipy.linalg import sqrtm

from scipy.linalg.lapack import dtrtri

from scipy.stats import ortho_group, norm, uniform

from pymanopt.manifolds.manifold import RiemannianSubmanifold
from pymanopt.manifolds.product import _ProductTangentVector

from .optimization_function import MappingBetweenManifolds
from .spd_manifold import sym




class FactorModel(RiemannianSubmanifold):
    """
    Manifold encapsulating the structure of the factor model.
    It consists in the sum of a symmetric positive semi-definite matrix of fixed rank
    and a diagonal positive definite matrix.
    To deal with the symmetric positive semi-definite matrix, we consider the quotient
    manifold of the product of the Stiefel and SPD manifolds.

    Parameters
    ----------
    n_features : int
        size of the matrices.
    rank : int
        rank of the symmetric positive semi-definite matrix.
    """
    def __init__(self, n_features, rank):
        self.n_features = n_features
        self.rank = rank
        name = f"Manifold for the factor model: {self.n_features}x{self.n_features} symmetric positive semi-definite matrix of rank {self.rank} \
              and {self.n_features}x{n_features} diagonal positive definite matrix"
        dimension = int(self.n_features*self.rank - self.rank*(self.rank-3)//2 + self.n_features)
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return np.sqrt(self.dim)

    def random_point(self):
        """
        Random point on the factor model manifold.

        Returns
        -------
        List[ndarray, ndarray, ndarray]
            random point.
        """
        U = ortho_group.rvs(self.n_features)
        U = U[:,range(self.rank)]
        #
        V = ortho_group.rvs(self.rank)
        d = norm.rvs(size=self.rank)**2
        #
        l = norm.rvs(size=self.n_features)**2
        return [U, V@np.diag(d)@V.T, l]

    def random_tangent_vector(self, point):
        return self.projection(point, [norm.rvs(size=(self.n_features,self.rank)), norm.rvs(size=(self.rank,self.rank)), norm.rvs(size=self.n_features)])

    def zero_vector(self, point):
        return _ProductTangentVector([np.zeros((self.n_features,self.rank)), np.zeros((self.rank,self.rank)), np.zeros(self.n_features)])
        
    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        inner_st =  np.tensordot( tangent_vector_a[0], (np.eye(self.n_features) - 0.5 * point[0] @ point[0].T) @ tangent_vector_b[0], axes=point[0].ndim)
        #
        L = la.cholesky(point[1]) # some replacement needed to optimize stuff a bit !
        iL, _ = dtrtri(L, lower=1)
        coor_a = iL @ tangent_vector_a[1] @ iL.T
        if tangent_vector_a[1] is tangent_vector_b[1]:
            coor_b = coor_a
        else:
            coor_b = iL @ tangent_vector_b[1] @ iL.T
        inner_spd = np.tensordot(coor_a, coor_b, axes=point[1].ndim)
        inner_dpd = np.tensordot(tangent_vector_a[2] / point[2], tangent_vector_b[2] / point[2], axes = point[2].ndim)
        return inner_st + inner_spd + inner_dpd

    def norm(self, point, tangent_vector):
        return np.sqrt(self.inner_product(point, tangent_vector, tangent_vector))
        
    def projection(self, point, vector):
        return _ProductTangentVector(self._projection_horizontal(point, self._projection_tangent(point, vector)))
    
    to_tangent_space = projection
        
    def _projection_tangent(self, point, vector):
        """
        Private function. Project from the ambient space onto the tangent space.
        Still need projection on horizontal space to respect factor model geometry.
        
        Parameters
        ----------
        point : List[ndarray, ndarray, ndarray]
            point on the factor model manifold.
        vector : List[ndarray, ndarray, ndarray]
            vector on the ambient space.

        Returns
        -------
        List[ndarray, ndarray, ndarray]
            tangent vector on the tangent space at point.
        """
        return _ProductTangentVector([vector[0] - point[0] @ sym(point[0].T @ vector[0]), sym(vector[1]), vector[2]])
        
    def _projection_horizontal(self, point, vector):
        """
        Private function. Project from the tangent space onto the horizontal space.

        Parameters
        ----------
        point : List[ndarray, ndarray, ndarray]
            point on the factor model manifold.
        vector : List[ndarray, ndarray, ndarray]
            vector on the tangent space at point.

        Returns
        -------
        List[ndarray, ndarray, ndarray]
            vector on the horizontal space at point.
        """
        iS = la.inv(point[1]) # to be replaced
        # find a way to efficiently solve this equation would be nice
        tmp1 = -3*np.eye(self.rank**2) + 2*(np.kron(point[1], iS) + np.kron(iS, point[1]))
        tmp2 = point[0].T @ vector[0] + 2*(vector[1] @ iS - iS @ vector[1])
        Omeg = la.solve(tmp1,tmp2.flatten('F'))
        Omeg = np.reshape(Omeg,(self.rank,self.rank),order='F')
        Omeg = Omeg - sym(Omeg)
        return _ProductTangentVector([vector[0] - point[0] @ Omeg, vector[1] + Omeg @ point[1] - point[1] @ Omeg, vector[2]])

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        return _ProductTangentVector([euclidean_gradient[0] - point[0] @ euclidean_gradient[0].T @ point[0],
                                      point[1] @ sym(euclidean_gradient[1]) @ point[1],
                                      point[2]**2 * euclidean_gradient[2]])
    
    def retraction(self, point, tangent_vector):
        v, _, w = la.svd(point[0] + tangent_vector[0], full_matrices=False)
        return [v @ w,
                sym( point[1] + tangent_vector[1] + 0.5 * tangent_vector[1] @ la.solve(point[1], tangent_vector[1])),
                point[2] + tangent_vector[2] + 0.5 * tangent_vector[2]**2 / point[2]]
        
    def transport(self, point_a, point_b, tangent_vector_a): # we can for sure do much better
        return self.proj(point_b, tangent_vector_a)


class FactorModel2SPD(MappingBetweenManifolds):
    """
    Mapping between the factor model manifold and the SPD manifold.
    """
    def __init__(self) -> None:
        pass

    def mapping(self, point):
        return point[0] @ point[1] @ point[0].T + np.diag(point[2])
    
    def differential(self, point, tangent_vector):
        return tangent_vector[0] @ point[1] @ point[0].T + point[0] @ point[1] @ tangent_vector[0].T + \
               point[0] @ tangent_vector[1] @ point[0].T + np.diag(tangent_vector[2])
    
    def differential_adjoint(self, point, tangent_vector_arrival):
        return _ProductTangentVector([2*tangent_vector_arrival @ point[0] @ point[1], point[0].T @ tangent_vector_arrival @ point[0], np.diag(tangent_vector_arrival)])
