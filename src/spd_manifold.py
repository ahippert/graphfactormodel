import numpy as np
import numpy.linalg as la
from scipy.linalg import sqrtm

from scipy.linalg.lapack import dtrtri

from scipy.stats import ortho_group, norm, uniform

from pymanopt.manifolds.manifold import RiemannianSubmanifold


def sym(X: np.ndarray) -> np.ndarray:
    """
    Returns the symmetrical part of its argument, i.e., :math: `sym(X)=(X+X^{T})/2`.

    Parameters
    ----------
    X : ndarray
        square matrix.

    Returns
    -------
    ndarray
        square matrix, symmetrical part of X.
    """
    return 0.5 * (X + X.T)


class SPD(RiemannianSubmanifold):
    """
    Manifold of SPD matrices, pymanopt format (inherits from pymanopt.manifolds.manifold.RiemannianSubmanifold).
    
    Parameters
    ----------
    n_features : int
        size of SPD matrices.
    alpha, beta : floats
        parameters for the general form of the affine-invariant Riemannian metric.
        One must have :math: `alpha>0` and :math: `alpha*n_features + beta \geq 0`. 
    """
    def __init__(self, n_features: int, alpha: int =1, beta: int =0) -> None:
        if (alpha <= 0 or alpha*n_features+beta <=0):
            raise NameError('value of alpha and/or beta invalid, must have alpha>0 and alpha*p+beta>0')

        self.n_features = n_features
        self.alpha = alpha
        self.beta = beta
        name = f"Manifold of symmetric positive definite {self.n_features}x{self.n_features} matrices"
        dimension = int(self.n_features*(self.n_features+1)//2)
        super().__init__(name, dimension)

    @property
    def typical_dist(self) -> float:
        return np.sqrt(self.dim)

    def random_point(self, condition_number: int =10) -> np.ndarray:
        """
        Random point on the manifold of SPD matrices through eigenvalue decomposition.

        Parameters
        ----------
        condition_number : float
            condition number of the point to be generated (maximum eigenvalue over minimum eigenvalue).

        Returns
        -------
        ndarray
            random SPD matrix, generated through :math: `U\Sigma U^T`,
            where :math: `U` is a random orthogonal matrix (uniformly distributed, computed with scipy.stats.ortho_group.rvs),
            and :math: `\Sigma`is a diagonal matrix with strictly positive values.
            The minimum and maximum diagonal values of :math: `\Sigma` are :math: `1/\sqrt{condition\_number}` and :math: `\sqrt{condition\_number}`.
            Values in between are uniformly distributed.
        """
        U = ortho_group.rvs(self.n_features)
        #
        d = np.zeros(self.n_features)
        if self.n_features>2:
            d[:self.n_features-2] = uniform.rvs(loc=1/np.sqrt(condition_number),scale=np.sqrt(condition_number)-1/np.sqrt(condition_number),size=self.n_features-2)
        d[self.n_features-2] = 1/np.sqrt(condition_number)
        d[self.n_features-1] = np.sqrt(condition_number)
        #
        return U @ np.diag(d) @ U.T

    def random_tangent_vector(self, point: np.ndarray) -> np.ndarray:
        """
        Random tangent vector at point.
        Since the tangent space at any point can be identified with the set of symmetric matrices,
        random tangent vectors can be any symmetric matrices.

        Parameters
        ----------
        point : ndarray
            reference SPD matrix (not actually used).

        Returns
        -------
        ndarray
            random symmetric matrix obtained by taking the symmetrical part of a random square matrix
            for which each element is drawn from the normal distribution (using scipy.stats.norm.rvs).
        """
        return self.projection(point, norm.rvs(size=(self.n_features,self.n_features)))

    def zero_vector(self, point: np.ndarray) -> np.ndarray:
        """
        Zero tangent vector at point. This is just the (n_features, n_features) matrix filled with zeros.

        Parameters
        ----------
        point : ndarray
            reference SPD matrix (not actually used).

        Returns
        -------
        ndarray
            matrix filled with zeros.
        """
        return np.zeros((self.n_features,self.n_features))

    def inner_product(self, point: np.ndarray, tangent_vector_a: np.ndarray, tangent_vector_b: np.ndarray) -> float:
        """
        Affine-invariant Riemannian metric on the manifold of SPD matrices.

        Parameters
        ----------
        point : ndarray
            reference SPD matrix.
        tangent_vector_a : ndarray
            first tangent vector at point.
        tangent_vector_b : ndarray
            second tangent vector at point.

        Returns
        -------
        float
            value of the affine-invariant Riemannian metric.
        """
        L = la.cholesky(point) # replace with dpptrf
        iL, _ = dtrtri(L, lower=1)
        coor_a = iL @ tangent_vector_a @ iL.T
        if tangent_vector_a is tangent_vector_b:
            coor_b = coor_a
        else:
            coor_b = iL @ tangent_vector_b @ iL.T
        return self.alpha * np.tensordot(coor_a, coor_b, axes=point.ndim) + self.beta * np.trace(coor_a) * np.trace(coor_b)

    def norm(self, point: np.ndarray, tangent_vector: np.ndarray) -> float:
        """
        Returns the norm of the tangent vector at the reference SPD point, computed through the Riemannian metric.

        Parameters
        ----------
        point : ndarray
            reference SPD matrix.
        tangent_vector : ndarray
            tangent vector at point, symmetric matrix.

        Returns
        -------
        float
            norm of tangent_vector at point according to Riemannian metric.
        """
        return np.sqrt(self.inner_product(point, tangent_vector, tangent_vector))

    def projection(self, point: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        Projection of any square matrix on the tangent space of some reference point on the SPD manifold.
        Here, returning the symmetrical part suffices.

        Parameters
        ----------
        point : ndarray
            reference SPD matrix (not actually used).
        vector : ndarray
            any (n_features, n_features) matrix. 

        Returns
        -------
        ndarray
            symmetrical part of vector.
        """
        return sym(np.real(vector))
    
    to_tangent_space = projection

    def euclidean_to_riemannian_gradient(self, point: np.ndarray, euclidean_gradient: np.ndarray) -> np.ndarray:
        """
        Euclidean to Riemannian gradient according to the most general form
        of the affine-invariant metric on the SPD manifold.

        Parameters
        ----------
        point : ndarray
            SPD matrix.
        euclidean_gradient : ndarray
            Euclidean gradient of the cost function at point.

        Returns
        -------
        ndarray
            Riemannian gradient of the cost function at point.
        """
        return (point @ sym(euclidean_gradient) @ point) / self.alpha - (self.beta / (self.alpha*(self.alpha + self.n_features * self.beta))) * np.trace(euclidean_gradient @ point) * point

    def retraction(self, point: np.ndarray, tangent_vector: np.ndarray) -> np.ndarray:
        """
        Retraction on the SPD manifold, i.e., a mapping from tangent spaces back onto the manifold.
        This is the second-order approximation of the Riemannian exponential mapping.

        Parameters
        ----------
        point : ndarray
            SPD matrix.
        tangent_vector : ndarray
            tangent vector at point.

        Returns
        -------
        ndarray
            resulting SPD matrix.
        """
        return np.real(sym( point + tangent_vector + 0.5 * tangent_vector @ la.solve(point, tangent_vector)))

    def transport(self, point_a: np.ndarray, point_b: np.ndarray, tangent_vector_a: np.ndarray) -> np.ndarray:
        """
        Transport of a tangent vector from the tangent space of a point onto the tangent space of another point.
        Here, this is the parallel transport on the SPD manifold.

        Parameters
        ----------
        point_a : ndarray
            SPD matrix, starting point.
        point_b : ndarray
            SPD matrix, arrival point.
        tangent_vector_a : ndarray
            symmetric matrix, tangent vector at point_a

        Returns
        -------
        ndarray
            symmetric matrix, resulting tangent vector at point_b
        """
        tmp = sqrtm(la.solve(point_a, point_b).T) # (point_b point_a^{-1})^{1/2}
        return tmp @ tangent_vector_a @ tmp.T

    def dist(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        """
        Riemannian distance on the SPD manifold corresponding to the affine-invariant metric.

        Parameters
        ----------
        point_a : ndarray
            SPD matrix.
        point_b : ndarray
            SPD matrix.

        Returns
        -------
        float
            Riemannian distance between point_a and point_b.
        """
        L = la.cholesky(point_a) # replace by dpptrf
        iL,_ = dtrtri(L, lower=1)
        tmp = iL @ point_b @ iL.T
        log_eigs = np.log(la.eigh(tmp)[0]) # replace by some Cholesky ??? YES
        return (self.alpha * np.sum(log_eigs**2) + self.beta * np.sum(log_eigs)**2)**0.5