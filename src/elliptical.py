import numpy as np

from scipy.linalg.lapack import dtrtri, dpptrf

from .optimization_function import OptimizationFunction



class EllipticalRealCentered(OptimizationFunction):
    """
    Optimization function of the family of multivariate real centered elliptical distribution
    (inherits OptimizationFunction class).

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        data used to estimate covariance matrix.
    log_density_generator : Callable
        logarithm of the density generator of the distribution.
    u : Callable
        :math: `u=-2h'/h`, where :math: `h` is the density generator of the distribution.

    Attributes
    ----------
    n_samples : int
        number of samples in data
    n_features : int
        number of features in data
    """
    def __init__(self, data, log_density_generator, u):
        self.data = data
        self.log_density_generator = log_density_generator
        self.u = u
        self.n_samples = data.shape[0]
        self.n_features = data.shape[1]
        self._trilix = np.tril_indices(self.n_features)

    def cost(self, R):
        """
        Log-likelihood of multivariate real centered elliptical distributions.

        Parameters
        ----------
        R : ndarray of shape (n_features, n_features)
            SPD matrix.

        Returns
        -------
        float
            log-likelihood evaluated at R.
        """
        L, iL = self._cholesky_and_inv(R)
        v = iL @ self.data.T
        a = np.einsum('ij,ji->i',v.T,v)
        return np.log(np.prod(np.diag(L))) - np.sum(self.log_density_generator(a)) / self.n_samples

    def euclidean_gradient(self,R):
        """
        Euclidean gradient of the log-likelihood of multivariate real centered elliptical distributions.

        Parameters
        ----------
        R : ndarray of shape (n_features, n_features)
            SPD matrix.

        Returns
        -------
        ndarray of shape (n_features, n_features)
            Euclidean gradient evaluated at R.
        """
        _, iL = self._cholesky_and_inv(R)
        v = iL @ self.data.T
        a = np.einsum('ij,ji->i',v.T,v)
        y = np.sqrt(self.u(a)) * self.data.T
        psi = y @ y.T
        iR = iL.T @ iL
        return iR @ ( R - psi / self.n_samples) / 2 @ iR
    
    def _cholesky_and_inv(self, R):
        """
        Private function to compute fast Cholesky decomposition and its inverse

        Parameters
        ----------
        R : ndarray of shape (n_features, n_features)
            SPD matrix.

        Returns
        -------
        ndarray of size (n_features, n_features)
            lower triangular matrix, inverse of the Cholesky decomposition of R. 
        """
        L = np.zeros((self.n_features,self.n_features))
        L[self._trilix] = dpptrf(self.n_features,R[self._trilix])[0] # since trilix is defined in init, should be quite faster than la.cholesky
        iL, _ = dtrtri(L, lower=1) # way faster than using solve_triangular
        return L, iL


class GaussianRealCentered(EllipticalRealCentered):
    """
    Optimization function corresponding to the multivariate real centered Gaussian distribution.
    
    It inherits EllipticalRealCentered. The euclidean_gradient method is overridden because
    it is quite simpler in the Gaussian case.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        data used to estimate covariance matrix.
    """
    def __init__(self, data):
        def log_density_generator(x): return -x/2
        def u(x): return 1
        super().__init__(data, log_density_generator, u)

    def euclidean_gradient(self, R):
        """
        Euclidean gradient of the log-likelihood of multivariate real centered Gaussian distribution.

        Parameters
        ----------
        R : ndarray of shape (n_features, n_features)
            SPD matrix.

        Returns
        -------
        ndarray of shape (n_features, n_features)
            Euclidean gradient evaluated at R.
        """
        # override Euclidean gradient because faster way
        L = np.zeros((self.n_features,self.n_features))
        L[self._trilix] = dpptrf(self.n_features,R[self._trilix],lower=1)[0]
        iL, _ = dtrtri(L, lower=1)
        iR = iL.T @ iL # faster than la.inv ?
        return iR @ (R - self.data @ self.data.T / self.n_samples) / 2 @ iR


class tRealCentered(EllipticalRealCentered):
    """
    Optimization function corresponding to the multivariate real centered t-distribution.
    It inherits EllipticalRealCentered.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        data used to estimate covariance matrix.
    df : float
        degrees of freedom of the t-distribution.
    """
    def __init__(self, data, df):
        n_features = data.shape[1]
        def log_density_generator(x): return -(df+n_features)/2*np.log(df+x)
        def u(x): return (df+n_features)/(df+x)
        super().__init__(data, log_density_generator, u)


class TylerReal(EllipticalRealCentered):
    """
    Optimization function corresponding to Tyler M-estimator.
    It inherits EllipticalRealCentered. 

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        data used to estimate covariance matrix.
    """
    def __init__(self, data):
        n_features = data.shape[1]
        def log_density_generator(x): return -n_features/2*np.log(x)
        def u(x): return n_features/x
        super().__init__(data, log_density_generator, u)



def SCM_estimator(data):
    return data.T @ data / data.shape[0]
