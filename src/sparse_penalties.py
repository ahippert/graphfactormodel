import os, sys
from typing import Any
sys.path.append(os.path.join(os.path.dirname(__file__), "./"))

import numpy as np
import numpy.linalg as la

from scipy.linalg.lapack import dtrtri, dpptrf

from .optimization_function import OptimizationFunction


class SparsePenalty_SPDInv_Elementwise(OptimizationFunction):
    """
    Optimization function to promote sparsity in the inverse of some SPD matrix.
    The penalty is applied elementwise.

    Parameters
    ----------
    elementwise_penalty : class
        penalty applied elementwise on any ndarray.
        The class needs to contain two methods: the function and its derivative.
    size : int
        size of the SPD matrices to handle.
        Needed for fast inversion purpose.
    """
    def __init__(self,elementwise_penalty,size):
        self.penalty = elementwise_penalty
        self.size = size
        # initialize structure for fast inverse
        self._trilix = np.tril_indices(self.size)
        # off diagonal indices
        self._ixu = np.triu_indices(self.size,k=1)
        self._ixl = np.tril_indices(self.size,k=-1) 

    def _inv(self, point: np.ndarray) -> np.ndarray:
        """
        Private function to compute fast inverse of SPD matrix.

        Parameters
        ----------
        point : ndarray of shape (size, size)
            SPD matrix.

        Returns
        -------
        ndarray of shape (size, size)
            SPD matrix, inverse of input.
        """
        L = np.zeros((self.size,self.size))
        L[self._trilix] = dpptrf(self.size,point[self._trilix])[0] # since trilix is defined in init, should be quite faster than la.cholesky
        # L = la.cholesky(R) # replace by dpptrf with np.tril_indices as attribute of the class
        iL, _ = dtrtri(L, lower=1) # way faster than using solve_triangular
        return iL.T @ iL

    def cost(self, point: np.ndarray) -> float:
        """
        Cost function of the sparse penalty.

        Parameters
        ----------
        point : ndarray of shape (size, size)
            SPD matrix.

        Returns
        -------
        float
            cost function evaluated at point.
        """
        A = self.penalty.function(self._inv(point))
        return np.sum(A[self._ixu]) + np.sum(A[self._ixl])
    
    def euclidean_gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Euclidean gradient of the sparse penalty.

        Parameters
        ----------
        point : np.ndarray of shape (size, size)
            SPD matrix.

        Returns
        -------
        np.ndarray of shape (size, size)
            Euclidean gradient evaluated at point.
        """
        inverse_point = self._inv(point)
        base_grad = np.zeros((self.size,self.size))
        base_grad[self._ixu] = self.penalty.derivative(inverse_point[self._ixu])
        base_grad[self._ixl] = self.penalty.derivative(inverse_point[self._ixl])
        return - inverse_point @ base_grad @ inverse_point


class l1Smooth():
    """
    Class encapsulating a smooth approximation of the l1 norm.
    It contains the function and its derivative.

    Parameters
    ----------
    eps : float
        parameter controlling how close the approximation is from l1 norm.
        The l1 norm is obtained as eps tends to 0.
    """
    def __init__(self, eps):
        self.eps = eps

    def function(self, x):
        """
        Approximation of the l1 norm.

        Parameters
        ----------
        x : scalar or ndarray

        Returns
        -------
        scalar or ndarray
            function applied elementwise.
        """
        res = self.eps * np.log(np.cosh(x/self.eps))
        ### fix eventual inf errors as best as possible
        if np.isscalar(res) and np.isinf(res):
            res = np.abs(x)
        else:
            ix = np.isinf(res)
            res[ix] = np.abs(x[ix])
        return res

    def derivative(self, x):
        """
        Derivative of the approximation of the l1 norm.

        Parameters
        ----------
        x : scalar or ndarray

        Returns
        -------
        scalar or ndarray
            derivative applied elementwise. 
        """
        return np.tanh(x/self.eps)


class ReluSmooth():
    """
    Class encapsulating a smooth approximation of the relu function.
    It contains the function and its derivative.

    WARNING: this has not actually been tested in the current form.

    Parameters
    ----------
    eps : float
        parameter controlling how close the approximation is from relu.
        The relu function is obtained as eps tends to 0.
    """
    def __init__(self, eps) -> None:
        self.eps = eps

    def function(self, x):
        res = self.eps * np.log(1+np.exp(x/self.eps))
        ### fix eventual inf errors as best as possible
        if np.isscalar(res) and np.isinf(res):
            res = x
        else:
            ix = np.isinf(res)
            res[ix] = x[ix]
        return res
    
    def derivative(self, x):
        res = np.exp(x/self.eps) / (1 + np.exp(x/self.eps))
        if np.isscalar(res) and np.isnan(res):
            res = 1
        else:
            res[np.isnan(res)] = 1 
        return res
