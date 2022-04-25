

import numpy as np
from .utils import invsqrtm


def negative_log_likelihood_complex_student_t(X, Sigma, df):
    """negative log-likelihood of complex student-t law.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data
    Sigma : array-like of shape (n_features, n_features)
        Covariance matrix
    df : float
        degrees of freedom of Student-t law. Must be > 0
    Returns
    -------
    float
        the negative log-likelihood
    """
    n, p = X.shape

    # Optimized
    temp = invsqrtm(Sigma)@X.T
    q = np.einsum('ij,ji->i', np.conjugate(temp.T), temp)
    return ((df+p)/(p*n))*np.sum(np.log(df+np.real(q))) +\
        (1/p)*np.log(np.real(np.linalg.det(Sigma)))