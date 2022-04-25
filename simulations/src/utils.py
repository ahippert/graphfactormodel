'''
File: utils.py
Created Date: Tuesday February 1st 2022 - 04:04pm
Author: Ammar Mian
Contact: ammar.mian@univ-smb.fr
-----
Last Modified: Thu Feb 03 2022
Modified By: Ammar Mian
-----
General purpose utils functions/classes
-----
Copyright (c) 2022 Université Savoie Mont-Blanc
'''
import numpy as np
from numpy.linalg import det
from scipy.stats import (
    unitary_group, multivariate_t, ortho_group,
    multivariate_normal
)
import logging
from contextlib import contextmanager
from functools import partialmethod
import tqdm

def hermitian(X):
    return .5*(X + np.conjugate(X.T))


def proj_shpd(A, xi):
    iA = invsqrtm(A)
    return hermitian(xi) - np.trace(iA@xi@iA)*A/len(A)


def MSE(Sigma_true, Sigma):
    return np.linalg.norm(Sigma_true-Sigma, 'fro')/np.linalg.norm(Sigma_true, 'fro')


def invsqrtm(M):
    """Inverse sqrtm for a SPD matirx

    Parameters
    ----------
    M : array-like of shape (n_features, n_features)
        input matrix
    """

    eigvals, eigvects = np.linalg.eigh(M)
    eigvals = np.diag(1/np.sqrt(eigvals))
    Out = np.dot(
        np.dot(eigvects, eigvals), np.conjugate(eigvects.T))
    return Out


def inv(M):
    eigvals, eigvects = np.linalg.eigh(M)
    eigvals = np.diag(1/eigvals)
    Out = np.dot(
        np.dot(eigvects, eigvals), np.conjugate(eigvects.T))
    return Out

def generate_covariance(n_features, unit_det=False,
                    random_state=None, dtype=float):
    """Generate random covariance of size n_features using EVD.

    Parameters
    ----------
    n_features : int
        number of features
    unit_det : bool, optional
        whether to have unit determinant or not, by default False
    random_state : None or a numpy random data generator
        for reproducibility matters, one can provide a random generator
    dtype : float or complex
        dtype of covariance wanted
    """
    if random_state is None:
        rng = np.random
    else:
        rng = random_state

    # Generate eigenvalues
    D = np.diag(1+rng.normal(size=n_features))**2
    if dtype is complex:
        Q = unitary_group.rvs(n_features, random_state=rng)
    else:
        Q = ortho_group.rvs(n_features, random_state=rng)
    Sigma = Q@D@Q.conj().T
    
    if unit_det:
        Sigma = Sigma/(np.real(det(Sigma))**(1/n_features))
    
    return Sigma.astype(dtype)



def sample_complex_gaussian(
    n_samples, location, shape, random_state=None
):
    """Sample from circular complex multivariate Gaussian distribution 

    Parameters
    ----------
    n_samples : int
        number of samples
    location : array-like of shape (n_features,)
        location (mean) of the distribution
    shape : array-like of shape (n_features, n_features)
        covariance of the distribution
    random_state : None or a numpy random data generator
        for reproducibility matters, one can provide a random generator
    """
    location_real = arraytoreal(location)
    shape_real = covariancetoreal(shape)
    return arraytocomplex(
        multivariate_normal.rvs(
            mean=location_real,
            cov=shape_real,
            size=n_samples,
            random_state=random_state
        )
    )


def sample_complex_multivariate_t(
    n_samples, location, shape, d, random_state=None
):
    """Sample from circular complex multivariate t-distribution 

    Parameters
    ----------
    n_samples : int
        number of samples
    location : array-like of shape (n_features,)
        location (mean) of the distribution
    shape : array-like of shape (n_features, n_features)
        covariance of the distribution
    d : float
        degrees of freedom of the distribution, must be > 0
    random_state : None or a numpy random data generator
        for reproducibility matters, one can provide a random generator
    """
    location_real = arraytoreal(location)
    shape_real = covariancetoreal(shape)
    return arraytocomplex(
        multivariate_t.rvs(
            loc=location_real,
            shape=shape_real,
            df=d,
            size=n_samples,
            random_state=random_state
        )
    )



def arraytoreal(a):
    """Returns a real equivalent of input complex array used in various taks.
    Parameters
    ----------
    a : array-like of shape (n_samples, n_features)
        Input array.
    Returns
    -------
    array-like of shape (n_samples, 2*n_features)
        Real equivalent array.
    Raises
    ------
    AttributeError
        when input array format is not of dimension 1 or 2.
    """
    if np.iscomplexobj(a):
        if a.ndim == 1:
            return np.concatenate([np.real(a), np.imag(a)])
        elif a.ndim == 2:
            return np.hstack([np.real(a), np.imag(a)])
        else:
            raise AttributeError("Input array format not supported.")
    else:
        logging.debug("Input array is not complex, returning input")
        return a


def arraytocomplex(a):
    """Returns complex array from real input array.
    Parameters
    ----------
    a : array-like of shape (n_samples, 2*n_features)
        Input array.
    Returns
    -------
    array-like of shape (n_samples, 2*n_features)
        Real equivalent array.
    Raises
    ------
    AttributeError
        when input array format is not of dimension 1 or 2.
    """
    if not np.iscomplexobj(a):
        if a.ndim == 1:
            p = int(len(a)/2)
            return a[:p] + 1j*a[p:]
        elif a.ndim == 2:
            p = int(a.shape[1]/2)
            return np.vstack(a[:, :p] + 1j*a[:, p:])
        else:
            raise AttributeError("Input array format not supported")
    else:
        return a

def covariancetoreal(a):
    """Return real equivalent of complex matrix input.
    Parameters
    ----------
    a : array-like of shape (n_features, n_features)
        Input array.
    Returns
    -------
    array-like of shape (2*n_features, 2*n_features)
        Real equivalent of input array.
    Raises
    ------
    AttributeError
        when input array is not a covariance matrix.
    """

    if np.iscomplexobj(a):
        if iscovariance(a):
            real_matrix = .5 * np.block([[np.real(a), -np.imag(a)],
                                        [np.imag(a), np.real(a)]])
            return real_matrix
        else:
            raise AttributeError("Input array is not a covariance.")
    else:
        logging.debug("Input array is not complex, returning input.")
        return a


def covariancetocomplex(a):
    """Return complex matrix from its real equivalent in input.
    Input can be any transform of a matrix obtained thanks to function
    covariancetoreal or any square amtrix whose shape is an even number.
    Parameters
    ----------
    a : array-like of shape (2*n_features, 2*n_features)
        Input array, real equivalent of a complex square matrix.
    Returns
    -------
    array-like of shape (n_features, n_features)
        Real equivalent of input array.
    Raises
    ------
    AttributeError
        when input array format is not of dimension 2 or shape is not even.
    """

    if not np.iscomplexobj(a):
        if iscovariance(a) and len(a) % 2 == 0:
            p = int(len(a)/2)
            complex_matrix = 2 * a[:p, :p] + 2j*a[p:, :p]
            return complex_matrix
        else:
            raise AttributeError("Input array format not supported.")

    else:
        logging.debug("Input is already a complex array, returning input.")
        return a


def iscovariance(a):
    """Check if Input array correspond to a square matrix.
    TODO: do more than square matrix.
    Parameters
    ----------
    a : array-like
        Input array to check.
    Returns
    -------
    bool
        Return True if the input array is a square matrix.
    """

    return (a.ndim == 2) and (a.shape[0] == a.shape[1])


def matprint(mat, fmt="g"):
    """ Pretty print a matrix in Python 3 with numpy.
    Source: https://gist.github.com/lbn/836313e283f5d47d2e4e
    """

    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col])
                 for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


def format_pipeline_name(pipeline):
    name = ''
    for step in pipeline.named_steps.keys():
        name += step + ' + '
    return name[:-3]


# -------------------------------------------------------------------------
# Disabling tqdm temporarily. Credits to liam-ly:
# https://github.com/tqdm/tqdm/issues/614
# -------------------------------------------------------------------------
@contextmanager
def monkeypatched(obj, name, patch):
    """Temporarily monkeypatch."""
    old_attr = getattr(obj, name)
    setattr(obj, name, patch(old_attr))
    try:
        yield
    finally:
        setattr(obj, name, old_attr)


@contextmanager
def disable_tqdm():
    """Context manager to disable tqdm."""

    def _patch(old_init):
        return partialmethod(old_init, disable=True)

    with (
        monkeypatched(tqdm.std.tqdm, "__init__", _patch)
    ):
        yield