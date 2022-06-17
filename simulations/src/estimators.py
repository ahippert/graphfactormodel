import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.covariance import EmpiricalCovariance, empirical_covariance, GraphicalLasso
from sklearn.cluster import SpectralClustering
from sknetwork.clustering import KMeans, Louvain
from StructuredGraphLearning.LearnGraphTopology import LearnGraphTopology
from .utils import disable_tqdm


# -------------------------------------------------------------------------
# sknetwork wrapper to have same arguments as sklearn
# -------------------------------------------------------------------------
class louvain(Louvain):
    def fit(self, X, y=None, force_bipartite=False):
        return super().fit(csr_matrix(X), force_bipartite)  

class kmeans(KMeans):
    def fit(self, X, y=None):
        return super().fit(csr_matrix(X))


# -------------------------------------------------------------------------
# Graphical Lasso transformer
# -------------------------------------------------------------------------
class GLasso(GraphicalLasso, TransformerMixin):
    def transform(self, X, **args):
        # TODO: GIVE THE ADJAACENCY AND NOT PRECISION  !!!!!!!!!!
        return np.abs(self.precision_)



# -------------------------------------------------------------------------
# Scikit-learn wrapper around StructuredGraphLearning
# -------------------------------------------------------------------------
def S_regularized_empirical_covariance(X, alpha):
    empirical_cov = empirical_covariance(X, assume_centered=True)
    return empirical_cov + alpha*np.eye(len(empirical_cov))


class SGLkComponents(EmpiricalCovariance, TransformerMixin):
    _LGT_ATTR_NAMES = [
        'S', 'is_data_matrix', 'alpha', 'maxiter', 'abstol',
        'reltol', 'record_objective', 'record_weights'
    ]
    _FIT_ATTR_NAMES = [
        'k', 'rho', 'beta', 'w0', 'fix_beta', 'beta_max',
        'lb', 'ub', 'eigtol', 'eps'
    ]
    def __init__(
        self,
        S, is_data_matrix=False, alpha=0, maxiter=10000, abstol = 1e-6, reltol = 1e-4,
        record_objective = False, record_weights = False,
        S_estimation_method=S_regularized_empirical_covariance, S_estimation_args=[0],
        k=1, rho=1e-2, beta=1e4, w0='naive', fix_beta=True, beta_max = 1e6,
        lb=0, ub=1e10, eigtol = 1e-9, eps = 1e-4,
        assume_centered=False, verbosity=1):

        super().__init__(assume_centered=assume_centered)
        self.verbosity = verbosity
        self.S_estimation_method = S_estimation_method
        self.S_estimation_args = S_estimation_args

        # Storing LearnGraphTopology attributes and creating estimator
        self._LGT_ATTR_VALUES = [
            S, is_data_matrix, alpha, maxiter, abstol, reltol,
            record_objective, record_weights,
        ]
        for key, value in zip(self._LGT_ATTR_NAMES, self._LGT_ATTR_VALUES):
            setattr(self, key, value)
        self.estimator = LearnGraphTopology(
            S, is_data_matrix, alpha, maxiter, 
            abstol, reltol, record_objective, record_weights
        )

        # Storing fit hyperparameters values as a list and attibutes
        self._FIT_ATTR_VALUES = [
            k, rho, beta, w0, fix_beta, beta_max,
            lb, ub, eigtol, eps
        ]
        for key, value in zip(self._FIT_ATTR_NAMES, self._FIT_ATTR_VALUES):
            setattr(self, key, value)

    def fit(self, X, y=None):
        # from (Kumar et al. 2020)
        if self.estimator.S is None:
            S = self.S_estimation_method(X, *self.S_estimation_args)
            self.estimator.S = S
            self.S = S

        # Doing estimation
        if self.verbosity>=1:
            results = self.estimator.learn_k_component_graph(*self._FIT_ATTR_VALUES)
        else:
            with disable_tqdm():
                results = self.estimator.learn_k_component_graph(*self._FIT_ATTR_VALUES)

        # Saving results
        self.precision_ = results['adjacency']
        self.covariance_ = np.linalg.inv(self.precision_)
        for key in results.keys():
            setattr(self, key+'_', results[key])
        
        return self

    def transform(self, X, **args):
        return self.adjacency_

