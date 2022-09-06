import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import EmpiricalCovariance, empirical_covariance
from StructuredGraphLearning.LearnGraphTopology import LearnGraphTopology
from StructuredGraphLearning.utils import Operators
from .utils import disable_tqdm
from networkx import from_numpy_matrix

# -------------------------------------------------------------------------
# Scikit-learn wrapper around StructuredGraphLearning
# -------------------------------------------------------------------------
def S_regularized_empirical_covariance(X, alpha):
    empirical_cov = empirical_covariance(X, assume_centered=True)
    return empirical_cov + alpha*np.eye(len(empirical_cov))


class SGLkComponents(EmpiricalCovariance):
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


# -------------------------------------------------------------------------
# Class Nonconvex Graph Learning
# -------------------------------------------------------------------------
class NGL(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        cov_type="scm",
        operator=Operators(), # instance of StructuredGraphLearning Operators
        alpha=0,
        S_estimation_method=S_regularized_empirical_covariance,
        S_estimation_args=[0],
        lamda=0.5,
        maxiter=50,
        reltol=0.0001,
        record_objective=False,
        backtrack=True,
        verbosity=1
    ):
        self.cov_type = cov_type
        self.operator = operator
        self.alpha = alpha
        self.S_estimation_method = S_estimation_method
        self.S_estimation_args = S_estimation_args
        self.lamda = lamda
        self.reltol = reltol
        self.maxiter = maxiter
        self.record_objective = record_objective
        self.backtrack = backtrack
        self.verbosity = verbosity

    def _compute_initial_weights(self, weights, Sinv, eta):
        """ This function computes initial graph weights using gradient descent
        """
        iteration = 0
        norm = np.inf

        while norm > self.reltol and iteration < self.maxiter:
            grad = self.operator.Lstar(self.operator.L(weights) - Sinv)
            w_new = weights - eta*grad
            w_new[w_new < 0] = 0

            norm = np.linalg.norm(weights - w_new)/np.linalg.norm(weights)
            weights = w_new

            iteration = iteration + 1

        return weights + 1e-4

    def _MCP_dummy(self, x, gamma=1.01):
        if x<0:
            print("Error: x must be positive")
        elif x>=0 and x<=gamma*self.lamda:
            return self.lamda - x/gamma
        else:
            return 0

    def _MCP(self, Lw, gamma=1.01):
        H = -(self.lamda + Lw/gamma) * (Lw >= -self.lamda*gamma)
        np.fill_diagonal(H, 0)
        return H

    def _SCAD_dummy(self, x, gamma=2.01):
        if x<0:
            print("Error: x must be positive")
        elif x>=0 and x<=self.lamda:
            return self.lamda
        elif x>=self.lamda and x<=gamma*self.lamda:
            return (gamma*self.lamda - x)/(gamma - 1)
        else:
            return 0

    def _SCAD(self, Lw, gamma=2.01):
        H = -self.lamda * (Lw >= -self.lamda)
        H = H + (-gamma * self.lamda - Lw) / (gamma - 1) * (Lw > -gamma*self.lamda) * (Lw < -self.lamda)
        np.fill_diagonal(H, 0)
        return H

    def _objective_function(self, Lw, J, K):
        matrix = Lw + J
        chol_factor, pd = sp.linalg.lapack.dpotrf(matrix)
        while pd > 0:
            print("Matrix Lw+J not PD!")
            matrix = matrix + np.eye(matrix.shape[0]) * 0.01
            #chol_factor = np.linalg.cholesky(Lw + J)
            chol_factor, pd = sp.linalg.lapack.dpotrf(matrix)
        return np.sum(Lw*K) - 2*np.sum(np.log(np.diag(chol_factor)))

    def _learn_graph(self, S):
        """Graph learning method
        """

        # Get feature dimension
        n_components, _ = S.shape

        # Compute inverse of covariance matrix
        Sinv = np.linalg.pinv(S)

        # Initialize learning rate
        eta = 1/(2*n_components)

        # Compute initial graph weights
        w0 = self.operator.Linv(Sinv)
        w0[w0<0] = 0
        w0 = self._compute_initial_weights(w0, Sinv, eta)
        Lw0 = self.operator.L(w0)

        # Useful matrices
        J = (1/n_components)*np.ones((n_components,n_components))
        I = np.eye(n_components)
        w = w0
        Lw = Lw0
        H = self._MCP(Lw0) # compute sparsity function
        K = S + H

        ########################
        # Estimation loop
        for i in range(self.maxiter):

            try:
                gradient = self.operator.Lstar(K - np.linalg.inv(Lw + J))
            except np.linalg.LinAlgError:
                #print("Matrix is non-invertible: might produce errors...")
                pass

            if self.backtrack:
                fun = self._objective_function(Lw, J, K)
                while(1):
                    wi = w - eta*gradient
                    wi[wi<0] = 0

                    Lwi = self.operator.L(wi)
                    fun_t = self._objective_function(Lwi, J, K)

                    if (fun < fun_t - np.sum(gradient*(wi-w)) - (.5/eta)*np.linalg.norm(wi-w)**2):
                        eta = .5 * eta
                    else:
                        eta = 2 * eta
                        break
            else:
                wi = w - eta * gradient
                wi[wi < 0] = 0

            norm = np.linalg.norm(self.operator.L(wi) - Lw, 'fro')/np.linalg.norm(Lw, 'fro')
            #print(norm)

            if (norm < self.reltol and i > 1):
                break

            w = wi
            Lw = self.operator.L(w)
            H = self._MCP(Lw) # compute sparsity function
            K = S + H

        #rel_error = np.linalg.norm(Lw - Lw_true, 'fro')/np.linalg.norm(Lw_true, 'fro')
        #print(rel_error)

        adjacency = self.operator.A(w)
        graph = from_numpy_matrix(adjacency)

        return {'adjacency': adjacency,
                'graph': graph}

    def fit(self, X, y=None):

        if self.cov_type == 'scm':
            S = self.S_estimation_method(X, *self.S_estimation_args)

        # Doing estimation
        if self.verbosity>=1:
            results = self._learn_graph(S)

        # Saving results
        self.precision_ = results['adjacency']
        self.covariance_ = np.linalg.inv(self.precision_)
        
        return self

    def transform(self, X, y=None):
        return self


# ### Test script ###
# from estimators import NGL
# ngl = NGL()
# from networkx import laplacian_matrix, barabasi_albert_graph, set_edge_attributes
# import scipy as sp
# from scipy.stats import multivariate_normal, multivariate_t
# from scipy.sparse import csr_matrix
# import numpy as np

# p = 50                                          # dimension
# n_samples = 300
# graph = barabasi_albert_graph(p, 1)
# set_edge_attributes(graph,
#                     {e: {'weight': np.random.uniform(2,5)} for e in graph.edges}
# )

# # Retrieve true laplacian matrix and convert it to numpy array
# Lw_true = csr_matrix.toarray(laplacian_matrix(graph))

# # Generate data with a LGMRF model
# X = multivariate_normal.rvs(
#     mean=np.zeros(p),
#     cov=np.linalg.pinv(Lw_true),
#     size=n_samples
# )

# ngl.fit(X)
