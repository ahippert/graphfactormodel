import numpy as np
import numpy.random as rnd

from scipy.stats import norm
from scipy.linalg.lapack import dpotrf

from sklearn.base import TransformerMixin, BaseEstimator

from pymanopt.function import Callable
from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient, SteepestDescent

from pymanopt import tools

# TODO: Use Python's enum module.
BetaTypes = tools.make_enum(
    "BetaTypes",
    "FletcherReeves PolakRibiere HestenesStiefel HagerZhang".split())

from functools import partial

from .manifold import SPD, manFactorModel, mapFactor2SPD, egradSPD2egradFactor
from .elliptical_estimation import (
    normal_cost, normal_egrad, normal_rgrad,
    t_cost, t_egrad, t_rgrad,
    tyler_cost, tyler_egrad, tyler_rgrad
)
from .sparse_penalties import (
    smooth_l1, smooth_l1_deriv,
    smooth_relu, smooth_relu_deriv,
    sparse_SPD_cost, sparse_SPD_egrad, sparse_SPD_rgrad,
    sparse_FactorLRpart_cost, sparse_FactorLRpart_egrad
    )

# -------------------------------------------------------------------------
# Class Elliptical Graph Learning
# -------------------------------------------------------------------------
def S_factor_model_sparse_covariance(p, k):
    '''
    '''
    # Low rank structured sparse matrix -- done through Cholesky decomposition
    # potential non-null indices
    triu_indices = np.full((p,p),False)
    triu_indices[np.triu_indices(p,k=1)] = True
    triu_indices[np.triu_indices(p-k,k=1)] = False

    diag_indices = np.full((p,p),False)
    diag_indices[np.diag_indices(p)] = True
    diag_indices[np.diag_indices(p-k)] = False

    # non-zeros elements
    # triu
    nb_triu = p*(p-1)//2 - (p-k)*((p-k)-1)//2
    proportion_nonzeros = 0.3
    nb_nonzeros = int(np.floor(proportion_nonzeros*nb_triu))
    triu = np.zeros(nb_triu)
    triu[:nb_nonzeros] = norm.rvs(size=nb_nonzeros)
    triu = rnd.permutation(triu)
    # diag
    #cond = 50
    l = norm.rvs(size=k)**2

    # Low rank matrix
    L = np.zeros((p,p))
    L[triu_indices] = triu
    L[diag_indices] = l
    low_rank_covariance = L@L.T

    # Full rank matrix
    d = norm.rvs(size=p)**2
    S = low_rank_covariance + np.diag(d)
    return S

class EllipticalGL(BaseEstimator, TransformerMixin):
    """
    Elliptical Graph Learning.
    """
    def __init__(
        self,
        S=None,
        geometry="SPD",
        cov_type="scm",
        df=None,
        k=None,
        S_estimation_method=S_factor_model_sparse_covariance,
        lambda_seq=[0.],
        maxiter=1000,
        eps=1e-12,
        init_S=None,
        record_objective=False,
        backtrack=True,
        verbosity=1
    ):
        self.S = S
        self.geometry = geometry
        self.cov_type = cov_type
        self.df = df
        self.k = k
        self.S_estimation_method = S_estimation_method
        self.lambda_seq = lambda_seq
        self.maxiter = maxiter
        self.eps = eps
        self.init_S = init_S
        self.record_objective = record_objective
        self.backtrack = backtrack
        self.verbosity = verbosity

    def _compute_initial_matrix(self, X, n, p):
        """
        """
        if self.geometry=="SPD":
            init_covariance = np.eye(p)
        else:
            SCM = X.T @ X / n
            ls, Us = np.linalg.eigh(SCM)
            init_covariance = [Us[:,p-self.k:], np.eye(self.k), np.ones(p)]
        #else:
        #    init_covariance = self.init_S
        return init_covariance


    def _learn_graph(self, X):
        """
        """
        # First of all, choose cost and egrad model
        cost_model = partial(t_cost, x=X.T, df=self.df)
        egrad_model = partial(t_egrad, x=X.T, df=self.df)
        #rgrad_model = partial(t_rgrad, x=x, df=df)

        # Choose cost, egrad penalty
        cost_penalty = partial(sparse_SPD_cost, h=partial(smooth_l1,eps=self.eps))
        egrad_penalty = partial(sparse_SPD_egrad, dh=partial(smooth_l1_deriv,eps=self.eps))
        # Initialize covariance matrix
        n, p = X.shape
        init_estimate = self._compute_initial_matrix(X, n, p)

        # Define manifolds
        manifold_SPD = SPD(p)
        if self.geometry is not "SPD":
            manifold_factor = manFactorModel(p, self.k)

        result_seq = []
        error_seq = []

        # Solver
        solver = ConjugateGradient(logverbosity=0, maxiter=self.maxiter)#, beta_type=BetaTypes.HagerZhang)
        #solver = SteepestDescent(logverbosity=0, maxiter=self.maxiter)
        # Define the different values of lambda
        #lamb_all = [0,1e-5,1e-4,5e-4,1e-3,2.5e-3,5e-3,7.5e-3,1e-2,2.5e-2,5e-2]
        for i, lamb in enumerate(self.lambda_seq):
            # Declare optimization functions
            @Callable#(man_SPD)
            def cost_SPD(R):
                #print(cost_model(R))
                return cost_model(R) + lamb * cost_penalty(R)
            @Callable#(man_SPD)
            def egrad_SPD(R):
                return egrad_model(R) + lamb * egrad_penalty(R)

            if self.geometry=="factor":
                @Callable#(man_SPD)
                def cost_factor(theta):
                    return cost_SPD(mapFactor2SPD(theta))
                @Callable#(man_SPD)
                def egrad_factor(theta):
                    return egradSPD2egradFactor(theta, egrad_SPD(mapFactor2SPD(theta)))
            elif self.geometry=="factor+penalty":
                @Callable
                def cost_factor_LRpenalty(theta):
                    return cost_model(mapFactor2SPD(theta)) + lamb * sparse_FactorLRpart_cost(theta, h=partial(smooth_l1,eps=self.eps))
                @Callable
                def egrad_factor_LRpenalty(theta):
                    return egradSPD2egradFactor(theta, egrad_model(mapFactor2SPD(theta))) + lamb * sparse_FactorLRpart_egrad(theta, dh=partial(smooth_l1_deriv, eps=self.eps))
            else:
                pass


            if self.geometry=="SPD":
                cost = cost_SPD
                egrad = egrad_SPD
            elif self.geometry=="factor":
                cost = cost_factor
                egrad = egrad_factor
            else:
                cost = cost_factor_LRpenalty
                egrad = egrad_factor_LRpenalty

            # Declare problem
            if self.geometry=="SPD":
                problem = Problem(manifold=manifold_SPD, cost=cost, egrad=egrad, verbosity=self.verbosity)
            else:
                problem = Problem(manifold=manifold_factor, cost=cost, egrad=egrad, verbosity=self.verbosity)

            # perform resolution
            tmp = solver.solve(problem, x=init_estimate)

            result_seq.append(tmp)

            # Compute distance between current estimate and initial model
            if self.geometry=="SPD":
                error_seq.append(10*np.log(manifold_SPD.dist(init_estimate, result_seq[i])**2))
                if self.verbosity: print(f"{self.geometry} - lambda={lamb}: {error_seq[i]}")
            else:
                error_seq.append(10*np.log(manifold_SPD.dist(mapFactor2SPD(init_estimate),
                                                             mapFactor2SPD(result_seq[i]))**2))
                if self.verbosity: print(f"{self.geometry} - lambda={lamb}: {error_seq[i]}")

            # Update initializations
            init_estimate = tmp

            #i = i+1

        # Get estimate whose distance is minimal and graph matrices
        ind_estimate = -1#np.argmin(np.abs(error_seq))
        if self.geometry=="SPD":
            final_estimate = result_seq[ind_estimate]
            precision_matrix = np.linalg.inv(final_estimate)
        elif self.geometry=="factor":
            U, Lambda, D = result_seq[ind_estimate][0], result_seq[ind_estimate][1], result_seq[ind_estimate][2]
            LR_matrix = U @ Lambda @ U.T
            final_estimate = LR_matrix + np.diag(D)
            precision_matrix = np.linalg.pinv(final_estimate)
        else:
            U, Lambda, D = result_seq[ind_estimate][0], result_seq[ind_estimate][1], result_seq[ind_estimate][2]
            LR_matrix = U @ Lambda @ U.T
            final_estimate = LR_matrix + np.diag(D)
            precision_matrix = np.linalg.pinv(LR_matrix)

        precision = precision_matrix
        #precision = np.abs(precision_matrix)
        #precision[np.abs(precision)<1e-2] = 0.
        #np.fill_diagonal(precision, 0.)

        return {"covariance": final_estimate, "precision": precision,
                "error_seq": error_seq, "result_seq": result_seq}



    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.

        The first step is the estimation of the covariance matrix.

        The second step consists in learning the graph.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_components, n_samples)
            Training vector, where 'n_samples' is the number of samples and
            'n_components' is the number of components or features.

        y :
        """
        # from (Kumar et al. 2020)
        # if self.S is None:
        #     k = self.k
        #     S = self.S_estimation_method(X.shape[1], k)
        #     self.S = S

        # Doing estimation
        #if self.verbosity>=1:
        results = self._learn_graph(X)

        # Saving results
        self.results_ = results
        self.precision_ = results["precision"]
        #self.covariance_ = np.linalg.inv(self.precision_)

        return self

    def transform(self, X, **args):
        """
        Does nothing. For scikit-learn compatibility purposes.
        """
        return self.results_

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, **fit_params).transform(X)
