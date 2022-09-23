import numpy as np
import scipy as sp

from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.covariance import EmpiricalCovariance, empirical_covariance, GraphicalLasso
from sklearn.cluster import SpectralClustering
from sknetwork.clustering import KMeans, Louvain, PropagationClustering
from StructuredGraphLearning.LearnGraphTopology import LearnGraphTopology
from StructuredGraphLearning.utils import Operators
from StructuredGraphLearning.optimizer import Optimizer
from .utils import disable_tqdm
from networkx import from_numpy_matrix
import time


# -------------------------------------------------------------------------
# sknetwork wrapper to have same arguments as sklearn
# -------------------------------------------------------------------------
class louvain(Louvain):
    def fit(self, X, y=None, force_bipartite=False):
        return super().fit(csr_matrix(X))  

class kmeans(KMeans):
    def fit(self, X, y=None):
        return super().fit(csr_matrix(X))

class propagation(PropagationClustering):
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
        #self.laplacian_ = results['laplacian']
        #self.precision_ = results['adjacency']
        self.results_ = results
        #self.covariance_ = np.linalg.inv(results['adjacency'])
        for key in results.keys():
            setattr(self, key+'_', results[key])

        return self

    def transform(self, X, **args):
        return self.results_

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, **fit_params).transform(X)


# -------------------------------------------------------------------------
# Class Nonconvex Graph Learning
# -------------------------------------------------------------------------
class NGL(BaseEstimator, TransformerMixin):
    """
    Nonconvex Graph Learning.

    This class is the Python implementation of the Nonconvex Graph Learning (NGL) algorithm
    as proposed in:

    Ying J., Cardoso J. and Palomar D.P. "Nonconvex Sparse Graph Learning under Laplacian
    Constrained Graphical Model". Neurips, 2020.

    For the initial R package, please visit https://github.com/mirca/sparseGraph

    Parameters
    ----------
    cov_type : {'scm'}, default='scm'
        Specify the type of covariance estimation:
        - 'scm' : sample covariance matrix.

    operator : type, default=Operators()
        Always initialized as an instance of the Operators() class. Useful to compute
        laplacian and other operators.

    alpha : float, default=0

    S_estimation_method : function type, default=S_regularized_empirical_covariance
        Defines the covariance estimation function.

    S_estimation_args : list, default=[0]
        Specify the arguments to compute the covariance matrix.

    lambda : float, default=0.

    maxiter : int, default=50

    reltol : float, default=0.0001

    record_objective : bool, default=False

    backtrack : bool, default=True

    verbosity : int, default=1
    """
    def __init__(
        self,
        cov_type="scm",
        operator=Operators(), # instance of StructuredGraphLearning Operators
        alpha=0.,
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

    def _MCP(self, Lw, gamma=1.01):
        """ Non-convex sparsity-promoting function.
        See: C.-H. Zhang et al. "Nearly unbiased variable selection under minimax concave penalty". The Annals of Statistics, 38(2):894–942, 2010.
        """
        H = -(self.lamda + Lw/gamma) * (Lw >= -self.lamda*gamma)
        np.fill_diagonal(H, 0)
        return H

    def _SCAD(self, Lw, gamma=2.01):
        """ Non-convex sparsity-promoting function.
        See: J. Fan and R. Li. "Variable selection via nonconcave penalized likelihood and its oracle properties". Journal of the American Statistical Association, 96(456):1348–1360, 2001.
        """
        H = -self.lamda * (Lw >= -self.lamda)
        H = H + (-gamma * self.lamda - Lw) / (gamma - 1) * (Lw > -gamma*self.lamda) * (Lw < -self.lamda)
        np.fill_diagonal(H, 0)
        return H

    def _objective_function(self, Lw, J, K):
        """ Computes the objective function as defined in (Ying et al., 2020)
        """
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

        # get adjacency matrix and graph
        adjacency = self.operator.A(w)
        graph = from_numpy_matrix(adjacency)

        return {'adjacency': adjacency,
                'graph': graph}

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
        if self.cov_type == 'scm':
            S = self.S_estimation_method(X, *self.S_estimation_args)

        # Doing estimation
        if self.verbosity>=1:
            results = self._learn_graph(S)

        # Saving results
        self.precision_ = results['adjacency']
        self.covariance_ = np.linalg.inv(self.precision_)

        return self

    def transform(self, X, **args):
        """
        Does nothing. For scikit-learn compatibility purposes.
        """
        return self.precision_

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, **fit_params).transform(X)

# -------------------------------------------------------------------------
# Class Heavy-Tail Graph Learning
# -------------------------------------------------------------------------
class HeavyTailGL(BaseEstimator, TransformerMixin):
    """
    Heavy Tail Graph Learning.

    This class is the Python implementation of the Nonconvex Graph Learning (NGL) algorithm
    as proposed in:

    Ying J., Cardoso J. and Palomar D.P. "Nonconvex Sparse Graph Learning under Laplacian
    Constrained Graphical Model". Neurips, 2020.

    For the initial R package, please visit https://github.com/mirca/sparseGraph

    Parameters
    ----------
    operator : type, default=Operators()
        Always initialized as an instance of the Operators() class. Useful to compute
        laplacian and other operators.
    """
    def __init__(
        self,
        heavy_type="gaussian",
        w0_type="naive",
        operator=Operators(), # Contains useful graph operators (laplacian, adjacency, etc.)
        optim=Optimizer(),
        nu=None,
        deg=1,
        rho=1,
        update_rho=True,
        maxiter=1000,
        reltol=1e-5,
        verbosity=1
    ):
        self.heavy_type = heavy_type
        self.w0_type = w0_type
        self.operator = operator
        self.optim = optim
        self.nu = nu
        self.deg = deg
        self.rho = rho
        self.update_rho = update_rho
        self.maxiter = maxiter
        self.reltol = reltol
        self.verbosity = verbosity

    # def _dstar(self, d):
    #     """d* operator (see Cardoso 2021, Neurips)
    #     """
    #     N = len(d)
    #     k = int(0.5*N*(N-1))
    #     dw = np.zeros(k)
    #     j, l = 0, 1
    #     for i in range(k):
    #         dw[i] = d[j] + d[l]
    #         if l==(N-1):
    #             j = j+1
    #             l = j+1
    #         else:
    #             l = l+1
    #     return dw

    def _dstar(self, d):
        """d* operator (see Cardoso 2021, Neurips)
        """
        return self.operator.Lstar(np.diag(d))

    def _Ainv(self, X):
        """Ainv operator (see Cardoso, 2021)
        
        C++ implementation available at:
        https://github.com/dppalomar/spectralGraphTopology/blob/master/src/operators.cc
        """
        n, _ = X.shape
        k = int(0.5*n*(n-1))
        w = np.zeros(k)
        l = 0
        for i in range(n):
            for j in range(i+1, n):
                w[l] = X[i,j]
                l = l+1
        return w

    def _compute_student_weights(self, w, LstarSq, p) :
        """ Compute graph weights for t-distributed data
        """
        return (p+self.nu) / (np.sum(w*LstarSq) + self.nu)


    def _compute_augmented_lagrangian(self, w, LstarSq, theta, J, Y, y, n, p):

        #print(np.sum(theta+J))
        eig = np.linalg.eigvals(theta + J)
        Lw = self.operator.L(w)
        Dw = np.diag(Lw)
        u_func = 0
        if self.heavy_type=="student":
            for q in range(n):
                u_func = u_func + (p+self.nu)*np.log(1 + n*np.sum(w*LstarSq[q])/self.nu)
        else:
            for q in range(n):
                u_func = u_func + np.sum(n*w*LstarSq[q])
  
        u_func = u_func/n
        return (u_func - np.sum(np.log(eig)) + np.sum(y*(Dw-self.deg)) \
                + np.sum(np.diag(Y@(theta - Lw))) \
                + .5*self.rho*(np.linalg.norm(Dw-self.deg)**2 \
                               + np.linalg.norm(Lw-theta, 'fro')**2))



    def _learn_regular_heavytail_graph(self, X):

        n, p = X.shape

        # store cross-correlations to avoid multiple computation of the same quantity
        xxt = [np.vstack(X[i])@X[i][np.newaxis,] for i in range(n)]
        Lstar_seq = [self.operator.Lstar(xxt[i])/(n-1) for i in range(n)]

        # Compute Sample covariance matrix from data
        cor = np.corrcoef(X.T)

        # Compute quantities on the initial guess
        w = self.optim.w_init(self.w0_type, np.linalg.pinv(cor))
        Aw0 = self.operator.A(w)
        Aw0 /= np.sum(Aw0, axis=1)
        Aw0[np.isnan(Aw0)] = 0.
        w = self._Ainv(Aw0)

        J = (1/p)*np.ones((p,p))

        # Initialization of the precision matrix (theta)
        Lw = self.operator.L(w)
        theta = Lw

        # Initialize dual variables by zero
        Y = np.zeros((p,p))
        y = np.zeros(p)

        # ADMM constants
        mu, tau = 2, 2

        # Residual vectors
        primal_lap_residual = []
        primal_deg_residual = []
        dual_residual = []
        rel_error_seq = []
    
        # Augmented lagrangian vector
        lagrangian = []
    
        elapsed_time = []
        start_time = time.time()

        for i in range(self.maxiter):

            # Update w
            LstarLw = self.operator.Lstar(Lw)
            DstarDw = self._dstar(np.diag(Lw))

            Lstar_S_weighted = np.zeros(int(0.5*p*(p-1)))

            if self.heavy_type=='student':
                for q in range(n):
                    Lstar_S_weighted = Lstar_S_weighted \
                        + Lstar_seq[q] * self._compute_student_weights(w, Lstar_seq[q], p)
            else:
                for q in range(n):
                    Lstar_S_weighted = Lstar_S_weighted + Lstar_seq[q]

            grad = Lstar_S_weighted - self.operator.Lstar(self.rho*theta + Y) \
                + self._dstar(y - self.rho*self.deg) + self.rho*(LstarLw + DstarDw)

            eta = 1/(2*self.rho*(2*p - 1))
            wi = w - eta*grad
            wi[wi<0] = 0.
            Lwi = self.operator.L(wi)

            # Update theta (precision matrix)
            Z = self.rho*(Lwi + J) - Y
            
            U, Lambda, _ = np.linalg.svd(Z)
            D = Lambda + np.sqrt(Lambda**2 + 4*self.rho)
            thetai = U@np.diag(D/(2*self.rho))@U.T - J

            # update Y
            R1 = thetai - Lwi
            Y = Y + self.rho*R1
            # update y
            R2 = np.diag(Lwi) - self.deg
            y = y + self.rho*R2

            # compute infinity norm of r for convergence
            primal_residual_R1 = np.linalg.norm(R1, 'fro')
            primal_residual_R2 = np.linalg.norm(R2)
            dual_residual = np.linalg.norm(self.rho*self.operator.Lstar(theta - thetai))

            lagrangian = self._compute_augmented_lagrangian(wi, Lstar_seq, thetai,
                                                            J, Y, y, n, p)

            # update rho
            if self.update_rho:
                s = self.rho * np.linalg.norm(self.operator.Lstar(theta - thetai))
                r = np.linalg.norm(R1, 'fro')
                if r > mu*s:
                    self.rho = self.rho*tau
                elif s > mu*r:
                    self.rho = self.rho/tau
                else:
                    self.rho = self.rho

            rel_error_seq.append(np.linalg.norm(Lw - Lwi, 'fro') / np.linalg.norm(Lw, 'fro'))
            self.has_converged = (rel_error_seq[i] < self.reltol) and (i > 0)
            elapsed_time = time.time() - start_time

            if self.has_converged:
                break

            w = wi
            Lw = Lwi
            theta = thetai

        return {"laplacian": self.operator.L(wi), "adjacency": self.operator.A(wi),
                "precision": thetai, "maxiter": i, "convergence": self.has_converged,
                "time": elapsed_time, "relative_error": rel_error_seq}


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
        #if self.cov_type == 'scm':
        #    S = self.S_estimation_method(X, *self.S_estimation_args)

        # Doing estimation
        if self.verbosity>=1:
            results = self._learn_regular_heavytail_graph(X)

        # Saving results
        self.precision_ = results['adjacency']
        self.covariance_ = np.linalg.inv(self.precision_)

        return self

    def transform(self, X, y=None):
        """
        Does nothing. For scikit-learn compatibility purposes.
        """
        return self.precision_

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, **fit_params).transform(X)

# -------------------------------------------------------------------------
# Class Heavy-Tail k-component Graph Learning
# -------------------------------------------------------------------------
class HeavyTailkGL(BaseEstimator, TransformerMixin):
    """
    Heavy Tail Graph Learning.

    This class is the Python implementation of the Nonconvex Graph Learning (NGL) algorithm
    as proposed in:

    Ying J., Cardoso J. and Palomar D.P. "Nonconvex Sparse Graph Learning under Laplacian
    Constrained Graphical Model". Neurips, 2020.

    For the initial R package, please visit https://github.com/mirca/sparseGraph

    Parameters
    ----------
    operator : type, default=Operators()
        Always initialized as an instance of the Operators() class. Useful to compute
        laplacian and other operators.
    """
    def __init__(
        self,
        heavy_type="gaussian",
        w0_type="naive",
        operator=Operators(), # Contains useful graph operators (laplacian, adjacency, etc.)
        optim=Optimizer(),
        k=1,
        nu=None,
        deg=1,
        rho=1,
        beta=1e-8,
        update_rho=True,
        update_beta=True,
        early_stopping=False,
        maxiter=1000,
        reltol=1e-5,
        verbosity=1
    ):
        self.heavy_type = heavy_type
        self.w0_type = w0_type
        self.operator = operator
        self.optim = optim
        self.k = k
        self.nu = nu
        self.deg = deg
        self.rho = rho
        self.beta = beta
        self.update_rho = update_rho
        self.update_beta = update_beta
        self.early_stopping = early_stopping
        self.maxiter = maxiter
        self.reltol = reltol
        self.verbosity = verbosity

    # def _dstar(self, d):
    #     """d* operator (see Cardoso 2021, Neurips)
    #     """
    #     N = len(d)
    #     k = int(0.5*N*(N-1))
    #     dw = np.zeros(k)
    #     j, l = 0, 1
    #     for i in range(k):
    #         dw[i] = d[j] + d[l]
    #         if l==(N-1):
    #             j = j+1
    #             l = j+1
    #         else:
    #             l = l+1
    #     return dw

    def _dstar(self, d):
        """d* operator (see Cardoso 2021, Neurips)
        """
        return self.operator.Lstar(np.diag(d))

    def _Ainv(self, X):
        """Ainv operator (see Cardoso, 2021)
        
        C++ implementation available at:
        https://github.com/dppalomar/spectralGraphTopology/blob/master/src/operators.cc
        """
        n, _ = X.shape
        k = int(0.5*n*(n-1))
        w = np.zeros(k)
        l = 0
        for i in range(n):
            for j in range(i+1, n):
                w[l] = X[i,j]
                l = l+1
        return w

    def _compute_student_weights(self, w, LstarSq, p) :
        """ Compute graph weights for t-distributed data
        """
        return (p+self.nu) / (np.sum(w*LstarSq) + self.nu)


    def _compute_augmented_lagrangian_kcomp(self, w, LstarSq, theta,
                                            U, Y, y, n, p):


        eigvals, _ = np.linalg.eigh(theta)
        eigvals = eigvals[self.k:p]
        Lw = self.operator.L(w)
        Dw = np.diag(Lw)
        u_func = 0
        if self.heavy_type=="student":
            for q in range(n):
                u_func = u_func + (p+self.nu)*np.log(1 + n*np.sum(w*LstarSq[q])/self.nu)
        else:
            for q in range(n):
                u_func = u_func + np.sum(n*w*LstarSq[q])
  
        u_func = u_func/n
        return (u_func - np.sum(np.log(eigvals)) + np.sum(y*(Dw-self.deg)) \
                + np.sum(np.diag(Y@(theta - Lw))) \
                + .5*self.rho*(np.linalg.norm(Dw-self.deg)**2 \
                               + np.linalg.norm(Lw-theta, 'fro')**2) \
                + self.beta*np.sum(w*self.operator.Lstar(U@U.T)))

    def _learn_kcomp_heavytail_graph(self, X):

        n, p = X.shape

        # store cross-correlations to avoid multiple computation of the same quantity
        xxt = [np.vstack(X[i])@X[i][np.newaxis,] for i in range(n)]
        Lstar_seq = [self.operator.Lstar(xxt[i])/n for i in range(n)]

        # Compute Sample covariance matrix from data
        cor = np.corrcoef(X.T)

        # Compute quantities on the initial guess
        w = self.optim.w_init(self.w0_type, np.linalg.pinv(cor))
        Aw0 = self.operator.A(w)
        Aw0 /= np.sum(Aw0, axis=1)
        #Aw0[np.isnan(Aw0)] = 0.
        w = self._Ainv(Aw0)

        # Initialization of the precision matrix (theta)
        Lw = self.operator.L(w)
        theta = Lw

        _, U = np.linalg.eigh(Lw)
        U = U[:, self.k:p]

        # Initialize dual variables by zero
        Y = np.zeros((p,p))
        y = np.zeros(p)

        # ADMM constants
        mu, tau = 2, 2

        # Residual vectors
        primal_lap_residual = []
        primal_deg_residual = []
        dual_residual = []
        beta_seq = []
        rel_error_seq = []
    
        # Augmented lagrangian vector
        lagrangian = []
    
        elapsed_time = []
        start_time = time.time()

        for i in range(self.maxiter):

            # Update w
            LstarLw = self.operator.Lstar(Lw)
            DstarDw = self._dstar(np.diag(Lw))

            Lstar_S_weighted = np.zeros(int(0.5*p*(p-1)))

            if self.heavy_type=='student':
                for q in range(n):
                    Lstar_S_weighted = Lstar_S_weighted \
                        + Lstar_seq[q] * self._compute_student_weights(w, Lstar_seq[q], p)
            else:
                for q in range(n):
                    Lstar_S_weighted = Lstar_S_weighted + Lstar_seq[q]

            grad = Lstar_S_weighted \
                + self.operator.Lstar(self.beta*U@U.T - Y - self.rho*theta) \
                + self._dstar(y - self.rho*self.deg) + self.rho*(LstarLw + DstarDw)

            eta = 1/(2*self.rho*(2*p - 1))
            wi = w - eta*grad
            wi[wi<0] = 0.
            Lwi = self.operator.L(wi)

            # Update U
            _, U = np.linalg.eigh(Lwi)
            U = U[:, :self.k]

            # Update theta (precision matrix)
            Z = self.rho*Lwi - Y
            
            Lambda, V = np.linalg.eigh(Z)
            V = V[:, self.k:p]
            Lambda = Lambda[self.k:p]
            D = Lambda + np.sqrt(Lambda**2 + 4*self.rho)
            thetai = V@np.diag(D/(2*self.rho))@V.T

            # update Y
            R1 = thetai - Lwi
            Y = Y + self.rho*R1
            # update y
            R2 = np.diag(Lwi) - self.deg
            y = y + self.rho*R2

            # compute infinity norm of r for convergence
            primal_residual_R1 = np.linalg.norm(R1, 'fro')
            primal_residual_R2 = np.linalg.norm(R2)
            dual_residual = np.linalg.norm(self.rho*self.operator.Lstar(theta - thetai))

            lagrangian = self._compute_augmented_lagrangian_kcomp(wi, Lstar_seq, thetai,
                                                                  U, Y, y, n, p)

            # update rho
            if self.update_rho:
                s = self.rho * np.linalg.norm(self.operator.Lstar(theta - thetai))
                r = np.linalg.norm(R1, 'fro')
                if r > mu*s:
                    self.rho = self.rho*tau
                elif s > mu*r:
                    self.rho = self.rho/tau
                else:
                    self.rho = self.rho

            # update beta
            if self.update_beta:
                _, eigvals, _ = np.linalg.svd(Lwi)
                n_zero_eigenvalues = np.sum(eigvals < 1e-9)
                if self.k < n_zero_eigenvalues:
                    self.beta = .5*self.beta
                elif self.k > n_zero_eigenvalues:
                    self.beta = 2*self.beta
                else:
                    if self.early_stopping:
                        has_converged = True
                        break
                beta_seq.append(self.beta)

            rel_error_seq.append(np.linalg.norm(Lw - Lwi, 'fro') / np.linalg.norm(Lw, 'fro'))
            self.has_converged = (rel_error_seq[i] < self.reltol) and (i > 0)
            elapsed_time = time.time() - start_time

            if self.has_converged:
                break

            w = wi
            Lw = Lwi
            theta = thetai

        return {"laplacian": self.operator.L(wi), "adjacency": self.operator.A(wi),
                "precision": thetai, "maxiter": i, "convergence": self.has_converged,
                "time": elapsed_time, "relative_error": rel_error_seq}


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
        #if self.cov_type == 'scm':
        #    S = self.S_estimation_method(X, *self.S_estimation_args)

        # Doing estimation
        if self.verbosity>=1:
            results = self._learn_kcomp_heavytail_graph(X)

        # Saving results
        self.precision_ = results['adjacency']
        self.covariance_ = np.linalg.inv(self.precision_)

        return self

    def transform(self, X, y=None):
        """
        Does nothing. For scikit-learn compatibility purposes.
        """
        return self.precision_

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, **fit_params).transform(X)


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
