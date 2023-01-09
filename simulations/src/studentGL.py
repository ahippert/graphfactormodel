import numpy as np

from StructuredGraphLearning.utils import Operators
from StructuredGraphLearning.optimizer import Optimizer
from StructuredGraphLearning.metrics import Metrics

import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import f1_score

# from sknetwork.data import erdos_renyi
# from sknetwork.visualization import svg_graph

from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal, multivariate_t

import visualization as vis

from utils import generate_covariance, invsqrtm

from models import negative_log_likelihood_complex_student_t

import time

def dstar(d):
    """d* operator (see Cardoso 2021, Neurips)
    """
    N = len(d)
    k = int(0.5*N*(N-1))
    dw = np.zeros(k)
    j, l = 0, 1
    for i in range(k):
        dw[i] = d[j] + d[l]
        if l==(N-1):
            j = j+1
            l = j+1
        else:
            l = l+1
    return dw

def Ainv(X):
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

def compute_student_weights(w, LstarSq, p, nu) :
    """ Compute graph weights for t-distributed data
    """
    return (p+nu) / (np.sum(w*LstarSq) + nu)

def compute_augmented_lagrangian(w, Lw, LstarSq, theta, J, Y, y, d,
                                 heavy_type, n, p, rho, nu):


    _, eig, _ = np.linalg.svd(theta + J)
    Dw = np.diag(Lw)
    u_func = 0
    if heavy_type=="student":
        for q in range(n):
            u_func = u_func + (p+nu)*np.log(1 + n*np.sum(w*LstarSq[q])/nu)
    else:
        for q in range(n):
            u_func = u_func + np.sum(n*w*LstarSq[q])
  
    u_func = u_func/n
    return (u_func - np.sum(np.log(eig)) + np.sum(y*(Dw-d)) \
           + np.sum(np.diag(Y@(theta - Lw))) \
           + .5*rho*(np.linalg.norm(Dw-d)**2 + np.linalg.norm(Lw-theta, 'fro')**2))

def compute_augmented_lagrangian_kcomp(w, Lw, LstarSq, theta, U, Y, y, d,
                                       heavy_type, n, p, k, rho, beta, nu):


    _, eig, _ = np.linalg.svd(theta)
    eig = eig[:p-k]
    Dw = np.diag(Lw)
    u_func = 0
    if heavy_type=="student":
        for q in range(n):
            u_func = u_func + (p+nu)*np.log(1 + n*np.sum(w*LstarSq[q])/nu)
    else:
        for q in range(n):
            u_func = u_func + np.sum(n*w*LstarSq[q])
  
    u_func = u_func/n
    return (u_func - np.sum(np.log(eig)) + np.sum(y*(Dw-d)) \
            + np.sum(np.diag(Y@(theta - Lw))) \
            + .5*rho*(np.linalg.norm(Dw-d)**2 + np.linalg.norm(Lw-theta, 'fro')**2) \
            + beta*np.sum(w*op.Lstar(U@U.T)))
    

def learn_regular_heavytail_graph(X,
                                  heavy_type='gaussian',
                                  nu=None,
                                  w0_type='naive',
                                  df=1,
                                  rho=1,
                                  update_rho=True,
                                  maxiter=1000,
                                  reltol=1e-5,
                                  verbose=True):

    n, p = X.shape

    # Instanciate useful classes
    optim = Optimizer()
    op = Operators()

    # store cross-correlations to avoid multiple computation of the same quantity
    xxt = [np.vstack(X[i])@X[i][np.newaxis,] for i in range(n)]
    Lstar_seq = [op.Lstar(xxt[i])/(n-1) for i in range(n)]

    # Compute Sample covariance matrix from data
    cov = np.cov(X.T)

    # Compute quantities on the initial guess
    w = optim.w_init(w0_type, np.linalg.pinv(cov))
    Aw0 = op.A(w)
    Aw0 /= np.sum(Aw0, 1)
    w = Ainv(Aw0)

    J = (1/p)*np.ones((p,p))

    # Initialization of the precision matrix (theta)
    Lw = op.L(w)
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

    for i in range(maxiter):

        # Update w
        LstarLw = op.Lstar(Lw)
        DstarDw = dstar(np.diag(Lw))

        Lstar_S_weighted = np.zeros(int(0.5*p*(p-1)))

        if heavy_type=='student':
            for q in range(n):
                Lstar_S_weighted = Lstar_S_weighted \
                    + Lstar_seq[q] * compute_student_weights(w, Lstar_seq[q], p, nu)
        else:
            for q in range(n):
                Lstar_S_weighted = Lstar_S_weighted + Lstar_seq[q]

        grad = Lstar_S_weighted - op.Lstar(rho*theta + Y) \
            + dstar(y - rho*df) + rho*(LstarLw + DstarDw)

        eta = 1/(2*rho*(2*p - 1))
        wi = w - eta*grad
        wi[wi<0] = 0
        Lwi = op.L(wi)

        # Update theta (precision matrix)
        Z = rho*(Lwi + J) - Y
        U, Lambda, _ = np.linalg.svd(Z)
        D = Lambda + np.sqrt(Lambda**2 + 4*rho)
        thetai = (1/2)*(1/rho)*U@np.diag(D)@U.T - J

        #thetai = U @ np.diag((Lambda + 

        # update Y
        R1 = thetai - Lwi
        Y = Y + rho*R1
        # update y
        R2 = np.diag(Lwi) - df
        y = y + rho*R2

        # compute infinity norm of r for convergence
        primal_residual_R1 = np.linalg.norm(R1, 'fro')
        primal_residual_R2 = np.linalg.norm(R2)
        dual_residual = np.linalg.norm(rho*op.Lstar(theta - thetai))

        lagrangian = compute_augmented_lagrangian(wi, Lwi, Lstar_seq, thetai, J, Y, y,
                                                  df, heavy_type, n, p, rho, nu)

        # update rho
        if update_rho:
            s = rho * np.linalg.norm(op.Lstar(theta - thetai))
            r = np.linalg.norm(R1, 'fro')
            if r > mu*s:
                rho = rho*tau
            elif s > mu*r:
                rho = rho / tau
            else:
                rho = rho

        rel_error_seq.append(np.linalg.norm(Lw - Lwi, 'fro') / np.linalg.norm(Lw, 'fro'))
        has_converged = (rel_error_seq[i] < reltol) and (i > 0)
        elapsed_time = time.time() - start_time

        if has_converged:
            break

        w = wi
        Lw = Lwi
        theta = thetai

    return {"laplacian": op.L(wi), "adjacency": op.A(wi), "precision": thetai,
            "maxiter": i, "convergence": has_converged, "time": elapsed_time,
            "relative_error": rel_error_seq}


def learn_kcomp_heavytail_graph(X,
                                k = 1,
                                heavy_type = "gaussian",
                                nu = None,
                                w0_type = "naive",
                                df = 1,
                                beta = 1e-8,
                                update_beta = True,
                                early_stopping = False,
                                rho = 1,
                                update_rho = False,
                                maxiter = 10000,
                                reltol = 1e-5,
                                verbose = True,
                                record_objective = False):

    n, p = X.shape

    # Instanciate useful classes
    optim = Optimizer()
    op = Operators()

    # store cross-correlations to avoid multiple computation of the same quantity
    xxt = [np.vstack(X[i])@X[i][np.newaxis,] for i in range(n)]
    Lstar_seq = [op.Lstar(xxt[i])/n for i in range(n)]

    # Compute Sample covariance matrix from data
    cov = np.cov(X.T)

    # Compute quantities on the initial guess
    w = optim.w_init(w0_type, np.linalg.pinv(cov))
    Aw0 = op.A(w)
    Aw0 /= np.sum(Aw0, 1)
    w = Ainv(Aw0)

    # Initialization of the precision matrix (theta)
    Lw = op.L(w)
    theta = Lw

    U, _, _ = np.linalg.svd(Lw)
    U = U[:, p-k+1:p]

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

    for i in range(maxiter):

        # Update w
        LstarLw = op.Lstar(Lw)
        DstarDw = dstar(np.diag(Lw))

        Lstar_S_weighted = np.zeros(int(0.5*p*(p-1)))

        if heavy_type=='student':
            for q in range(n):
                Lstar_S_weighted = Lstar_S_weighted \
                    + Lstar_seq[q] * compute_student_weights(w, Lstar_seq[q], p, nu)
        else:
            for q in range(n):
                Lstar_S_weighted = Lstar_S_weighted + Lstar_seq[q]

        grad = Lstar_S_weighted + op.Lstar(beta*U@U.T - Y - rho*theta) \
            + dstar(y - rho*df) + rho*(LstarLw + DstarDw)

    
        eta = 1/(2*rho*(2*p - 1))
        wi = w - eta*grad
        wi[wi<0] = 0
        Lwi = op.L(wi)

        # Update U
        U, _, _ = np.linalg.svd(Lwi)
        U = U[:, p-k+1:p]

        # Update theta (precision matrix)
        Z = rho*Lwi - Y
        U, Lambda, _ = np.linalg.svd(Z)
        U = U[:, :p-k]
        Lambda = Lambda[:p-k]
        D = Lambda + np.sqrt(Lambda**2 + 4*rho)
        thetai = (1/2)*(1/rho)*U@np.diag(D)@U.T

        # update Y
        R1 = thetai - Lwi
        Y = Y + rho*R1
        # update y
        R2 = np.diag(Lwi) - df
        y = y + rho*R2

        # compute infinity norm of r for convergence
        primal_residual_R1 = np.linalg.norm(R1, 'fro')
        primal_residual_R2 = np.linalg.norm(R2)
        dual_residual = np.linalg.norm(rho*op.Lstar(theta - thetai))

        lagrangian = compute_augmented_lagrangian_kcomp(wi, Lwi, Lstar_seq, thetai, U, Y, y, df,
                                                        heavy_type, n, p, k, rho, beta, nu)

        # update rho
        if update_rho:
            s = rho * np.linalg.norm(op.Lstar(theta - thetai))
            r = np.linalg.norm(R1, 'fro')
            if r > mu*s:
                rho = rho*tau
            elif s > mu*r:
                rho = rho / tau
            else:
                rho = rho

        # update beta
        if update_beta:
            _, eigvals, _ = np.linalg.svd(Lwi)
            n_zero_eigenvalues = np.sum(eigvals < 1e-9)
            if k < n_zero_eigenvalues:
                beta = .5*beta
            elif k > n_zero_eigenvalues:
                beta = 2*beta
            else:
                if early_stopping:
                    has_converged = True
                    break
            beta_seq.append(beta)
        
        
        rel_error_seq.append(np.linalg.norm(Lw - Lwi, 'fro') / np.linalg.norm(Lw, 'fro'))
        has_converged = (rel_error_seq[i] < reltol) and (i > 0)
        elapsed_time = time.time() - start_time

        if has_converged:
            break

        w = wi
        Lw = Lwi
        theta = thetai

    return {"laplacian": op.L(wi), "adjacency": op.A(wi), "precision": thetai,
            "maxiter": i, "convergence": has_converged, "time": elapsed_time,
            "relative_error": rel_error_seq}
        
        

n_samples = 100                                 # number of observations
n_components = 3                                # number of components
p = 20                                          # dimension

# graph = block_model([n1,n2,n3], p_in=[0.5,0.4,0.3], p_out=0.01, metadata=True)
# adjacency = erdos_renyi(20, 0.2)
# adj_true = csr_matrix(adjacency, dtype=np.int8).toarray()
# L_true = adj_true * (-1)
# L_inv_true = np.linalg.pinv(L_true)

########################
# Data generation
op = Operators()

# w0 = np.array([1.,1.,1.,1.,1.,1.])/3

# laplacian_true = op.L(w0)
# laplacian_inv_true = np.linalg.pinv(laplacian_true)

# p, _ = laplacian_true.shape

# # Multivariate normal data
# X1 = multivariate_t.rvs(
#     loc=np.zeros(p),
#     shape=laplacian_inv_true,
#     df=np.inf,
#     size=p*100
# )

# results = learn_regular_heavytail_graph(X1, rho=100, heavy_type="student", nu=1e3)

# print('Convergence:', bool(results['convergence']))
# print('Iterations:', results['maxiter'])
# metrics = Metrics(laplacian_true, results["laplacian"])
# print('Rel error:', metrics.relative_error())
# print('F1-score:', metrics.f1_score())


# # Multivariate t-distributed data
# nu = 4
# X2 = multivariate_t.rvs(
#     loc=np.zeros(p),
#     shape=((nu-2)/nu)*laplacian_inv_true,
#     df=nu,
#     size=p*500
# )

# results = learn_regular_heavytail_graph(X2, rho=1, heavy_type="student", nu=4)
# print('Convergence:', bool(results['convergence']))
# print('Iterations:', results['maxiter'])
# metrics = Metrics(laplacian_true, results["laplacian"])
# print('Rel error:', metrics.relative_error())
# print('F1-score:', metrics.f1_score())

# plt.plot(results['relative_error'])
# plt.xlabel('Iteration number')
# plt.ylabel('Relative error')
# plt.show()

# k-component graph learning test
w1 = np.array([1.,1.,1.,1.,1.,1.])/3
w2 = np.array([1.,1.,1.,1.,1.,1.])/3
w3 = np.array([1.,1.,1.,1.,1.,1.])/3
w4 = np.array([1.,1.,1.,1.,1.,1.])/3
w5 = np.array([1.,1.,1.,1.,1.,1.])/3

laplacian_true = block_diag(op.L(w1), op.L(w2), op.L(w3), op.L(w4), op.L(w5))
laplacian_inv_true = np.linalg.pinv(laplacian_true)

p, _ = laplacian_true.shape

# Multivariate gaussian data
# nu = 4
# X3 = multivariate_t.rvs(
#     loc=np.zeros(p),
#     shape=laplacian_inv_true,
#     df=np.inf,
#     size=p*100
# )

# results = learn_kcomp_heavytail_graph(X3, k=5, rho=1e2)

# Multivariate t-distributed data
nu = 4
X3 = multivariate_t.rvs(
    loc=np.zeros(p),
    shape=((nu-2)/nu)*laplacian_inv_true,
    df=nu,
    size=p*500
)

results = learn_kcomp_heavytail_graph(X3, k=5, rho=1e2, heavy_type="student", nu=nu, reltol=1e-4)

# Print results
print('Convergence:', bool(results['convergence']))
print('Iterations:', results['maxiter'])
metrics = Metrics(laplacian_true, results["laplacian"])
print('Rel error:', metrics.relative_error())
print('F1-score:', metrics.f1_score())

plt.plot(results['relative_error'])
plt.xlabel('Iteration number')
plt.ylabel('Relative error')
plt.show()



# d = np.ones(p)                                  # degree vector
# rho = 1                                         # penalty parameter
# df = 5                                          # degree of freedom
# tol = 1e-2                                      # convergence tolerance

# # Initialize dual variables by zero
# Y = np.zeros((p,p))
# y = np.zeros(p)

# iteration = 0

# J = (1/p)*np.ones((p,p))
# I = np.eye(p)
# inf_norm = np.inf

# # compute quantities on the initial guess
# count = 0

# # store cross-correlations to avoid multiple computation of the same quantity
# xxt = [np.vstack(X[i])@X[i][np.newaxis,] for i in range(n_samples)]
# Lstar = [op.Lstar(xxt[i])/(n_samples-1) for i in range(n_samples)]

# ########################
# # Estimation loop
# # --> Diverge pour l'instant !!!
# while (inf_norm > tol):

#     if count == 0:
#         w = w0
#         Lw = Lw0
#         dw = np.diag(Lw0)

#     # Update precision matrix
#     Z = rho*(Lw + J) - Y
#     U, Lambda, _ = np.linalg.svd(Z)
#     D = Lambda + np.sqrt(Lambda**2 + 4*rho*I)
#     precision = (1/2)*(1/rho)*U@D@U.T - J

#     # Update graph weights
#     for j in range(10):
#         # weighted sample covariance matrix
#         total = 0.
#         for i in range(n_samples):
#             total += ((p + df)*xxt[i])/(np.dot(w, Lstar[i]) + df)

#         S = (1/n_samples)*total
#         #print(S)

#         a = op.Lstar(S - Y - rho*(precision - Lw))
#         b = dstar(y - rho*(d - dw))

#         w_new = np.maximum(0., w - (a+b)/(2*rho*(2*p-1)))

#         Lw = op.L(w_new)
#         dw = np.diag(Lw)

#         w = w_new

#     Lw = op.L(w)
#     dw = np.diag(Lw)

#     # Compute residuals
#     r = precision - Lw
#     s = dw - d

#     # Update dual variables Y and y
#     Y_new = Y + rho*r
#     y_new = y + rho*s

#     Y = Y_new
#     y = y_new

#     # compute infinity norm of r for convergence
#     inf_norm = np.linalg.norm(s, np.inf)
#     #print(inf_norm)

#     # compute MLE
#     temp = X@Lw
#     q = np.einsum('ij,ji->i', temp, X.T)

#     MLE = ((df+p)/n_samples)*np.sum(np.log(1+q/df)) -\
#           np.log(np.linalg.det(precision + J))

#     #print(negative_log_likelihood_complex_student_t(X, Lw, df))

#     print(MLE)

#     dot_d = np.dot(y, dw-d)
#     norm_d = np.linalg.norm(dw - d)
#     dot_L = np.dot(Y, precision-Lw)
#     norm_L = np.linalg.norm(precision - Lw, 'fro')

#     #lagrangian = MLE + dot_d + (rho/2)*norm_d + dot_L + (rho/2)*norm_L
#     #print(MLE)

#     count = count + 1

# # --> Calculer le MLE
