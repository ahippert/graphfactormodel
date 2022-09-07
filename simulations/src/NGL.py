import numpy as np

from StructuredGraphLearning.utils import Operators

import matplotlib.pyplot as plt

import scipy as sp
from scipy.sparse import csr_matrix
from scipy.stats import multivariate_normal, multivariate_t

from utils import invsqrtm

from networkx import laplacian_matrix, barabasi_albert_graph, set_edge_attributes, \
    to_numpy_array, from_numpy_matrix, draw_networkx

def compute_initial_weights(w, Sinv, eta, norm_exit=1e-4, iter_max=50):
    """ This function computes initial graph weights using gradient descent
    """
    iteration = 0
    norm = np.inf

    while norm > norm_exit and iteration < iter_max:
        grad = op.Lstar(op.L(w) - Sinv)
        w_new = w - eta*grad
        w_new[w_new < 0] = 0

        norm = np.linalg.norm(w - w_new)/np.linalg.norm(w)
        w = w_new

        iteration = iteration + 1

    return w + 1e-4

def MCP_dummy(x, lamda, gamma=1.01):
    if x<0:
        print("Error: x must be positive")
    elif x>=0 and x<=gamma*lamda:
        return lamda - x/gamma
    else:
        return 0

def MCP(Lw, lamda, gamma=1.01):
    H = -(lamda + Lw/gamma) * (Lw >= -lamda*gamma)
    np.fill_diagonal(H, 0)
    return H

def SCAD_dummy(x, lamda, gamma=2.01):
    if x<0:
        print("Error: x must be positive")
    elif x>=0 and x<=lamda:
        return lamda
    elif x>=lamda and x<=gamma*lamda:
        return (gamma*lamda - x)/(gamma - 1)
    else:
        return 0

def SCAD(Lw, lamda, gamma=2.01):
    H = -lamda * (Lw >= -lamda)
    H = H + (-gamma * lamda - Lw) / (gamma - 1) * (Lw > -gamma*lamda) * (Lw < -lamda)
    np.fill_diagonal(H, 0)
    return H

def objective_function(Lw, J, K):
    matrix = Lw + J
    chol_factor, pd = sp.linalg.lapack.dpotrf(matrix)
    while pd > 0:
        print("Matrix Lw+J not PD!")
        matrix = matrix + np.eye(matrix.shape[0]) * 0.01
        #chol_factor = np.linalg.cholesky(Lw + J)
        chol_factor, pd = sp.linalg.lapack.dpotrf(matrix)
    return np.sum(Lw*K) - 2*np.sum(np.log(np.diag(chol_factor)))


n_samples = 300                                 # number of observations
n_components = 3                                # number of components
p = 50                                          # dimension
n_edges = 49                                    # number of graph edges

########################
# Initialization terms
lamda = 0.5                                     # lambda in sparsity function
tol = 1e-5                                      # error tolerance
iter_max = 50                                   # maximum iterations of the algorithm
df = 5                                          # degrees of freedom
eta = 1/(2*p)                                   # learning rate
backtrack = True                                # update learning rate using backtrack line search

# Generate Barabas-Albert graph with uniformly sampled weights from U(2,5)
graph = barabasi_albert_graph(p, 1)
set_edge_attributes(graph,
                    {e: {'weight': np.random.uniform(2,5)} for e in graph.edges}
)

plt.figure()
draw_networkx(graph)

# Retrieve true laplacian matrix and convert it to numpy array
Lw_true = csr_matrix.toarray(laplacian_matrix(graph))

# Generate data with a LGMRF model
X = multivariate_normal.rvs(
    mean=np.zeros(p),
    cov=np.linalg.pinv(Lw_true),
    size=n_samples
)

# Sample covariance matrix and its inverse
S = np.cov(X.T)
Sinv = np.linalg.pinv(S)

# Compute initial graph weights
op = Operators()
w0 = op.Linv(Sinv)
w0[w0<0] = 0
w0 = compute_initial_weights(w0, Sinv, eta)
Lw0 = op.L(w0)

# Useful matrices
J = (1/p)*np.ones((p,p))
I = np.eye(p)

# store cross-correlations to avoid multiple computation of the same quantity
#xxt = [np.vstack(X[i])@X[i][np.newaxis,] for i in range(n_samples)]
#Lstar = [op.Lstar(xxt[i]) for i in range(n_samples)]

w = w0
Lw = Lw0
H = MCP(Lw0, lamda)    # Compute sparsity function
K = S + H

########################
# Estimation loop
for i in range(iter_max):

    try:
        gradient = op.Lstar(K - np.linalg.inv(Lw + J))
    except np.linalg.LinAlgError:
        #print("Matrix is non-invertible: might produce errors...")
        pass

    #for t in range(50):
    # total = 0.
    # for i in range(n_samples):
    #     total += ((p + df)*xxt[i])/(np.dot(w, Lstar[i]) + df)
    # S = (1/n_samples)*total

    if backtrack:
        fun = objective_function(Lw, J, K)
        while(1):
            wi = w - eta*gradient
            wi[wi<0] = 0

            Lwi = op.L(wi)
            fun_t = objective_function(Lwi, J, K)

            if (fun < fun_t - np.sum(gradient*(wi-w)) - (.5/eta)*np.linalg.norm(wi-w)**2):
                eta = .5 * eta
            else:
                eta = 2 * eta
                break
    else:
        wi = w - eta * gradient
        wi[wi < 0] = 0

    norm = np.linalg.norm(op.L(wi) - Lw, 'fro')/np.linalg.norm(Lw, 'fro')
    print(norm)

    if (norm < tol and i > 1):
        break

    w = wi
    Lw = op.L(w)
    H = MCP(Lw, lamda)    # Compute sparsity function
    K = S + H

rel_error = np.linalg.norm(Lw - Lw_true, 'fro')/np.linalg.norm(Lw_true, 'fro')
print(rel_error)
adjacency = op.A(w)
graph = from_numpy_matrix(adjacency)
plt.figure()
draw_networkx(graph)
plt.show()
