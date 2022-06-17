import numpy as np

from StructuredGraphLearning.utils import Operators

import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, make_circles

# from sknetwork.data import erdos_renyi
# from sknetwork.visualization import svg_graph

from scipy.sparse import csr_matrix
from scipy.stats import multivariate_normal

import visualization as vis


# n = 30  # number of nodes per cluster
# k = 2   # number of components
# dataset = make_moons(n_samples=n*k, noise=.05, shuffle=True)
# X, y = dataset
# X = X - np.mean(X, axis=0)

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

X = multivariate_normal.rvs(
    mean=np.zeros(p),
    cov=np.eye(p),#L_inv_true,
    size=n_samples
)
covariance = np.cov(X.T)


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

########################
# Initialization terms
w0 = np.random.uniform(0.1, 3, size=p*(p-1)//2) # initial estimate of graph weights
d = np.ones(p)                                  # degree vector
rho = 1                                         # penalty parameter
nu = 5                                          # degree of freedom
tol = 1e-2                                      # convergence tolerance

# Initialize dual variables by zero
Y = np.zeros((p,p))
y = np.zeros(p)

iteration = 0

J = (1/p)*np.ones((p,p))
I = np.eye(p)
inf_norm = np.inf

# compute quantities on the initial guess
op = Operators()
Lw0 = op.L(w0)
count = 0

# store cross-correlations to avoid multiple computation of the same quantity
xxt = [np.vstack(X[i])@X[i][np.newaxis,] for i in range(n_samples)]
Lstar = [op.Lstar(xxt[i]) for i in range(n_samples)]

########################
# Estimation loop
# --> Diverge pour l'instant !!!
while (inf_norm > tol):

    if count == 0:
        w = w0
        Lw = Lw0
        dw = np.diag(Lw0)

    # Update precision matrix
    Z = rho*(Lw + J) - Y
    U, Lambda, _ = np.linalg.svd(Z)
    D = Lambda + np.sqrt(Lambda**2 + 4*rho*I)
    precision = (1/2)*(1/rho)*U@D@U.T - J

    # Update graph weights
    for j in range(20):
        # weighted sample covariance matrix
        total = 0.
        for i in range(n_samples):
            #total += ((p + nu)*xxt[i])/(np.dot(w, Lstar[i]) + nu)
            total += xxt[i]

        S = (1/n_samples)*total

        a = op.Lstar(S - Y - rho*(precision - Lw))
        b = dstar(y - rho*(d - dw))

        w_new = np.maximum(0., w - (a+b)/(2*rho*(2*p-1)))

        Lw = op.L(w_new)
        dw = np.diag(Lw)

        w = w_new

    Lw = op.L(w)
    dw = np.diag(Lw)

    # Compute residuals
    r = precision - Lw
    s = dw - d

    # Update dual variables Y and y
    Y_new = Y + rho*r
    y_new = y + rho*s

    Y = Y_new
    y = y_new

    # compute infinity norm of r for convergence
    inf_norm = np.linalg.norm(s, np.inf)
    print(inf_norm)

    # weighted sample covariance matrix
    # tot = 0.
    # for i in range(n_samples):
    #     xLwx = X[i][np.newaxis,]@Lw@np.vstack(X[i])
    #     tot += np.log(1 + xLwx/nu)

    # # voir si c'est "precision" ou Lw qu'il faut mettre
    # MLE = ((p+nu)/n_samples)*tot - np.log(np.linalg.det(precision + J))
    # print(MLE)

    count = count + 1

# --> Calculer le MLE
