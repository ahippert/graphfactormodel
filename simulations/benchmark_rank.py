import numpy as np

from StructuredGraphLearning.utils import Operators
from StructuredGraphLearning.metrics import Metrics

import networkx as nx

from scipy.sparse import csr_matrix
from scipy.stats import multivariate_normal, multivariate_t
from scipy.linalg import block_diag

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.covariance import empirical_covariance
from sklearn.datasets import make_sparse_spd_matrix

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from networkx import laplacian_matrix, adjacency_matrix, barabasi_albert_graph, \
    set_edge_attributes, erdos_renyi_graph, to_numpy_array, from_numpy_matrix, draw_networkx

from src.estimators import SGLkComponents, NGL, GLasso, HeavyTailkGL, EllipticalGL

from joblib import Parallel, delayed
from tqdm import tqdm
import time

def S_estimation_method(X, *args):
    return np.dot(X, X.T)

def block_structured_graph(n1, n2):
    """ 
    """
    op = Operators()
    w1 = np.random.uniform(0, 1, n1)
    w2 = np.random.uniform(0, 1, n2)
    return block_diag(op.L(w1), op.L(w2))

def BA_graph(n, w_min, w_max):
    """Generate Barabasi-Albert graph with sampled weights from U(w_min, w_max)
    """
    graph = barabasi_albert_graph(n, 1)
    weights = np.random.uniform(w_min, w_max)
    set_edge_attributes(graph,
                        {e: {'weight': weights} for e in graph.edges}
                        )
    # Retrieve true laplacian matrix and convert it to numpy array
    return csr_matrix.toarray(laplacian_matrix(graph))

def ER_graph(n, p, w_min, w_max):
    """Generate Erdos-Renyi graph with sampled weights from U(w_min, w_max)
    """
    graph = erdos_renyi_graph(n, p)
    weights = np.random.uniform(w_min, w_max, size=len(graph.edges))
    set_edge_attributes(graph,
                        {e: {'weight': weights[i]} for i, e in enumerate(graph.edges)}
                        )
    # Retrieve true laplacian and adjacency matrices and convert it to numpy array
    laplacian = csr_matrix.toarray(laplacian_matrix(graph))
    adjacency = csr_matrix.toarray(adjacency_matrix(graph))
    return {"laplacian": laplacian, "adjacency": adjacency}

def generate_gmrf(precision, n):
    covariance = np.linalg.inv(precision)
    X = np.random.multivariate_normal(np.zeros(len(precision)), covariance, n)
    return X

# def generate_studentmrf(precision, n, df):
#     covariance = np.linalg.inv(precision)
#     X = multivariate_t(np.zeros(len(precision), covariance, n)))
#     return X
                                      

#multivariate_normal.rvs(
#        mean=np.zeros(len(precision)),
#        cov=np.linalg.pinv(precision),
#        size=n
#    )

def one_monte_carlo_rank(trial_no, precision, n_seq, p, lamb, rank):

    np.random.seed(trial_no)

    # Pre-processing
    pre_processing = StandardScaler(with_mean=True, with_std=False)

    # Estimators compared
    EGFM = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Graph Estimation', EllipticalGL(geometry="factor", k=rank,
                                          lambda_seq=[lamb],
                                          df=5, verbosity=False)),
        #('Clustering', louvain(shuffle_nodes=True, n_aggregations=n_clusters))
        ]
    )
    
    pipeline = EGFM

    delta_seq = []

    for n in n_seq:
        X = multivariate_t.rvs(
            loc=np.zeros(len(precision)),
            shape=np.linalg.inv(precision),
            df=5,
            size=n,
            random_state=np.random.RandomState(n*trial_no)
        )

        pipeline.fit_transform(X)

        # Get adjacency matrix and metrics
        adjacency = pipeline['Graph Estimation'].precision_
        metrics = Metrics(precision_true, adjacency)
        delta_seq.append(metrics.relative_error())
        f1_score = metrics.f1_score()
    
    return delta_seq

def parallel_monte_carlo(precision, n, p, lamb, rank, n_threads, n_trials, Multi):

    # Looping on Monte Carlo Trials
    if Multi:
        results_parallel = Parallel(n_jobs=n_threads)(delayed(one_monte_carlo_rank)(iMC, precision, n, p, lamb, rank) for iMC in range(n_trials))
        results_parallel = np.array(results_parallel)
        return results_parallel
    else:
        # Results container
        results = []
        for iMC in range(n_trials):
            results.append(one_monte_carlo(iMC, precision, n, p))
        results = np.array(results)
        return results

if __name__ == "__main__":

    np.random.seed(0)

    n_samples = 50*5                               # number of observations
    n_components = 2                               # number of components
    p = 50                                         # dimension

    #precision_true = make_sparse_matrix(p, 0.8, -.9, True)

    # Generate sparse SPD matrix
    # prng = np.random.RandomState(1)
    # precision_true = make_sparse_spd_matrix(
    #     p, alpha=0.9, norm_diag=False, smallest_coef=0.3,
    #     largest_coef=0.7, random_state=prng
    # )
    # np.fill_diagonal(precision_true, 0.)
    # precision_true = -np.abs(precision_true)
    # np.fill_diagonal(precision_true, np.max(np.sum(abs(precision_true), axis=1)) + 0.01)

    # adjacency = ER_graph(p, 0.05, 0, 0.45)["adjacency"]
    # prec_tilde = (1+1e-3)*np.max(np.linalg.eigvals(adjacency))*np.eye(p) - adjacency
    # prec_inv = np.linalg.inv(prec_tilde)
    # D = np.diag(np.sqrt(np.diag(prec_inv)))
    # prec_true = D@prec_tilde@D


    # Pre-processing
    pre_processing = StandardScaler(with_mean=True, with_std=False)

    NGL = Pipeline(steps=[
        #('Centering', pre_processing),
        ('Graph Estimation', NGL(maxiter=1000, record_objective=True)),
        ]
    )

    StudentGL = Pipeline(steps=[
        ('Graph Estimation', HeavyTailkGL(
            heavy_type='student', k=n_components, nu=1e3))
        ]
    )

    EGFM = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Graph Estimation', EllipticalGL(geometry="factor", k=10,
                                          lambda_seq=[1e-4],
                                          df=5, verbosity=False)),
        ]
    )

    GGFM = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Graph Estimation', EllipticalGL(geometry="factor", k=10,
                                          lambda_seq=[3.5],
                                          df=1e3)),
        ]
    )

    EGM = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Graph Estimation', EllipticalGL(geometry="SPD", k=10,
                                          lambda_seq=[0, 0.1, 0.2],
                                          df=5)),
        ]
    )

    GGM = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Graph Estimation', EllipticalGL(geometry="SPD", k=n_components,
                                          lambda_seq=[1e-4],#1e-4,5e-4,1e-3,5e-3,1e-2],
                                          df=1e3, verbosity=False)),
        ]
    )

    pipeline_seq = [StudentGL, EGFM]

    
    # Load sparse SPD matrix (for reproducibility)
    #precision_true = np.load("precision_true_erdos_reyni.npy")
    precision_true = np.load("precision_true_erdos_reyni_mtp2_best.npy")
    d = 1/np.sqrt(np.diag(precision_true))
    precision_true = np.diag(d) @ precision_true @ np.diag(d)
    #plt.imshow(precision_true)
    #precision_true = prec_true
    
    #cov = linalg.inv(prec)
    #d = np.sqrt(np.diag(cov))
    #precision_true *= d
    #precision_true *= d[:, np.newaxis]

    # generate various graph Laplacian
    # laplacian_true = block_structured_graph(4, 8)
    # #laplacian_true = BA_graph(p, 2, 5)
    # precision_true = abs(laplacian_true)
    graph_true = from_numpy_matrix(precision_true)

    #laplacian_ER = ER_graph(7, 0.35, 0, 0.45)
    #precision_ER = abs(laplacian_ER)
    #precision_noisy = precision_true + precision_ER

    #list_names = ['GGM']#, 'NGL', 'StudentGL', 'EllGL']
    #list_names = ['EllGL']
    #list_pipelines = [GGM]#, NGL, StudentGL, EllGL]
    #list_pipelines = [EllGL]

    lambda_seq = np.linspace(0,1,6)
    lambda_seq = np.logspace(-5,0,6)
    lambda_seq = np.logspace(-3,-.5,10)
    lamb = 0.1
    
    rank_seq = np.unique(np.logspace(0.5,1.4,6).astype(int)) 
    error_seq = []
                           
    number_of_threads = -1              # to use the maximum number of threads
    Multi = True
    sample_seq = np.unique(np.logspace(1,2.5,5).astype(int))
    sample_seq = np.array([25, 50, 100, 150]) #n=[2*k, p, 2*p, 4*p]
    number_of_trials = 50

    # # # -------------------------------------------------------------------------------
    # # # Doing estimation for an increasing N and saving the natural distance to true value
    # # # -------------------------------------------------------------------------------
    print( '|￣￣￣￣￣￣￣￣￣￣|')
    print( '|   Launching        |')
    print( '|   Monte Carlo      |')
    print( '|   simulation       |' )
    print( '|＿＿＿＿＿＿＿＿＿＿|')
    print( ' (\__/) ||')
    print( ' (•ㅅ•) || ')
    print( ' / 　 づ')
    print(u"Parameters: p=%d, n=%s" % (p, rank_seq))

    t_beginning = time.time()

    # Distance container cube
    number_of_δ = 1
    δ_precision_container = np.zeros((len(rank_seq), number_of_trials, len(sample_seq)))
    #δ_precision_container = np.zeros((len(sample_seq), number_of_trials, number_of_δ))
    
    for i_l, n in enumerate(tqdm(rank_seq)):
        δ_precision_container[i_l] = parallel_monte_carlo(precision_true, sample_seq, p, lamb, rank_seq[i_l], number_of_threads, number_of_trials, Multi)
            

    print(δ_precision_container)
    print('Done in %f s'%(time.time()-t_beginning))

    np.save("delta_rank.npy", δ_precision_container)
    plt.figure()
    labels = ["n=p/2", "n=p", "n=2p", "n=4p"]
    #labels_pipeline = ["StGL", "EGFM", "SGL"]
    for n in range(len(sample_seq)):
        plt.loglog(rank_seq,
                   np.mean(δ_precision_container[:,:,n], axis=1),
                   marker='o', label=labels[n])
    plt.legend()
    plt.show()
