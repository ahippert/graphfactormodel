import numpy as np

from StructuredGraphLearning.utils import Operators
from StructuredGraphLearning.metrics import Metrics

import networkx as nx
from pyvis.network import Network

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

GLasso = Pipeline(steps=[('Graph Estimation', GLasso(alpha=0.25))]
                  )
    
NGL = Pipeline(steps=[('Graph Estimation', NGL(lamda=0.5, maxiter=1000, record_objective=True))]
               )

StGL = Pipeline(steps=[
    ('Graph Estimation', HeavyTailkGL(heavy_type='student', maxiter=1000, nu=3))]
                )

def one_monte_carlo(trial_no, precision, n, p, lambda_EGFM, lambda_GGFM, lambda_EGM, lambda_GGM):

    np.random.seed(trial_no)

    # X = multivariate_normal.rvs(
    #         mean=np.zeros(len(precision)),
    #         cov=np.linalg.pinv(precision),
    #         size=n,
    #         random_state=np.random.RandomState(trial_no)
    # )


    X = multivariate_t.rvs(
        loc=np.zeros(len(precision)),
        shape=np.linalg.pinv(precision),
        df=3,
        size=n,
        random_state=np.random.RandomState(trial_no)
    )

    # Pre-processing
    pre_processing = StandardScaler(with_mean=True, with_std=False)
    

    # Estimate sample covariance matrix
    S = empirical_covariance(X, assume_centered=True)
    
    # Pre-processing
    pre_processing = StandardScaler(with_mean=True, with_std=False)

    # Estimators compared
    # SGL is the only one changing, it's why it is here
    SGL = Pipeline(steps=[
        #('Centering', pre_processing),
        ('Graph Estimation', SGLkComponents(
            S, maxiter=1000, record_objective=True, record_weights=True,
            alpha=.1, beta=1000, verbosity=1)),
        ]
    )
    
    EGFM = Pipeline(steps=[
        #('Centering', pre_processing),
        ('Graph Estimation', EllipticalGL(geometry="factor", k=10,
                                          lambda_seq=[0.1],
                                          df=3, verbosity=False)),
        ]
    )

    GGFM = Pipeline(steps=[
        #('Centering', pre_processing),
        ('Graph Estimation', EllipticalGL(geometry="factor", k=10,
                                          lambda_seq=[0.1], df=100,
                                          verbosity=False)),
        ]
    )

    EGM = Pipeline(steps=[
        #('Centering', pre_processing),
        ('Graph Estimation', EllipticalGL(geometry="SPD", k=10,
                                          lambda_seq=[0.1],
                                          df=3, verbosity=False)),
        ]
    )

    GGM = Pipeline(steps=[
        #('Centering', pre_processing),
        ('Graph Estimation', EllipticalGL(geometry="SPD", k=10,
                                          lambda_seq=[0.1],#1e-4,5e-4,1e-3,5e-3,1e-2],
                                          df=100, verbosity=False)),
        ]
    )


    pipeline_seq = [EGFM, GGFM, EGM, GGM, StGL, SGL]
    pipeline_seq = [NGL]
    delta_seq = []

    for pipeline in pipeline_seq:
        pipeline.fit_transform(X)

        # Get adjacency matrix to get the graph and clustered labels to compute modularity
        precision_est = pipeline['Graph Estimation'].precision_
        
        d = 1/np.sqrt(np.diag(precision_est))
        precision_est = np.diag(d) @ precision_est @ np.diag(d)

        d = 1/np.sqrt(np.diag(precision))
        precision_true = np.diag(d) @ precision @ np.diag(d)

        # ça fait juste remonter l'erreur !
        #np.fill_diagonal(precision_est, 0.) 
        #np.fill_diagonal(precision_true, 0.)

        error = np.linalg.norm(precision_true - precision_est, 'fro')/np.linalg.norm(precision_true, 'fro')
        
        #metrics = Metrics(precision_true, adjacency)
        
        delta_seq.append(error)
    return delta_seq


def parallel_monte_carlo(precision, n, p, lambda_EGFM, lambda_GGFM, lambda_EGM, lambda_GGM, n_threads, n_trials, Multi):

    # Looping on Monte Carlo Trials
    if Multi:
        results_parallel = Parallel(n_jobs=n_threads)(delayed(one_monte_carlo)(iMC, precision, n, p, lambda_EGFM, lambda_GGFM, lambda_EGM, lambda_GGM) for iMC in range(n_trials))
        results_parallel = np.array(results_parallel)
        return results_parallel
    else:
        # Results container
        results = []
        for iMC in range(n_trials):
            results.append(one_monte_carlo(iMC, precision, n, p, pipelines))
        results = np.array(results)
        return results

if __name__ == "__main__":

    np.random.seed(0)

    n_samples = 50*5                               # number of observations
    n_components = 10                              # number of components
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

    # pipeline_seq = [EGFM, GGFM, EGM, GGM, StGL]
    # pipeline_seq = [GLasso, NGL]
    # pipeline_seq = [EGFM, GGFM, EGFM2, GGFM2]

    
    # Load sparse SPD matrix (for reproducibility)
    #precision_true = np.load("precision_true_erdos_reyni.npy")
    precision_true = np.load("precision_true_erdos_reyni_mtp2_best.npy")
    
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

    # lambda_seq_EGFM = np.array([0.17, 0.15, 0.08, 0.02, 0.01, 0.007, 0.005])
    # lambda_seq_GGFM = np.array([0.1, 0.08, 0.04, 0.01, 0.004, 0.003, 0.002])
    # lambda_seq_EGM = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    # lambda_seq_GGM = np.array([0.11, 0.09, 0.08, 0.06, 0.045, 0.04])

    lambda_seq_EGFM = np.array([0.3, 0.1, 0.07, 0.03, 0.01, 0.009])
    lambda_seq_GGFM = np.array([0.6, 0.45, 0.3, 0.27, 0.2, 0.1])
    lambda_seq_EGM = np.array([0.2, 0.15, 0.12, 0.08, 0.065, 0.06])
    lambda_seq_GGM = np.array([0.34, 0.32, 0.3, 0.2, 0.1, 0.1])
    
    rank_seq = np.unique(np.logspace(0.5,1.4,6).astype(int)) 
    error_seq = []
                           
    number_of_threads = -1              # to use the maximum number of threads
    Multi = True
    sample_seq = np.unique(np.logspace(1,2.5,5).astype(int))
    sample_seq = np.array([120, 140, 170]) #n=[2*k, p, 2*p, 4*p]
    #sample_seq = np.array([20, 50, 120]) #n=[2*k, p, 2*p, 4*p]
    number_of_trials = 100

    # # # -------------------------------------------------------------------------------
    # # # Doing estimation for an increasing N and saving the distance to true value
    # # # -------------------------------------------------------------------------------
    print( '|￣￣￣￣￣￣￣￣￣￣|')
    print( '|   Launching        |')
    print( '|   Monte Carlo      |')
    print( '|   simulation       |' )
    print( '|＿＿＿＿＿＿＿＿＿＿|')
    print( ' (\__/) ||')
    print( ' (•ㅅ•) || ')
    print( ' / 　 づ')
    print(u"Parameters: p=%d, n=%s" % (p, sample_seq))

    t_beginning = time.time()

    # Distance container cube
    number_of_δ = 1
    δ_precision_container = np.zeros((len(sample_seq), number_of_trials, number_of_δ))
    #δ_precision_container = np.zeros((len(sample_seq), number_of_trials, number_of_δ))
    
    for i_n, n in enumerate(tqdm(sample_seq)):
        δ_precision_container[i_n] = parallel_monte_carlo(precision_true, sample_seq[i_n], p, lambda_seq_EGFM[i_n], lambda_seq_GGFM[i_n], lambda_seq_GGM[i_n], lambda_seq_GGM[i_n], number_of_threads, number_of_trials, Multi)
            

    print(δ_precision_container)
    print('Done in %f s'%(time.time()-t_beginning))

    np.save("delta_samples_student_NGL.npy", δ_precision_container)
    # plt.figure()
    # labels = ["EGFM", "GGFM", "EGM", "GGM", "StGL", "SGL"]
    # for n in range(number_of_δ):
    #     plt.loglog(sample_seq,
    #                np.nanmean(δ_precision_container[:,:,n], axis=1),
    #                marker='o', label=labels[n])
    # plt.legend()
    # plt.show()
    
# # Doing estimation
#     for pipeline, name in zip(list_pipelines, list_names):

#         for i, lamb in enumerate(lambda_seq):

#             for i, n_samples in enumerate(sample_seq):

#                 X = multivariate_normal.rvs(
#                     mean=np.zeros(len(precision_true)),
#                     cov=np.linalg.pinv(precision_true),
#                     size=n_samples
#                 )
                
#                 pipeline['Graph Estimation'].lambda_seq = [lamb]
#                 pipeline.fit_transform(X)

#                 # Get adjacency matrix to get the graph and clustered labels to compute modularity
#                 adjacency = pipeline['Graph Estimation'].precision_
#                 #labels = pipeline['Clustering'].labels_
#                 #np.fill_diagonal(adjacency, 0.)
#                 graph = nx.from_numpy_matrix(adjacency)
#                 #graph = nx.relabel_nodes(graph, dict_names)

#                 #laplacian_est = pipeline['Graph Estimation'].results_['laplacian']
#                 metrics = Metrics(precision_true, adjacency)

#                 print('Graph statistics:')
#                 print('Nodes: ', graph.number_of_nodes(), 'Edges: ', graph.number_of_edges() )
#                 #print('Modularity: ', np.round(get_modularity(adjacency, labels), 2))
#                 error_seq.append(metrics.relative_error())
#                 print('Rel error:', error_seq[i])
#                 print('F1-score:', metrics.f1_score())
#                 print("----------------------------------------------------------")
#                 print('\n\n')
#     plt.plot(error_seq)
#     plt.show()

            # nt = Network('100%', '100%')
    #         #nt.show_buttons()
    #         nt.set_options("""
    #         var options = {
    #         "nodes": {
    #         "borderWidth": 0,
    #             "shadow": {
    #             "enabled": true
    #             },
    #             "size": 18
    #         },
    #         "edges": {
    #             "arrowStrikethrough": false,
    #             "color": {
    #             "inherit": true
    #             },
    #             "dashes": true,
    #             "font": {
    #             "size": 0
    #             },
    #             "shadow": {
    #             "enabled": true
    #             },
    #             "smooth": {
    #             "type": "continuous",
    #             "forceDirection": "none",
    #             "roundness": 1
    #             }
    #         },
    #         "physics": {
    #             "forceAtlas2Based": {
    #             "springLength": 100
    #             },
    #             "minVelocity": 0.75,
    #             "solver": "forceAtlas2Based"
    #         }
    #         }
    #         """)
        
    #         nt.from_nx(graph)

    #         # # Adding labels if possible
    #         # if hasattr(pipeline[-1], 'labels_'):
    #         #     for i_animal, i_cluster in enumerate(pipeline[-1].labels_):
    #         #         nt.nodes[i_animal]['color'] = cluster_colors[i_cluster]

    #         nt.show(f'synthetic_graphs.html')

    # precision_est = adjacency
    # # Visualize learned Laplacian matrix
    # fig, ax = plt.subplots(1, 2, figsize=(9,4))
    # laplacian_seq = {"$\Theta_{true}$": precision_true,
    #                  #"$\Theta_{noisy}$": precision_noisy,
    #                  "$\Theta_{learned}$": precision_est}
    # min_val = np.min(list(laplacian_seq.values()))
    # max_val = np.max(list(laplacian_seq.values()))
    # for i, (key, laplacian) in enumerate(laplacian_seq.items()):
    #     im = ax[i].imshow(laplacian, vmin=min_val-.1, vmax=max_val+.1, cmap='inferno')
    #     ax[i].set_title(key)
    #     ax[i].set_axis_off()

    # cax = fig.add_axes([ax[1].get_position().x1+0.02, ax[1].get_position().y0,
    #                     0.02, ax[1].get_position().height])
   
    # plt.colorbar(im, cax=cax)


    # # Visualize true graph
    # nt = Network('100%', '100%')
    # nt.set_options("""
    #     var options = {
    #     "nodes": {
    #             "borderWidth": 0,
    #             "color": {
    #             "background": "#F3325B"
    #             },
    #             "shadow": {
    #             "enabled": true
    #             },
    #             "size": 18
    #         },
    #         "edges": {
    #             "arrowStrikethrough": false,
    #             "color": {
    #             "color": "#F3325B",
    #             "inherit": true
    #             },
    #             "dashes": true,
    #             "font": {
    #             "size": 0
    #             },
    #             "shadow": {
    #             "enabled": true
    #             },
    #             "smooth": {
    #             "type": "continuous",
    #             "forceDirection": "none",
    #             "roundness": 1
    #             }
    #         },
    #         "physics": {
    #             "forceAtlas2Based": {
    #             "springLength": 100
    #             },
    #             "minVelocity": 0.75,
    #             "solver": "forceAtlas2Based"
    #         }
    #         }
    # """)
        
    # nt.from_nx(graph_true)

    # nt.show(f'true_graph.html')

    #plt.show()

    
