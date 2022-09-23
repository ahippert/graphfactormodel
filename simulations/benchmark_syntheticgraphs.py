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

import matplotlib.pyplot as plt

from networkx import laplacian_matrix, barabasi_albert_graph, set_edge_attributes, \
    to_numpy_array, from_numpy_matrix, draw_networkx

from src.estimators import SGLkComponents


def S_estimation_method(X, *args):
    return np.dot(X, X.T)

if __name__ == "__main__":

    np.random.seed(0)

    n_samples = 500*9                                 # number of observations
    n_components = 4                                # number of components
    p = 20                                          # dimension

    op = Operators()

    # Generate Barabas-Albert graph with uniformly sampled weights from U(2,5)
    # graph = barabasi_albert_graph(p, 1)
    # set_edge_attributes(graph,
    #                     {e: {'weight': np.random.uniform(2,5)} for e in graph.edges}
    #                     )
    # plt.figure()
    # draw_networkx(graph)

    # # Retrieve true laplacian matrix and convert it to numpy array
    # Lw_true = csr_matrix.toarray(laplacian_matrix(graph))

    # k-component graph learning test
    w1 = np.random.uniform(0, 1, 3)
    w2 = np.random.uniform(0, 1, 6)

    laplacian_true = block_diag(op.L(w1), op.L(w2))
    #np.fill_diagonal(laplacian_true, 0.)
    
    precision_true = abs(laplacian_true)

    plt.figure()
    plt.imshow(precision_true)

    graph = from_numpy_matrix(laplacian_true)
    plt.figure()
    draw_networkx(graph)
    
    # Generate data with a LGMRF model
    X = multivariate_normal.rvs(
        mean=np.zeros(len(laplacian_true)),
        cov=np.linalg.pinv(laplacian_true),
        size=n_samples
    )

    #
    S = empirical_covariance(X, assume_centered=True)
    
    # Pre-processing
    pre_processing = StandardScaler(with_mean=True, with_std=False)

    SGL = Pipeline(steps=[
        #('Centering', pre_processing),
        ('Graph Estimation', SGLkComponents(
            S, maxiter=1000, record_objective=True, record_weights=True,
            beta = 1e4, k=2, verbosity=1)),
        ]
    )

    list_names = ['SGL']
    list_pipelines = [SGL]

    # Doing estimation
    for pipeline, name in zip(list_pipelines, list_names):

        pipeline.fit_transform(X)

        # Get adjacency matrix to get the graph and clustered labels to compute modularity
        adjacency = pipeline['Graph Estimation'].results_['adjacency']
        #labels = pipeline['Clustering'].labels_
        graph = nx.from_numpy_matrix(adjacency)
        # graph = nx.relabel_nodes(graph, dict_names)

        laplacian_est = pipeline['Graph Estimation'].results_['laplacian']
        metrics = Metrics(laplacian_true, laplacian_est)

        print('Graph statistics:')
        print('Nodes: ', graph.number_of_nodes(), 'Edges: ', graph.number_of_edges() )
        #print('Modularity: ', np.round(get_modularity(adjacency, labels), 2))
        print('Rel error:', metrics.relative_error())
        print('F1-score:', metrics.f1_score())
        print("----------------------------------------------------------")
        print('\n\n')

        nt = Network('100%', '100%')
        # nt.show_buttons()
        nt.set_options("""
        var options = {
        "nodes": {
                "shadow": {
                "enabled": true
                },
                "size": 23
            },
            "edges": {
                "arrowStrikethrough": false,
                "color": {
                "inherit": true
                },
                "dashes": true,
                "font": {
                "size": 0
                },
                "shadow": {
                "enabled": true
                },
                "smooth": {
                "type": "continuous",
                "forceDirection": "none",
                "roundness": 1
                }
            },
            "physics": {
                "forceAtlas2Based": {
                "springLength": 100
                },
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based"
            }
            }
        """)
        nt.from_nx(graph)

        # # Adding labels if possible
        # if hasattr(pipeline[-1], 'labels_'):
        #     for i_animal, i_cluster in enumerate(pipeline[-1].labels_):
        #         nt.nodes[i_animal]['color'] = cluster_colors[i_cluster]

        nt.show(f'animaldata_{name}.html')

    plt.show()

    
