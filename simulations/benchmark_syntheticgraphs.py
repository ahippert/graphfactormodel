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
from mpl_toolkits.axes_grid1 import make_axes_locatable

from networkx import laplacian_matrix, barabasi_albert_graph, set_edge_attributes, \
    erdos_renyi_graph, to_numpy_array, from_numpy_matrix, draw_networkx

from src.estimators import SGLkComponents, NGL, GLasso, HeavyTailkGL, EllipticalGL


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
    weights = np.random.uniform(w_min, w_max)
    set_edge_attributes(graph,
                        {e: {'weight': weights} for e in graph.edges}
                        )
    # Retrieve true laplacian matrix and convert it to numpy array
    return csr_matrix.toarray(laplacian_matrix(graph))
    

if __name__ == "__main__":

    np.random.seed(0)

    n_samples = 30*8                               # number of observations
    n_components = 2                                # number of components
    p = 20                                          # dimension


    # generate various graph Laplacian
    laplacian_true = block_structured_graph(4, 8)
    #laplacian_true = BA_graph(p, 2, 5)
    precision_true = abs(laplacian_true)
    graph_true = from_numpy_matrix(laplacian_true)

    laplacian_ER = ER_graph(7, 0.35, 0, 0.45)
    precision_ER = abs(laplacian_ER)
    precision_noisy = precision_true + precision_ER
    
    # Generate data with a LGMRF model
    X = multivariate_normal.rvs(
        mean=np.zeros(len(laplacian_true)),
        cov=np.linalg.pinv(precision_true),
        size=n_samples
    )

    print(np.linalg.eigh(np.linalg.pinv(laplacian_true)))
    #print(np.linalg.cond(laplacian_true))

    # Estimate sample covariance matrix
    S = empirical_covariance(X, assume_centered=True)
    
    # Pre-processing
    pre_processing = StandardScaler(with_mean=True, with_std=False)

    # Estimators compared
    # Note that GLasso cannot be compared here since it does not estimate a Laplacian matrix within
    # the MTP2 model.

    SGL = Pipeline(steps=[
        #('Centering', pre_processing),
        ('Graph Estimation', SGLkComponents(
            S, maxiter=1000, record_objective=True, record_weights=True,
            alpha=.1, beta=400, k=n_components, verbosity=1)),
        ]
    )
    
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
                                          lambda_seq=[10],
                                          df=5)),
        #('Clustering', louvain(shuffle_nodes=True, n_aggregations=n_clusters))
        ]
    )
    #Modularity=0.8

    GGFM = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Graph Estimation', EllipticalGL(geometry="factor", k=10,
                                          lambda_seq=[3.5],
                                          df=1e3)),
        #('Clustering', louvain(shuffle_nodes=True, n_aggregations=n_clusters))
        ]
    )
    # Modularity=0.79

    EGM = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Graph Estimation', EllipticalGL(geometry="SPD", k=10,
                                          lambda_seq=[0, 0.1, 0.2],
                                          df=5)),
        #('Clustering', louvain(shuffle_nodes=True, n_aggregations=n_clusters))
        ]
    )
    # Modularity=0.44

    GGM = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Graph Estimation', EllipticalGL(geometry="SPD", k=n_components,
                                          lambda_seq=[0],
                                          df=1e3)),
        #('Clustering', louvain(shuffle_nodes=True, n_aggregations=n_clusters))
        ]
    )

    list_names = ['GGM']#, 'NGL', 'StudentGL', 'EllGL']
    #list_names = ['EllGL']
    list_pipelines = [GGM]#, NGL, StudentGL, EllGL]
    #list_pipelines = [EllGL]

    # Doing estimation
    for pipeline, name in zip(list_pipelines, list_names):

        pipeline.fit_transform(X)

        # Get adjacency matrix to get the graph and clustered labels to compute modularity
        adjacency = pipeline['Graph Estimation'].precision_
        #labels = pipeline['Clustering'].labels_
        np.fill_diagonal(adjacency, 0.)
        graph = nx.from_numpy_matrix(adjacency)
        #graph = nx.relabel_nodes(graph, dict_names)

        #laplacian_est = pipeline['Graph Estimation'].results_['laplacian']
        metrics = Metrics(precision_true, adjacency)

        print('Graph statistics:')
        print('Nodes: ', graph.number_of_nodes(), 'Edges: ', graph.number_of_edges() )
        #print('Modularity: ', np.round(get_modularity(adjacency, labels), 2))
        print('Rel error:', metrics.relative_error())
        print('F1-score:', metrics.f1_score())
        print("----------------------------------------------------------")
        print('\n\n')

        nt = Network('100%', '100%')
        #nt.show_buttons()
        nt.set_options("""
        var options = {
        "nodes": {
                "borderWidth": 0,
                "shadow": {
                "enabled": true
                },
                "size": 18
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

        nt.show(f'synthetic_graphs.html')

    precision_est = adjacency
    # Visualize learned Laplacian matrix
    fig, ax = plt.subplots(1, 3, figsize=(9,4))
    laplacian_seq = {"$\Theta_{true}$": precision_true,
                     "$\Theta_{noisy}$": precision_noisy,
                     "$\Theta_{learned}$": precision_est}
    min_val = np.min(list(laplacian_seq.values()))
    max_val = np.max(list(laplacian_seq.values()))
    for i, (key, laplacian) in enumerate(laplacian_seq.items()):
        im = ax[i].imshow(laplacian, vmin=min_val-.1, vmax=max_val+.1, cmap='inferno')
        ax[i].set_title(key)
        ax[i].set_axis_off()

    cax = fig.add_axes([ax[2].get_position().x1+0.02, ax[2].get_position().y0,
                        0.02, ax[2].get_position().height])
   
    plt.colorbar(im, cax=cax)


    # Visualize true graph
    nt = Network('100%', '100%')
    nt.set_options("""
        var options = {
        "nodes": {
                "borderWidth": 0,
                "color": {
                "background": "#F3325B"
                },
                "shadow": {
                "enabled": true
                },
                "size": 18
            },
            "edges": {
                "arrowStrikethrough": false,
                "color": {
                "color": "#F3325B",
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
        
    nt.from_nx(graph_true)
    
    # # Adding labels if possible
    # if hasattr(pipeline[-1], 'labels_'):
    #     for i_animal, i_cluster in enumerate(pipeline[-1].labels_):
    #         nt.nodes[i_animal]['color'] = cluster_colors[i_cluster]

    nt.show(f'true_graph.html')

    plt.show()

    
