from fileinput import filename
import os, sys
import numpy as np
import pyreadr

import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt

from sklearn.covariance import GraphicalLasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.cluster import SpectralClustering

from sknetwork.clustering import get_modularity, Louvain

import networkx.algorithms.community as nx_comm

from src.estimators import SGLkComponents, NGL, GLasso, HeavyTailGL, HeavyTailkGL, EllipticalGL

if __name__ == "__main__":

    # Load crypto dataset (Cardoso et al., 2020)
    file_path = os.path.dirname(__file__)
    data_path = os.path.join(file_path, "../data/crypto-prices")

    data = pyreadr.read_r(data_path)

    names_array = data[None].keys() # get keys of dict
    # Form list of dictionary in the form [(0: 'name1), (1: 'name2'), ...]
    # Used to label each node
    names_array = np.hstack(np.hstack(names_array))
    dict_names = {}
    for i, name in enumerate(names_array):
        dict_names[i] = name

    X = data[None]
    X = np.diff(np.log(X), axis=0)
    
    #y = data['features']
    n_samples, n_features = X.shape
    n_clusters = 7

    from scipy.stats import t
    result = t.fit(X[:,0])
    print(result)

    cluster_colors = [
        'red', 'blue', 'black', 'orange',
        'green', 'gray', 'purple', 'cyan',
        'yellow', 'brown', 'magenta',
        'yellow', 'brown', 'magenta',
        'yellow', 'brown', 'magenta',
        'yellow', 'brown', 'magenta'
    ]

    # Pre-processing
    pre_processing = StandardScaler(with_mean=True, with_std=True)

    # Estimators compared
    # GLasso = Pipeline(steps=[
    #         ('Centering', pre_processing),
    #         ('Graph Estimation', GLasso(alpha=0.05)),
    #         #('Louvain', louvain(shuffle_nodes=True, n_aggregations=n_clusters))
    #     ]
    # )
    SGL = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Graph Estimation', SGLkComponents(
            None, maxiter=1000, record_objective=True, record_weights=True,
            S_estimation_args=[0], k=7, beta=400, verbosity=1))
        ]
    )
    NGL = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Graph Estimation', NGL(maxiter=1000, record_objective=True))
        ]
    )

    StudentGL = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Graph Estimation', HeavyTailkGL(
            heavy_type='student', k=7, nu=3))
        ]
    )

    EllGL = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Graph Estimation', EllipticalGL(geometry="factor", k=7,
                                          lambda_seq=[0.4], df=1e3)),
        #('Clustering', louvain(shuffle_nodes=True, n_aggregations=n_clusters))
        ]
    )
    #modularity=0.78

    EllGL2 = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Graph Estimation', EllipticalGL(geometry="factor", k=7,
                                          lambda_seq=[0.5], df=3)),
        #('Clustering', louvain(shuffle_nodes=True, n_aggregations=n_clusters))
        ]
    )
    #modularity=0.79

    list_names = ['EllGL2']
    list_pipelines = [EllGL2]

    # Doing estimation
    for pipeline, name in zip(list_pipelines, list_names):
        print("----------------------------------------------------------")
        print("----------------------------------------------------------")
        print("Doing estimation with :", name)
        print(pipeline)
        pipeline.fit_transform(X)
        adjacency = pipeline['Graph Estimation'].precision_
        
        graph = nx.from_numpy_matrix(adjacency)
        graph = nx.relabel_nodes(graph, dict_names)
        
        print('Graph statistics:')
        print('Nodes: ', graph.number_of_nodes(), 'Edges: ', graph.number_of_edges() )
        print('Modularity: ', nx_comm.modularity(graph, nx_comm.greedy_modularity_communities(graph)))
        print("----------------------------------------------------------")
        print('\n\n')
        
        cluster_seq = nx_comm.label_propagation_communities(graph)
        print(cluster_seq)
    
        nt = Network(height='900px', width='100%')
        # nt.show_buttons()
        nt.set_options("""
            var options = {
            "nodes": {
                "shadow": {
                "enabled": true
                },
                "size": 23,
                "font": {
                "size": 23
                }
            },
            "edges": {
                "arrowStrikethrough": false,
                "color": {
                "inherit": true
                },
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

        # Adding labels if possible
        #if hasattr(pipeline[-1], 'labels_'):
        if len(cluster_seq) > 0:
            for i_color, i_cluster in enumerate(cluster_seq):
                for i_crypto in i_cluster:
                    nt.get_node(i_crypto)['color'] = cluster_colors[i_color]

        nt.show(f'cryptodata_{name}.html')

    plt.show()
