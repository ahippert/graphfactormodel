
from fileinput import filename
import os, sys
import numpy as np
import scipy.io

import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import SpectralClustering
from sklearn import cluster

#from sknetwork.embedding import PCA, GSVD
from sknetwork.clustering import get_modularity, Louvain

import networkx.algorithms.community as nx_comm

from src.estimators import SGLkComponents, NGL, GLasso, HeavyTailkGL, EllipticalGL

if __name__ == "__main__":

    # Load animal dataset (Osherson et al., 1991; Lake and Tenenbaum, 2010)
    file_path = os.path.dirname(__file__)
    data_path = os.path.join(file_path, "./data/animals.mat")
    data = scipy.io.loadmat(data_path)

    names_array = data['names']
    # Form list of dictionary in the form [(0: 'name1), (1: 'name2'), ...]
    # Used to label each node
    names_array = np.hstack(np.hstack(names_array))
    dict_names = {}
    for i, name in enumerate(names_array):
        dict_names[i] = name
    X = data['data'].T
    y = data['features']
    n_samples, n_features = X.shape
    n_clusters = 6

    cluster_colors = [
        'red', 'blue', 'black', 'orange',
        'green', 'gray', 'purple', 'cyan',
        'yellow', 'magenta', 'brown'
    ]

    # Pre-processing
    pre_processing = StandardScaler(with_mean=True, with_std=False)

    # Estimators compared
    GLasso = Pipeline(steps=[
            ('Centering', pre_processing),
            ('Graph Estimation', GLasso(alpha=0.05)),
            #('Clustering', louvain(modularity='Newman', n_aggregations=n_clusters))
        ]
    )
    SGL = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Graph Estimation', SGLkComponents(
            None, maxiter=1000, record_objective=True, record_weights=True,
            beta = 0.5, k=8, S_estimation_args=[1./3], verbosity=1)),
        #('Clustering', louvain(modularity='Newman', n_aggregations=n_clusters))
        ]
    )
    
    NGL = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Graph Estimation', NGL(S_estimation_args=[1./3], maxiter=100, record_objective=True)),
        #('Clustering', louvain(shuffle_nodes=True, n_aggregations=n_clusters))
        ]
    )

    StudentGL = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Graph Estimation', HeavyTailkGL(
            heavy_type='student', k=8, nu=1e3)),
        #('Clustering', louvain(shuffle_nodes=True, n_aggregations=n_clusters))
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
        ('Graph Estimation', EllipticalGL(geometry="SPD",
                                          lambda_seq=[0.075],
                                          df=1e3)),
        #('Clustering', louvain(shuffle_nodes=True, n_aggregations=n_clusters))
        ]
    )
    # Modularity=0.44


    list_names = ['GLasso', 'SGL', 'NGL', 'StudentGL',
                  'GGM', 'EGM', 'GGFM', 'EGFM']
    list_pipelines = [GLasso, SGL, NGL, StudentGL,
                      GGM, EGM, GGFM, EGFM]

    # Doing estimation
    for pipeline, name in zip(list_pipelines, list_names):
        print("----------------------------------------------------------")
        print("----------------------------------------------------------")
        print("Doing estimation with :", name)
        print(pipeline)
        pipeline.fit_transform(X)

        # Get adjacency matrix to get the graph and clustered labels to compute modularity
        precision = pipeline['Graph Estimation'].precision_
        adjacency = np.abs(precision)
        if name=='GGFM' or name=='EGFM':
            tol = 1e-2
            adjacency[np.abs(adjacency)<tol] = 0.
        elif name=='GGM' or name=='EGM' or name=='StudentGL':
            tol = 1e-1
            adjacency[np.abs(adjacency)<tol] = 0.
        else:
            pass
        np.fill_diagonal(adjacency, 0.)
        
        graph = nx.from_numpy_array(adjacency)
        graph = nx.relabel_nodes(graph, dict_names)
        
        print('Graph statistics:')
        print('Nodes: ', graph.number_of_nodes(), 'Edges: ', graph.number_of_edges() )
        print('Modularity: ', nx_comm.modularity(graph, nx_comm.greedy_modularity_communities(graph)))

        print("----------------------------------------------------------")
        print('\n\n')

        cluster_seq = nx_comm.label_propagation_communities(graph)
        print(cluster_seq)

        # Put uniform weights
        for u,v,d in graph.edges(data=True):
            d['weight'] = 1.

        nt = Network(height='900px', width='100%')
        #nt.show_buttons()
        nt.set_options("""
            var options = {
            "nodes": {
                "shadow": {
                "enabled": true
                },
                "size": 20,
                "font": {
                "size": 24
                }
            },
            "edges": {
                "arrowStrikethrough": false,
                "color": {
                "inherit": true
                },
                "font": {
                "size": 20
                },
                "shadow": {
                "enabled": true
                },
                "smooth": {
                "type": "continuous",
                "forceDirection": "none",
                "roundness": 1
                },
                "width": 3
            },
            "physics": {
                "forceAtlas2Based": {
                "springLength": 80
                },
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based"
            }
            }
        """)
        nt.from_nx(graph)

        # Adding labels if possible
        if len(cluster_seq) > 0:
            for i_color, i_cluster in enumerate(cluster_seq):
                for i_crypto in i_cluster:
                    nt.get_node(i_crypto)['color'] = cluster_colors[i_color]

        nt.show(f'animaldata_{name}.html')

    plt.show()

