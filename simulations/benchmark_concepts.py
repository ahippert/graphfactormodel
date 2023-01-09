
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

from sknetwork.clustering import get_modularity

import networkx.algorithms.community as nx_comm

from src.estimators import SGLkComponents, NGL, GLasso, kmeans, louvain, propagation, HeavyTailkGL, EllipticalGL

if __name__ == "__main__":

    # Load animal dataset (Osherson et al., 1991; Lake and Tenenbaum, 2010)
    file_path = os.path.dirname(__file__)
    data_path = os.path.join(file_path, "../data/intel_data.mat")
    data = scipy.io.loadmat(data_path)

    names_array = data['names']
    # Form list of dictionary in the form [(0: 'name1), (1: 'name2'), ...]
    # Used to label each node
    list_names = [(i, names_array[i,0][0]) for i in range(len(names_array))]
    names_array = np.hstack(np.hstack(names_array))
    dict_names = {}
    for i, name in enumerate(names_array):
        dict_names[i] = name
    X = data['data'].T
    y = data['names']
    n_samples, n_features = X.shape
    X = X[:,:500]
    n_clusters = 10

    cluster_colors = [
        'red', 'blue', 'black', 'orange',
        'green', 'gray', 'purple', 'cyan',
        'yellow', 'brown', 'magenta','red', 'blue', 'black', 'orange',
        'green', 'gray', 'purple', 'cyan',
        'yellow', 'brown', 'magenta','red', 'blue', 'black', 'orange',
        'green', 'gray', 'purple', 'cyan',
        'yellow', 'brown', 'magenta'
    ]

    # Pre-processing
    pre_processing = StandardScaler(with_mean=True, with_std=False)

    # Estimators compared
    # GLasso = Pipeline(steps=[
    #     ('Centering', pre_processing),
    #     ('Graph Estimation', GLasso(alpha=0.2, max_iter=1000, verbose=True)),
    #         #('Clustering', louvain(modularity='Newman', n_aggregations=n_clusters))
    #     ]
    # )
    SGL = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Graph Estimation', SGLkComponents(
            None, maxiter=1000, record_objective=True, record_weights=True,
            beta = 0.07, k=10, verbosity=1)),
        #('Clustering', louvain(modularity='Newman', n_aggregations=n_clusters))
        ]
    )
    
    # NGL = Pipeline(steps=[
    #     ('Centering', pre_processing),
    #     ('Graph Estimation', NGL(S_estimation_args=[1./3], maxiter=100, record_objective=True)),
    #     ('Clustering', louvain(shuffle_nodes=True, n_aggregations=n_clusters))
    #     ]
    # )

    # StudentGL = Pipeline(steps=[
    #     ('Centering', pre_processing),
    #     ('Graph Estimation', HeavyTailkGL(
    #         heavy_type='student', k=6, nu=1e3)),
    #     ('Clustering', louvain(shuffle_nodes=True, n_aggregations=n_clusters))
    #     ]
    # )

    EllGL = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Graph Estimation', EllipticalGL(geometry="factor", k=10,
                                          lambda_seq=[0.0045],#[4,5,6,7,8,9,10],
                                          df=1e3, maxiter=1e4)),
        #('Clustering', louvain(shuffle_nodes=True, n_aggregations=n_clusters))
        ]
    )

    list_names = ['EllGL']#, 'NGL', 'StudentGL']
    list_pipelines = [EllGL]#, NGL, StudentGL]

    # Doing estimation
    for pipeline, name in zip(list_pipelines, list_names):
        print("----------------------------------------------------------")
        print("----------------------------------------------------------")
        print("Doing estimation with :", name)
        print(pipeline)
        pipeline.fit_transform(X)

        # Get adjacency matrix to get the graph and clustered labels to compute modularity
        adjacency = pipeline['Graph Estimation'].precision_
        #adjacency = np.abs(adjacency)
        
        graph = nx.from_numpy_matrix(adjacency/100)
        graph = nx.relabel_nodes(graph, dict_names)

        print('Graph statistics:')
        print('Nodes: ', graph.number_of_nodes(), 'Edges: ', graph.number_of_edges() )
        
        print('Modularity: ', nx_comm.modularity(graph, nx_comm.greedy_modularity_communities(graph)))
        print("----------------------------------------------------------")
        print('\n\n')

        cluster_seq = nx_comm.label_propagation_communities(graph)

        # Select graph with components having more than one node
        hidden_nodes = [', '.join(i_cluster) for i_cluster in cluster_seq if len(i_cluster)==1]
        graph.remove_nodes_from(hidden_nodes)

        # Put uniform weights
        for u,v,d in graph.edges(data=True):
            d['weight'] = 1.
        
        nt = Network(height="950px", width='100%')
        #nt.show_buttons()
        nt.set_options("""
            var options = {
            "nodes": {
                "shadow": {
                "enabled": true
                },
                "size": 20,
                "font": {
                "size": 20
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
                "minVelocity": 0.5,
                "solver": "forceAtlas2Based"
            }
            }
        """)
        nt.from_nx(graph)

        # Adding labels if possible
        if len(cluster_seq) > 0:
            i_color = 0
            for i_cluster in cluster_seq:
                if len(i_cluster)>1:
                    for i_crypto in i_cluster:
                        nt.get_node(i_crypto)['color'] = cluster_colors[i_color]
                    i_color += 1

        nt.show(f'conceptdata_{name}.html')

        #import pdfkit
        #pdfkit.from_url(f'conceptdata_{name}.html', f'conceptdata_{name}.pdf')

    plt.show()

