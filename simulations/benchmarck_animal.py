
from fileinput import filename
import os, sys
import numpy as np
import scipy.io

import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt

from sklearn.covariance import GraphicalLasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.cluster import SpectralClustering

from sknetwork.embedding import PCA
from src.estimators import SGLkComponents, GLasso, kmeans, louvain


if __name__ == "__main__":

    # Load animal dataset (Osherson et al., 1991; Lake and Tenenbaum, 2010)
    file_path = os.path.dirname(__file__)
    data_path = os.path.join(file_path, "../data/animals.mat")
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
    y = data['features']
    n_samples, n_features = X.shape
    n_clusters = 6

    cluster_colors = [
        'red', 'blue', 'black', 'orange',
        'green', 'gray', 'purple', 'cyan',
        'yellow'
    ]

    # Pre-processing
    pre_processing = StandardScaler(with_mean=True, with_std=False)

    # Estimators compared
    # GLasso = Pipeline(steps=[
    #         ('Centering', pre_processing),
    #         ('Graph Estimation', GLasso(alpha=0.05)),
    #         ('KMeans', kmeans(n_clusters=n_clusters, embedding_method=PCA()))
    #     ]
    # )
    # SGL = Pipeline(steps=[
    #         ('Centering', pre_processing),
    #         ('Graph Estimation', SGLkComponents(
    #             None, maxiter=1000, record_objective=True, record_weights=True,
    #             beta = 0.5, k=n_clusters, S_estimation_args=[1./3], verbosity=1
    #         )),
    #         ('KMeans', kmeans(n_clusters=n_clusters, embedding_method=PCA()))
    #     ]
    # )
    GLasso = Pipeline(steps=[
            ('Centering', pre_processing),
            ('Graph Estimation', GLasso(alpha=0.05)),
            ('Louvain', louvain(shuffle_nodes=True, n_aggregations=n_clusters))
        ]
    )
    SGL = Pipeline(steps=[
            ('Centering', pre_processing),
            ('Graph Estimation', SGLkComponents(
                None, maxiter=1000, record_objective=True, record_weights=True,
                beta = 0.5, k=n_clusters, S_estimation_args=[1./3], verbosity=1
            )),
            ('Louvain', louvain(shuffle_nodes=True, n_aggregations=n_clusters))
        ]
    )

    list_names = ['GLasso', 'SGL']
    list_pipelines = [GLasso, SGL]

    # Doing estimation
    for pipeline, name in zip(list_pipelines, list_names):
        print("----------------------------------------------------------")
        print("----------------------------------------------------------")
        print("Doing estimation with :", name)
        print(pipeline)
        pipeline.fit(X)
        graph = nx.from_numpy_matrix(pipeline['Graph Estimation'].precision_)
        graph = nx.relabel_nodes(graph, dict_names)
        print('Graph statistics:')
        print('Nodes: ', graph.number_of_nodes(), 'Edges: ', graph.number_of_edges() )
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

        # Adding labels if possible
        if hasattr(pipeline[-1], 'labels_'):
            for i_animal, i_cluster in enumerate(pipeline[-1].labels_):
                nt.nodes[i_animal]['color'] = cluster_colors[i_cluster]

        nt.show(f'animaldata_{name}.html')

