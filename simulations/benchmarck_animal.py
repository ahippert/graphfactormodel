
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

import src.visualization as vis
from src.estimators import SGLkComponents, GLasso


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

    cluster_colors = ['red', 'blue', 'black', 'orange', 'green', 'gray']

    # Pre-processing
    pre_processing = StandardScaler(with_mean=True, with_std=False)

    # Estimators compared
    GLasso = Pipeline(steps=[
            ('Centering', pre_processing),
            ('Graph Estimation', GLasso(alpha=0.05)),
            # ('Spectral Clustering', SpectralClustering(
            #     affinity='precomputed', n_clusters=n_clusters))
        ]
    )
    SGL = Pipeline(steps=[
            ('Centering', pre_processing),
            ('Graph Estimation', SGLkComponents(
                None, maxiter=1000, record_objective=True, record_weights=True,
                beta = 0.5, k=n_clusters, S_estimation_args=[1./3], verbosity=1
            )),
            ('Spectral Clustering', SpectralClustering(
                affinity='precomputed', n_clusters=n_clusters))
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

        nt = Network('500px', '500px')
        # nt.show_buttons()
        nt.set_options("""
        var options = {
        "edges": {
            "color": {
            "inherit": true
            },
            "font": {
            "color": "rgba(52,52,52,0.52)",
            "size": 7,
            "align": "bottom"
            },
            "smooth": false
        },
        "interaction": {
            "hover": true
        },
        "physics": {
            "enabled": true,
            "minVelocity": 0.05
        }
        }
        """)
        nt.from_nx(graph)

        # Adding labels if possible
        if hasattr(pipeline[-1], 'labels_'):
            for i_animal, i_cluster in enumerate(pipeline[-1].labels_):
                nt.nodes[i_animal]['color'] = cluster_colors[i_cluster]

        nt.show(f'animaldata_{name}.html')

