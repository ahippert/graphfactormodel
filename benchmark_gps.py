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
from sklearn.impute import SimpleImputer

from sknetwork.clustering import get_modularity, Louvain

import networkx.algorithms.community as nx_comm

from src.estimators import SGLkComponents, NGL, GLasso, HeavyTailkGL, EllipticalGL

if __name__ == "__main__":

    # Load GNSS dataset (Smitarello et al., 2019
    file_path = os.path.dirname(__file__)
    data_path = os.path.join(file_path, "./data/gps_up.npy")
    #data_path = os.path.join(file_path, "./data/gps_up_2014_2022.npy")
    data = np.load(data_path)

    names_array = ['BOMG', 'BORG', 'CASG', 'CRAG', 'DERG', 'DSRG', 'ENCG', 'ENOG', 'FERG', 'FJAG', 'FOAG', 'FREG', 'GBNG', 'GBSG', 'GITG', 'GPNG', 'GPSG','HDLG', 'PRAG', 'RVLG', 'SNEG', 'TRCG']
    #names_array = ['BOMG', 'BORG', 'C98G', 'CASG', 'CRAG', 'DERG', 'DSRG', 'ENCG', 'ENOG', 'FERG', 'FEUG', 'FJAG', 'FOAG', 'FREG', 'GB1G', 'GBNG', 'GBSG', 'GITG', 'GPNG', 'GPSG','HDLG', 'MAIG', 'PBRG', 'PRAG', 'PVDG', 'REUN', 'RVAG', 'RVLG', 'SNEG', 'SROG', 'STAN','STJS', 'TRCG']

    # Form list of dictionary in the form [(0: 'name1), (1: 'name2'), ...]
    # Used to label each node
    names_array = np.hstack(np.hstack(names_array))
    dict_names = {}
    for i, name in enumerate(names_array):
        dict_names[i] = name
    data = data[:, ~np.all(np.isnan(data), axis=0)]
    X = data.T
    n_samples, n_features = X.shape
    print(n_samples)
    n_clusters = 6

    cluster_colors = [
        'red', 'blue', 'black', 'orange',
        'green', 'gray', 'purple', 'cyan',
        'yellow', 'magenta', 'brown','red', 'blue', 'black', 'orange',
        'green', 'gray', 'purple', 'cyan',
        'yellow', 'magenta', 'brown',
    ]

    # Pre-processing
    pre_processing = StandardScaler(with_mean=True, with_std=True)

    # Estimators compared
    StudentGL = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('Graph Estimation', HeavyTailkGL(
            heavy_type='student', k=3, nu=5)),
        ]
    )

    # EGFM = Pipeline(steps=[
    #     ('Centering', pre_processing),
    #     ('Imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    #     ('Graph Estimation', EllipticalGL(geometry="factor", k=4,
    #                                       lambda_seq=[1],
    #                                       df=5, maxiter=1e4)),
    #     ]
    # )
    EGFM = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('Graph Estimation', EllipticalGL(geometry="factor", k=4,
                                          lambda_seq=[1.5],
                                          df=5)),
        ]
    )
    

    GGFM = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('Graph Estimation', EllipticalGL(geometry="factor", k=4,
                                          lambda_seq=[0.5],
                                          df=1e3)),
        ]
    )

    EGM = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('Graph Estimation', EllipticalGL(geometry="SPD", k=10,
                                          lambda_seq=[0, 0.56],
                                          df=5)),
        ]
    )

    GGM = Pipeline(steps=[
        ('Centering', pre_processing),
        ('Imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('Graph Estimation', EllipticalGL(geometry="SPD", k=10,
                                          lambda_seq=[0., 1],
                                          df=1e3)),
        ]
    )

    list_names = ['StudentGL', 'GGM', 'EGM', 'GGFM', 'EGFM']
    list_pipelines = [StudentGL, GGM, EGM, GGFM, EGFM]

    # Doing estimation
    for pipeline, name in zip(list_pipelines, list_names):
        print("----------------------------------------------------------")
        print("----------------------------------------------------------")
        print("Doing estimation with :", name)
        print(pipeline)
        pipeline.fit_transform(X)

        # Get adjacency matrix to get the graph and clustered labels to compute modularity
        adjacency = pipeline['Graph Estimation'].precision_
        if name=='GGFM' or name=='EGFM' or name=='StudentGL':
            tol = 1e-2
        else:
            tol = 2.5e-2
        adjacency[np.abs(adjacency)<tol] = 0.
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

        # Put uniform weights and save edges for tikz drawing
        path_to_file = os.path.dirname(__file__)
        path_to_results = path_to_file + '/results'
        print(path_to_results)
        if not os.path.isdir(path_to_results):
            os.makedirs(path_to_results)
        edges = open(path_to_results + '/edges_GNSS_{}.dat'.format(name), "w")

        #print(graph.edges())
        for i, (u, v, d) in enumerate(graph.edges(data=True)):
            d['weight'] = 1.
            if i<len(graph.edges())-1:
                edges.write(u+'/'+v+','+'\n')
            else:
                edges.write(u+'/'+v+'\n')
                
        edges.close()

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

        nt.show('gpsdata_{}.html'.format(name))

    plt.show()
