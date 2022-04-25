
import os, sys
import numpy as np
import scipy.io

import networkx as nx
import matplotlib.pyplot as plt

from sklearn.covariance import GraphicalLasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import cluster, manifold

import src.visualization as vis
from src.estimators import SGLkComponents
from src.utils import format_pipeline_name

if __name__ == "__main__":

    # Load animal dataset (Osherson et al., 1991; Lake and Tenenbaum, 2010)
    file_path = os.path.dirname(__file__)
    data_path = os.path.join(file_path, "../data/animals.mat")
    data = scipy.io.loadmat(data_path)

    names_array = data['names']
    # Form list of dictionary in the form [(0: 'name1), (1: 'name2'), ...]
    # Used to label each node
    list_names = [(i, names_array[i,0][0]) for i in range(len(names_array))]
    X = data['data'].T
    y = data['features']
    n_samples, n_features = X.shape

    # Pre-processing
    pre_processing = StandardScaler(with_mean=True, with_std=False)

    # Estimators compared
    Glasso = make_pipeline(
        pre_processing, GraphicalLasso(alpha=0.05)
    )
    SGL = make_pipeline(
        pre_processing, 
        SGLkComponents(
            None, maxiter=1000, record_objective=True, record_weights=True,
            beta = 0.5, k=6, S_estimation_args=[1./3], verbosity=1
        )
    )

    # Doing estimation
    for pipeline in [Glasso, SGL]:
        print("----------------------------------------------------------")
        print("----------------------------------------------------------")
        name = format_pipeline_name(pipeline)
        print("Doing estimation with :", name)
        print(pipeline)
        pipeline.fit(X)
        graph = nx.from_numpy_matrix(pipeline[-1].precision_)
        print('Graph statistics:')
        print('Nodes: ', graph.number_of_nodes(), 'Edges: ', graph.number_of_edges() )
        print("----------------------------------------------------------")
        print('\n\n')


        #########################################################################
        # Visualization 1 - manual visualization
        vis.visualize_graph(
            X.T, pipeline[-1].covariance_, pipeline[-1].precision_, list_names, name
        )

        #########################################################################
        # Visualization 2 - using networkx
        vis.visualize_graph2(
            X.T, graph, pipeline[-1].covariance_, list_names, name
        )

    plt.show()