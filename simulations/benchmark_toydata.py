import os, sys
import numpy as np
from copy import deepcopy

from StructuredGraphLearning.LearnGraphTopology import LearnGraphTopology

import networkx as nx
import matplotlib.pyplot as plt

from sklearn.covariance import GraphicalLasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import cluster, manifold
from sklearn.datasets import make_moons, make_circles

import src.visualization as vis
from src.estimators import SGLkComponents
from src.utils import format_pipeline_name


def S_estimation_method(X, *args):
    return np.dot(X, X.T)

if __name__ == "__main__":


    np.random.seed(0)

    # Toy datasets
    n = 30  # number of nodes per cluster
    k = 2   # number of components
    datasets =[
        make_moons(n_samples=n*k, noise=.05, shuffle=True),
        make_circles(n_samples=n*k, factor=0.5, noise=0.05)
    ]

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
            beta = 0.1, k=k, verbosity=1, S_estimation_method=S_estimation_method
        )
    )

    # Iterate over datasets
    for ds_cnt, ds in enumerate(datasets):

        print("\n")
        print("----------Dataset %d----------" %(ds_cnt+1))
        X, y = ds
        pos = {}
        for i in range(n*k):
            pos[i] = X[i]

        # Doing estimation
        for pipeline in [Glasso, SGL]:
            pipeline_dataset = deepcopy(pipeline)
            print("----------------------------------------------------------")
            name = format_pipeline_name(pipeline_dataset)
            print("Doing estimation with :", name)
            print(pipeline_dataset)
            pipeline_dataset.fit(X)
            graph = nx.from_numpy_matrix(pipeline_dataset[-1].precision_)
            # Visualization - using networkx
            vis.visualize_simple_graph(X.T, graph, pos)
            print('Graph statistics:')
            print('Nodes: ', graph.number_of_nodes(), 'Edges: ', graph.number_of_edges() )
            print("----------------------------------------------------------")
            print('\n') 

        print('\n\n') 

    plt.show()

