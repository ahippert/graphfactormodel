import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

import numpy as np

from StructuredGraphLearning.LearnGraphTopology import LearnGraphTopology

import networkx as nx
import matplotlib.pyplot as plt

from sklearn.covariance import GraphicalLasso
from sklearn import cluster, manifold
from sklearn.datasets import make_moons, make_circles

import visualization as vis

if __name__ == "__main__":
    # plots_dir = './plots'
    # if not os.path.exists(plots_dir):
    #     os.makedirs(plots_dir)

    np.random.seed(0)

    # Toy datasets
    n = 40  # number of nodes per cluster
    k = 2   # number of components
    datasets =[make_moons(n_samples=n*k, noise=.05, shuffle=True),
               make_circles(n_samples=n*k, factor=0.5, noise=0.05)
    ]

    # Graph learning algorithms
    names = [
        #"Graphical Lasso",
        "SGL"
    ]

    algorithms = [
        #GraphicalLasso(alpha=0.05),
        LearnGraphTopology(np.eye(1), maxiter=1000, record_objective = True, record_weights = True)
    ]

    # Iterate over datasets
    for ds_cnt, ds in enumerate(datasets):

        print("\n")
        print("----------Dataset %d----------" %(ds_cnt+1))
        X, y = ds
        pos = {}
        for i in range(n*k):
            pos[i] = X[i]

        # Iterate over graph learning algorithms
        for name, alg in zip(names, algorithms):

            if name=="Graphical Lasso":
                print("%s processing..." %name)
                alg.fit(X.T)
                precision = alg.precision_.copy()
                covariance = alg.covariance_.copy() # used for clustering

            elif name=='SGL':
                print("%s processing..." %name)
                beta = 0.1
                alg.S = np.dot(X, X.T) # set covariance to empirical covariance
                graph = alg.learn_k_component_graph(k=k, beta=beta)
                precision = graph['adjacency']
            else:
                print("Exit")

            graph = nx.from_numpy_matrix(precision)
            print('Graph statistics:')
            print('Nodes: ', graph.number_of_nodes(), 'Edges: ', graph.number_of_edges() )

            # Visualization - using networkx
            vis.visualize_simple_graph(X, graph, pos)

    plt.show()

