
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))
import numpy as np

from StructuredGraphLearning.LearnGraphTopology import LearnGraphTopology

import networkx as nx
import matplotlib.pyplot as plt

from sklearn.covariance import GraphicalLasso
from sklearn import cluster, manifold

import visualization as vis

import scipy.io

if __name__ == "__main__":
    # plots_dir = './plots'
    # if not os.path.exists(plots_dir):
    #     os.makedirs(plots_dir)

    # Load animal dataset (Osherson et al., 1991; Lake and Tenenbaum, 2010)
    file_path = os.path.dirname(__file__)
    data_path = os.path.join(file_path, "../../data/animals.mat")
    animals_mat = scipy.io.loadmat(data_path)

    real_datasets = {'animals': animals_mat
    }

    names_array = real_datasets['animals']['names']
    X = real_datasets['animals']['data']
    y = real_datasets['animals']['features']
    p, n = X.shape

    # Graph learning algorithms
    names = [
        "Graphical Lasso",
        "SGL"
    ]

    algorithms = [
        GraphicalLasso(alpha=0.05),
        LearnGraphTopology(np.eye(1), maxiter=1000, record_objective = True, record_weights = True)
    ]

    # De-mean data /!\ important
    mean = np.mean(X, axis=0)
    X = X - mean

    # Iterate over graph learning algorithms
    for name, alg in zip(names, algorithms):

        if name=="Graphical Lasso":
            alg.fit(X.T)
            precision = alg.precision_.copy()
            covariance = alg.covariance_.copy() # used for clustering

        elif name=='SGL':
            n_components = 6
            beta = 0.5                              # from (Kumar et al. 2020)
            alg.S = (1./n)*X@X.T + (1./3)*np.eye(p) # from (Kumar et al. 2020)
            graph = alg.learn_k_component_graph(k=n_components, beta=beta)
            precision = graph['adjacency']
        else:
            print("Exit")

        graph = nx.from_numpy_matrix(precision)
        print('Graph statistics:')
        print('Nodes: ', graph.number_of_nodes(), 'Edges: ', graph.number_of_edges() )

        # Form list of dictionary in the form [(0: 'name1), (1: 'name2'), ...]
        # Used to label each node
        list_names = [(i, names_array[i,0][0]) for i in range(len(names_array))]

        #########################################################################
        # Visualization 1 - manual visualization
        vis.visualize_graph(X, covariance, precision, list_names, name)

        #########################################################################
        # Visualization 2 - using networkx
        vis.visualize_graph2(X, graph, covariance, list_names, name)


    plt.show()

