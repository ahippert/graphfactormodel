import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import networkx as nx
import pandas as pd

from sklearn import cluster, covariance, manifold


def visualize_graph(data, covariance, precision, list_names, name):
    """Plot a graph with matplotlib libraries.
    Nodes positions are learned from low-dimension embedding.
    Adapted from: https://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html?highlight=graph#visualization
    """

    labels = dict([(k, v) for k, v in list_names])
    names = np.array(list_names)[:,1]

    # ##########################################################################
    # Cluster using affinity propagation
    _, labels = cluster.affinity_propagation(covariance, random_state=0)
    n_labels = labels.max()

    for i in range(n_labels + 1):
        print("Cluster %i: %s" % ((i + 1), ", ".join(names[labels == i])))

    # ##########################################################################
    # Find a low-dimension embedding for visualization: find the best position of
    # the nodes on a 2D plane

    # We use a dense eigen_solver to achieve reproducibility (arpack is
    # initiated with random vectors that we don't control). In addition, we
    # use a large number of neighbors to capture the large-scale structure.
    node_position_model = manifold.LocallyLinearEmbedding(
        n_components=2, n_neighbors=7
    )
    
    embedding = node_position_model.fit_transform(data).T

    # ##########################################################################
    # Visualization
    plt.figure(figsize=(10, 8))
    #plt.clf()
    ax = plt.axes([0.0, 0.0, 1.0, 1.0])

    # Display a graph of the partial correlations
    partial_correlations = precision.copy()
    #d = 1 / np.sqrt(np.diag(partial_correlations))
    #partial_correlations *= d
    #partial_correlations *= d[:, np.newaxis]
    non_zero = np.abs(np.triu(partial_correlations, k=1)) > 0

    # Plot the nodes using the coordinates of our embedding
    plt.scatter(
        embedding[0], embedding[1], s=80, c=labels,
        cmap=plt.cm.nipy_spectral)

    plt.title("Learned graph for animal dataset")
    #plt.suptitle('components k=' + str(n_components) + ', beta=' + str(beta))
    plt.suptitle('Algorithm: ' + name)

    # Plot the edges
    start_idx, end_idx = np.where(non_zero)
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [
        [embedding[:, start], embedding[:, stop]] for start, stop in zip(start_idx, end_idx)
    ]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(
        segments, zorder=0, cmap=plt.cm.binary, norm=plt.Normalize(0, 0.7 * values.max())
    )
    lc.set_array(values)
    lc.set_linewidths(values)
    ax.add_collection(lc)

    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    for index, (name, label, (x, y)) in enumerate(zip(names, labels, embedding.T)):
        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = "left"
            x = x + 0.002
        else:
            horizontalalignment = "right"
            x = x - 0.002
        if this_dy > 0:
            verticalalignment = "bottom"
            y = y + 0.002
        else:
            verticalalignment = "top"
            y = y - 0.002
        plt.text(
            x,
            y,
            name,
            size=10,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            #bbox=dict(
            #    facecolor="w",
            #    edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
            #    alpha=0.6,
            #),
        )

    plt.xlim(
        embedding[0].min() - 0.15 * embedding[0].ptp(),
        embedding[0].max() + 0.10 * embedding[0].ptp(),
    )
    plt.ylim(
        embedding[1].min() - 0.03 * embedding[1].ptp(),
        embedding[1].max() + 0.03 * embedding[1].ptp(),
    )

def visualize_simple_graph(data, graph, pos):
    # Compute normalized weights corresponding to each node's width
    all_weights = []
    for (node1,node2,data) in graph.edges(data=True):
        all_weights.append(data['weight'])
    max_weight = max(all_weights)
    norm_weights = [2*w / max_weight for w in all_weights]

    # Plot graph using draw_networkx
    fig = plt.figure(figsize=(15,8))

    nx.draw_networkx(graph, pos, width=norm_weights)
    plt.title("Learned graph")
    #plt.suptitle('components k=' + str(n_components) + ', beta=' + str(beta))


def visualize_graph2(data, graph, covariance, list_names, name, n_components=None, beta=None):
    """Plot a graph with the networkx library.
    Nodes positions are according to the default layout of draw_networkx.
    With pieces of code from: https://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html?highlight=graph#visualization
    """

    labels = dict([(k, v) for k, v in list_names])
    names = np.array(list_names)[:,1]

    # ##########################################################################
    # Cluster using affinity propagation
    _, labels_no = cluster.affinity_propagation(covariance,
                                                random_state=0
    )
    n_labels = labels_no.max() + 1

    for i in range(n_labels):
        print("Cluster %i: %s" % ((i + 1), ", ".join(names[labels_no == i])))

    # we need a list of RGB colors for each node (one cluster = one color)
    colors = plt.cm.get_cmap('nipy_spectral', n_labels)
    colors = colors(range(n_labels))[:,:3] # get RGB components only
    nodecolor = [(colors[i][0], colors[i][1], colors[i][2]) for i in labels_no]

    # Compute normalized weights corresponding to each node's width
    all_weights = []
    for (node1,node2,data) in graph.edges(data=True):
        all_weights.append(data['weight'])
    max_weight = max(all_weights)
    norm_weights = [2*w / max_weight for w in all_weights]

    # Plot graph using draw_networkx
    fig = plt.figure(figsize=(15,8))
    pos = nx.spring_layout(graph)

    nx.draw_networkx(graph, pos, node_size=80, node_color=nodecolor,
                     with_labels=False, labels=labels, font_size=10,
                     width=norm_weights)
    plt.title("Learned graph")
    #plt.suptitle('components k=' + str(n_components) + ', beta=' + str(beta))
    plt.suptitle('Algorithm: ' + name)

    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    embedding = [pos[i].tolist() for i in pos.keys()]
    embedding = np.array(embedding).T
    for index, (name, label, x, y) in enumerate(zip(names, labels_no, embedding[0], embedding[1])):
        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = "left"
            x = x + 0.003
        else:
            horizontalalignment = "right"
            x = x - 0.003
        if this_dy > 0:
            verticalalignment = "bottom"
            y = y + 0.003
        else:
            verticalalignment = "top"
            y = y - 0.003
        plt.text(
            x,
            y,
            name,
            size=10,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            #bbox=dict(
            #    facecolor="w",
            #    edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
            #    alpha=0.6,
            #),
        )

    plt.xlim(
        embedding[0].min() - 0.15 * embedding[0].ptp(),
        embedding[0].max() + 0.10 * embedding[0].ptp(),
    )
    plt.ylim(
        embedding[1].min() - 0.03 * embedding[1].ptp(),
        embedding[1].max() + 0.03 * embedding[1].ptp(),
    )
