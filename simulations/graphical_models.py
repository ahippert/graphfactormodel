import numpy as np

import sys

from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt

from networkx import laplacian_matrix, adjacency_matrix, barabasi_albert_graph, \
    set_edge_attributes, draw_networkx, erdos_renyi_graph, watts_strogatz_graph, \
    random_geometric_graph, stochastic_block_model

class Graphical_models():

    def __init__(self, number_of_nodes, weight_min, weight_max, probability, sampled_coeff=True):
        self.number_of_nodes = number_of_nodes
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.probability = probability
        self.sampled_coeff = sampled_coeff

    def BA_graph(self):
        """Generate Barabasi-Albert graph with sampled weights from U(w_min, w_max)
        """
        # Generate the graph with n nodes and 1 edge to attach from a new node to existing nodes
        graph = barabasi_albert_graph(self.number_of_nodes, 1)

        if self.sampled_coeff:
            weights = np.random.uniform(self.weight_min, self.weight_max)
            set_edge_attributes(graph,
                                {e: {'weight': weights} for e in graph.edges}
                                )

        # Retrieve true laplacian and adjacency matrices and convert it to numpy array
        return {"graph": graph,
                "laplacian": csr_matrix.toarray(laplacian_matrix(graph)),
                "adjacency": csr_matrix.toarray(adjacency_matrix(graph)),
                "model_name": "Barabasi-Albert"}

    def ER_graph(self):
        """Generate Erdos-Renyi graph with sampled weights from U(w_min, w_max)
        """
        graph = erdos_renyi_graph(self.number_of_nodes, self.probability)

        if self.sampled_coeff:
            weights = np.random.uniform(self.weight_min, self.weight_max, size=len(graph.edges))
            set_edge_attributes(graph,
                                {e: {'weight': weights[i]} for i, e in enumerate(graph.edges)}
                                )
        # Retrieve true laplacian and adjacency matrices and convert it to numpy array
        laplacian = csr_matrix.toarray(laplacian_matrix(graph))
        adjacency = csr_matrix.toarray(adjacency_matrix(graph))
        return {"graph": graph, "laplacian": laplacian, "adjacency": adjacency,
                "model_name": "Erdos-Renyi"}

    def WS_graph(self, num_of_neighbors):
        """Generate Watts-Strogatz graph with sampled weights from U(w_min, w_max)
        """
        graph = watts_strogatz_graph(self.number_of_nodes, num_of_neighbors, self.probability)

        if self.sampled_coeff:
            weights = np.random.uniform(self.weight_min, self.weight_max, size=len(graph.edges))
            set_edge_attributes(graph,
                                {e: {'weight': weights[i]} for i, e in enumerate(graph.edges)}
                                )

        # Retrieve true laplacian and adjacency matrices and convert it to numpy array
        laplacian = csr_matrix.toarray(laplacian_matrix(graph))
        adjacency = csr_matrix.toarray(adjacency_matrix(graph))
        return {"graph": graph, "laplacian": laplacian, "adjacency": adjacency,
                "model_name": "Watts-Strogatz"}

    def RG_graph(self, radius):
        """Generate random geometric graph with sampled weights from U(w_min, w_max)
        """
        graph = random_geometric_graph(self.number_of_nodes, radius)

        if self.sampled_coeff:
            weights = np.random.uniform(self.weight_min, self.weight_max, size=len(graph.edges))
            set_edge_attributes(graph,
                                {e: {'weight': weights[i]} for i, e in enumerate(graph.edges)}
                                )

        # Retrieve true laplacian and adjacency matrices and convert it to numpy array
        laplacian = csr_matrix.toarray(laplacian_matrix(graph))
        adjacency = csr_matrix.toarray(adjacency_matrix(graph))
        return {"graph": graph, "laplacian": laplacian, "adjacency": adjacency,
                "model_name": "Random geometric graph"}

    def block_structured_graph(self, block_size1, block_size2):
        """
        """
        op = Operators()
        w1 = np.random.uniform(0, 1, block_size1)
        w2 = np.random.uniform(0, 1, block_size2)
        return block_diag(op.L(w1), op.L(w2))

if __name__ == "__main__":

    np.random.seed(0)

    n_samples = 105                               # number of observations
    n_components = 10                              # number of components
    p = 50                                         # dimension
    wmin, wmax = 2, 5
    proba = 0.1

    # 1. Generate graph with different models as in [Ying, 2020, Neurips]
    model = Graphical_low_rank_models(p, wmin, wmax, proba)

    BA_model = model.BA_graph()    # Barabasi-Albert graph
    ER_model = model.ER_graph()    # Erdos-Reyni graph
    WS_model = model.WS_graph(5)    # Watts-Strogatz graph with 5 neighbors
    RG_model = model.RG_graph(0.2)    # Random geometric graph with radius=0.2

    model_list = [BA_model, ER_model, WS_model, RG_model]
    model_names = ["Barabasi-Albert", "Erdos-Reyni", "Watts-Strogatz", "Random geometric"]
    gra_seq, lapl_seq, adj_seq = [], [], []

    # 2. Retrieve graph object (networkx object), structured-laplacian and adjacency matrices
    for i, model in enumerate(model_list):
        G, L, A = model["graph"], model["laplacian"], model["adjacency"]
        gra_seq.append(G)
        lapl_seq.append(L)
        adj_seq.append(A)

        print("approx. rank of adjacency matrix: {:2}".format(np.linalg.matrix_rank(adj_seq[i])))

        if np.mean((lapl_seq[i] - (np.diag(np.diag(lapl_seq[i])) - adj_seq[i]))) != 0.:
            print('error in Laplacian')

    # 3. Ensure that Laplacian's are invertible
    lapl_inv_seq = []
    for L in lapl_seq:
        cond = np.linalg.cond(L)
        print(cond)
        if cond < 1/sys.float_info.epsilon:
            L_inv = np.linalg.inv(L)
            lapl_inv_seq.append(L_inv)
        else:
            L_inv = L + 0.1*np.eye(p)
            lapl_inv_seq.append(L_inv)


    fig, axes = plt.subplots(3, len(model_list), figsize=(2.1*len(model_list),1.7*len(model_list)))
    for i in range(len(model_list)):
        draw_networkx(gra_seq[i], node_size=50, font_size=5, ax=axes[0,i])
        axes[0,i].set_title("{} graph".format(model_names[i]))
        axes[1,i].imshow(adj_seq[i]); axes[1,i].set_title("adjacency")
        axes[2,i].imshow(lapl_seq[i]); axes[2,i].set_title("laplacian")
        axes[1,i].tick_params(
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,
            labelbottom=False,
            labelleft=False) #
        axes[2,i].tick_params(
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            left=False,         # ticks along the top edge are off
            labelbottom=False,
            labelleft=False) #
    plt.tight_layout()
    plt.show()




