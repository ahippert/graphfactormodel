import numpy as np
import sys, os
import argparse
import networkx as nx

from scipy.sparse import csr_matrix
from scipy.stats import multivariate_normal, multivariate_t

from sklearn.pipeline import Pipeline
from sklearn.covariance import empirical_covariance
from sklearn import metrics

import matplotlib.pyplot as plt

from src.estimators import SGLkComponents, NGL, HeavyTailkGL, GLasso

from src.elliptical_graph_model import EllipticalGL

from joblib import Parallel, delayed
from tqdm import tqdm
import time

from graphical_models import Graphical_models

argParser = argparse.ArgumentParser()
argParser.add_argument("--graph", type=str, default='BA', help="graph structure to be considered. Possible values are: BA, ER, WS, RG (default: BA)")
argParser.add_argument("--method", type=str, default='GGM', help=". Possible values are: GGM, EGM, GGFM, EGFM, all")
argParser.add_argument("--roc", type=str, default='lambda', help="type of ROC curve to be computed. Possible values are: lambda, rank, compare (default: lambda)")
argParser.add_argument("--lambda_val", type=float, default=0.05, help="regularization parameter (default: 0.05)")
argParser.add_argument("--rank", type=int, default=20, help="rank of the factor model (default: 20)")
argParser.add_argument("-n", "--samples", type=int, default=105, help="number of samples (default: 105 (~2*p))")
argParser.add_argument("--multi", type=bool, default=True, help="wheter to use multi-threading or not (default: True)")
argParser.add_argument("--save", type=bool, default=False, help="save plot in pdf format (default: False)")
args = argParser.parse_args()


NGL = Pipeline(steps=[
        ('Graph Estimation', NGL(lamda=0.1, maxiter=1000, record_objective=True)),
        ]
    )

GLasso = Pipeline(steps=[
            ('Graph Estimation', GLasso(alpha=0.05))
        ]
    )

def S_estimation_method(X, *args):
    return np.dot(X, X.T)

def generate_gmrf(precision, n):
    covariance = np.linalg.inv(precision)
    X = np.random.multivariate_normal(np.zeros(len(precision)), covariance, n)
    return X

def one_monte_carlo(trial_no, model, parameter, n, p):

    np.random.seed(trial_no)

    # Initialize graph structure
    if args.graph == 'BA':
        model = model.BA_graph()    # Barabasi-Albert graph
    elif args.graph == 'ER':
        model = model.ER_graph()    # Erdos-Reyni graph
    elif args.graph == 'WS':
        model = model.WS_graph(5)   # Watts-Strogatz graph with 5 neighbors
    elif args.graph == 'RG':
        model = model.RG_graph(0.2) # Random geometric graph with radius=0.2
    else:
        pass # do nothing

    # Retrieve structured-laplacian and adjacency matrices
    L, A = model["laplacian"], model["adjacency"]

    # print approximate rank
    print("approx. rank of adjacency matrix: {:2}".format(np.linalg.matrix_rank(A)))

    # do some matching test
    if np.mean((L - (np.diag(np.diag(L)) - A))) != 0.:
        print('error in Laplacian')

    # Ensure that Laplacian's are invertible
    cond = np.linalg.cond(L)
    if cond < 1/sys.float_info.epsilon:
        L_inv = np.linalg.pinv(L)
    else:
        L += 0.1*np.eye(p)
        L_inv = np.linalg.pinv(L)

    # Sample non-Gaussian data with the pseudo-inverse of the precision (Elliptical Markov Random Field)
    X = multivariate_t.rvs(
        loc=np.zeros(len(L)),
        shape=L_inv,
        df=3,
        size=n,
        random_state=np.random.RandomState(trial_no)
    )

    # Estimate sample covariance matrix
    S = empirical_covariance(X, assume_centered=True)

    # Estimators compared
    if args.roc == 'lambda':
        rank_val = args.rank
        lambda_val = parameter
    elif args.roc == 'rank':
        rank_val = parameter
        lambda_val = args.lambda_val
    else:
        rank_val = args.rank
        lambda_val =  args.lambda_val


    GGM = Pipeline(steps=[
        ('Graph Estimation', EllipticalGL(geometry="SPD",
                                          lambda_seq=[lambda_val],
                                          df=1e3, verbosity=False)),
    ]
                   )

    EGM = Pipeline(steps=[
        ('Graph Estimation', EllipticalGL(geometry="SPD",
                                          lambda_seq=[lambda_val],
                                          df=3, verbosity=False)),
    ]
                   )


    EGFM = Pipeline(steps=[
        ('Graph Estimation', EllipticalGL(geometry="factor", k=rank_val,
                                          lambda_seq=[lambda_val],
                                          df=3, verbosity=False)),
    ]
                    )

    GGFM = Pipeline(steps=[
        ('Graph Estimation', EllipticalGL(geometry="factor", k=rank_val,
                                          lambda_seq=[lambda_val], df=1e3,
                                          verbosity=False)),
    ]
                    )


    if args.method == 'GGM':
        pipeline_seq = [GGM]
    elif args.method == "EGM":
        pipeline_seq = [EGM]
    elif args.method == 'GGFM':
        pipeline_seq = [GGFM]
    elif args.method == 'EGFM':
        pipeline_seq = [EGFM]
    else:
        if args.graph == 'BA' or args.graph == 'ER':
            lambda_GGM = 0.05
            lambda_GGFM = 0.01
            lambda_EGFM = lambda_GGFM
        elif args.graph == 'WS':
            lambda_GGM = 0.1
            lambda_GGFM = 0.01
            lambda_EGFM = lambda_GGFM
        else:
            lambda_GGM = 0.1
            lambda_GGFM = 0.01
            lambda_EGFM = 0.05
        GGM = Pipeline(steps=[
            ('Graph Estimation', EllipticalGL(geometry="SPD",
                                              lambda_seq=[lambda_GGM],#1e-4,5e-4,1e-3,5e-3,1e-2],
                                              df=1e3, verbosity=False)),
        ]
                       )

        EGFM = Pipeline(steps=[
            ('Graph Estimation', EllipticalGL(geometry="factor", k=20,
                                              lambda_seq=[lambda_EGFM],
                                              df=3, verbosity=False)),
        ]
                        )
        GGFM = Pipeline(steps=[
            ('Graph Estimation', EllipticalGL(geometry="factor", k=20,
                                              lambda_seq=[lambda_GGFM], df=1e3,
                                              verbosity=False)),
        ]
                        )
        EGM = Pipeline(steps=[
            ('Graph Estimation', EllipticalGL(geometry="SPD",
                                              lambda_seq=[lambda_GGM],
                                              df=3, verbosity=False)),
        ]
                       )
        SGL = Pipeline(steps=[
            ('Graph Estimation', SGLkComponents(
                S, maxiter=1000, record_objective=True, record_weights=True,
                alpha=0.1, beta=1000, verbosity=1)),
        ]
                       )

        StudentGL = Pipeline(steps=[
            ('Graph Estimation', HeavyTailkGL(
                heavy_type='student', k=1, nu=3))
        ]
                             )

        pipeline_seq = [GGM, EGM, GGFM, EGFM, GLasso, SGL, NGL, StudentGL]


    fpr, tpr = [], []
    for pipeline in pipeline_seq:
        pipeline.fit_transform(X)

        # Get estimated precision matrix
        precision_est = pipeline['Graph Estimation'].precision_

        # Normalize it to get 1's on the diagonal
        d = 1/np.sqrt(np.diag(precision_est))
        precision_est = np.diag(d) @ precision_est @ np.diag(d)

        # Same for the true precision
        d = 1/np.sqrt(np.diag(L))
        precision_true = np.diag(d) @ L @ np.diag(d)

        # Get predicted labels using upper triangular matrix only (symmetric matrix)
        precision_est_vec = precision_est[np.triu_indices(p, k=1)]
        y_pred = abs(precision_est_vec)

        # Get true labels
        precision_true_vec = precision_true[np.triu_indices(p, k=1)]
        y_true = np.where(precision_true_vec < 0, 1., 0.) # minus sign

        # Compute ROC parameters
        y_pred[np.isnan(y_pred)] = 0.
        tmp = metrics.roc_curve(y_true, y_pred)
        fpr.append(tmp[0])
        tpr.append(tmp[1])

    return fpr, tpr


def parallel_monte_carlo(model,param, n, p, n_threads, n_trials, Multi):

    # Looping on Monte Carlo Trials
    fpr_seq, tpr_seq = [], []
    if Multi:
        tmp = Parallel(n_jobs=n_threads)(delayed(one_monte_carlo)(iMC, model, param, n, p) for iMC in range(n_trials))
        for i in range(n_trials):
            fpr_seq.extend(tmp[i][0])
            tpr_seq.extend(tmp[i][1])
        return fpr_seq, tpr_seq
    else:
        for iMC in range(n_trials):
            tmp = one_monte_carlo(iMC, model, param, n, p)
            fpr_seq.append(tmp[0][0])
            tpr_seq.append(tmp[1][0])
        return fpr_seq, tpr_seq

def main():

    np.random.seed(0)

    n_samples = args.samples                                 # number of observations
    p = 50                                                  # dimension
    wmin, wmax = 2, 5                                       # U(wmin, wmax) for graph weight sampling
    proba = 0.1                                             # probability of Erdos-Renyi graph

    # generate graph instance
    model = Graphical_models(p, wmin, wmax, proba)

    # define which type parameter varies in the ROC curve (lambda, rank or None)
    if args.roc == "lambda":
        parameter_seq = np.array([0.001, 0.01, 0.05])
    elif args.roc == "rank":
        parameter_seq = np.array([5, 10, 20, 30])
    else:
        parameter_seq = np.array([0])

    # # # -------------------------------------------------------------------------------
    # # # Doing estimation and saving the distance to true value
    # # # -------------------------------------------------------------------------------
    print( '|￣￣￣￣￣￣￣￣￣￣|')
    print( '|   Launching        |')
    print( '|   Monte Carlo      |')
    print( '|   simulation       |' )
    print( '|＿＿＿＿＿＿＿＿＿＿|')
    print( ' (\__/) ||')
    print( ' (•ㅅ•) || ')
    print( ' / 　 づ')
    print(u"Parameters: p={}, n={}, {}={}".format(p, n_samples, args.roc, parameter_seq))

    # to use the maximum number of threads
    number_of_threads = -1

    # number of trials of the MC experiment
    number_of_trials = 1

    t_beginning = time.time()

    # Distance container cube
    if args.roc == 'compare':
        methods = ["GGM", "EGM", "GGFM", "EGFM", "GLasso", "SGL", "NGL", "StudentGL"]
        number_of_δ = len(methods)
    else:
        number_of_δ = 1


    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(5, 5))
    plt.axes().set_aspect('equal', 'datalim')
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

    c = 0
    for i_n, parameter in enumerate(tqdm(parameter_seq)):
        fpr_seq, tpr_seq = parallel_monte_carlo(model, parameter, n_samples, p, number_of_threads, number_of_trials, args.multi)

        tprs = []
        aucs = []
        #c = 0
        for i in range(number_of_δ):
            for j in range(number_of_trials):

                if args.roc == 'lambda' or args.roc == 'rank':
                    interp_tpr = np.interp(mean_fpr, fpr_seq[j], tpr_seq[j])
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)
                    auc = metrics.auc(fpr_seq[j], tpr_seq[j])
                    aucs.append(auc)
                    plt.plot(fpr_seq[j], tpr_seq[j], alpha=0.1, lw=1, color=colors[c])
                else:
                    interp_tpr = np.interp(mean_fpr, fpr_seq[number_of_δ*j + i], tpr_seq[number_of_δ*j + i])
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)
                    auc = metrics.auc(fpr_seq[number_of_δ*j + i], tpr_seq[number_of_δ*j + i])
                    aucs.append(auc)


            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = metrics.auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            if args.roc == 'lambda':
                plt.plot(mean_fpr, mean_tpr, lw=3, label=r"$\lambda$ = {:3} (AUC = {:1.2} $\pm$ {:1.2})".format(parameter, mean_auc, std_auc), color=colors[c])
            elif args.roc == 'rank':
                plt.plot(mean_fpr, mean_tpr, lw=3, label=r"$k$ = {:2} (AUC = {:1.2} $\pm$ {:1.2})".format(parameter, mean_auc, std_auc), color=colors[c])
            else:
                plt.plot(mean_fpr, mean_tpr, label=r"{} (AUC = {:1.2} $\pm$ {:1.2})".format(methods[i], mean_auc, std_auc), color=colors[c])

            c += 1

    print('Done in %f s'%(time.time()-t_beginning))
 
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.legend(loc="lower right", fontsize=12.5)
    # plt.tick_params(
    #     axis='both',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     left=False,      # ticks along the bottom edge are off
    #     bottom=False,
    #     top=False,         # ticks along the top edge are off
    #     labelleft=False,
    #     labelbottom=False) # labels along the bottom edge are off

    plt.title("{} - {} graph - n={}".format(args.method, args.graph, args.samples))
    if args.save:
        path_to_file = os.path.dirname(__file__)
        path_to_results = path_to_file + '/results'
        if not os.path.isdir(path_to_results):
            os.makedirs(path_to_results)
        plt.savefig(path_to_results + "/{}_{}_n={}_{}_ROC.pdf".format(args.method, args.graph, args.samples, args.roc))
    plt.show()

if __name__ == "__main__":
    main()
