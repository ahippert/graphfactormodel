import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

# Remove these two lines if you want minor ticks to show up
matplotlib.rcParams['xtick.minor.size'] = 0
matplotlib.rcParams['xtick.minor.width'] = 0

if __name__ == "__main__":

    np.random.seed(0)

    n_samples = 50*5                               # number of observations
    p = 50                                         # dimension

    # Load sparse SPD matrix (for reproducibility)
    precision_true = np.load("precision_true_erdos_reyni.npy")
    #precision_true = prec_true
    #plt.imshow(prec_true)
    #plt.show()
    
    lambda_seq = np.linspace(0,1,6)
    lambda_seq = np.logspace(-3,-.5,10)
    rank_seq = np.unique(np.logspace(0.5,1.4,6).astype(int)) 
    error_seq = []

    x_locations = rank_seq
    xlabels = ["3", "4", "7", "10", "16", "25"]
                           
    sample_seq = np.array([20, 50, 100, 200]) #n=[2*k, p, 2*p, 4*p]
    number_of_trials = 100

    δ_precision_sparsity = np.load("delta_lambda.npy")
    δ_precision_rank = np.load("delta_rank.npy")
    δ_precision_sparsity = np.round(δ_precision_sparsity, 4)

    labels = ["$n=p/2$", "$n=p$", "$n=2p$", "$n=4p$"]
    colors = ["C0", "C1", "C2", "C3"]
    
    fig, ax = plt.subplots(1,2,figsize=(5.5,3), sharey=True)
    for i, n in enumerate(range(len(sample_seq))):
        delta_sparsity = np.mean(δ_precision_sparsity[:,:,n], axis=1)
        delta_rank = np.mean(δ_precision_rank[:,:,n], axis=1)
        ax[0].plot(lambda_seq, delta_sparsity,
                   marker='o', markersize=8, label=labels[n])
        ax[1].plot(rank_seq, delta_rank,
                   marker='o', markersize=8, label=labels[n])
        plt.text(rank_seq[-2]-5, delta_rank[-2]+0.007, f'{labels[n]}', color=colors[n])

    ax[0].set_title('$k=10$, $p=50$')
    ax[1].set_title('$\lambda=10^{-1}$, $p=50$')
    ax[0].set_ylim(0.348, 0.482)
    ax[1].set_ylim(0.348, 0.482)
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[1].set_xticks(x_locations)
    ax[1].set_xticklabels(xlabels)
    ax[0].set_xlabel('sparsity $\lambda$')
    ax[0].set_ylabel('Relative error')
    ax[1].set_xlabel('rank $k$')
    plt.tight_layout()
    fig.savefig("student_precision_sparsity_rank.pdf", bbox_inches='tight')

    # ***********************************************************************************
    # Second graph: MSE versus number of samples
    # ***********************************************************************************

    sample_seq = np.array([20, 30, 50, 80, 120, 170])
    number_of_δ = 6
    δ_precision_student_other_methods = np.load("delta_samples_student.npy")
    δ_precision_student_NGL = np.load("delta_samples_student_NGL.npy")
    # δ_precision_gaussian_GLasso = np.load("delta_samples_gaussian_GLasso.npy")

    #δ_precision_gaussian = np.concatenate((δ_precision_gaussian_other_methods,
    #                                       δ_precision_gaussian_GLasso_NGL), axis=0)
    
    labels = ["EGFM", "GGFM", "EGM", "GGM", "StGL", "SGL", "NGL"]
    x_locations = sample_seq
    xlabels = ["20", "30", "$n=p$", "80", "120", "170"]
    markers = ["o", "o", "o", "o", "*", "^", "d"]
    colors = ["C0", "C0", "C1", "C1", "C2", "C3", "k"]
    linestyles= ["-", "--", "-", "--", "-", "-", "-"]
    
    fig, ax = plt.subplots(figsize=(3.5,3))
    plt.axvline(x=50, color="k", lw=.8)
    for n in range(number_of_δ):
        delta_gaussian = np.nanmean(δ_precision_student_other_methods[:,:,n], axis=1)
        ax.plot(sample_seq, delta_gaussian, color=colors[n], linestyle=linestyles[n],
                marker=markers[n], markersize=8, label=labels[n])
    ax.plot(np.array([120, 140, 170]), np.nanmean(δ_precision_student_NGL[:,:,0], axis=1),
                   color=colors[6], marker=markers[6], markersize=8, label=labels[6])

    ax.set_title('$p=50$, $\lambda=0.1$, $k=10$')
    ax.set_xscale('log')
    ax.set_xticks(x_locations)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel('$n$')
    ax.set_ylabel('Relative error')
    ax.legend(loc=0, ncols=2)
    plt.tight_layout()
    fig.savefig("student_precision_sampels.pdf", bbox_inches='tight')
    plt.show()    
