## Do some stats
import numpy as np
from scipy import stats
from mne.stats import spatio_temporal_cluster_1samp_test, fdr_correction

def do_stats(X, method='FDR', adjacency=None, FDR_alpha=.025, h0=0, 
             cluster_alpha = .025, p_threshold = .05, n_permutations=500):
    
    n_subjects = X.shape[0]
    if len(X.shape) == 3:
        X = X.transpose(0,2,1)

    t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
    if method == 'montecarlo':
        print('Clustering.')
        tvals, clusters, cluster_p_values, H0 = \
            spatio_temporal_cluster_1samp_test(X-h0, adjacency=adjacency, n_jobs=-1,
                                                threshold=t_threshold, buffer_size=None,
                                                verbose=True, n_permutations = n_permutations,
                                                out_type='mask')
        good_cluster_inds = np.where(cluster_p_values <= cluster_alpha)[0]
        gclust = np.array([clusters[c] for c in good_cluster_inds])
        gmask = np.zeros(X.shape[1:])

        if gclust.shape[0] > 0:
            for tc in range(gclust.shape[0]):
                gmask = gmask + gclust[tc].astype(float)
                
        if len(X.shape) == 3:
            X = X.transpose(0,2,1)
        
        stat_results = {'mask': gmask.T, 'tvals': tvals.T, 'pvals': cluster_p_values,
                        'data_mean': np.mean(X,0), 'data_sd': np.std(X,0),'clusters': clusters,
                        'n': n_subjects,'alpha': cluster_alpha,'p_threshold': p_threshold}
    elif method == 'FDR':
        print('\nPerforming FDR correction\n.')
        tvals, pvals = stats.ttest_1samp(X, popmean=h0,axis=0)
        gmask, adj_pvals = fdr_correction(pvals, FDR_alpha)
        print(np.sum(gmask))
        stat_results = {'mask': gmask, 'tvals': tvals, 'pvals': pvals, 'qvals': adj_pvals,
                        'data_mean': np.mean(X,0), 'data_sd': np.std(X,0), 'n': n_subjects,
                        'alpha': FDR_alpha}
    return stat_results