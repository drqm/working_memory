from os import path as op
from stormdb.access import Query
import pickle
import numpy as np
from scipy import stats
from mne.stats import spatio_temporal_cluster_1samp_test, spatio_temporal_cluster_test, fdr_correction, ttest_1samp_no_p
from sklearn import linear_model
from sklearn.impute import SimpleImputer as Imputer
from functools import partial

def load_ERF_sensor(subs, suffix='', exclude=[]):   
    project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
    project_dir = '/projects/' + project
    avg_path = project_dir + '/scratch/working_memory/averages/data/'
    sdata, scodes = {}, []
    qr = Query(project)
    sub_codes = qr.get_subjects()
    for sub in subs:
        sub_code = sub_codes[sub-1]
        if sub not in exclude:
            try:
                print('loading subject {}'.format(sub))
                evkd_fname = op.join(avg_path,sub_code,sub_code + '_evoked_' + suffix + '.p')
                evkd_file = open(evkd_fname,'rb')
                evokeds = pickle.load(evkd_file)
                evkd_file.close()
                for c in evokeds:
                    times = evokeds[c].times
                    sdata.setdefault(c,[])
                    sdata[c] += [evokeds[c]]
                scodes += [sub]
            except Exception as e:
                print('could not load subject {}'.format(sub_code))
                print(e)
                continue
    try:
        print('converting to array')
        scodes = np.array(scodes)
        print('converted to array')
        
    except Exception as ee:
        print(ee)
    print('loaded data for {} subjects'.format(len(sdata[c])))
    return sdata, scodes, np.array(times)

def load_scores(suffix, subs, exclude=[]):
    
    project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
    project_dir = '/projects/' + project
    avg_path = project_dir + '/scratch/working_memory/averages/data/'
    sdata, scodes = {}, []
    qr = Query(project)
    sub_codes = qr.get_subjects()
    for sub in subs:
        sub_code = sub_codes[sub-1]
        if sub not in exclude:
            try:
                print('loading subject {}'.format(sub))
                
                #score_fname = op.join(avg_path,sub_code + '_scores_imagined_smoothing25_50_hp005.p')
                score_fname = op.join(avg_path,sub_code,sub_code + '_scores_' + suffix + '.p')
                score_file = open(score_fname,'rb')
                score = pickle.load(score_file)
                if len(scodes) == 0:
                    times_fname = op.join(avg_path,sub_code,sub_code + '_times_' + suffix + '.p')
                    times_file = open(times_fname,'rb')
                    times = pickle.load(times_file)
                score_file.close()
                times_file.close()
                for c in score:
                    sdata.setdefault(c,[])
                    sdata[c].append(score[c].data)
                scodes += [sub]
            except Exception as e:
                print('could not load subject {}'.format(sub_code))
                print(e)
                continue
    try:
        print('converting to array')
        sdata = {s: np.array(sdata[s]) for s in sdata}
        scodes = np.array(scodes)
        print('converted to array')
        
    except Exception as ee:
        print(ee)
    print('loaded data for {} subjects'.format(sdata[c].shape[0]))
    return sdata, scodes, times

def load_connectivity(subs, suffix='', exclude=[]):
    
    project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
    project_dir = '/projects/' + project
    avg_path = project_dir + '/scratch/working_memory/averages/data/'
    sdata, scodes = {}, []
    qr = Query(project)
    sub_codes = qr.get_subjects()
    for sub in subs:
        sub_code = sub_codes[sub-1]
        if sub not in exclude:
            try:
                print('loading subject {}'.format(sub))
                conn_fname = op.join(avg_path,sub_code,sub_code + '_conn' + suffix + '.p')
                conn_file = open(conn_fname,'rb')
                conn = pickle.load(conn_file)
                conn_file.close()
      
                for c in conn:
                    for p in conn[c]:
                        ccond = c+'_'+p
                        times = conn[c][p].times
                        names = conn[c][p].names
                        sdata.setdefault(ccond,[])
                        sdata[ccond].append(conn[c][p].get_data())
                scodes += [sub]
            except Exception as e:
                print('could not load subject {}'.format(sub_code))
                print(e)
                continue
    try:
        print('converting to array')
        sdata = {s: np.array(sdata[s]) for s in sdata}
        scodes = np.array(scodes)
        print('converted to array')
        
    except Exception as ee:
        print(ee)
    print('loaded data for {} subjects'.format(sdata[ccond].shape[0]))
    return sdata, scodes, np.array(times), names

def load_scores_compare(block, mode, pars, subs, rois=[''], exclude=[], mask_type='include'):
    
    project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
    project_dir = '/projects/' + project
    avg_path = project_dir + '/scratch/working_memory/averages/data/'
    sdata, scodes, times = {}, {}, {}
    qr = Query(project)
    sub_codes = qr.get_subjects()
    for sub in subs:
        sub_code = sub_codes[sub-1]
        if sub not in exclude:
            for roi in rois:    
                try:
                    print('loading subject {} roi {}'.format(sub,roi))
                    if len(roi) > 0:
                        roi = '_' + mask_type + '_' + roi
                    #score_fname = op.join(avg_path,sub_code + '_scores_imagined_smoothing25_50_hp005.p')
                    fend = block + '_' + mode + '_' + pars + roi + '.p'
                    score_fname = op.join(avg_path,sub_code,sub_code + '_scores_' + fend )
                    score_file = open(score_fname,'rb')
                    score = pickle.load(score_file).copy()
                    score_file.close()
                    for c in score:
                        cc = c + roi
                        sdata.setdefault(cc,[])
                        scodes.setdefault(cc,[]) 
                        if len(scodes[cc]) == 0:
                            times_fname = op.join(avg_path,sub_code,sub_code + '_times_'+ fend)
                            times_file = open(times_fname,'rb')
                            times = pickle.load(times_file)
                            times_file.close()
                        scodes[cc] += [sub]
                        sdata[cc].append(score[c].data)

                except Exception as e:
                    print('could not load subject {} roi {}'.format(sub_code, roi))
                    print(e)
                    continue
    try:
        print('converting to array')
        sdata = {s: np.array(sdata[s]) for s in sdata}
        scodes = {s: np.array(scodes[s]) for s in scodes}
        print('converted to array')
        
    except Exception as ee:
        print(ee)
    print('loaded data for {} subjects'.format(scodes[cc].shape[0]))
    return sdata, scodes, times

def load_predicted_proba(subs, exclude=[]):
    
    project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
    project_dir = '/projects/' + project
    avg_path = project_dir + '/scratch/working_memory/averages/data/'
    sprobs, sevents, sbetas, stimes, sdelays, scodes = {}, {}, {}, {}, {}, []
    qr = Query(project)
    sub_codes = qr.get_subjects()
    for sub in subs:
        sub_code = sub_codes[sub-1]
        if sub not in exclude:
            try:
                print('loading subject {}'.format(sub))
                
                #score_fname = op.join(avg_path,sub_code + '_scores_imagined_smoothing25_50_hp005.p')
                score_fname = op.join(avg_path,sub_code,sub_code + '_predicted_proba.p')
                score_file = open(score_fname,'rb')
                score = pickle.load(score_file)
                score_file.close()
                for c in score['probs']:
                    sprobs.setdefault(c,[])
                    sevents.setdefault(c,[])
                    sbetas.setdefault(c,[])
                    stimes.setdefault(c,score['times'])
                    sdelays.setdefault(c,score['delays'])

                    sprobs[c] += [score['probs'][c]]
                    sevents[c] += [np.array(score['events'][c][:,2])]
                    sbetas[c] += [score['betas'][c]]
                    
                scodes += [sub]
            except Exception as e:
                print('could not load subject {}'.format(sub_code))
                print(e)
                continue
    try:
        print('converting to array')
        probs = {s: np.concatenate(sprobs[s],axis=1) for s in sprobs}
        betas = {s: np.concatenate(sbetas[s],axis=1) for s in sbetas}
        events = {s: np.concatenate(sevents[s]) for s in sevents}
        codes = np.array(scodes)
        print('converted to array')
        
    except Exception as ee:
        print(ee)
    print('loaded data for {} subjects'.format(codes.shape[0]))
    return probs, betas, events, stimes, sdelays, codes


def grand_avg_scores(sdata):
    
    # grand averages:
    smean, sstd, smedian, sci_lower, sci_upper, siqr_lower, siqr_upper = {},{},{},{},{},{},{}
    for s in sdata:
        smean[s] = np.mean(sdata[s],0)
        smedian[s] = np.median(sdata[s],0)
        sstd[s] = np.std(sdata[s],0)
        sci_lower[s] = smean[s] - 1.96*sstd[s]/np.sqrt(sdata[s].shape[0]-1)
        sci_upper[s] = smean[s] + 1.96*sstd[s]/np.sqrt(sdata[s].shape[0]-1)
        siqr_lower[s] = np.percentile(sdata[s],25,0)
        siqr_upper[s] = np.percentile(sdata[s],75,0)
        
    return smean, sstd, sci_lower, sci_upper, smedian, siqr_lower, siqr_upper

def do_stats(X, method='FDR', adjacency=None, FDR_alpha=.025, h0=0,sigma=1e-3,n_jobs=-1,
             cluster_alpha = .05, p_threshold = .05, n_permutations=500, cluster_method = 'normal'):
    
    n_subjects = X.shape[0]
    if cluster_method == 'normal':
        t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
        print('Two sided alpha level: {} - t threshold: {}'.format(p_threshold,t_threshold))

    elif cluster_method == 'TFCE':
        t_threshold = dict(start=0, step=0.2)
        print('Two sided alpha level: {} - Using TFCE'.format(p_threshold))

    stat_fun = partial(ttest_1samp_no_p, sigma=sigma)
    if method == 'montecarlo':
        print('Clustering.')
        if len(X.shape) == 3:
            X = X.transpose(0,2,1)
            
        tvals, clusters, cluster_p_values, H0 = \
            spatio_temporal_cluster_1samp_test(X-h0, adjacency=adjacency, n_jobs=n_jobs,
                                                threshold=t_threshold, buffer_size=None,
                                                verbose=True, n_permutations = n_permutations,
                                                out_type='mask', stat_fun=stat_fun)
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
        print('\nPerforming FDR correction\n')
        tvals, pvals = stats.ttest_1samp(X, popmean=h0,axis=0)
        gmask, adj_pvals = fdr_correction(pvals, FDR_alpha)
        print(np.sum(gmask==0))
        stat_results = {'mask': gmask, 'tvals': tvals, 'pvals': pvals, 'qvals': adj_pvals,
                        'data_mean': np.mean(X,0), 'data_sd': np.std(X,0), 'n': n_subjects,
                        'alpha': FDR_alpha}
    return stat_results


class LinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def __init__(self, *args, **kwargs):
        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = True
        super(LinearRegression, self)\
                .__init__(*args, **kwargs)

    def fit(self, X2, y, n_jobs=1):
        # Deal with missing values
        imp = Imputer() 
        Ximp = imp.fit_transform(X2.copy())

        # fit
        self = super(LinearRegression, self).fit(Ximp, y)#, n_jobs)

        # Calculate standard error
        df = float(X2.shape[0] - X2.shape[1] - 1)
        se = np.sqrt(np.sum((self.predict(Ximp) - y) ** 2, axis=0) / (np.sum((X2-np.mean(X2,axis=0,keepdims=True))**2,axis=0) * df)) 
        print(se)
        # sse = np.sum((self.predict(Ximp) - y) ** 2, axis=0) / float(Ximp.shape[0] - Ximp.shape[1]+1)
        # se = np.array([
        #     np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(Ximp.T, Ximp))))
        #                                             for i in range(sse.shape[0])
        #             ])
        # calculate tvals
        #print('negative betas: ', np.sum(self.coef_ < 0))
        self.t = self.coef_ / np.squeeze(se)
        #self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - Ximp.shape[1]))
        return self

def reg_fun(Y, preds):
    lm = LinearRegression(positive=False)
    lm.fit(X2=preds,y=Y)
    return lm.t

def do_regression(X, preds, adjacency=None, threshold_alpha = .025,  cluster_alpha = .05, n_permutations=500):
    xshape = X.shape
    n_subjects = X.shape[0]
    
    def c_reg_fun(X1):
        tvals = reg_fun(X1, preds)
        return tvals#.reshape((xshape[1],xshape[2],-1))
    
    threshold = stats.distributions.t.ppf(1 - threshold_alpha, preds.shape[0]-preds.shape[1])
    print('threshold: ', threshold)
    print('Clustering.')
    if len(X.shape) == 3:
        X = X.transpose(0,2,1)
        tvals, clusters, cluster_p_values, H0 = \
                   spatio_temporal_cluster_test([X], adjacency=adjacency, stat_fun=c_reg_fun, n_jobs=-1,
                                                threshold=threshold, buffer_size=None,
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
                        'n': n_subjects,'alpha': cluster_alpha,'threshold': threshold}
        return stat_results