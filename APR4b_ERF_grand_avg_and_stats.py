#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import mne
import os
import os.path as op
import numpy as np
from scipy import stats
import pickle
from warnings import filterwarnings
from sys import argv
import matplotlib.pyplot as plt
from stormdb.access import Query
from do_stats import do_stats
filterwarnings("ignore", category=DeprecationWarning)

project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
project_dir = '/projects/' + project
os.environ['MINDLABPROJ']= project
os.environ['MNE_ROOT']= '~/miniconda3/envs/mne' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

avg_path = project_dir + '/scratch/working_memory/averages/data/'
stats_dir = project_dir + '/scratch/working_memory/results/stats/'
qr = Query(project)
sub_codes = qr.get_subjects()

## load data
sub_Ns = np.arange(11,91) #[2,11,12,13,14,15,16]#np.arange(8) + 1
exclude = np.array([55,60,73,82]) # subjects with low maintenance accuracy
gdata = {}
garray = {}
scount = 0
for sub in sub_Ns:
    sub_code = sub_codes[sub-1]
    if sub not in exclude:
        try:
            print('loading subject {}'.format(sub_code))
            evkd_fname = op.join(avg_path,sub_code + '_evoked.p')
            evkd_file = open(avg_path + sub_code + '_evoked.p','rb')
            evokeds = pickle.load(evkd_file)
            evkd_file.close()
            scount = scount +1
            for e in evokeds:
                if scount == 1:
                    gdata[e] = []
                    garray[e] = []
                gdata[e].append(evokeds[e].data)
                garray[e].append(evokeds[e])
        except:
            print('could not load subject {}'.format(sub_code))
            continue
for g in gdata:
    gdata[g] = np.array(gdata[g])
    
## Do some stats
# def do_stats(X,method,adjacency=None,FDR_alpha=.025):
#     n_subjects = X.shape[0]
#     p_threshold = 0.001
#     t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
#     if method == 'montecarlo':
#         print('Clustering.')
#         T_obs, clusters, cluster_p_values, H0 = \
#             spatio_temporal_cluster_1samp_test(X, adjacency=adjacency, n_jobs=1,
#                                                 threshold=t_threshold, buffer_size=None,
#                                                 verbose=True, n_permutations = 100,out_type='mask')
#         good_cluster_inds = np.where(cluster_p_values < 0.025)[0]
#         gclust = np.array([clusters[c] for c in good_cluster_inds])
#         gmask = np.zeros(X.shape[1:]).T

#         if gclust.shape[0] > 0:
#           for tc in range(gclust.shape[0]):
#               gmask = gmask + gclust[tc].astype(float).T
#         stat_results = {'mask': gmask, 'tvals': T_obs, 'pvals': cluster_p_values,
#                         'data_mean': np.mean(X,0).T, 'data_sd': np.std(X,0).T}

#     elif method == 'FDR':
#         print('\nPerforming FDR correction\n.')
#         tvals, pvals = stats.ttest_1samp(X, 0)
#         gmask, adj_pvals = mne.stats.fdr_correction(pvals.T, FDR_alpha)
#         stat_results = {'mask': gmask.T, 'tvals': tvals, 'pvals': pvals.T, 'qvals': adj_pvals.T,
#                         'data_mean': np.mean(X,0), 'data_sd': np.std(X,0), 'n': X.shape[0],
#                         'FDR_alpha': FDR_alpha}
#     return stat_results

### Stats on main comparisons
conds = ['maint', 'manip', 'difference']
gdata['difference'] = gdata['manip'] - gdata['maint']
cnames = ['maintenance','manipulation','difference']
alphas = [.001,.001,.025]
stat_results = {}
for cidx, cnd in enumerate(conds):
    cname = cnames[cidx]
    cdata = np.array(gdata[cnd].copy())
    print(cdata.shape)
    stat_results[cname] = do_stats(cdata, 'FDR', FDR_alpha=alphas[cidx])
    print('reporting stats for {}:\n\n'.format(cname))
    print('minimum pval = ', np.round(np.min(stat_results[cname]['pvals']),2))
    print('minimum qval = ', np.round(np.min(stat_results[cname]['qvals']),2))
    print('minimum tstat = ', np.round(np.min(stat_results[cname]['tvals']),2))
    print('maximum tstat = ', np.round(np.max(stat_results[cname]['tvals']),2))

print('saving stats results')
stats_fname = op.join(stats_dir,'ERF_sensor_stats.p')
sfile = open(stats_fname,'wb')
pickle.dump(stat_results,sfile)
sfile.close()

## Stats on target types
tidx = [x and y for x,y in zip(garray['maint'][0].times >= 4, garray['maint'][0].times <= 6)]
conds = {'maint': ['same', 'diff1', 'diff2'],
         'manip' : ['inv', 'other1', 'other2']}
target_results = {}
for cnd in conds:
    for eix1, e1 in enumerate(conds[cnd]):
        for eix2, e2 in enumerate(conds[cnd]):
            if eix2 > eix1:
                cdata = (np.mean([gdata[cnd + '/mel1/' + e2],gdata[cnd + '/mel2/' + e2]],axis=0) - 
                         np.mean([gdata[cnd + '/mel1/' + e1],gdata[cnd + '/mel2/' + e1]],axis=0))
                cdata = cdata[:, :, tidx]
                dname = e1 + '-' + e2
                target_results[cnd + '/' + dname] = do_stats(cdata, 'FDR')
                print('reporting stats for {}, {} - {}:\n'.format(cnd,e2,e1))
                print('minimum pval = ', np.round(np.min(target_results[cnd + '/' + dname]['pvals']),2))
                print('minimum qval = ', np.round(np.min(target_results[cnd + '/' + dname]['qvals']),2))
                print('minimum tstat = ', np.round(np.min(target_results[cnd + '/' + dname]['tvals']),2))
                print('maximum tstat = ', np.round(np.max(target_results[cnd + '/' + dname]['tvals']),2))

print('saving stats results')
target_fname = op.join(stats_dir,'ERF_target_sensor_stats.p')
tfile = open(target_fname, 'wb')
pickle.dump(target_results,tfile)
tfile.close()

tidx = [x and y for x,y in zip(garray['maint'][0].times >= -.5, garray['maint'][0].times <= 4)]
conds = {'maint': ['mel1', 'mel2'],
         'manip' : ['mel1', 'mel2']}            
mel_results = {}
ch_types = ['mag','grad']
for cht in ch_types:
    adjacency, ch_names = mne.channels.find_ch_adjacency(garray['maint'][0].info,ch_type=cht)
    chix = [cch in ch_names for cch in garray['maint'][0].ch_names]
    mel_results[cht] = {}
    for cnd in conds:
        for eix1, e1 in enumerate(conds[cnd]):
            cdata = np.array(gdata[cnd + '/' + e1])
            cdata = cdata[:, chix, :]
            cdata = cdata[:, :, tidx]
            dname = e1
#                mel_results[cnd + '/' + dname] = do_stats(cdata, 'FDR')
            mel_results[cht][cnd + '/' + dname] = do_stats(cdata, method='montecarlo', adjacency=adjacency,
                                                          n_permutations = 2000)
            print('reporting stats for {}, {}:\n'.format(cnd,e1))
            print('minimum pval = ', np.round(np.min(mel_results[cht][cnd + '/' + dname]['pvals']),2))
            #print('minimum qval = ', np.round(np.min(mel_results[cnd + '/' + dname]['qvals']),2))
            print('minimum tstat = ', np.round(np.min(mel_results[cht][cnd + '/' + dname]['tvals']),2))
            print('maximum tstat = ', np.round(np.max(mel_results[cht][cnd + '/' + dname]['tvals']),2))

conds = {'maint': ['mel1', 'mel2'],
         'manip' : ['mel1', 'mel2']}            
ch_types = ['mag','grad']
for cht in ch_types:
    adjacency, ch_names = mne.channels.find_ch_adjacency(garray['maint'][0].info,ch_type=cht)
    chix = [cch in ch_names for cch in garray['maint'][0].ch_names]
    #mel_results[cht] = {}
    for cnd in conds:
        for eix1, e1 in enumerate(conds[cnd]):
            for eix2, e2 in enumerate(conds[cnd]):
                if eix2 > eix1:
                    cdata = np.array(gdata[cnd + '/' + e2]) - np.array(gdata[cnd + '/' + e1])
                    cdata = cdata[:, chix, :]
                    cdata = cdata[:, :, tidx]
                    dname = e1 + '-' + e2
    #                mel_results[cnd + '/' + dname] = do_stats(cdata, 'FDR')
                    mel_results[cht][cnd + '/' + dname] = do_stats(cdata, method='montecarlo', adjacency=adjacency,
                                                                  n_permutations = 5000)
                    print('reporting stats for {}, {} - {}:\n'.format(cnd,e2,e1))
                    print('minimum pval = ', np.round(np.min(mel_results[cht][cnd + '/' + dname]['pvals']),2))
                    #print('minimum qval = ', np.round(np.min(mel_results[cnd + '/' + dname]['qvals']),2))
                    print('minimum tstat = ', np.round(np.min(mel_results[cht][cnd + '/' + dname]['tvals']),2))
                    print('maximum tstat = ', np.round(np.max(mel_results[cht][cnd + '/' + dname]['tvals']),2))

print('saving stats results')
mels_fname = op.join(stats_dir,'ERF_mels_sensor_stats.p')
mfile = open(mels_fname, 'wb')
pickle.dump(mel_results,mfile)
mfile.close()

# print('\nsaving stats file\n\n')
# stats_file = '{}TFR_{}_{}-{}.py'.format(stats_dir, b, np.round(times[0],2), np.round(times[1],2))
# sfile = open(stats_file,'wb')
# pickle.dump(stat_results,sfile)
# sfile.close()

# Grand averages
# grand_avg = {}
# for e in garray:
#     grand_avg[e] = {}
#     for c in garray[e]:
#         grand_avg[e][c] = mne.grand_average(garray[e][c])
#         grand_avg[e][c].data = np.mean(np.array(gdata[e][c]),0)
#         grand_avg[e][c].comment = garray[e][c][0].comment
#
# mne.viz.plot_evoked_topo([grand_avg['main']['all'].copy().pick_types('mag').crop(-0.5,2),
#                           grand_avg['inv']['all'].copy().pick_types('mag').crop(-0.5,2)])
#
# mne.viz.plot_evoked_topo(grand_avg['difference']['all'].copy().pick_types('mag').crop(-0.5,2))
# grand_avg['difference']['all'].copy().pick_types('grad').crop(-0.5,2).plot_joint()
#
# mne.viz.plot_evoked_topo([grand_avg['main']['all'].copy().pick_types('mag').crop(2,4.5),
#                           grand_avg['inv']['all'].copy().pick_types('mag').crop(2,4.5)])
# mne.viz.plot_evoked_topo(grand_avg['difference']['all'].copy().pick_types('grad').crop(2,4))
# grand_avg['difference']['all'].copy().pick_types('grad').crop(2,4.5).plot_joint()
#
# mne.viz.plot_evoked_topo([grand_avg['main']['different1'].copy().pick_types('mag').crop(4,6.5),
#                           grand_avg['main']['different2'].copy().pick_types('mag').crop(4,6.5)])
# mne.viz.plot_evoked_topo([grand_avg['inv']['other1'].copy().pick_types('mag').crop(4,6.5),
#                           grand_avg['inv']['other2'].copy().pick_types('mag').crop(4,6.5)])
#
# mne.viz.plot_evoked_topo(grand_avg['difference']['all'].copy().pick_types('grad').crop(4,6.5))
# grand_avg['difference']['all'].copy().pick_types('grad').crop(2,4.5).plot_joint()
#
# mne.viz.plot_evoked_topo([grand_avg['main']['all'].copy().pick_types('grad'),
#                           grand_avg['inv']['all'].copy().pick_types('grad')],
#                           merge_grads=True)
# mne.viz.plot_evoked_topo(grand_avg['difference']['all'].copy().pick_types('grad'),
#                          merge_grad=True)
#
# grand_avg['difference']['all'].copy().pick_types('grad').crop(-0.5,2).plot_joint
#
# # Encoding plots:
# chans = ['MEG1341','MEG0311','MEG2241']
# for ch in chans:
#     mne.viz.plot_compare_evokeds([grand_avg['main']['all'].copy().crop(-0.5,2),
#                                   grand_avg['inv']['all'].copy().crop(-0.5,2)],
#                                   picks=ch)
#
# grand_avg['difference']['all'].copy().pick_types('mag').plot_topomap(times=[0.25,0.5,0.75,1])
# grand_avg['difference']['all'].copy().pick_types('grad').plot_topomap(times=[0.25,0.5,0.75,1])
# grand_avg['main']['all'].copy().pick_types('mag').plot_topomap(times=[0.125,0.21,0.45])
# grand_avg['main']['all'].copy().pick_types('grad').plot_topomap(times=[0.125,0.21,0.45])
#
# # Mainteance/manipulation plots:
# chans = ['MEG1341','MEG0311','MEG2241']
# for ch in chans:
#     mne.viz.plot_compare_evokeds([grand_avg['main']['all'].copy().crop(2,4),
#                                   grand_avg['inv']['all'].copy().crop(2,4)],
#                                   picks=ch)
# grand_avg['difference']['all'].copy().pick_types('mag').plot_topomap(times=np.arange(2,4,0.25),
#                                                                      average=0.25)
# grand_avg['difference']['all'].copy().pick_types('grad').plot_topomap(times=np.arange(2,4,0.25),
#                                                                      average=0.25)
# grand_avg['main']['all'].copy().pick_types('mag').plot_topomap(times=np.arange(2,4,0.25),
#                                                                      average=0.25)
# grand_avg['main']['all'].copy().pick_types('grad').plot_topomap(times=np.arange(2,4,0.25),
#                                                                      average=0.25)
# grand_avg['main']['all'].copy().pick_types('mag').plot_topomap(times=3)
# grand_avg['main']['all'].copy().pick_types('grad').plot_topomap(times=3)
#
# # retrieval plots:
# chans = ['MEG1341','MEG0311','MEG2241']
# for ch in chans:
#     mne.viz.plot_compare_evokeds([grand_avg['main']['all'].copy().crop(4,6.5),
#                                   grand_avg['inv']['all'].copy().crop(4,6.5)],
#                                   picks=ch)
# grand_avg['difference']['all'].copy().pick_types('mag').plot_topomap(times=[4.75,5,5.25,5.5,5.75])
# grand_avg['difference']['all'].copy().pick_types('grad').plot_topomap(times=[4.75,5,5.25,5.5,5.75])
# grand_avg['main']['all'].copy().pick_types('mag').plot_topomap(times=5.4)
# grand_avg['main']['all'].copy().pick_types('grad').plot_topomap(times=5.4)
#
# # Dev type plots:
# chans = ['MEG0431']#,'MEG0311','MEG2241']
# for ch in chans:
#     mne.viz.plot_compare_evokeds([grand_avg['main']['same'].copy().crop(4,6.5),
#                                   grand_avg['main']['different1'].copy().crop(4,6.5),
#                                   grand_avg['main']['different2'].copy().crop(4,6.5)],
#                                   picks=ch)
#     mne.viz.plot_compare_evokeds([grand_avg['inv']['inverted'].copy().crop(4,6.5),
#                               grand_avg['inv']['other1'].copy().crop(4,6.5),
#                               grand_avg['inv']['other2'].copy().crop(4,6.5)],
#                               picks=ch)

## compute group sources:

# Download fsaverage files


# The files live in:
# subjects_dir = op.join(project_dir,'scratch/fs_subjects_dir')
# inv_file = open(avg_path + '0002_BYG_evoked_inverse.p','rb')
# inv = pickle.load(inv_file)
# inv_file.close()

# src_file = open(avg_path + '0002_BYG_evoked_sources.p','rb')
# sources = pickle.load(src_file)
# src_file.close()

# SNR = 3
# src_diff = sources['inv']['all']-sources['main']['all']
# sources['main']['all'].plot(subjects_dir=subjects_dir,initial_time=0.11,hemi = 'split',
#                             time_viewer=True, views=['lateral','medial'])
# sources['inv']['all'].plot(subjects_dir=subjects_dir,initial_time=0.11,hemi = 'split',
#                             time_viewer=True, views=['lateral','medial'])

# src_diff.plot(subjects_dir=subjects_dir,initial_time=0.11,hemi = 'split',
#               time_viewer=True, views=['lateral','medial'])



# src_diff2 = sources['main']['different2']-sources['main']['different1']
# src_diff3 = sources['inv']['other2']-sources['inv']['other1']
# src_diff2.plot(subjects_dir=subjects_dir,initial_time=5.22,hemi = 'split',
#                             time_viewer=True, views=['lateral','medial'])
# src_diff3.plot(subjects_dir=subjects_dir,initial_time=5.22,hemi = 'split',
#                             time_viewer=True, views=['lateral','medial'])
