#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import mne
import numpy as np
from matplotlib import pyplot as plt
from stormdb.access import Query
from pickle import load
from scipy import stats
from mne.datasets import sample
from mne.stats import spatio_temporal_cluster_1samp_test
import os
import os.path as op
import pickle
from copy import deepcopy
from sys import argv
import src.group_stats as gs
os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt5'

proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
wdir = '/projects/' + proj_name + '/scratch/working_memory/'
data_dir = wdir + 'averages/data/'
subs_dir = '/projects/' + proj_name + '/scratch/fs_subjects_dir/'
sample_path = sample.data_path()
sample_subjects_dir = sample_path + '/subjects'
src_sample = mne.read_source_spaces(subs_dir +
                                     '/fsaverage/bem/fsaverage-vol-5-src.fif')
# src_sample = mne.read_source_spaces(subs_dir +
#                                     '/fsaverage/bem/fsaverage_ico4_vol-src.fif')
# src_sample = mne.read_source_spaces(subs_dir +
#                                     '/fsaverage/bem/fsaverage-vol-8-src.fif')
print(src_sample)
stats_dir = wdir + 'results/stats/'
#Get subjects:
qr = Query(proj_name)
subjects = qr.get_subjects()

subs = range(11,91)#91) #, 27, 28, 29, 30, 31, 32, 33, 34, 35]
avg = False
periods = {'encoding': [0, 2],'delay': [2, 4]}#, 'retrieval': [4, 6]}
mode = 'patterns'
suffix = 'sources_task_sensor_lf_0.05_hf_None_tstep_0.025_twin_0.05_localized'

if len(argv) > 1:
    period = argv[1]

if len(argv) > 2:
    mode = argv[2]
    


print('grand average analyses for lsitening vs imagining')

all_data = {}
for sidx,s in enumerate(subs):
    try:
        scode = subjects[s-1]
        morph = mne.read_source_morph(subs_dir + scode + '/bem/' + scode + '_vol-morph.h5')
        morph_mat = morph.vol_morph_mat
        dfname = op.join(data_dir,scode, scode + '_' + mode + '_' + suffix + '.p')
        print('loading file {}'.format(dfname))
        dfile = open(dfname,'rb')
        curdata = load(dfile)
        for cd in ['maintenance1','manipulation1']:
            times1, times2 = periods['encoding'], periods['delay']
            cdata = (morph_mat.dot(curdata[cd].crop(times2[0],times2[1]).data).mean(axis=1,keepdims=True) - 
                     morph_mat.dot(curdata[cd].crop(times1[0],times1[1]).data).mean(axis=1,keepdims=True))#deepcopy(cmorphed.data) #morph_mat.dot(c.data)
       
            cd_name = cd+ '_difference'
            print('appending subject {} condition {}'.format(scode,cd_name))
            all_data.setdefault(cd_name,np.array([cdata]))
            #all_data[cd][cd2].append([cdata])
            if sidx > 0:
                all_data[cd_name] = np.vstack((all_data[cd_name],np.array([cdata])))
        all_data['interaction'] = all_data['manipulation1_difference'] - all_data['maintenance1_difference']
    except Exception as e:
        print(e)
        continue
        
adjacency = mne.spatial_src_adjacency(src_sample)

### Stats on main comparisons
conds = [k for k in all_data]
#conds = ['maintenance','manipulation','melody','melody_maintenance','melody_manipulation','block','interaction']

#alphas = [.025,.025,.025,.025,.025,.025,.025]
stat_results = {}
for cidx, cnd in enumerate(all_data.keys()):
    cdata = all_data[cnd]
    print(cdata.shape)
    #stat_results[cnd] = gs.do_stats(cdata, 'FDR', adjacency=adjacency, FDR_alpha=.025)
    stat_results[cnd] = gs.do_stats(cdata, 'montecarlo', adjacency=adjacency, FDR_alpha=.025)
    print('reporting stats for {}:\n\n'.format(cnd))
#     print('minimum pval = ', np.round(np.min(stat_results[cnd]['pvals']),2))
#     print('minimum qval = ', np.round(np.min(stat_results[cnd]['qvals']),2))
    print('minimum tstat = ', np.round(np.min(stat_results[cnd]['tvals']),2))
    print('maximum tstat = ', np.round(np.max(stat_results[cnd]['tvals']),2))
    
print('saving stats results')
stats_fname = op.join(stats_dir,'{}_source_stats_list_vs_im.p'.format(mode))
sfile = open(stats_fname,'wb')
pickle.dump(stat_results,sfile)
sfile.close()

# ## Stats on target types
# if period == 'retrieval':
#     conds = {'maint': ['same', 'diff1', 'diff2'],
#              'manip' : ['inv', 'other1', 'other2']}
#     target_results = {}
#     for cnd in conds:
#         for eix1, e1 in enumerate(conds[cnd]):
#             for eix2, e2 in enumerate(conds[cnd]):
#                 if eix2 > eix1:
#                     cdata = (np.mean([all_data[cnd + '/mel1/' + e2],all_data[cnd + '/mel2/' + e2]],axis=0) - 
#                              np.mean([all_data[cnd + '/mel1/' + e1],all_data[cnd + '/mel2/' + e1]],axis=0))
#                     dname = e1 + '-' + e2
#                     target_results[cnd + '/' + dname] = do_stats(cdata, 'FDR')
#                     print('reporting stats for {}, {} - {}:\n'.format(cnd,e2,e1))
#                     print('minimum pval = ', np.round(np.min(target_results[cnd + '/' + dname]['pvals']),2))
#                     print('minimum qval = ', np.round(np.min(target_results[cnd + '/' + dname]['qvals']),2))
#                     print('minimum tstat = ', np.round(np.min(target_results[cnd + '/' + dname]['tvals']),2))
#                     print('maximum tstat = ', np.round(np.max(target_results[cnd + '/' + dname]['tvals']),2))

#     print('saving stats results')
#     target_fname = op.join(stats_dir,'ERF_target_source_stats.p')
#     tfile = open(target_fname, 'wb')
#     pickle.dump(target_results, tfile)
#     tfile.close()

# conds = {'maint': ['mel1', 'mel2'],
#          'manip' : ['mel1', 'mel2']}            
# mel_results = {}
# for cnd in conds:
#     for eix1, e1 in enumerate(conds[cnd]):
#         cdata = all_data[cnd + '/' + e1]
#         dname = e1
# #                mel_results[cnd + '/' + dname] = do_stats(cdata, 'FDR')
#         mel_results[cnd + '/' + dname] = do_stats(cdata, method='FDR')
#         print('reporting stats for {}, {}:\n'.format(cnd,e1))
#         print('minimum pval = ', np.round(np.min(mel_results[cnd + '/' + dname]['pvals']),2))
#         print('minimum qval = ', np.round(np.min(mel_results[cnd + '/' + dname]['qvals']),2))
#         print('minimum tstat = ', np.round(np.min(mel_results[cnd + '/' + dname]['tvals']),2))
#         print('maximum tstat = ', np.round(np.max(mel_results[cnd + '/' + dname]['tvals']),2))

# conds = {'maint': ['mel1', 'mel2'],
#          'manip' : ['mel1', 'mel2']}            
# for cnd in conds:
#     for eix1, e1 in enumerate(conds[cnd]):
#         for eix2, e2 in enumerate(conds[cnd]):
#             if eix2 > eix1:
#                 cdata = all_data[cnd + '/' + e2] - all_data[cnd + '/' + e1]
#                 dname = e1 + '-' + e2
# #                mel_results[cnd + '/' + dname] = do_stats(cdata, 'FDR')
#                 mel_results[cnd + '/' + dname] = do_stats(cdata, method='FDR')
#                 print('reporting stats for {}, {} - {}:\n'.format(cnd,e2,e1))
#                 print('minimum pval = ', np.round(np.min(mel_results[cnd + '/' + dname]['pvals']),2))
#                 print('minimum qval = ', np.round(np.min(mel_results[cnd + '/' + dname]['qvals']),2))
#                 print('minimum tstat = ', np.round(np.min(mel_results[cnd + '/' + dname]['tvals']),2))
#                 print('maximum tstat = ', np.round(np.max(mel_results[cnd + '/' + dname]['tvals']),2))

# print('saving stats results')
# mels_fname = op.join(stats_dir,'ERF_mels_source_stats_{}.p'.format(period))
# mfile = open(mels_fname, 'wb')
# pickle.dump(mel_results,mfile)
# mfile.close()

# stat_results = {}
# for cd in all_data:
#     stat_results[cd] = {}
#     for cd2 in all_data[cd]:
#         print('Computing stats for the condition: ' + cd + ' ' + cd2)
#         cdata = all_data[cd][cd2]in
#         stat_results[cd][cd2] = do_stats(cdata, 'FDR',adjacency)
#         print(stat_results[cd][cd2])
#
# print('\nsaving stats file\n\n')
# stats_file = '{}TFR_{}_{}-{}.py'.format(stats_dir, b, np.round(times[0],2), np.round(times[1],2))
# sfile = open(stats_file,'wb')
# pickle.dump(stat_results,sfile)
# sfile.close()

# band = bands[0]
#times = [2,4]
# brain=(grand_avg[band]['same'] + grand_avg[band]['different']).plot(subjects_dir=subs_dir,#time_viewer=True,
#                          subject='fsaverage', mode = 'stat_map',src=src_sample)
# brain=(grand_avg[band]['inverted'] + grand_avg[band]['other']).plot(subjects_dir=subs_dir,#time_viewer=True,
#                          subject='fsaverage', mode = 'stat_map',src=src_sample)
#
# brain=(grand_avg[band]['inverted'] + grand_avg[band]['other'] -
#        grand_avg[band]['same'] - grand_avg[band]['different']).plot(subjects_dir=subs_dir,#time_viewer=True,
#                          subject='fsaverage',src=src_sample)
# adjacency = mne.spatial_src_adjacency(src_sample)
# results = do_stats((all_data[band]['same'] + all_data[band]['different'])/2,'FDR',adjacency)
# # results = do_stats((all_data[band]['inverted'] + all_data[band]['other'])/2,'FDR',adjacency)
# # results = do_stats((all_data[band]['inverted'] + all_data[band]['other'] -
# #                     all_data[band]['same'] - all_data[band]['different'])/2,'FDR',adjacency)
# dummy = deepcopy(morph.apply(curdata[cd].crop(times[0],times[1])))
# dummy.data=results['tvals']*results['mask']
# dummy.plot(subjects_dir=subs_dir,#time_viewer=True,
#                   subject='fsaverage',src=src_sample, mode='glass_brain')

# comparisons = [['blocal',[]],
#             ['llocal',[]],
#             ['glocal',[]],
#             ['bmixed',[]],
#             ['lmixed',[]],
#             ['gmixed',[]],
#             ['bglobal',[]],
#             ['lglobal',[]],
#             ['gglobal',[]],
#             ['blocal',['llocal']],
#             ['blocal',['glocal']],
#             ['llocal',['glocal']],
#             ['bmixed',['lmixed']],
#             ['bmixed',['gmixed']],
#             ['lmixed',['gmixed']],
#             ['bglobal',['lglobal']],
#             ['bglobal',['gglobal']],
#             ['lglobal',['gglobal']]]
#
# sresults = {}
# adjacency = mne.spatial_src_adjacency(src_sample)
# for cmp in comparisons:
#     X = deepcopy(all_data[cmp[0]])
#     cname = cmp[0]
#     if cmp[1]:
#        X = X - deepcopy(all_data[cmp[1][0]])
#        cname = cmp[0] + '-' + cmp[1][0]
#     #X = np.transpose(X[:,:,50:], [0, 2, 1])
#     X = np.transpose(X[:,:,10:], [0, 2, 1])
#     sresults[cname] = do_stats(X,method = 'montecarlo', adjacency = adjacency)
#
# stats_fname = os.path.join(wdir,'results/stats/grand_avg_patterns_stats.p')
# stats_file = open(stats_fname,'wb')
# pickle.dump(sresults,stats_file)
# stats_file.close()
# print('stats file saved')
#
# csource = deepcopy(cmorphed)
# csource.subject = 'fsaverage'
# csource.tmin = 0

# comparisons = [['blocal',['llocal']]]
# for cmp in comparisons:
#     cname = cmp[0]
#     if cmp[1]:
#        cname = cmp[0] + '-' + cmp[1][0]
#     csource.data = sresults[cname]['data_mean'] * sresults[cname]['mask']
#     splt = csource.plot_3d(src=src_sample, views = ['medial','axial'], view_layout = 'horizontal',
#                             size = (800, 300), show_traces = False, smoothing_steps = 1)

# splt.save_movie(time_dilation = 10, tmin = 0, tmax = 0.600, interpolation = 'nearest',
#                 filename = wdir + 'results/figures/stats_movie.gif')

# #splt.set_clim(0,1)
# print('Visualizing clusters.')

# #    Now let's build a convenient representation of each cluster, where each
# #    cluster becomes a "time point" in the SourceEstimate
# stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=10,
#                                               vertices=src_sample,
#                                               subject='fsaverage')

# #    Let's actually plot the first "time point" in the SourceEstimate, which
# #    shows all the clusters, weighted by duration.
# # blue blobs are for condition A < condition B, red for A > B
# brain = stc_all_cluster_vis.plot(src=src_sample, mode = 'glass_brain',#initial_time=0.51,
#                                  subjects_dir=sample_subjects_dir)
