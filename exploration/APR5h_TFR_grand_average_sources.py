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
import pickle
from copy import deepcopy
from sys import argv
from do_stats import do_stats
os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt5'

proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
wdir = '/projects/' + proj_name + '/scratch/working_memory/'
data_dir = wdir + 'averages/data/'
subs_dir = '/projects/' + proj_name + '/scratch/fs_subjects_dir/'
sample_path = sample.data_path()
sample_subjects_dir = sample_path + '/subjects'
src_sample = mne.read_source_spaces(sample_subjects_dir +
                                '/fsaverage/bem/fsaverage-vol-5-src.fif')
stats_dir = wdir + 'results/stats/'
#Get subjects:
qr = Query(proj_name)
subjects = qr.get_subjects()

subs = range(21,91) #, 27, 28, 29, 30, 31, 32, 33, 34, 35]
b = 'delta'
times = [2, 4]
if len(argv) > 1:
    b = argv[1]

if len(argv) > 2:
   times = [float(argv[2]), float(argv[3])]

method = 'montecarlo'
all_data = {}
for sidx,s in enumerate(subs):
    try:
        scode = subjects[s-1]
        dfname = data_dir + scode + '_TFR_src2_' + b + '.p'
        print('loading file {}'.format(dfname))
        dfile = open(dfname,'rb')
        curdata = load(dfile)
        dfile.close()
        morph = mne.read_source_morph(subs_dir + scode + '/bem/' + scode + '_vol-morph.h5')
        morph_mat = morph.vol_morph_mat
        if sidx == 0:
            all_data =  {}
        for cd in curdata:
            cdata = morph_mat.dot(curdata[cd].crop(times[0],times[1]).data)#deepcopy(cmorphed.data) #morph_mat.dot(c.data)
            print('appending subject {} band {} condition {}'.format(scode,b,cd))
            all_data.setdefault(cd,np.array([cdata]))
            if sidx > 0:
                all_data[cd] = np.vstack((all_data[cd],np.array([cdata])))
    except Exception as e:
        print(e)
        continue

# Do grand average
grand_avg = {}
for cd in all_data:
    grand_avg[cd] = deepcopy(morph.apply(curdata[cd].crop(times[0],times[1])))
    #grand_avg[b][cd].subject = 'sample'
    grand_avg[cd].data = np.mean(all_data[cd],0)
    print(grand_avg[cd].data.shape)

adjacency = mne.spatial_src_adjacency(src_sample)
stats_names = ['maintenance','manipulation','difference']
conds_math = [['same','different'],['inverted','other'],['same','different','inverted','other']]
conds_op = [['+',''],['+',''],['+','-','-','']]

stat_results = {}
for sidx, sn in enumerate(stats_names):
    print('Computing stats for the comparison: ' + sn + '\n\n')
    math_cmd = 'cdata = ('
    for cidx, cd in enumerate(conds_math[sidx]):
        math_cmd += 'all_data["' + cd + '"]' + conds_op[sidx][cidx]
    math_cmd += ')/2'
    print('executing the following command:\n\n' + math_cmd)
    exec(math_cmd)
    print(cdata.shape)
    #stat_results[sn] = do_stats(cdata, method='FDR',adjacency=adjacency)
    stat_results[sn] = do_stats(cdata, method=method,adjacency=adjacency,n_permutations=1000)

print('\nsaving stats file\n\n')
stats_file = '{}TFR_{}_{}_{}-{}.py'.format(stats_dir, b, method, np.round(times[0],2), np.round(times[1],2))
sfile = open(stats_file,'wb')
pickle.dump(stat_results,sfile)
sfile.close()

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
