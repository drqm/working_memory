#!/usr/bin/env python3
# -*- coding: utf-8 -*-
proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
wdir = '/projects/' + proj_name + '/scratch/working_memory/'
scripts_dir = '/projects/' + proj_name + '/scripts/working_memory/'
import sys
sys.path.append(scripts_dir)

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
from random import choices

os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt5'

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
stats_dir = wdir + 'results/stats/'
#Get subjects:
qr = Query(proj_name)
subjects = qr.get_subjects()

subs = range(11,91)#91)#91) #, 27, 28, 29, 30, 31, 32, 33, 34, 35]
performance_exc = [55,58,60,73,76,82]
no_source_exc = [30,51,42]
noise_exc = [15]
no_data_exc = [32,33]

exclude = np.array(performance_exc + no_source_exc + noise_exc + no_data_exc)
subs = np.array([s for s in subs if s not in exclude])
subs.shape
avg = 1
add_flip = False
periods = {'all': [0, 4],
           'encoding': [0, 2],
           'S1a': [.15,.25],
           'S1b': [.3,.5],
           'S2a': [.65,.75],
           'S2b': [.8,1.],
           'S3a': [1.15,1.25],
           'S3b': [1.3,1.5],
           'L1': [.2,.5],
           'L2': [.7,1],
           'L3': [1.2,1.5],
           'I1': [2,4],
           'I2': [2.5,3.5]}

period = 'I1'
mode = 'patterns'
suffix = 'sources_task_sensor_lf_0.05_hf_None_tstep_0.025_twin_0.05_localized'
fsuffix = 'sources_task_sensor_lf_0.05_hf_None_tstep_0.025_twin_0.05_flipped_localized'
suffix2 = '_peaks'

if len(argv) > 1:
    period = argv[1]

if len(argv) > 2:
    mode = argv[2]

if len(argv) > 3:
    avg = int(argv[3])

times = periods[period]

print('grand average analyses for {} ({}-{} s)'.format(period, times[0], times[1]))

all_data = {}
for sidx,s in enumerate(subs):
    try:
        scode = subjects[s-1]
        morph = mne.read_source_morph(subs_dir + scode + '/bem/' + scode + '_vol-morph.h5')
        morph_mat = morph.vol_morph_mat
        dfname = op.join(data_dir,scode, scode + '_' + mode + '_' + suffix + '.p')
        print('\n\nloading file {}\n'.format(dfname))
        dfile = open(dfname,'rb')
        curdata = load(dfile)

        if add_flip:
            flipped_fname = op.join(data_dir,scode, scode + '_' + mode + '_' + fsuffix + '.p')
            print('\nloading file {}\n'.format(flipped_fname))
            flipped_file = open(flipped_fname,'rb')
            fdata = load(flipped_file)
            curdata['flipped1'] = fdata['flipped1'].copy()
            del fdata

        for cd in curdata:
            cdata = morph_mat.dot(curdata[cd].crop(times[0],times[1]).data)#deepcopy(cmorphed.data) #morph_mat.dot(c.data)
            if avg == 1:
                cdata = cdata.mean(axis=1,keepdims=True)
            print('appending subject {} condition {}'.format(scode,cd))
            all_data.setdefault(cd,np.array([cdata]))
            #all_data[cd][cd2].append([cdata])
            if sidx > 0:
                all_data[cd] = np.vstack((all_data[cd],np.array([cdata])))
    except Exception as e:
        print(e)
        continue
        
adjacency = mne.spatial_src_adjacency(src_sample)

### Stats on main comparisons
#conds = [k for k in all_data]
conds = ['interaction','maintenance1','manipulation1']
#alphas = [.025,.025,.025,.025,.025,.025,.025]
stat_results = {}
for cidx, cnd in enumerate(conds):
    cdata = all_data[cnd]
    print(cdata.shape)
    if cnd == 'interaction':
         cdata = all_data['manipulation1'] - all_data['maintenance1']
    #stat_results[cnd] = gs.do_stats(cdata, 'FDR', adjacency=adjacency, FDR_alpha=.025)
    stat_results[cnd] = gs.do_stats(cdata, 'montecarlo', adjacency=adjacency,
                                     FDR_alpha=.025, n_permutations=5000, cluster_alpha=.05)#, cluster_method='TFCE')
    print('reporting stats for {}:\n'.format(cnd))
    print('minimum pval = ', np.round(np.min(stat_results[cnd]['pvals']),2))
#     print('minimum qval = ', np.round(np.min(stat_results[cnd]['qvals']),2))
    print('minimum tstat = ', np.round(np.min(stat_results[cnd]['tvals']),2))
    print('maximum tstat = ', np.round(np.max(stat_results[cnd]['tvals']),2),'\n')
    
print('saving stats results')
stats_fname = op.join(stats_dir,'{}_source_stats_{}{}.p'.format(mode,period,suffix2))
sfile = open(stats_fname,'wb')
pickle.dump(stat_results,sfile)
sfile.close()

