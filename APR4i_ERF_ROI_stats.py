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
figures_dir = wdir + 'results/figures/'
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

#suffix = 'patterns_sources_task_sensor_lf_0.05_hf_None_tstep_0.025_twin_0.05_localized_ROI'
suffix = 'evoked_sources_lf_0.05_hf_None_tstep_0.025_twin_0.05_ROI'

print('ROI grand average analyses')

all_data = {}
for sidx,s in enumerate(subs):
    try:
        scode = subjects[s-1]
        dfname = op.join(data_dir,scode, scode + '_' + suffix + '.p')
        print('\n\nloading file {}\n'.format(dfname))
        dfile = open(dfname,'rb')
        curdata = load(dfile)
        for cd in curdata:
            cdata = curdata[cd]['data']#deepcopy(cmorphed.data) #morph_mat.dot(c.data)
            all_data.setdefault(cd,{})
            for ROI in cdata:
                print('appending subject {} condition {} ROI {}'.format(scode,cd,ROI))
                all_data[cd].setdefault(ROI,np.array([cdata[ROI]]))
                #all_data[cd][cd2].append([cdata])
                if sidx > 0:
                    all_data[cd][ROI] = np.vstack((all_data[cd][ROI],np.array([cdata[ROI]])))
    except Exception as e:
        print(e)
        continue

n = all_data[cd][ROI].shape[0]
times = curdata[cd]['times']
### Stats on main comparisons
#conds = [k for k in all_data]
conds = ['interaction',
         'maintenance/mel1',
         'maintenance/mel2',
         'manipulation/mel1',
         'manipulation/mel2',
         'melody_maintenance',
         'melody_manipulation']

#alphas = [.025,.025,.025,.025,.025,.025,.025]
stat_results = {}
for cidx, cnd in enumerate(conds):
    stat_results[cnd] = {}
    for ROI in all_data[cnd]:
        cdata = all_data[cnd][ROI]
        if cnd == 'interaction':
            cdata = cdata*-1
        print(cdata.shape)
        #stat_results[cnd][ROI] = gs.do_stats(cdata, 'FDR', FDR_alpha=.025)
        stat_results[cnd][ROI] = gs.do_stats(cdata, 'montecarlo', #adjacency=adjacency,
                                        FDR_alpha=.025, n_permutations=5000, cluster_alpha=.05)#, cluster_method='TFCE')
        print('reporting stats for cond {} ROI {}:\n'.format(cnd,ROI))
        print('minimum pval = ', np.round(np.min(stat_results[cnd][ROI]['pvals']),2))
    #     print('minimum qval = ', np.round(np.min(stat_results[cnd]['qvals']),2))
        print('minimum tstat = ', np.round(np.min(stat_results[cnd][ROI]['tvals']),2))
        print('maximum tstat = ', np.round(np.max(stat_results[cnd][ROI]['tvals']),2),'\n')
    
print('saving stats results')
stats_fname = op.join(stats_dir,'ROI_source_stats_{}.p'.format(suffix))
sfile = open(stats_fname,'wb')
pickle.dump(stat_results,sfile)
sfile.close()

### Make plots
cnd2plot = ['melody_maintenance','melody_manipulation','interaction']
additional = [['maintenance/mel1','maintenance/mel2'],
              ['manipulation/mel1','manipulation/mel2'],
              ['melody_maintenance','melody_manipulation']]
ROI2plot =  ['Right Auditory',#'Left Auditory',
            'Right Memory',# 'Left Ventro-Medial',
            'Left Dorsal Cognitive Control',
            'Right Dorsal Cognitive Control',
            'Right Ventral Cognitive Control']#, 'Left Dorso-Lateral',
            #'Right Postero-Medial','Left Postero-Medial']
ncols = 5
for cndix, cnd in enumerate(cnd2plot):
        nrows = np.ceil(len(ROI2plot)/ncols).astype(int)
        fig, axes = plt.subplots(ncols=ncols,nrows=nrows,
                                  figsize = (4*ncols,nrows*3))
        for ROIx, ROI in enumerate(ROI2plot):
            cplts = []
            rix, cix = ROIx//ncols,ROIx%ncols

            if nrows == 1:
                cax = axes[cix]
            else:
                cax = axes[rix,cix]
            
            #axes[rix, cix].plot(times,stc_labels[b + '/' + 'mel1'][lidx],color='b')
            #axes[rix, cix].plot(times,stc_labels[b + '/' + 'mel2'][lidx],color='r')
            ci_upper = np.squeeze(stat_results[cnd][ROI]['data_mean'] + 2*stat_results[cnd][ROI]['data_sd']/np.sqrt(n-1))
            ci_lower = np.squeeze(stat_results[cnd][ROI]['data_mean'] - 2*stat_results[cnd][ROI]['data_sd']/np.sqrt(n-1))
            ccmask = np.squeeze(np.array(stat_results[cnd][ROI]['mask']).astype(float))
            ccmask[ccmask==0.] = np.nan
            cax.fill_between(times, ci_lower, ci_upper, color='k', alpha=.05)
            cax.plot(times,np.squeeze(stat_results[cnd][ROI]['data_mean']),color='k',
                                alpha=.8,label='difference')
            cax.plot(times,np.squeeze(stat_results[cnd][ROI]['data_mean'])*ccmask, 
                                color='k',linewidth=4,alpha = .4)
            for a in additional[cndix]:
                cax.plot(times,np.squeeze(stat_results[a][ROI]['data_mean']),label=a,alpha=.9)
            cax.set_title(ROI)
            cax.set_ylim(-1,1)
            cax.set_xlim(-.1,4)
            cax.set_xlabel('time (s)')
            cax.set_ylabel('source activation (a.u.)')
            cax.axhline(0., color='k')
            cax.axvline(0., color='k')
            cax.axvline(2., color='k',linestyle='--')
            cax.axvline(.5, color='k',linestyle=':')
            cax.axvline(1, color='k',linestyle=':')
            cax.legend()
            #axes[rix, cix].legend(['patterns'])#,'difference'])
        plt.tight_layout()
        plt.savefig(figures_dir + 'ERF_mels_ROI_{}.pdf'.format(cnd), orientation='portrait')