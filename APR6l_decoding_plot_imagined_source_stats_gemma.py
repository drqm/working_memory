import mne
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import pathlib
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
import warnings
from do_stats import do_stats
warnings.filterwarnings("ignore", category=DeprecationWarning) 

os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt5'

proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
wdir = '/projects/' + proj_name + '/scratch/working_memory/'
data_dir = wdir + 'averages/data/'
subs_dir = '/projects/' + proj_name + '/scratch/fs_subjects_dir/'

sample_path = sample.data_path()
sample_subjects_dir = sample_path + '/subjects'
# src_sample = mne.read_source_spaces(sample_subjects_dir +
#                                 '/fsaverage/bem/fsaverage-vol-5-src.fif')
src_sample = mne.read_source_spaces(subs_dir +
                                'fsaverage/bem/fsaverage-vol-5-src.fif')
#label_file = subs_dir + 'fsaverage/mri/aparc.DKTatlas+aseg.mgz'
label_file = sample_subjects_dir + '/fsaverage/mri/aparc.a2009s+aseg.mgz'

stats_dir = wdir + 'results/stats/'
figures_dir = wdir + 'results/figures/'

## Get subjects:
qr = Query(proj_name)
subjects = qr.get_subjects()
subs = range(11,91) #, 27, 28, 29, 30, 31, 32, 33, 34, 35]

all_data = {}
for sidx,s in enumerate(subs):
    try:
        scode = subjects[s-1]
        #dfname = data_dir + scode + '_patterns_imagined_s40_lp4.p'
        dfname = data_dir + scode + '_patterns_loc_source.p' 
        #imagined_smoothing25_50_hp005.p'
        print('loading file {}'.format(dfname))
        dfile = open(dfname,'rb')
        curdata = load(dfile)
        morph = mne.read_source_morph(subs_dir + scode + '/bem/' + scode + '_vol-morph.h5')
        morph_mat = morph.vol_morph_mat
        for cd in curdata:
            for tone in curdata[cd]:
                cdata = morph_mat.dot(curdata[cd][tone].data)#deepcopy(cmorphed.data) #morph_mat.dot(c.data)
                print('appending subject {} condition {}'.format(scode,cd))
                condname = cd + '_' + tone
                all_data.setdefault(condname, np.array([cdata]))
                #all_data[cd][cd2].append([cdata])
                if sidx > 0:
                    all_data[condname] = np.vstack((all_data[condname],np.array([cdata])))
    except Exception as e:
        print(e)
        continue
    
print(all_data[condname].shape)

grand_avg = {}
for cd in all_data:
    grand_avg[cd] = deepcopy(morph.apply(curdata['listened']['1']))
    #grand_avg[b][cd].subject = 'sample'
    grand_avg[cd].data = np.mean(all_data[cd],0)
    grand_avg[cd].tmin = -0.1
    grand_avg[cd].tstep = 0.01
    print(grand_avg[cd].data.shape)

adjacency = mne.spatial_src_adjacency(src_sample)

stats_results = {}
for cd in all_data:
    stats_results[cd] = do_stats(all_data[cd], method='FDR', adjacency=adjacency)
    stats_fname = op.join(stats_dir,cd + '_loc_source_FDR_stats.p')             
    stats_file = open(stats_fname,'wb')
    pickle.dump(cd,stats_file)
    stats_file.close()
    print('stats file saved')
  
crop_times = np.arange(-0.25,2,0.25)
matplotlib.use('Qt5Agg')
for p in stats_results:
    print(grand_avg[p].tmin)
    for tidx, t in enumerate(crop_times):        
        if tidx < (len(crop_times) - 1):
            t2 = crop_times[tidx + 1]
            print('{} ({} {}):'.format(p, t, t2) )
            cdata = grand_avg[p].copy()
            cdata.data = stats_results[p]['tvals'].T * stats_results[p]['mask'].T 
            cdata.copy().crop(t,t2).mean().plot(src = src_sample, subjects_dir = subs_dir, mode = 'glass_brain',#'glass_brain',#'glass_brain',#initial_time = 1.75,
                              initial_pos = None)#[.007,.05,.023])

for p in stats_results:
    print(p)
    cdata = grand_avg[p].copy()
    cdata.data = stats_results[p]['tvals'].T #* stats_results[p]['mask'].T 
    cdata.plot(src = src_sample, subjects_dir = subs_dir, mode = 'stat_map')#'glass_brain')#[.007,.05,.023])    

# crop_times = np.arange(-0.5,2.5,0.5)
# for p in grand_avg:
#     print(grand_avg[p].tmin)
#     for tidx, t in enumerate(crop_times):        
#         if tidx < (len(crop_times) - 1):
#             t2 = crop_times[tidx + 1]
#             print('{} ({} {}):'.format(p, t, t2) )
#             grand_avg[p].copy().crop(t,t2).mean().plot(src = src_sample, subjects_dir = subs_dir, mode = 'stat_map',#'glass_brain',#'glass_brain',#initial_time = 1.75,
#                               initial_pos = None)#[.007,.05,.023])

coords = [[.005,-.005,.005]]
for p in grand_avg:
    print(p)
    cdata = grand_avg[p].copy()
    cdata.data = stats_results[p]['data_mean'].T #* stats_results[p]['mask'].T 
    for c in coords:
        cdata.plot(src = src_sample, subjects_dir = subs_dir, mode = 'stat_map', initial_pos=c,
                                clim = {'kind': 'value', 'pos_lims': [0.01,0.05,0.5]})#'glass_brain')#[.007,.05,.023])
    
# ts = [0.3,1.3]
# for p in grand_avg:
#     for t in ts:
#         print(grand_avg[p].tmin)
#         print('{} ({}):'.format(p, t) )
#         cdata = grand_avg[p].copy()
#         cdata.data = stats_results[p]['data_mean'].T * stats_results[p]['mask'].T 
#         cdata.plot(src = src_sample, subjects_dir = subs_dir, mode = 'stat_map',#initial_pos=[.005,-.005,.005], #stat_map',#'glass_brain',#'glass_brain',#initial_time = 1.75,
#                                   initial_time = t, clim = {'kind': 'value', 'pos_lims': [0.001,0.05,0.6]})#[.007,.05,.023])

## Plot regions of interest:
labels = mne.get_volume_labels_from_aseg(label_file)

for l in labels:
    print(l)

clabels = ['Left-Thalamus-Proper',
           'Right-Thalamus-Proper',
           #'Left-Caudate',
           #'Right-Caudate',
           'ctx_lh_G_and_S_cingul-Ant',
           'ctx_rh_G_and_S_cingul-Ant',
           'ctx_lh_G_and_S_cingul-Mid-Ant',
           'ctx_rh_G_and_S_cingul-Mid-Ant',
           'ctx_lh_G_and_S_cingul-Mid-Post',
           'ctx_rh_G_and_S_cingul-Mid-Post',
           'ctx_lh_G_temp_sup-G_T_transv',
           'ctx_rh_G_temp_sup-G_T_transv',
           'ctx_lh_G_temp_sup-Lateral',
           'ctx_rh_G_temp_sup-Lateral']

stc_labels = {}
se_labels = {}
for c in grand_avg:
    cdata = grand_avg[c].copy()
    sedata = grand_avg[c].copy()
    cdata.data = stats_results[c]['data_mean'].T #* stats_results[c]['mask'].T
    sedata.data = stats_results[c]['data_sd'].T / np.sqrt(stats_results[c]['n']) #* stats_results[c]['mask'].T
    stc_labels[c] = cdata.extract_label_time_course(labels = [label_file,clabels], src = src_sample, mode = 'auto')
    se_labels[c] = sedata.extract_label_time_course(labels = [label_file,clabels], src = src_sample, mode = 'auto')

times = grand_avg['listened_1'].times 
cur_labs = ['Left thalamus','Left A1']
#change titles
titles = ['maintenance (delay)', 'manipulation (delay)','maintenance (1st melody)','manipulation (1st melody)']
conds = ['inv_fmel','inv_delay']
fig, axes = plt.subplots(ncols=3,nrows=2, figsize = (10,10))
for sidx, sl in enumerate(stc_labels):
    times = grand_avg[sl].times 
    print(sidx,sl)
    rix, cix = sidx//3,sidx%3
    for lidx in range(len(stc_labels[sl])):
        if lidx in [0,8]:
            axes[rix,cix].plot(times,stc_labels[sl][lidx])
    axes[rix, cix].set_title(sl)
    axes[rix, cix].set_ylim(-.8,.5)
    axes[rix, cix].set_xlim(times[0],times[-1])
    axes[rix, cix].set_xlabel('time (s)')
    axes[rix, cix].set_ylabel('difference (a.u.)')
    axes[rix, cix].axhline(0., color='k')
    axes[rix, cix].axvline(0., color='k')
    #axes[rix, cix].legend([clabels[cls] for cls in [0,8]])
    axes[rix, cix].legend(cur_labs)
plt.tight_layout()
plt.savefig(figures_dir + 'pattern_sources_labels_lh.pdf',orientation='portrait')

fig, axes = plt.subplots(ncols=2,nrows=2, figsize = (10,10))
cur_labs = ['Right A1','Right cingulate']
lab_ix = [9,7]
conds = ['inv_fmel','inv_delay']
titles = ['manipulation (1st melody)','manipulation (delay)']
sidx = -1
for lx, lab in enumerate(lab_ix):
    lidx = lab_ix[lx]
    for slx, sl in enumerate(conds):
        sidx += 1
        times = grand_avg[sl].times.copy()
        if sl == 'inv_delay':
            times = times + 2 
        rix, cix = sidx//2,sidx%2
        ci_upper = stc_labels[sl][lidx] + 1.96*se_labels[sl][lidx]
        ci_lower = stc_labels[sl][lidx] - 1.96*se_labels[sl][lidx]
        axes[rix, cix].fill_between(times, ci_lower, ci_upper, color='k', alpha=.2)
        axes[rix,cix].plot(times,stc_labels[sl][lidx],color = 'k')
        axes[rix, cix].set_title(titles[slx] + ' - ' + cur_labs[lx])
        axes[rix, cix].set_ylim(-1,1)
        axes[rix, cix].set_xlim(times[0],times[-1]+0.01)
        axes[rix, cix].set_xlabel('time (s)')
        axes[rix, cix].set_ylabel('difference (a.u.)')
        axes[rix, cix].axhline(0., color='k')
        axes[rix, cix].axvline(0., color='k')        
        if sl == 'inv_fmel':
            axes[rix, cix].axvline(.5, color='k',linestyle=':')
            axes[rix, cix].axvline(1, color='k',linestyle=':')
        if sl == 'inv_delay':
            axes[rix, cix].axvline(2, color='k',linestyle='--')
        #axes[rix, cix].legend([clabels[cls] for cls in [1,9]])
        #axes[rix, cix].legend(cur_labs)
plt.tight_layout()
plt.savefig(figures_dir + 'pattern_sources_labels_rh.pdf',orientation='portrait')

import statsmodels.api as sm

reg_ix = [0,1,8,9]
ccor = {}
for cd in stc_labels:
    ccor[cd] = {}
    for a1 in reg_ix:
        for a2 in reg_ix:
            if a2 > a1:  
                cur_cmp = clabels[a1] + ' ' + clabels[a2]
                print(cur_cmp)
                ccor[cd][cur_cmp] = np.correlate(stc_labels[cd][a1],
                                                         stc_labels[cd][a2],
                                                         'full')

lags = np.arange(-len(times)+1,len(times))*0.02
plt.plot(lags,ccor['inv_delay']['Right-Thalamus-Proper ctx_rh_G_temp_sup-G_T_transv'])

