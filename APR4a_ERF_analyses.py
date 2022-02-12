#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import mne
import os
import os.path as op
import numpy as np
import pickle
from warnings import filterwarnings
from sys import argv
import matplotlib.pyplot as plt
from stormdb.access import Query

filterwarnings("ignore", category=DeprecationWarning)

project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
project_dir = '/projects/' + project
os.environ['MINDLABPROJ']= project
#os.environ['MNE_ROOT']='/users/david/miniconda3/envs/mne3d' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

raw_path = project_dir + '/scratch/maxfiltered_data/tsss_st16_corr96'
ica_path = project_dir + '/scratch/working_memory/ICA'
avg_path = project_dir + '/scratch/working_memory/averages'
subjects_dir = project_dir + '/scratch/fs_subjects_dir'
fwd_path = project_dir + '/scratch/forward_models'
event_ids = [[['same',1],['different1',2],['different2',3]],
             [['inverted',1],['other1',2],['other2',3]]]

qy = Query(project)
subs = qy.get_subjects()
scode = 16
sub = subs[scode-1]#'0002_BYG'#'0002_BYG'#'0008_HMD'#'0002_BYG'
conds = ['main','inv'] #  ['mainv2','invv2']
save_averages = True#False#True
plot_topo = True#True
compute_sources = True # False
plot_sources = False
# sub, conds, save_averages,  = argv[0], argv[1], argv[2]
# plot_topo, compute_sources, plot_sources = argv[3], argv[4]

epochs = {}
evokeds = {}

print('\n epoching \n')
for cidx, c in enumerate(conds):
    #load and preprocess:
    fname = os.path.join(raw_path, sub, c + '_raw_tsss.fif')
    icaname = os.path.join(ica_path,sub, c + '_raw_tsss-ica.fif')
    raw = mne.io.read_raw_fif(fname, preload = True) # load data
    ica = mne.preprocessing.read_ica(icaname) # load ica solution
    raw = ica.apply(raw) # apply ICA
    raw.resample(100)
    raw.filter(0.1,40,fir_design = 'firwin')

    # Get, correct and recode triggers:
    events = mne.find_events(raw, shortest_event = 1)
    events = events[np.append(np.diff(events[:,0]) >2,True)] # delete spurious t
    events2 = events.copy()
    events3 = events.copy()
    events2 = events2[events[:,2] < 20]
    events3 = events3[np.isin(events[:,2]//10, [12,22]),:]
    events2[:,2] = events[np.isin(events[:,2]//10,[11,21]),2]//100 # recode events
    events2[np.isin(events3[:,2],[221,223]),2]=3 # add subcategory of other/different

    # Epoching:
    event_id = dict(event_ids[cidx])
    picks = mne.pick_types(raw.info, meg = True)
    tmin, tmax = -2.5,6.5 #epoch time
    baseline = (-1,0) # baseline time
    reject = dict(mag = 4e-12, grad = 4000e-13)#eeg = 200e-6, #, eog = 250e-6)
    epochs[c] = mne.Epochs(raw, events = events2, event_id = event_id,
                    tmin = tmin, tmax = tmax, picks = picks,
                    baseline = baseline)#, reject = reject)

    # Averaging:
    evokeds[c] = dict((cond,epochs[c][cond].average()) for
                        cond in sorted(event_id.keys()))
    evokeds[c]['all'] = epochs[c].average()
    evokeds[c]['all'].comment = c
    print('\n done with block {}\n'.format(c))

#compute difference between conditions:
evokeds['difference'] = {}
evokeds['difference']['all'] = evokeds[conds[1]]['all'].copy()
evokeds['difference']['all'].data = (evokeds[conds[0]]['all'].data -
                                     evokeds[conds[1]]['all'].data)
evokeds['difference']['all'].comment = 'difference'

#save output:
if save_averages:
    evkd_fname = op.join(avg_path,'data',sub + '_evoked.p')
    evkd_file = open(evkd_fname,'wb')
    pickle.dump(evokeds,evkd_file)
    evkd_file.close()
    print('evoked file saved')

print('done epoching')

# Plot some figures
fig1, axis = plt.subplots(nrows=1,ncols=1,figsize=(60,30))
mne.viz.plot_evoked_topo([evokeds[conds[0]]['all'].copy().pick_types('mag'),
                          evokeds[conds[1]]['all'].copy().pick_types('mag')],axes = axis)
plt.tight_layout()
plt.savefig(avg_path + '/figures/{}_ERFs_topo_mag.pdf'.format(sub))

fig2, axis = plt.subplots(nrows=1,ncols=1,figsize=(60,30))
mne.viz.plot_evoked_topo([evokeds[conds[0]]['all'].copy().pick_types('grad'),
                          evokeds[conds[1]]['all'].copy().pick_types('grad')],
                         axes = axis,merge_grads=True)
plt.tight_layout()
plt.savefig(avg_path + '/figures/{}_ERFs_topo_grad.pdf'.format(sub))

fig3, axis = plt.subplots(nrows=1,ncols=1,figsize=(60,30))
mne.viz.plot_evoked_topo(evokeds['difference']['all'].copy().pick_types('grad'),
                         axes = axis,merge_grads=True)
plt.tight_layout()
plt.savefig(avg_path + '/figures/{}_ERFs_diff_topo_grad.pdf'.format(sub))

fig4, axis = plt.subplots(nrows=1,ncols=1,figsize=(60,30))
mne.viz.plot_evoked_topo(evokeds['difference']['all'].copy().pick_types('mag'),
                         axes = axis)
plt.tight_layout()
plt.savefig(avg_path + '/figures/{}_ERFs_diff_topo_mag.pdf'.format(sub))

plt.close('all')
# if plot_topo:
#   for e in evokeds:
#       for c in evokeds[e]:
#         times = np.arange(0.11,6.11,0.1)
#         #evokeds[e][c].plot_joint(topomap_args ={'average': 0.5},picks='mag',times = times)
#         evokeds[e][c].plot_topo()


## Source analysis
if compute_sources:
    print('\n computing sources \n')
    fwd_fn = op.join(fwd_path, sub + '_main_session-fwd.fif')
    fwd = mne.read_forward_solution(fwd_fn)
    #compute noise covariance
    noise_cov = mne.compute_covariance([epochs[e] for e in epochs],tmin = -1,
                                       tmax=0, rank='info')
    data_cov = mne.compute_covariance([epochs[e].load_data().copy().pick_types('mag')
                                       for e in epochs],
                                       tmin= 0,
                                       tmax = 6.5,rank ='info')
    ## mne solution
    inv = mne.beamformer.make_lcmv(epochs[conds[0]].info,fwd,data_cov, reg=0.05,
                              #noise_cov=noise_cov,#pick_ori='max-power',depth = 0.95,
                              weight_norm= 'nai', rank = 'info')
    # inv = mne.minimum_norm.make_inverse_operator(epochs[conds[0]].info,fwd,
    #                                               noise_cov,loose = 1)
    SNR = 3
    sources = {}
    for e in evokeds:
        sources[e] = {}
        for c in evokeds[e]:
            # sources[e][c] = mne.minimum_norm.apply_inverse(evokeds[e][c],inv,
            #                                                 lambda2=1/SNR**2)
            sources[e][c] = mne.beamformer.apply_lcmv(evokeds[e][c],inv)#,max_ori_out='signed')

            if plot_sources:
                brain = sources[e][c].plot(subjects_dir=subjects_dir,initial_time=-1,hemi = 'split',
                                           time_viewer=False, views=['lateral','medial'],
                                           title = 'ERF- {} {} {}'.format(sub,e,c))
                brain.save_movie(avg_path + '/figures/{}_ERF_sources_{}_{}.mov'.format(sub,e,c),
                                 framerate=4,time_dilation = 10)
                brain.close()

    src_fname = op.join(avg_path,'data',sub + '_evoked_sources.p')
    src_file = open(src_fname,'wb')
    pickle.dump(sources,src_file)
    src_file.close()

    inv_fname = op.join(avg_path,'data',sub + '_evoked_inverse.p')
    inv_file = open(inv_fname,'wb')
    pickle.dump(inv,inv_file)
    inv_file.close()
    print('\n sources file saved')
