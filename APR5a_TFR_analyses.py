#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import mne
import os
import os.path as op
import numpy as np
from mne.time_frequency import tfr_morlet, csd_morlet
import matplotlib.pyplot as plt
import pickle
from sys import argv
from stormdb.access import Query
#import matplotlib
from warnings import filterwarnings
#matplotlib.use('Qt4Agg')
filterwarnings("ignore", category=DeprecationWarning)
#import pickle
#import pandas as pd

project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
project_dir = '/projects/' + project
os.environ['MINDLABPROJ']=project
os.environ['MNE_ROOT']='/users/david/miniconda3/envs/mne3d' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

raw_path = project_dir + '/scratch/maxfiltered_data/tsss_st16_corr96'
ica_path = project_dir + '/scratch/working_memory/ICA'
avg_path = project_dir + '/scratch/working_memory/averages'
subjects_dir = project_dir + '/scratch/fs_subjects_dir'
fwd_path = project_dir + '/scratch/forward_models'

save_averages = True
compute_sources = True
save_sources = True
plot_sources = True
suffix = 'tw_2-4'
source_tw = [2,3.5]

qr = Query(project)
sub_codes = qr.get_subjects()
sub = sub_codes[25]

if len(argv) > 1:
    sub = sub_codes[int(argv[1])]

#sub = '0012_VK2' #'0002_BYG'#'0008_HMD'#'0007_ESO'#'0006_TNV'#'0005_AO8'#'0003_S5V'#'0002_BYG' #'0004_LXY'#'0001_VLC' #
conds = ['main','inv']
# args = argv[0]
# sub, conds, suffix = args['sub'], args['conds'], args['suffix']
# save_averages, compute_sources = args['save_averages'], args['compute_sources']
# save_sources, plot_sources = args['save_sources','plot_sources']

event_ids = [[['same',1],['different',2]],[['inverted',3],['other',4]]]

epochs = {}
power = {}
for cidx, c in enumerate(conds):
    fname = os.path.join(raw_path, sub, c + '_raw_tsss.fif')
    icaname = os.path.join(ica_path,sub, c + '_raw_tsss-ica.fif')
    raw = mne.io.read_raw_fif(fname, preload = True) # load data
    ica = mne.preprocessing.read_ica(icaname) # load ica solution
    raw = ica.apply(raw) # apply ICA
    #raw.plot()
    raw.resample(100)

    # Get and correct triggers:
    events = mne.find_events(raw, shortest_event = 1)
    events = events[np.append(np.diff(events[:,0]) >2,True)] # delete spurious t
    events2 = events.copy()
    events2 = events2[events[:,2] < 20]
    events2[:,2] = events[np.isin(events[:,2]//10,[11,21]),2]//100 + cidx*2# recode events

    event_id = dict(event_ids[cidx])
    picks = mne.pick_types(raw.info, meg = True)
    tmin, tmax = -1,6.5 #epoch time
    baseline = (-1,0) # baseline time
    reject = dict(mag = 8e-12, grad = 4000e-13)#eeg = 200e-6, #, eog = 250e-6)

    #epoching:
    epochs[c] = mne.Epochs(raw, events = events2, event_id = event_id,
                    tmin = tmin, tmax = tmax, picks = picks,
                    baseline = baseline, reject = reject)

#TFR analyses

all_epochs = mne.epochs.concatenate_epochs([epochs[c] for c in epochs])
freqs = np.append(np.arange(1,14,1),np.arange(15,30,2))
#freqs = np.logspace(*np.log10([2,100]),20)
n_cycles = freqs/2.

power = dict((cond,
                tfr_morlet(all_epochs[cond], freqs = freqs, n_cycles= n_cycles,
                           use_fft = True,return_itc = False, decim = 1,
                           average = False, n_jobs=4)) for
                cond in sorted(all_epochs.event_id.keys()))

for p in power:
    power[p].data = np.log(power[p].data)

for p in power:
    cdata = np.transpose(power[p].data[:,:,:,0:int(power[p].info['sfreq']*1)],[0,3,1,2])
    cdata = cdata.reshape((cdata.shape[0]*cdata.shape[1], cdata.shape[2], cdata.shape[3]))
    bmean = np.mean(cdata,0).reshape((1,cdata.shape[1],cdata.shape[2],1))
    bsd = np.std(cdata,0).reshape((1,cdata.shape[1],cdata.shape[2],1))
    del(cdata)
    power[p].data = power[p].data - bmean
    power[p].data = power[p].data / bsd

power =  {cond: power[cond].average() for cond in power}

power[conds[0]] = power['same'].copy()
power[conds[1]] = power['same'].copy()
power['difference'] = power['same'].copy()
# power['difference2'] = power['same'].copy()
power[conds[0]].data = np.mean([power['same'].data,power['different'].data],0)
power[conds[1]].data = np.mean([power['inverted'].data,power['other'].data],0)
power['difference'].data = power[conds[1]].data - power[conds[0]].data
# power['difference2'].data = (power[conds[1]].copy().apply_baseline((-2,0)).data -
#                              power[conds[0]].copy().apply_baseline((-2,0)).data)

if save_averages:
    TFR_fname = op.join(avg_path,'data',sub + '_TFR'+ suffix + '.p')
    TFR_file = open(TFR_fname,'wb')
    pickle.dump(power,TFR_file)
    TFR_file.close()
    print('TFR file saved')

# # make some plots:
# tfreqs={(0.1,11): (0.1,2),(0.3,2): (0.05,2),(1.2,11): (0.4,2),(2.35,11): (0.2,2),
#         (2.4,2): (0.2,2),(4,11): (0.2,2), (5.4,11): (0.4,2)}

# tfreqs2={(-1.4,11): (0.5,2), (-1.4,2): (0.05,2),(0.25,11): (0.25,2),(0.25,2): (0.25,2),
#         (1,11): (0.25,2),(1,2): (0.25,2), (2,11): (0.25,2),(2,2): (0.25,2),
#         (3,11): (0.25,2), (3,2): (0.25,2), (3.9,11): (0.25,2), (3.9,2): (0.25,2)}

# topomap_args = {'vmax': 8e-22, 'vmin':-8e-22}
# topomap_args2 = {'vmax': 3e-22, 'vmin':-3e-22}
# mfig = power[conds[0]].copy().pick_types(meg = 'grad').plot_joint(baseline=(-1,0),#(-2,0),#None,#(-3,0),
#                                                          timefreqs=tfreqs,
#                                                          topomap_args = topomap_args,
#                                                          title = 'maintenance block')
# mfig.set_figwidth(20)
# mfig.set_figheight(10)
# plt.savefig(avg_path + '/figures/{}_TFR_maintenance{}.pdf'.format(sub,suffix),
#             orientation='landscape')

# ifig = power[conds[1]].copy().pick_types(meg = 'grad').plot_joint(baseline=(-1,0),
#                                                         timefreqs=tfreqs,
#                                                         topomap_args = topomap_args,
#                                                         title = 'manipulation block')
# ifig.set_figwidth(20)
# ifig.set_figheight(10)
# plt.savefig(avg_path + '/figures/{}_TFR_manipulation{}.pdf'.format(sub,suffix),
#             orientation='landscape')

# dfig = power['difference'].copy().pick_types(meg = 'grad').plot_joint(baseline=None,
#                                                                timefreqs=tfreqs2,
#                                                                topomap_args = topomap_args2,
#                                             title = 'difference (manipulation - maintenance)')

# dfig.set_figwidth(20)
# dfig.set_figheight(10)
# plt.savefig(avg_path + '/figures/{}_TFR_difference1{}.pdf'.format(sub,suffix),
#             orientation='landscape')

# dfig2 = power['difference'].copy().pick_types(meg = 'grad').plot_joint(baseline=(-1,0),
#                                                                timefreqs=tfreqs2,
#                                                                topomap_args = topomap_args2,
#                                             title = 'difference (manipulation - maintenance)')

# dfig2.set_figwidth(20)
# dfig2.set_figheight(10)
# plt.savefig(avg_path + '/figures/{}_TFR_difference2{}.pdf'.format(sub,suffix),
#             orientation='landscape')

# ## Source analyses
# if compute_sources:
#     print('\n computing sources \n')
#     fwd_fn = op.join(fwd_path, sub + '_main_session-fwd.fif')
#     fwd = mne.read_forward_solution(fwd_fn)

#     bands = {'delta': np.array([1,1.5,2,2.5]),
#               'theta': np.array([3,4,5,6,7]),
#               'alpha': np.array([8,9,10,11,12]),
#               'beta':  np.logspace(np.log10(12), np.log10(30), 9),
#               'gamma': np.logspace(np.log10(30), np.log10(50), 9)}
#     #bands = {'alpha': np.array([8,9,10,11,12])}
#     src_pwr = {}
#     csd = {}
#     invs = {}
#     for bnd in bands:
#         freqs2 = bands[bnd]
#         csd[bnd] ={}
#         n_cycles = freqs2/2.
#         csd[bnd]['all'] = csd_morlet(all_epochs,freqs2,tmin = -2.5,tmax=6.5,
#                                 decim=20,n_jobs=4,n_cycles= n_cycles)
#         csd[bnd]['inv'] = csd_morlet(epochs[conds[1]].load_data(),freqs2,
#                                      tmin = source_tw[0],tmax=source_tw[1],
#                                 decim=20,n_jobs=4,n_cycles= n_cycles)
#         csd[bnd]['main'] = csd_morlet(epochs[conds[0]].load_data(),freqs2,
#                                       tmin = source_tw[0],tmax=source_tw[1],
#                                  decim=20,n_jobs=4,n_cycles= n_cycles)
#         csd[bnd]['inv_base'] = csd_morlet(epochs[conds[1]].load_data(),freqs2,
#                                           tmin = -1.5,tmax=0,
#                                      decim=20,n_jobs=4,n_cycles= n_cycles)
#         csd[bnd]['main_base'] = csd_morlet(epochs[conds[0]].load_data(),freqs2,
#                                            tmin = -1.5,tmax=0,
#                                       decim=20,n_jobs=4,n_cycles= n_cycles)
#         csd[bnd]['baseline'] = csd_morlet(all_epochs,freqs2,tmin = -1.5,tmax=0,
#                                      decim=20,n_jobs=4,n_cycles= n_cycles)
#         for cii in csd[bnd]:
#             csd[bnd][cii] = csd[bnd][cii].mean()

#         invs[bnd] = mne.beamformer.make_dics(all_epochs.info, fwd, csd[bnd]['all'],
#                                        noise_csd=csd[bnd]['baseline'],rank='info',
#                                        pick_ori='max-power', reduce_rank=True)

#         csd_names = ['main','inv','main_base','inv_base']
#         src_pwr[bnd] = {}
#         for csii in csd_names:
#             src_pwr[bnd][csii], freqs3 = mne.beamformer.apply_dics_csd(csd[bnd][csii], invs[bnd])

#         src_pwr[bnd]['inv_change'] = src_pwr[bnd]['inv'] - src_pwr[bnd]['inv_base']
#         src_pwr[bnd]['main_change'] = src_pwr[bnd]['main']- src_pwr[bnd]['main_base']
#         src_pwr[bnd]['difference1'] = src_pwr[bnd]['inv'] - src_pwr[bnd]['main']
#         src_pwr[bnd]['difference2'] = src_pwr[bnd]['inv_change'] - src_pwr[bnd]['main_change']

#     if save_sources:
#         TFR_src_fname = op.join(avg_path,'data',sub + '_TFR_sources' + suffix + '.p')
#         TFR_src_file = open(TFR_src_fname,'wb')
#         pickle.dump(src_pwr,TFR_src_file)
#         TFR_src_file.close()
#         print('TFR sources file saved')

#         TFR_inv_fname = op.join(avg_path,'data',sub + '_TFR_inverse'+suffix+'.p')
#         TFR_inv_file = open(TFR_inv_fname,'wb')
#         pickle.dump(invs,TFR_inv_file)
#         TFR_src_file.close()
#         print('TFR sources file saved')

#     if plot_sources:
#         plots_to_save = ['inv_change','main_change','difference1','difference2']
#         #plots_to_save = ['inv','main','difference']#,'difference2']
#         for bnd in bands:
#             for ps in plots_to_save:
#                 brain =[]
#                 brain=src_pwr[bnd][ps].plot(hemi='split',
#                                             views=['lateral','medial','axial'],
#                                             subjects_dir=subjects_dir, subject=sub,
#                                             title = '{} {}'.format(bnd,ps))
#                 brain.save_image('{}/figures/{}_TFR_source_{}_{}{}.png'.format(avg_path,
#                                                                                sub,bnd,ps,
#                                                                                suffix))
#                 brain.close()
