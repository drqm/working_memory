#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import mne
import fooof
import os
# import os.path as op
import numpy as np
# from mne.time_frequency import tfr_morlet, csd_morlet
# import matplotlib.pyplot as plt
# import pickle
from sys import argv
#import matplotlib
from warnings import filterwarnings
from stormdb.access import Query
#matplotlib.use('Qt4Agg')
filterwarnings("ignore", category=DeprecationWarning)
#import pickle
import pandas as pd

project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
project_dir = '/projects/' + project
os.environ['MINDLABPROJ']=project
os.environ['MNE_ROOT']='/users/david/miniconda3/envs/mne3d' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

raw_path = project_dir + '/scratch/maxfiltered_data/tsss_st16_corr96'
ica_path = project_dir + '/scratch/working_memory/ICA'
avg_path = project_dir + '/scratch/working_memory/averages'
fooof_path = project_dir + '/scratch/working_memory/FOOOF'
subjects_dir = project_dir + '/scratch/fs_subjects_dir'
fwd_path = project_dir + '/scratch/forward_models'

TWs = [[-2,0], [0,2], [2,4], [4,6.5]]
TWnames = ['baseline','encoding','delay','recall']

sub = qy = Query(project)
subs = qy.get_subjects()
scode = 21
if len(argv) > 1:
    scode = argv[1]
sub = subs[int(scode)-1]
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
    raw.resample(250)
    #raw.notch_filter([50,100])
    # Get and correct triggers:
    events = mne.find_events(raw, shortest_event = 1)
    events = events[np.append(np.diff(events[:,0]) >2,True)] # delete spurious t
    events2 = events.copy()
    events2 = events2[events[:,2] < 20]
    events2[:,2] = events[np.isin(events[:,2]//10,[11,21]),2]//100 + cidx*2# recode events

    event_id = dict(event_ids[cidx])
    picks = mne.pick_types(raw.info, meg = True)
    tmin, tmax = -2,6.5 #epoch time
    baseline = (-2,0) # baseline time
    reject = dict(mag = 8e-12, grad = 4000e-13)#eeg = 200e-6, #, eog = 250e-6)

    #epoching:
    epochs[c] = mne.Epochs(raw, events = events2, event_id = event_id,
                    tmin = tmin, tmax = tmax, picks = picks,
                    baseline = baseline, reject = reject)

#TFR analyses

all_epochs = mne.epochs.concatenate_epochs([epochs[c] for c in epochs])
fmin, fmax = 1,120
n_fft = int(all_epochs.info['sfreq'] * 2)
fms = {}
for twix, tw in enumerate(TWs):
    print('\n\n###############################################################\n'
          'Separating periodic and aperiodic components of {} period\n'
          '###############################################################\n\n'.format(TWnames[twix]))
    all_psd, freqs = mne.time_frequency.psd_welch(all_epochs, fmin=fmin,fmax=fmax,
                                                  tmin=tw[0], tmax = tw[1],
                                                  n_fft = n_fft, average = 'median')
    all_psd = np.squeeze(all_psd.mean(0))
    fms[TWnames[twix]] = fooof.FOOOF(aperiodic_mode='knee')
    fms[TWnames[twix]].fit(freqs,all_psd.mean(0))

peaks = {'subject': [], 'period': [],'CF': [], 'PW': [], 'BW': []}
for prdix, prd in enumerate(fms):
    report_fname = fooof_path + '/' + sub + '_' + prd + '.pdf'
    fms[prd].save_report(report_fname)
    #fms[prd].plot(plot_peaks='shade')
    for npeak in range(fms[prd].peak_params_.shape[0]):
        peaks['CF'] += [np.round(fms[prd].peak_params_[npeak,0],2)]
        peaks['PW'] += [np.round(fms[prd].peak_params_[npeak,1],2)]
        peaks['BW'] += [np.round(fms[prd].peak_params_[npeak,2],2)]
        peaks['subject'] += [sub]
        peaks['period'] += [prd]

peaks_df = pd.DataFrame(peaks)
csv_path = fooof_path + '/' + sub + '_peaks.csv'
peaks_df.to_csv(csv_path,index=False)

# periodic = np.zeros((all_psd.shape[0],len(freqs)))
# for ch in range(periodic.shape[0]):
#     print(ch)
#     fm = []
#     fm = fooof.FOOOF()
#     fm.fit(freqs,all_psd[ch,:])
#     periodic[ch,:] = fm._spectrum_flat


# freqs = np.append([0.3,0.5,0.8],np.arange(1,50,1))
# #freqs = np.logspace(*np.log10([2,100]),20)
# n_cycles = freqs/2.

# power = dict((cond,
#                 tfr_morlet(all_epochs[cond], freqs = freqs, n_cycles= n_cycles,
#                            use_fft = True,return_itc = False, decim = 3, n_jobs=4)) for
#                 cond in sorted(all_epochs.event_id.keys()))

# power[conds[0]] = power['same'].copy()
# power[conds[1]] = power['same'].copy()
# power['difference'] = power['same'].copy()
# # power['difference2'] = power['same'].copy()
# power[conds[0]].data = np.mean([power['same'].data,power['different'].data],0)
# power[conds[1]].data = np.mean([power['inverted'].data,power['other'].data],0)
# power['difference'].data = power[conds[1]].data - power[conds[0]].data
# # power['difference2'].data = (power[conds[1]].copy().apply_baseline((-2,0)).data -
# #                              power[conds[0]].copy().apply_baseline((-2,0)).data)

# if save_averages:
#     TFR_fname = op.join(avg_path,'data',sub + '_TFR'+ suffix + '.p')
#     TFR_file = open(TFR_fname,'wb')
#     pickle.dump(power,TFR_file)
#     TFR_file.close()
#     print('TFR file saved')

# # make some plots:
# tfreqs={(0.1,11): (0.1,2),(0.3,2): (0.05,2),(1.2,11): (0.4,2),(2.35,11): (0.2,2),
#         (2.4,2): (0.2,2),(4,11): (0.2,2), (5.4,11): (0.4,2)}

# tfreqs2={(-1.4,11): (0.5,2), (-1.4,2): (0.05,2),(0.25,11): (0.25,2),(0.25,2): (0.25,2),
#         (1,11): (0.25,2),(1,2): (0.25,2), (2,11): (0.25,2),(2,2): (0.25,2),
#         (3,11): (0.25,2), (3,2): (0.25,2), (3.9,11): (0.25,2), (3.9,2): (0.25,2)}

# topomap_args = {'vmax': 8e-22, 'vmin':-8e-22},
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

## Source analyses
# if compute_sources:

#     print('\n computing sources \n')
#     fwd_fn = op.join(fwd_path, sub + '_vol-fwd.fif')
#     label_path = op.join(subjects_dir, sub + '/mri/aparc.DKTatlas+aseg.mgz')
#     src_fn = op.join(subjects_dir,sub,'bem',sub + '_vol-src.fif')

#     fwd = mne.read_forward_solution(fwd_fn)
#     src = mne.read_source_spaces(src_fn)
#     lbls0 = mne.get_volume_labels_from_aseg(label_path) #,subjects_dir=subjects_dir)

#     # Include / exclude labels
#     lbls_exc = ['Unknown', 'unknown', 'White','Ventricle','Vent','CSF',
#                 'choroid','Chiasm','hypointensities']
#     lbls_inc = ['ctx','Cerebellum','Thalamus','Caudate','Putamen',
#                  'Pallidum','Hippocampus','Amygdala','Accumbens']

#     lbls = []
#     for ll in lbls0:
#         mtch = np.array(0)
#         for linc in lbls_inc:
#             for lexc in lbls_exc:
#                 cmtch = int(linc in ll) - int(lexc in ll)
#                 #print(ll + ' ' + linc + ' ' + lexc + ' ' + str(cmtch))
#                 mtch += cmtch
#         if mtch > 0:
#             lbls = lbls + [ll]

#     ##
#     data_cov = mne.compute_covariance(all_epochs.load_data().copy().pick_types('grad'),
#                                        tmin= 0,
#                                        tmax = 2.5,rank ='info')
#     inv = mne.beamformer.make_lcmv(all_epochs.info,fwd,data_cov, reg=0.05,
#                               #noise_cov=noise_cov,
#                               pick_ori='max-power',depth = 0.95,
#                               weight_norm= 'nai', rank = 'info')

#     source_epochs = mne.beamformer.apply_lcmv_epochs(all_epochs, inv)


#     source_psds = np.array([])
#     for sidx, se in enumerate(source_epochs):
#         print('computing psd for epoch {}'.format(sidx+1))
#         csource_psds, cfreqs = mne.time_frequency.psd_array_welch(se.data[:,125:], sfreq = 250,
#                                                           fmin=fmin,fmax=fmax,
#                                                           n_fft=500,
#                                                           average = 'median')
#         if sidx == 0:
#             source_psds = np.zeros((len(source_epochs),
#                                     csource_psds.shape[0],
#                                     csource_psds.shape[1]))
#         source_psds[sidx,:,:] = csource_psds.copy()

#     mean_source_psds = source_psds.mean(0)

#     sdummy = source_epochs[0].copy()
#     sdummy.tmin = cfreqs[0]
#     sdummy.tmax = cfreqs[-1]
#     sdummy.tstep = cfreqs[1] - cfreqs[0]
#     sdummy.data = mean_source_psds
#     ## Extract labels
#     parc = mne.extract_label_time_course(sdummy, [label_path,lbls], src, mode = 'mean', allow_empty=True)

#     parc_periodic = np.zeros(parc.shape)
#     for ppidx in range(parc_periodic.shape[0]):
#         print('fitting ROI {}'.format(ppidx+1))
#         fm2 = []
#         fm2 = fooof.FOOOF(aperiodic_mode='knee')
#         fm2.fit(cfreqs,parc[ppidx,:])
#         parc_periodic[ppidx,:] = fm2._spectrum_flat

#     parc_periodic_src = mne.labels_to_stc([label_path,lbls],parc_periodic,tmin= cfreqs[0],
#                                     tstep = cfreqs[1] - cfreqs[0] , src = src)

#     brain = parc_periodic_src.plot( subjects_dir=subjects_dir,initial_time=11.5,
#                                    mode='glass_brain', src=src,
#                                    clim = {'kind': 'value', 'lims': [0,0.15,0.3]})


    #all_epochs.filter(10,14)

#     bands = {'delta': np.array([1,1.5,2,2.5]),
#               'theta': np.array([3,4,5,6,7]),
#               'alpha': np.array([10,11,12,13,14]),
#               'beta':  np.logspace(np.log10(12), np.log10(30), 9),
#               'gamma': np.logspace(np.log10(30), np.log10(50), 9)}
#    bands = {'alpha': np.array([10,11,12,13,14])}
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
