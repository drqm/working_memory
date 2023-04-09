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
#project_dir = 'Z:/Desktop/' + project
os.environ['MINDLABPROJ']=project
os.environ['MNE_ROOT']='~/miniconda3/envs/mne' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

raw_path = project_dir + '/scratch/maxfiltered_data/tsss_st16_corr96'
ica_path = project_dir + '/scratch/working_memory/ICA'
avg_path = project_dir + '/scratch/working_memory/averages'
subjects_dir = project_dir + '/scratch/fs_subjects_dir'
fwd_path = project_dir + '/scratch/forward_models'

qr = Query(project)
sub_codes = qr.get_subjects()
sub = sub_codes[26]

if len(argv) > 1:
    sub = sub_codes[int(argv[1])]

fwd_fn = op.join(fwd_path, sub + '_vol-fwd.fif')
fwd = mne.read_forward_solution(fwd_fn)

band = 'theta'
# inv_fname = op.join(avg_path,'data', sub + '_TFR_inv.p')
src_fname = op.join(avg_path,'data', sub + '_TFR_src2_' + band +'.p')
vol_src_fname = op.join(subjects_dir,sub,'bem',sub + '_vol-src.fif')

# inv_file = open(inv_fname,'rb')
# inv = pickle.load(inv_file)
# inv_file.close()
# print('inv file loaded')

src_file = open(src_fname,'rb')
src = pickle.load(src_file)
src_file.close()
print('src file loaded')

vol_src = mne.read_source_spaces(vol_src_fname)
# for b in src:
#     for c in ['same']:
#         brain=src[b][c].plot(subjects_dir=subjects_dir,#time_viewer=True,
#                                  subject=sub, src = vol_src)

brain=(src['inverted'] + src['other']).crop(-1,6.25).plot(subjects_dir=subjects_dir,#time_viewer=True,
                         subject=sub, src = vol_src, mode = 'glass_brain')

brain=(src['inverted'] + src['other'] - 
       src['same'] - src['different']).crop(-1,6.25).plot(subjects_dir=subjects_dir,#time_viewer=True,
                         subject=sub, src = vol_src)

#     print('\n computing sources \n')

# for p in power:
#     cdata = np.transpose(power[p].data[:,:,:,0:int(power[p].info['sfreq']*1)],[0,3,1,2])
#     cdata = cdata.reshape((cdata.shape[0]*cdata.shape[1], cdata.shape[2], cdata.shape[3]))
#     bmean = np.mean(cdata,0).reshape((1,cdata.shape[1],cdata.shape[2],1))
#     bsd = np.std(cdata,0).reshape((1,cdata.shape[1],cdata.shape[2],1))
#     del(cdata)
#     power[p].data = power[p].data - bmean
#     power[p].data = power[p].data / bsd


# power =  {cond: power[cond].average() for cond in power}
#
# power[conds[0]] = power['same'].copy()
# power[conds[1]] = power['same'].copy()
# power['difference'] = power['same'].copy()
# # power['difference2'] = power['same'].copy()
# power[conds[0]].data = np.mean([power['same'].data,power['different'].data],0)
# power[conds[1]].data = np.mean([power['inverted'].data,power['other'].data],0)
# power['difference'].data = power[conds[1]].data - power[conds[0]].data
# # power['difference2'].data = (power[conds[1]].copy().apply_baseline((-2,0)).data -
# #                              power[conds[0]].copy().apply_baseline((-2,0)).data)
#
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
