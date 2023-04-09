#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import mne
import os
import os.path as op
import numpy as np
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
import pickle
from sys import argv
from stormdb.access import Query
from warnings import filterwarnings
from copy import deepcopy
#matplotlib.use('Qt4Agg')
filterwarnings("ignore", category=DeprecationWarning)

# set paths and other variables
project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
project_dir = '/projects/' + project
os.environ['MINDLABPROJ']=project
os.environ['MNE_ROOT']='~/miniconda3/envs/mne' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

raw_path = project_dir + '/scratch/maxfiltered_data/tsss_st16_corr96'
ica_path = project_dir + '/scratch/working_memory/ICA'
avg_path = project_dir + '/scratch/working_memory/averages'
subjects_dir = project_dir + '/scratch/fs_subjects_dir'
fwd_path = project_dir + '/scratch/forward_models'

## Get subject codes
qr = Query(project)
sub_codes = qr.get_subjects()
sub_n = 24

if len(argv) > 1:
    sub_n = int(argv[1])
sub = sub_codes[sub_n-1]

## Load forward solution
fwd_fn = op.join(fwd_path, sub + '_vol-fwd.fif')
fwd = mne.read_forward_solution(fwd_fn)

## Conditions to analyse
conds = ['main','inv']
event_ids = [[['same',1],['different',2]],[['inverted',3],['other',4]]]
chans = 'mag' # select channels to do the inversion
# Preprocess data looping over blocks:
epochs = {}
power = {}
for cidx, c in enumerate(conds):

    # Load data, apply ICA and resample
    fname = os.path.join(raw_path, sub, c + '_raw_tsss.fif')
    icaname = os.path.join(ica_path,sub, c + '_raw_tsss-ica.fif')
    raw = mne.io.read_raw_fif(fname, preload = True) # load data
    ica = mne.preprocessing.read_ica(icaname) # load ica solution
    raw = ica.apply(raw) # apply ICA
    raw.resample(400)

    # Get and correct triggers:
    events = mne.find_events(raw, shortest_event = 1)
    events = events[np.append(np.diff(events[:,0]) >2,True)] # delete spurious t
    events2 = events.copy()
    events2 = events2[events[:,2] < 20]
    events2[:,2] = events[np.isin(events[:,2]//10,[11,21]),2]//100 + cidx*2# recode events

    # Epoching parameters:
    event_id = dict(event_ids[cidx])
    picks = mne.pick_types(raw.info, meg = True)
    tmin, tmax = -1.25, 6.5 #epoch time
    baseline = (-1.25, 0) # baseline time
    reject = dict(mag = 8e-12, grad = 4000e-13) #eeg = 200e-6, #, eog = 250e-6)

    #epoching:
    epochs[c] = mne.Epochs(raw, events = events2, event_id = event_id,
                    tmin = tmin, tmax = tmax, picks = picks,
                    baseline = baseline, reject = reject)
### TFR analyses
# put all epochs together and select channels
all_epochs = mne.epochs.concatenate_epochs([epochs[c] for c in epochs])
all_epochs.pick_types(chans)
sfreq = all_epochs.info['sfreq']
del epochs # free memory
del raw
## Transform to source space:
## calculate data covariance
data_cov = mne.compute_covariance(all_epochs, tmin= -1, tmax = 6.5, rank =None)

## compute inverse solution
inv = mne.beamformer.make_lcmv(all_epochs.info, fwd, data_cov, reg=0.05,
                                  weight_norm= 'nai', rank = None, pick_ori='max-power')

print('processing HFA')
freqs = np.arange(50, 160, 10)
print(freqs)
n_cycles = freqs / 2.
n_cycles[np.where(n_cycles < 2)] = 2
time_bandwidth = 7
stc_pwr = {}
sbpwr = {} 
for fidx in range(len(freqs)):
    freq = [freqs[fidx]]
    cpwr = np.zeros((len(all_epochs.events), inv['n_sources'], 1, int(np.ceil(len(all_epochs.times)/4))))
    print(cpwr.shape)
    for eix, epoch in enumerate(all_epochs.iter_evoked()):
          if eix < 2:
            print('decomposing epoch {}'.format(eix+1))
            stc = mne.beamformer.apply_lcmv(epoch, inv, max_ori_out = 'signed')
            cdata = np.expand_dims(stc.data, axis = 0)
            cpwr[eix, :, :, :] = np.log(mne.time_frequency.tfr_array_multitaper(cdata,
                                                                    sfreq, freq, 
                                                                    n_cycles=n_cycles[fidx],
                                                                    time_bandwidth=time_bandwidth,
                                                                    use_fft = True,
                                                                    output='power',
                                                                    decim = 4,
                                                                    n_jobs=-1))
    del cdata
    stc2 = deepcopy(stc)
    stc2.data = np.squeeze(cpwr[0,:,0,:])
    stc2.tmin = stc.tmin
    stc2.tstep = stc.tstep * 4
    print(stc2.times)
    del stc
    
    btidx = np.where([x and y for x, y in zip(stc2.times >= -1, stc2.times <= 0)])
    print(btidx)
    bmean = np.mean(np.squeeze(cpwr[:,:,:,btidx], axis = 3),  axis = (0,3), keepdims = True)
    bstd =  np.std(np.squeeze(cpwr[:,:,:,btidx], axis = 3), axis = (0,3), keepdims = True)
    print(bmean.shape)
    print(bstd.shape)
    for trix in range(cpwr.shape[0]):
        print('z scoring epoch {}'.format(trix+1))
        cpwr[trix,:,:,:] = cpwr[trix,:,:,:] - bmean
        cpwr[trix,:,:,:] = cpwr[trix,:,:,:] / bstd
    del bmean
    del bstd
    print(cpwr.shape)
    for eid in all_epochs.event_id.keys():
        print('averaging source power for HFA condition {} frequency {}'.format(eid,freq))
        eidx = all_epochs.events[:,2] == all_epochs.event_id[eid]
        sbpwr.setdefault(eid, np.zeros((inv['n_sources'], len(freqs), len(stc2.times))))
        mnpwr = np.mean(cpwr[eidx,:,:,:],axis=0)
        sbpwr[eid][:,fidx,:] = np.squeeze(mnpwr)
    del cpwr
    del mnpwr

for eid in sbpwr:
    stc_pwr[eid] = deepcopy(stc2)
    stc_pwr[eid].data = sbpwr[eid].mean(axis=1)    
del sbpwr

src_fname = op.join(avg_path,'data', sub + '_TFR_src2_HFA.p')
src_file = open(src_fname,'wb')
pickle.dump(stc_pwr, src_file)
src_file.close()
print('src file saved')
del stc
del stc_pwr
