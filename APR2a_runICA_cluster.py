#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from sys import argv

def runICA(in_fname,out_fname,ICA_dir,bname,bads = [],rawfilter = True, l_freq = 1,
               h_freq = 100,filter_method = 'iir'):   
    raw = mne.io.read_raw_fif(in_fname,preload = True)
    if bads:
        raw.info['bads'] = bads
    raw.interpolate_bads(reset_bads = True)
    raw.pick_types(eeg = False, meg = True, ecg = True, eog = True)

    if rawfilter:
        raw.filter(l_freq=l_freq,h_freq=h_freq,method=filter_method)
    
    ica = []
    n_max_ecg = 3 
    ica = ICA(n_components=0.90,noise_cov=None)
    ica.fit(raw)
    
    ecg_source_idx, ecg_scores, ecg_exclude = [], [], []
    eog_source_idx, eog_scores, eog_exclude = [], [], []
    
    eog_picks = mne.pick_types(raw.info, eeg=False, meg = False, ecg=False, eog=True)
    ecg_picks = mne.pick_types(raw.info, eeg=False, meg = False, ecg=True, eog=False)[0]
    
    # ica_picks = mne.pick_types(raw.info,eeg=True, meg = True, eog=False, ecg=False,
    #                            exclude='bads')
    ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5)#, #picks=ica_picks,
                                  # ch_name = raw.ch_names[ecg_picks])
    ecg_evoked = ecg_epochs.average()

    eog_epochs = create_eog_epochs(raw, tmin=-.5, tmax=.5, #picks=ica_picks, 
                                   ch_name=[raw.ch_names[n] for n in eog_picks], 
                                   verbose=False)
    eog_evoked = eog_epochs.average()
    ecg_source_idx, ecg_scores = ica.find_bads_ecg(ecg_epochs,method = 'ctps')#, method='ctps')
    eog_source_idx, eog_scores = ica.find_bads_eog(eog_epochs,ch_name=[raw.ch_names[n] for n in eog_picks])#eog_epochs)
    
    title = 'Sources related to %s artifacts (red)'
    source_idx = range(0, ica.n_components_)

    ica_plot_mag = ica.plot_components(source_idx, ch_type="mag")
    ica_plot_grad = ica.plot_components(source_idx, ch_type="grad")
  
    if not ecg_source_idx:
        print("No ECG components above threshold were identified " +
              " - selecting the component with the highest score under threshold")
        ecg_exclude = [np.absolute(ecg_scores).argmax()]
        ecg_source_idx=[np.absolute(ecg_scores).argmax()]
    elif ecg_source_idx:
        ecg_exclude += ecg_source_idx[:n_max_ecg]
    ica.exclude += ecg_exclude
    
    if not eog_source_idx:
        if sum(np.absolute(eog_scores)>0.3) > 0:
            eog_exclude=[np.absolute(eog_scores).argmax()]
            eog_source_idx=[np.absolute(eog_scores).argmax()]
            print("No EOG components above threshold were identified " +
                  " - selecting the component with the highest score under threshold above 0.3")
        elif not sum(np.absolute(eog_scores)>0.3) > 0:
            eog_exclude=[]
            print("No EOG components above threshold were identified")
    elif eog_source_idx:
         eog_exclude += eog_source_idx

    ica.exclude += eog_exclude
                                           

    mag_name = '{}{}_comps_eog_{}-ecg_{}_mag.pdf'.format(ICA_dir,bname,
                                                       str(eog_exclude),
                                                       str(ecg_exclude))
    grad_name = '{}{}_comps_eog_{}-ecg_{}_grad.pdf'.format(ICA_dir,bname,
                                               str(eog_exclude),
                                               str(ecg_exclude))
    ica_plot_mag.savefig(mag_name,format='pdf')
    ica_plot_grad.savefig(grad_name,format='pdf')
    
    scores_plots_ecg=ica.plot_scores(ecg_scores, exclude=ecg_source_idx, title=title % 'ecg')
    scores_plots_ecg.savefig(ICA_dir + bname + '_ecg_scores.pdf', format = 'pdf')
    scores_plots_eog=ica.plot_scores(eog_scores, exclude=eog_source_idx, title=title % 'eog')
    scores_plots_eog.savefig(ICA_dir + bname + '_eog_scores.pdf', format = 'pdf')
    
    source_source_ecg=ica.plot_sources(ecg_evoked)#, exclude=ecg_source_idx)
    source_source_ecg.savefig(ICA_dir + bname + '_ecg_source.pdf', format = 'pdf')
    source_clean_ecg=ica.plot_overlay(ecg_evoked)#, exclude=ecg_source_idx)
    source_clean_ecg.savefig(ICA_dir + bname + '_ecg_clean.pdf', format = 'pdf')
    
    source_source_eog=ica.plot_sources(eog_evoked)#, exclude=eog_source_idx)
    source_source_eog.savefig(ICA_dir + bname + '_eog_source.pdf', format = 'pdf')
    source_clean_eog=ica.plot_overlay(eog_evoked, exclude=eog_source_idx)
    source_clean_eog.savefig(ICA_dir + bname + '_eog_clean.pdf', format = 'pdf')
    
    overl_plot = ica.plot_overlay(raw)
    overl_plot.savefig(ICA_dir + bname + '_overl.pdf', format = 'pdf')
    plt.close('all')
    ica.save(out_fname)
       
runICA(in_fname = argv[1],out_fname = argv[2], ICA_dir = argv[3],bname = argv[4])