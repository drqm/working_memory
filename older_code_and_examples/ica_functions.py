# -*- coding: utf-8 -*-
"""
Adapted on Aug 2015)
(Orig created on Wed Mar 19 09:34:24 2014)

@orig_author: lau
@author: andreas & niels christian
"""

'''ANALYSIS PIPELINE FUNCTIONS'''
' Many of these are NOT used!!! '

## import libraries
import mne
import os
import numpy as np
import scipy as sci
#==============================================================================
# import sklearn
# import sklearn.svm
# import sklearn.pipeline
# import sklearn.feature_selection
# import sklearn.cross_validation
# import sklearn.preprocessing
# import sklearn.linear_model
#==============================================================================
#import matplotlib
#matplotlib.use('Qt4Agg')
#matplotlib.use('TkAgg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mne.io import Raw
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne.viz import plot_evoked_topomap


''' PREPROCESSING PART '''
    
       
def readRawList(fileList,preload=False):
    '''Read in fileList, and choose whether to preload. Returns "raw"'''
    raw = Raw(fileList,preload=preload) ## read in files
    return raw
 
def filterRaw(raw,l_freq,h_freq,method='iir',save=False):
    '''Filter raw file at the given frequencies and with the given method'''
    raw.filter(l_freq=l_freq,h_freq=h_freq,method=method)
    if save:
        raw.save('filt' + str(l_freq) + '_' + str(h_freq) +\
                'raw_tsss_at_mc.fif')
                
def splitSaveRaw(raw,splits,fnamePattern):
    for iSplit, (tmin,tmax) in enumerate(splits):
        raw.copy().crop(tmin,tmax).save(fnamePattern % iSplit)
    
def findEvents(raw,stim_channel='STI101',verbose=False,
                        min_duration=0.001):
     '''Find Target ids'''                           
     events = mne.find_events(raw=raw,stim_channel=stim_channel,
                                verbose=verbose,min_duration=min_duration) 
     return events                                   
                                                                     
''' re adjustTimeLineBy, cf. spm-preproc-script (meg_context_pipeline_func) '''
def getEpochs(raw,tmin,tmax,baseline,reject,event_id,picks,verbose=False,
              decim=4,adjustTimeLineBy=-13.6):
    epochsList = []

    events = mne.find_events(raw, stim_channel='STI101', shortest_event=1)
    print(events[:,0])
    ## adjust trigger timeline
    events[:,0] = [x - np.round(adjustTimeLineBy*10**-3 * raw.info['sfreq']) for x in events[:,0]]
    print(events[:,0])        
    
    ## epoching   
    epochs = mne.Epochs(raw,events,event_id,tmin,tmax,picks=picks,
                        baseline=baseline,reject=reject,preload=True,
                        decim=decim,verbose=verbose)
    epochs.drop_bad_epochs()
    epochsList.append(epochs)
    return epochsList

''' cf. http://martinos.org/mne/stable/auto_examples/preprocessing/plot_ica_from_raw.html for help on integrating ecg-identification as well '''    
def runICA(raw,saveRoot,name):

    saveRoot = saveRoot    
    icaList = [] 
    ica = []
    n_max_ecg = 3   # max number of ecg components 
#    n_max_eog_1 = 2 # max number of vert eog comps
#    n_max_eog_2 = 2 # max number of horiz eog comps          
    ecg_source_idx, ecg_scores, ecg_exclude = [], [], []
    eog_source_idx, eog_scores, eog_exclude = [], [], []
    #horiz = 1       # will later be modified to horiz = 0 if no horizontal EOG components are identified                   
    ica = ICA(n_components=0.90,noise_cov=None)

    #raw = raw.drop_channels(['ECG003','EOG001'])
    ica.fit(raw)
    #*************
    #eog_picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=False, ecg=False, eog=True, emg=False)[0]
    ecg_picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=False, ecg=True, eog=False, emg=False)[0]
    ica_picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, ecg=False,
                   stim=False, exclude='bads')
    ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5, picks = ica_picks)
    ecg_evoked = ecg_epochs.average()

    eog_evoked = create_eog_epochs(raw, tmin=-.5, tmax=.5, picks=ica_picks, 
                                   ch_name='EOG001',#raw.ch_names[eog_picks].encode('UTF8'), 
                                   verbose=False).average()
    
    ecg_source_idx, ecg_scores = ica.find_bads_ecg(ecg_epochs,
                                                   #ch_name='ECG001',#raw.ch_names[ecg_picks].encode('UTF8'),
                                                   method='ctps')
            
    print('\n\ntest\n\n')
    eog_source_idx, eog_scores = ica.find_bads_eog(raw,ch_name = 'EOG001')#ch_name=raw.ch_names[eog_picks].encode('UTF8'))
            
    print('\n\ntest2\n\n')
#    eog_source_idx, eog_scores = ica.find_bads_eog(raw,ch_name='EEG002')
#    if not eog_source_idx_2:
#        horiz = 0
    
    #show_picks = np.abs(scores).argsort()[::-1][:5]
    #ica.plot_sources(raw, show_picks, exclude=ecg_inds, title=title % 'ecg')
    
        
    # defining a title-frame for later use
    title = 'Sources related to %s artifacts (red)'

    # extracting number of ica-components and plotting their topographies
    source_idx = range(0, ica.n_components_)
    print('\n\nso far so good\n\n')
    ica_plot = ica.plot_components(source_idx, ch_type="mag")                                           


    # select ICA sources and reconstruct MEG signals, compute clean ERFs
    # Add detected artefact sources to exclusion list
    # We now add the eog artefacts to the ica.exclusion list
    print('\n\ndone with 1st plot\n\n')
    if not ecg_source_idx:
        print("No ECG components above threshold were identified for subject " + name +
        " - selecting the component with the highest score under threshold")
        ecg_exclude = [np.absolute(ecg_scores).argmax()]
        ecg_source_idx=[np.absolute(ecg_scores).argmax()]
    elif ecg_source_idx:
        ecg_exclude += ecg_source_idx[:n_max_ecg]
    ica.exclude += ecg_exclude

    if not eog_source_idx:
        if np.sum(np.absolute(eog_scores)>0.3) > 0:
            eog_exclude=[np.absolute(eog_scores).argmax()]
            eog_source_idx=[np.absolute(eog_scores).argmax()]
            print("No EOG components above threshold were identified " + name +
            " - selecting the component with the highest score under threshold above 0.3")
        elif not np.sum(np.absolute(eog_scores)>0.3) > 0:
            eog_exclude=[]
            print("No EOG components above threshold were identified" + name)
    elif eog_source_idx:
         eog_exclude += eog_source_idx

    ica.exclude += eog_exclude

    print('########## saving')
    save_name = saveRoot + '/' + name + 'comps_eog_{}-ecg_{}.pdf'.format(str(eog_exclude),
                                                                        str(ecg_exclude))
    ica_plot.savefig(save_name,format='pdf')
    
    scores_plots_ecg=ica.plot_scores(ecg_scores, exclude=ecg_source_idx, title=title % 'ecg')
    scores_plots_ecg.savefig(saveRoot + name + '_ecg_scores.pdf', format = 'pdf')
    scores_plots_eog=ica.plot_scores(eog_scores, exclude=eog_source_idx, title=title % 'eog')
    scores_plots_eog.savefig(saveRoot + name + '_eog_scores.pdf', format = 'pdf')
    
    source_source_ecg=ica.plot_sources(ecg_evoked)
    source_source_ecg.savefig(saveRoot + name + '_ecg_source.pdf', format = 'pdf')
    source_clean_ecg=ica.plot_overlay(ecg_evoked)
    source_clean_ecg.savefig(saveRoot + name + '_ecg_clean.pdf', format = 'pdf')
        
    source_source_eog=ica.plot_sources(eog_evoked)
    source_source_eog.savefig(saveRoot + name + '_eog_source.pdf', format = 'pdf')
    source_clean_eog=ica.plot_overlay(eog_evoked)
    source_clean_eog.savefig(saveRoot + name + '_eog_clean.pdf', format = 'pdf')
     
    overl_plot = ica.plot_overlay(raw)
    overl_plot.savefig(saveRoot + name + '_overl.pdf', format = 'pdf')
 
    plt.close('all')
    ## restore sensor space data
    icaList = ica.apply(raw)
    return(icaList, ica)

    
# def getEvokedFields(epochsList,event_id,trigs):
#     ''' get evoked fields for each epoch object '''
#     evokedList = []
#     conds = event_id.keys()
#     for i in xrange(len(epochsList)):
#         for j in xrange(len(conds)):
#             evokedList.append(epochsList[i][conds[j]].average())
#     return evokedList
    
# def adjustTriggerTimeLine(evokedList,adjustBy):
#     ''' adjust trigger time line for 'at' and 'ad' by the vowel duration '''
#     for i in xrange(len(evokedList)):
#         ## mne_doc: "When relative is True, positive value of tshift moves the data forward while negative tshift moves it backward." Hence, negative here to account for the initial vowel. 
#         evokedList[i].shift_time(tshift=adjustBy,relative=True)
        
#     return evokedList
    
#def getEvokedContrast(evokedList):
#    ''' calculate contrasts for each comparison '''
#    contrastList = []
#    for i in xrange(len(evokedList)/2):
#        contrastList.append(evokedList[2*i+1] - evokedList[2*i])
#    return contrastList
    
def saveRaw(raw,ica,saveRoot,name):
    raw.save(saveRoot + name + '_ica-raw.fif', overwrite=True)
 #   ica.save(saveRoot + name + '-ica.fif')
    
def saveICA(raw,ica,saveRoot,name):
    #raw.save(saveRoot + name + '_ica-raw.fif', overwrite=True)
    ica.save(saveRoot + name + '-ica.fif')