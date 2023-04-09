# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 13:15:38 2016

@author: kousik
"""
import os
from os.path import join
#import matplotlib
#matplotlib.use('Qt4Agg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sys import argv
from warnings import filterwarnings

import mne
import numpy as np
from stormdb.access import Query

from mne.preprocessing import create_ecg_epochs, create_eog_epochs
#from sys import argv
filterwarnings("ignore", category=DeprecationWarning)

proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'

#plt.interactive(True)
#plt.ioff()
qy = Query(proj_name)
subs = qy.get_subjects()
scode = 34
if len(argv) > 1:
    scode = int(argv[1])
cur_sub = subs[scode-1]#'0002_BYG' #argv[1]

#cur_cond = argv[2]

## which steps to run
## first steps
read = 1 ## can't save filtered files, so don't run this time \
                  ## consuming process every time
Filter = 1
## epochs and sensor space processing
epochIca = 0
evokeds = 0
saveICA = 1
ICAraw = 1


## set path and subs
rawRoot     = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/maxfiltered_data/tsss_st16_corr96/'
icaRoot     = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/working_memory/ICA/'
artRejRoot  = join(icaRoot, cur_sub)
resultsRoot = join(icaRoot, cur_sub)

if not os.path.exists(resultsRoot):
    os.makedirs(resultsRoot)

#regex = re.compile(cur_cond + '.*-ica.fif')
#icaFileList = [f for f in os.listdir(artRejRoot) if f.endswith('-ica.fif')]
#icaFileList = ['bass2_raw_tsss-ica.fif']#,'bass2_raw_tsss-ica.fif','bass3_raw_tsss-ica.fif','melody1_raw_tsss-ica.fif','melody2_raw_tsss-ica.fif','melody3_raw_tsss-ica.fif','dichotic1_raw_tsss-ica.fif','dichotic2_raw_tsss-ica.fif','dichotic3_raw_tsss-ica.fif','high1_raw_tsss-ica.fif','high2_raw_tsss-ica.fif','high3_raw_tsss-ica.fif']
icaFileList  =  ['loc','inv','main']#'loc','inv','main']#['loc']#,'inv','main']
if len(argv) > 2:
    icaFileList  = argv[2:]
icaFileList.sort()

#icaFileList = [f for f in os.listdir(artRejRoot) if re.match(regex,f)]
#icaFileList.sort()

#rawFileList = [f for f in os.listdir(artRejRoot) if f.endswith('tsss.fif')]
#rawFileList.sort()

#artRejFileList = [f for f in os.listdir(artRejRoot) if f.endswith('_ica-raw.fif')]
#artRejFileList.sort()

#rawFileList=[]
artRejFileList=[]

for f in np.arange(0,np.size(icaFileList)):
    artRejFileList.insert(f,icaFileList[f]+'_raw_tsss.fif')
    #rawFileList.insert(f,icaFileList[f][:16]+'_tsss.fif')

for k in icaFileList:
    name=k+"_raw_tsss"
    j=k+"_raw_tsss-ica.fif"
    print(name)
    icacomps = mne.preprocessing.read_ica(join(artRejRoot,j))
    if icacomps.exclude:
        print('##################')
        print('Pre-selected comps: '+str(icacomps.exclude))
        print('##################')
        icacomps.excludeold=icacomps.exclude
        icacomps.exclude=[]
    if not icacomps.exclude:
        print('Old components copied. Exclude field cleared')

    raw = mne.io.Raw(join(rawRoot,cur_sub,name+'.fif'), preload=True)
    ecg_picks = mne.pick_types(raw.info, meg=False, eeg=False, eog=False, ecg=True,
                   stim=False, exclude='bads')
    #eog_picks = mne.pick_types(raw.info, meg=False, eeg=False, ecg=False, eog=True,
#                   stim=False, exclude='bads')[0]
    meg_picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, ecg=False,
                       stim=False, exclude='bads')

    ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5,picks=meg_picks, verbose=False)                                   #ch_name=raw.ch_names[ecg_picks].encode('UTF8'))
    ecg_evoked = ecg_epochs.average()
    #eog_evoked = create_eog_epochs(raw, tmin=-.5, tmax=.5,picks=meg_picks,
#                           ch_name=raw.ch_names[eog_picks].encode('UTF8'), verbose=False).average()
    eog_evoked = create_eog_epochs(raw, tmin=-.5, tmax=.5,picks=meg_picks,
                                   ch_name="EOG001").average()
    # ica topos
    source_idx = range(0, icacomps.n_components_)
    ica_plot_mag = icacomps.plot_components(source_idx, ch_type="mag")
    ica_plot_grad = icacomps.plot_components(source_idx, ch_type="grad")
    plt.waitforbuttonpress(1)
    #ica_plot.canvas.manager.window.attributes('-topmost',1)

    title = 'Sources related to %s artifacts (red)'

    #ask for comps ECG
    prompt = '> '
    ecg_done = 'N'
    eog_done = 'N'
    exclude_all = icacomps.exclude.copy()
    while ecg_done.strip() != 'Y' and ecg_done.strip() != 'y':
        icacomps.exclude =  exclude_all.copy()
        ecg_source_idx = []
        print('##################')
        print('Pre-selected comps (both ECG and EOG): '+str(icacomps.excludeold))
        print('##################')
        print('What components should be rejected as ECG comps?')
        print('If more than one, list them each separated by a comma and a space')
        try:
            ecg_source_idx = map(int, input(prompt).split(','))
        except ValueError:
            ecg_source_idx = []
            print('##################')
            print('Exiting ECG - No components selected')
            break

        print(ecg_source_idx)

        if ecg_source_idx:
            icacomps.exclude += ecg_source_idx
            print(ecg_source_idx)
            source_plot_ecg = icacomps.plot_sources(ecg_evoked)
            plt.waitforbuttonpress(1)
            clean_plot_ecg=icacomps.plot_overlay(ecg_evoked)
            plt.waitforbuttonpress(1)
            print('##################')
            print('Clean enough?[Y/N]: ')
            print('')
            print('To terminate without selecting any components, type "N" now')
            print('and then don''t select any components pressing ENTER')
            ecg_done = input(prompt)
            plt.close(source_plot_ecg)
            plt.close(clean_plot_ecg)

    ecg_exclude = ecg_source_idx

    if ecg_source_idx:
       # icacomps.exclude += ecg_source_idx
        source_plot_ecg.savefig(join(resultsRoot,name + '_ecg_source_vis.pdf'), format = 'pdf')
        plt.waitforbuttonpress(1)
        clean_plot_ecg.savefig(join(resultsRoot,name + '_ecg_clean_vis.pdf'), format = 'pdf')
        plt.waitforbuttonpress(1)
  #      scores_plot_ecg.savefig(resultsRoot + name + 'scores_plot_ecg_vis.pdf', format = 'pdf')
        plt.close(source_plot_ecg)
        plt.close(clean_plot_ecg)
    else:
        print('*** No ECG components rejected...')
    exclude_all = icacomps.exclude.copy()
    while eog_done.strip() != 'Y' and eog_done.strip() != 'y':
        icacomps.exclude =  exclude_all.copy()
        eog_source_idx = []
        print('##################')
        print('Pre-selected comps (both ECG and EOG): '+str(icacomps.excludeold))
        print('##################')
        print('What components should be rejected as EOG comps?')
        print('If more than one, list them each separated by a comma and a space')
        print('And if none, just hit ENTER')
        try:
            eog_source_idx = map(int, input(prompt).split(','))
        except ValueError:
            eog_source_idx = []
            print('##################')
            print('Exiting EOG - No components selected')
            break

        print(eog_source_idx)

        if eog_source_idx:
            icacomps.exclude += eog_source_idx
            print(eog_source_idx)
            source_plot_eog = icacomps.plot_sources(eog_evoked)
            plt.waitforbuttonpress(1)
            clean_plot_eog=icacomps.plot_overlay(eog_evoked)
            plt.waitforbuttonpress(1)
            print('##################')
            print('Clean enough?[Y/N]: ')
            print('')
            print('To terminate without selecting any components, type "N" now')
            print('and then don''t select any components pressing ENTER')
            eog_done = input(prompt)
            plt.close(source_plot_eog)
            plt.close(clean_plot_eog)
    eog_exclude = eog_source_idx

    if eog_source_idx:
        #icacomps.exclude += eog_source_idx
        source_plot_eog.savefig(join(resultsRoot,name + '_eog_source_vis.pdf'), format = 'pdf')
#        plt.waitforbuttonpress(1)
        clean_plot_eog.savefig(join(resultsRoot,name + '_eog_clean_vis.pdf'), format = 'pdf')
#        plt.waitforbuttonpress(1)
#        scores_plot_eog.savefig(resultsRoot + name + 'scores_plot_eog_vis.pdf', format = 'pdf')
        plt.close(source_plot_eog)
        plt.close(clean_plot_eog)
    else:
        print('*** No EOG components rejected...')

    print('############')
    print('*** Excluding following components: ', icacomps.exclude)
    print('')
    #ica_plot_mag.savefig(join(resultsRoot,name + ('comps_eog%s-ecg%s_vis_mag.pdf' % ('_'.join(map(str,eog_exclude)),'_'.join(map(str,ecg_exclude))))),format='pdf')
    #ica_plot_grad.savefig(join(resultsRoot,name + ('comps_eog%s-ecg%s_vis_grad.pdf' % ('_'.join(map(str,eog_exclude)),'_'.join(map(str,ecg_exclude))))),format='pdf')
    ica_plot_mag.savefig(join(resultsRoot,'{}comps_eog_{}-ecg_{}_vis_mag.pdf'.format(name,eog_exclude,ecg_exclude)))
    ica_plot_grad.savefig(join(resultsRoot,'{}comps_eog_{}-ecg_{}_vis_grad.pdf'.format(name,eog_exclude,ecg_exclude)))

    plt.close('all')

    #raw_ica = icacomps.apply(raw)
    #raw_ica.save(join(resultsRoot,name + '_ica-vis-raw.fif'), overwrite=True,verbose=False)
    ica = icacomps.copy()
    ica.save(join(resultsRoot, name + '-ica.fif'),overwrite=True)
