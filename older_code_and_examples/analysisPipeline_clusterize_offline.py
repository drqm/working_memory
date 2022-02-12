# -*- coding: utf-8 -*-
"""
Adapted on Aug 2015)
(Orig created on Wed Mar 19 09:34:24 2014)

@orig_author: lau
@author: andreas & niels christian
"""

''' ANALYSIS PIPELINE '''
### see also MNE_script.py for related processing steps

## read in functions from 'analysisPipelineFunctions'

import os
os.environ['MINDLABPROJ']='MINDLAB2020_MEG-AuditoryPatternRecognition'
os.chdir('/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scripts/invmain')
from ica_functions import runICA, readRawList, filterRaw
from stormdb.access import Query
import matplotlib
#matplotlib.use('Qt4Agg')
#matplotlib.use('TkAgg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.environ['MINDLABPROJ']='MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MNE_ROOT']='/users/david/miniconda3/envs/mne' # for create_bem_surfaces
proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'

# import pandas as pd
# import mne
# import numpy as np
# from subprocess import check_output     # in order to be able to call the bash command 'find' and use the output
# from sys import argv

matplotlib.interactive(True)

plt.close('all')

printStatus = 1
ignoreDepWarnings = 1
if ignoreDepWarnings:
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

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
#
##print argv
#file_name        = argv[2]
qy = Query(proj_name)
subs = qy.get_subjects()
scode = 16
subj= subs[scode-1]#'0012_VK2'#'0002_BYG'#'0004_LXY'#'0001_VLC' '0002_BYG' #'0003_S5V' '0006_TNV' '0007_ESO' '0008_HMD'
wd = "/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/invmain_analyses/"
#subs = pd.read_csv(wd + 'table.csv')
#subidx = subs['subject'].str.contains(subj)
#
#####cond='08
####cond            = argv[1]
#rawRoot          = argv[3]
#resultsRoot      = argv[4]
#print file_name
#print rawRoot
#print resultsRoot

#root = '/projects/MINDLAB2011_39-STN-DBS-Effect-Cortex-MEG/scratch/for_tSSS/'
#resultsRoot = '/projects/MINDLAB2011_39-STN-DBS-Effect-Cortex-MEG/scratch/for_tSSS'#+\
#                'control_move_artRej'

#rawRoot = '/projects/MINDLAB2016_MEG-DichoticMMN/scratch/maxfilter/tsss_st16_corr96/{:s}'.format(subj)
rawRoot = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/maxfiltered_data/tsss_st16_corr96/{:s}/'.format(subj)
resultsRoot= '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/invmain_analyses/ICA/{:s}'.format(subj)

if not(os.path.exists(resultsRoot)):
    os.mkdir(resultsRoot)
    print('*** Directory created') 
    
os.chdir(rawRoot) ## set series directory
#fileList = [f for f in os.listdir(rawRoot) if f.endswith('.fif')]
#fileList = ['loc_raw_tsss.fif']#,'unfamiliar2_raw.fif']
fileList = ['inv']#,'loc','main']#,'main'] #['inv']#
#fileList = ['hh1','hh2','hl1','hl2','lh1','lh2','ll1','ll2']
#,'dichotic3_raw_tsss.fif','high1_raw_tsss.fif','high2_raw_tsss.fif','high3_raw_tsss.fif','melody1_raw_tsss.fif','melody2_raw_tsss.fif','melody3_raw_tsss.fif','bass1_raw_tsss.fif','bass2_raw_tsss.fif','bass3_raw_tsss.fif']
# fileList = ['dichotic1_raw_tsss.fif','dichotic2_raw_tsss.fif','dichotic3_raw_tsss.fif','high1_raw_tsss.fif','high2_raw_tsss.fif','high3_raw_tsss.fif','melody1_raw_tsss.fif','melody2_raw_tsss.fif','melody3_raw_tsss.fif','bass1_raw_tsss.fif','bass3_raw_tsss.fif']

#fileList = ['high2_raw_tsss.fif','high3_raw_tsss.fif','melody2_raw_tsss.fif','melody3_raw_tsss.fif']
# fileList = ['bass2_raw_tsss.fif']

#fileList.sort()

###
#
for j in fileList:
    file_name = j + '_raw_tsss.fif'
    #file_name = subj +'Session' +cond +'_tSSS.fif'
#    file_name = rawRoot + '/' + j
    new_name = file_name[0:-4]
   # chars = ["u","'","[","]"," "]
    
    ''' READ AND FILTER'''
    if read:
        print('Reading subject: ' +file_name)
        ## read raws
        raw = readRawList(file_name,preload=True)
       
    if Filter:
        ## filter
        l_freq, h_freq = 1, 100.0
        filterRaw(raw,l_freq,h_freq)   
    
    ''' INDEPENDENT COMPONENT ANALYSIS (EYE BLINKS) '''
    if ICAraw:
        saveRoot = (resultsRoot +  '/')
        _,ICAcomps = runICA(raw,saveRoot,new_name)
    del raw
    '''SAVE'T ALL'''
    if saveICA:
        if printStatus:
            print('Saving Files for subject: ' +  new_name)
    #                os.chdir(root + allSeries[i] + '/mne') ## set saving directory
        saveRoot = (resultsRoot + '/')
        ICAcomps.save(saveRoot + new_name + '-ica.fif')
        #saveICA(ICAList,ICAcomps,saveRoot,new_name)
    del ICAcomps   #                os.chdir(root + allSeries[i]) ## set back to working directory