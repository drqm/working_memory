#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:56:14 2021

@author: david
"""
proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
wdir = '/projects/' + proj_name + '/scratch/working_memory/'
scripts_dir = '/projects/' + proj_name + '/scripts/working_memory/'
import sys
sys.path.append(scripts_dir)

import mne
from stormdb.access import Query

from mne.datasets import sample, fetch_fsaverage

ignoreDepWarnings = 1
if ignoreDepWarnings:
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

#from pickle import dump,load

proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
wdir = '/projects/' + proj_name + '/scratch/working_memory/'
subjects_dir = '/projects/' + proj_name + '/scratch/fs_subjects_dir/'
data_dir = wdir + 'averages/'
#sample_path = sample.data_path()
#sample_subjects_dir = sample_path + '/subjects'
#fetch_fsaverage(sample_subjects_dir)

qr = Query(proj_name)
subjects = qr.get_subjects()

subs = [68]#[3,6,7,8,9,10,11,12,14,15,16,17,18,19,20,22,
        #23,24,26,27,28,29,30,31,32,33,34,35,36,37,
        #39,41,42,44,45,46,47,48,49,50,51]
src_to = mne.read_source_spaces(subjects_dir +
                                'fsaverage/bem/fsaverage-vol-5-src.fif')
if len(sys.argv) > 1:
    subs = [int(sys.argv[1])]

for s in subs:
    scode = subjects[s-1]
    print('morhping subject {} \n'.format(scode))

    srcfname = subjects_dir + scode + '/bem/' + scode + '_vol-src.fif'
    csrc = mne.read_source_spaces(srcfname)
    morph = mne.compute_source_morph(src = csrc, subject_to = 'fsaverage',src_to = src_to,
                                     subjects_dir = subjects_dir,smooth=20)#, spacing='ico4')
    # morph = mne.compute_source_morph(src = csrc, subject_to = 'fsaverage',
    #                                  subjects_dir = subjects_dir)#, spacing=8.)
    morph.compute_vol_morph_mat()
    print(morph.vol_morph_mat)
    morph.save(subjects_dir + scode + '/bem/' + scode + '_smooth_vol-morph.h5',
               overwrite = True)
