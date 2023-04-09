#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:56:14 2021

@author: david
"""
import mne
from stormdb.access import Query
import sys
from mne.datasets import sample, fetch_fsaverage
#from pickle import dump,load

proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
wdir = '/projects/' + proj_name + '/scratch/working_memory/'
subjects_dir = '/projects/' + proj_name + '/scratch/fs_subjects_dir/'
data_dir = wdir + 'averages/'
sample_path = sample.data_path()
sample_subjects_dir = sample_path + '/subjects'
fetch_fsaverage(sample_subjects_dir)

qr = Query(proj_name)
subjects = qr.get_subjects()

bem_file = subjects_dir + 'fsaverage/bem/inner_skull.surf'
#src_to = sample_subjects_dir + '/fsaverage/bem/fsaverage-vol-8-src.fif'
src_to = subjects_dir + 'fsaverage/bem/fsaverage_ico4_vol-src.fif'
fsavg_src = mne.setup_volume_source_space(subject='fsaverage', pos = 6.2, surface = bem_file)
print(fsavg_src)
fsavg_src.save(src_to)
