#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:08:04 2021

@author: david
"""
import os
import os.path as op
from stormdb.process import Freesurfer
from stormdb.access import Query
from sys import argv

proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MINDLABPROJ']= proj_name

# change if necessary:
#os.environ['MNE_ROOT']='~/miniconda3/envs/mne'
os.environ['MNE_ROOT']='~/miniconda3/envs/mne3d'

subj_dir = op.join('/projects',proj_name,'scratch','fs_subjects_dir')
fwd_dir = op.join('/projects',proj_name,'scratch','forward_models')

qy = Query(proj_name)
subs = qy.get_subjects()
subno = [11]

if len(argv)>1:
    subno = argv[1:]

subjects = [subs[int(s)-1] for s in subno]

bem_jobs = {}
for subject in subjects:
    bem_jobs[subject] = Freesurfer(proj_name= proj_name,subjects_dir = subj_dir)
    #bem_jobs[subject].create_bem_surfaces?
    bem_jobs[subject].create_bem_surfaces(subject=subject,make_coreg_head = True)
    bem_jobs[subject].submit(fake=True)
    bem_jobs[subject].submit()
