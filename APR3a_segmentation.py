#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:08:04 2021

@author: david
"""
import os
import os.path as op
from stormdb.process import Freesurfer
from stormdb.base import mkdir_p
from stormdb.access import Query
from sys import argv

proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MINDLABPROJ']= proj_name

subj_dir = op.join('/projects',proj_name,'scratch','fs_subjects_dir')
fwd_dir = op.join('/projects',proj_name,'scratch','forward_models')

mkdir_p(subj_dir)
mkdir_p(fwd_dir)

qy = Query(proj_name)
subs = qy.get_subjects()
subno = [11]

if len(argv)>1:
    subno = argv[1:]

subjects = [subs[int(s)-1] for s in subno]
t1_name = '*t1_mpr*'

fs = {}
for subject in subjects:
    inst = op.join('/projects',proj_name,'scratch','maxfiltered_data',
                   'tsss_st16_corr96/',subject,'loc_raw_tsss.fif')
    trans = op.join('/projects',proj_name,'scratch','trans',subject+'-trans.fif')
    try:
        fs[subject] = Freesurfer(proj_name= proj_name,subjects_dir = subj_dir)
        fs[subject].recon_all(subject = subject, t1_series = t1_name)
        fs[subject].submit(fake=True)
        fs[subject].submit()
    except Exception as e:
        print(e)
        continue
