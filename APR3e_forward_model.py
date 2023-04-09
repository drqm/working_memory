#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:00:04 2021

@author: david
"""
import os
import os.path as op
from stormdb.process import MNEPython
from stormdb.access import Query
from sys import argv

proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MINDLABPROJ']= proj_name

subj_dir = op.join('/projects',proj_name,'scratch','fs_subjects_dir')
fwd_dir = op.join('/projects',proj_name,'scratch','forward_models')

qy = Query(proj_name)
subs = qy.get_subjects()
subno = [11]
if len(argv)>1:
    subno = argv[1:]
subjects = [subs[int(s)-1] for s in subno]

# compute volume forward model
for subject in subjects:
    bem_fn = op.join(subj_dir,subject,'bem',subject + '-1LBEM-sol.fif')
    src_fn = op.join(subj_dir,subject,'bem',subject + '_vol-src.fif')
    inst = op.join('/projects',proj_name,'scratch','maxfiltered_data',
                   'tsss_st16_corr96/',subject,'loc_raw_tsss.fif')
    trans = op.join('/projects',proj_name,'scratch','trans',subject+'-trans.fif')
    fwd_fn = op.join(fwd_dir,subject + '_vol-fwd.fif')
    mp_fwd = MNEPython(proj_name)
    mp_fwd.make_forward_solution(inst,trans,bem_fn,src_fn,fwd_fn)
    mp_fwd.submit(fake = True)
    mp_fwd.submit()

# compute surface forward model
for subject in subjects:
    bem_fn = op.join(subj_dir,subject,'bem',subject + '-1LBEM-sol.fif')
    src_fn = op.join(subj_dir,subject,'bem',subject + '-src.fif')
    inst = op.join('/projects',proj_name,'scratch','maxfiltered_data',
                   'tsss_st16_corr96/',subject,'loc_raw_tsss.fif')
    trans = op.join('/projects',proj_name,'scratch','trans',subject+'-trans.fif')
    fwd_fn = op.join(fwd_dir,subject + '-fwd.fif')
    mp_fwd = MNEPython(proj_name)
    mp_fwd.make_forward_solution(inst,trans,bem_fn,src_fn,fwd_fn)
    mp_fwd.submit(fake = True)
    mp_fwd.submit()
