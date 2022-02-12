#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:59:01 2021

@author: david
"""
import os
import os.path as op
from stormdb.cluster import ClusterBatch
from stormdb.access import Query
from sys import argv

proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MINDLABPROJ']= proj_name
#os.environ['MNE_ROOT']='/users/gemma/miniconda3/envs/mne' # for create_bem_surfaces

subj_dir = op.join('/projects',proj_name,'scratch','fs_subjects_dir')
fwd_dir = op.join('/projects',proj_name,'scratch','forward_models')

qy = Query(proj_name)
subs = qy.get_subjects()
subno = [11]
if len(argv)>1:
    subno = argv[1:]
subjects = [subs[int(s)-1] for s in subno]

# bem model
cb = ClusterBatch(proj_name)
for subject in subjects:
    bem_fn = op.join(subj_dir,subject,'bem',subject + '-1LBEM-sol.fif')
    script = ("from mne import make_bem_model, make_bem_solution, write_bem_solution; "
              "surfs = make_bem_model(subject = '{}', ico=4, "
              "subjects_dir='{}', conductivity = [0.3]); "
              "bem = make_bem_solution(surfs); "
              "write_bem_solution('{}', bem)")
    cmd = "python -c \""
    cmd += script.format(subject,subj_dir,bem_fn)
    cmd += "\""
    print(cmd)
    cb.add_job(cmd = cmd, queue='all.q',n_threads = 2,cleanup = False)
cb.submit()

#surface source space
cb = ClusterBatch(proj_name)
for subject in subjects:
    src_fn = op.join(subj_dir,subject,'bem',subject + '-src.fif')
    script = ("import mne; src = mne.setup_source_space(subject='{}',"
              "subjects_dir = '{}'); mne.write_source_spaces('{}',src=src)")
    cmd = "python -c \""
    cmd += script.format(subject,subj_dir,src_fn)
    cmd += "\""
    print(cmd)
    cb.add_job(cmd = cmd,queue='all.q',n_threads = 2,cleanup = False)
cb.submit()

# volume source space
cb = ClusterBatch(proj_name)
for subject in subjects:
    src_fn = op.join(subj_dir,subject,'bem',subject + '_vol-src.fif')
    in_dir = op.join(subj_dir,subject,'bem','inner_skull.surf')
    script = ("import mne; src = mne.setup_volume_source_space(subject='{}',"
              "subjects_dir = '{}', surface= '{}'); mne.write_source_spaces('{}',src=src)")
    cmd = "python -c \""
    cmd += script.format(subject,subj_dir,in_dir,src_fn)
    cmd += "\""
    print(cmd)
    cb.add_job(cmd = cmd,queue='all.q',n_threads = 2,cleanup = False)
cb.submit()
