#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from warnings import filterwarnings
from stormdb.cluster import ClusterBatch
from stormdb.access import Query

project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MINDLABPROJ']=project
os.environ['MNE_ROOT']='~/miniconda3/envs/mne' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'
script_dir = '/projects/{}/scripts/working_memory/'.format(project)

subNs = range(11,91)#91)

#subNs = [23,27,29,30,32,33,37,38,39,42,44,47,49,51,52,53,54,56,57,65,67,68,71,74,75,76,78,79,80,81,83,85]
cb = ClusterBatch(project)
for s in subNs:
    #sub = sub_codes[s-1]
    submit_cmd = 'python {}APR6i_decoding_imagined_seq_sources.py {}'.format(script_dir,s)
    cb.add_job(cmd=submit_cmd, queue='all.q',n_threads = 6, cleanup = False)

cb.submit()
