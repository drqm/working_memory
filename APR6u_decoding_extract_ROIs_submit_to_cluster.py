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
subNs = range(40,91)
         #'include': [['rA1','lA1'], ['rThal','lThal'],['rPCC','lPCC'], ['rHC','lHC'], ['rPCN','lPCN']]}
         #'exclude': [['rA1','lA1'], ['rThal','lThal'],['rPCC','lPCC'], ['rHC','lHC'], ['rPCN','lPCN']]}

cb = ClusterBatch(project)
for sub in subNs:
    submit_cmd_base = 'python {}APR6t_decoding_extract_ROIs.py {}'.format(script_dir,sub)
    cb.add_job(cmd=submit_cmd_base, queue='short.q',n_threads = 1, cleanup = False)
cb.submit()

