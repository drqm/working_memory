#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import os
from warnings import filterwarnings
from stormdb.cluster import ClusterBatch

project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
#os.environ['MINDLABPROJ']=project
#os.environ['MNE_ROOT']='~/miniconda3/envs/mne' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'
script_dir = '/projects/{}/scripts/working_memory/'.format(project)

subNs = [79] #range(11,21) #25,26,27,29,31

cb = ClusterBatch(project)
for s in subNs:
    #sub = sub_codes[s-1]
    submit_cmd = 'python {}APR6h_decoding_localizer_source_gemma.py {}'.format(script_dir,s)
    cb.add_job(cmd=submit_cmd, queue='all.q',n_threads = 8, cleanup = False)

cb.submit()