#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import os.path as op
import numpy as np
import pickle
from warnings import filterwarnings
from sys import argv
from stormdb.cluster import ClusterBatch
from stormdb.access import Query

project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MINDLABPROJ']=project
os.environ['MNE_ROOT']='~/miniconda3/envs/mne' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'
script_dir = '/projects/{}/scripts/working_memory/'.format(project)

qr = Query(project)
sub_codes = qr.get_subjects()

subNs = range(11,91)
cb = ClusterBatch(project)
for s in subNs:
    submit_cmd = 'python {}APR10a_connectivity.py {}'.format(script_dir,s)
    cb.add_job(cmd=submit_cmd, queue='short.q',n_threads = 4, cleanup = False)

cb.submit()
