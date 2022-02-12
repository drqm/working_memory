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
#os.environ['MNE_ROOT']='/users/david/miniconda3/envs/mne3d' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'
script_dir = '/projects/{}/scripts/working_memory/'.format(project)
# args = {}
# args['save_averages'] = True
# args['compute_sources'] = True
# args['save_sources'] = True
# args['plot_sources'] = True
# args['suffix'] = ''
# args['conds'] = ['main','inv']

qr = Query(project)
sub_codes = qr.get_subjects()

#subNs = np.arange(8) + 1
subNs = [7]#25,26,27,29,31]#,11,12,13,14,15,16]
cb = ClusterBatch(project)
for s in subNs:
    sub = sub_codes[s-1]
    submit_cmd = 'python {}APR6a_decoding_pipeline.py {}'.format(script_dir,sub)
    cb.add_job(cmd=submit_cmd, queue='highmem.q',n_threads = 4,cleanup = False)

cb.submit()
