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
os.environ['MNE_ROOT']='/users/david/miniconda3/envs/mne' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'
script_dir = '/projects/{}/scripts/working_memory/'.format(project)

qr = Query(project)
sub_codes = qr.get_subjects()

#subNs = np.arange(8) + 1
subNs = range(11,91)#,11,12,13,14,15,16]
#subNs = [24]
#bands = ['delta','theta','alpha','beta1','beta2']#,'HFA']
bands = ['HFA']
cb = ClusterBatch(project)
for s in subNs:
    for b in bands:
        #sub = sub_codes[s-1]
        print(s,b)
        if b == 'HFA':
            submit_cmd = 'python {}exploration/APR5k_TFR_analyses_source_HFA.py {}'.format(script_dir,s)
        else:
            submit_cmd = 'python {}exploration/APR5j_TFR_analyses_source2.py {} {}'.format(script_dir,s,b)
        #
        cb.add_job(cmd=submit_cmd, queue='all.q',n_threads = 4,cleanup = False)

cb.submit()
