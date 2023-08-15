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
blocks = ['task']#'localizer','task']
subNs = range(11,21)
modes = ['sensor']#['sensor','source']
masks = {'': ['']}#, # No mask
         #'include': [['rA1','lA1'], ['rThal','lThal'],['rPCC','lPCC'], ['rHC','lHC'], ['rPCN','lPCN']]}
         #'exclude': [['rA1','lA1'], ['rThal','lThal'],['rPCC','lPCC'], ['rHC','lHC'], ['rPCN','lPCN']]}

cb = ClusterBatch(project)
for b in blocks:
        for m in modes:
              
            # Loop over subjects:
            for sub in subNs:
                submit_cmd_base = 'python {}APR6a_decoding.py {} {} {}'.format(script_dir,b,sub,m)
                
                # If source, check and apply masks:
                if m == 'source':
                    # Loop over masks:
                    for mt in masks:
                        # Copy base command and add ROIs
                        for rs in masks[mt]:
                            submit_cmd = submit_cmd_base + ' ' + mt
                            for srs in rs:
                                submit_cmd += ' ' + srs
                            # Append job
                            cb.add_job(cmd=submit_cmd, queue='highmem.q',n_threads = 6, cleanup = False)
                else:
                    #If only sensor, then apply no mask and add job
                    cb.add_job(cmd=submit_cmd_base, queue='highmem.q',n_threads = 6, cleanup = False)
cb.submit()
