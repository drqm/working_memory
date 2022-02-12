#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 13:01:34 2020

@author: david
"""
import mne
import os
import os.path as op
from stormdb.access import Query
from sys import argv

# set a few environmental variables to make sure it works properly
proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
#os.environ['SUBJECTS_DIR']= '../../scratch/fs_subjects_dir'
os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt5'

#os.environ['MINDLABPROJ']= proj_name
#os.environ['MNE_ROOT']='/users/david/miniconda3/envs/mne3d' # for create_bem_surfaces

#necessary for coreg gui

os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

#subj_dir = op.join('/projects',proj_name,'scratch','fs_subjects_dir')
#fwd_dir = op.join('/projects',proj_name,'scratch','forward_models')

#subj_dir = '/projects/MINDLAB2020_MEG-UncertaintyModulation/scratch/APR_subjects_dir/fs_subjects_dir'
subj_dir = '../../scratch/fs_subjects_dir'
fwd_dir = '../../scratch/forward_models'
#
qy = Query(proj_name)
subs = qy.get_subjects()
# t1_name = '*t1_mpr*'
#
# # coregistration
subno = [11]
if len(argv)>1:
    subno = argv[1:]
# subjects = ['0011_U7X']
subjects = [subs[int(s)-1] for s in subno]
for subject in subjects:
    inst = op.join('../../scratch/maxfiltered_data',
               'tsss_st16_corr96/',subject,'loc_raw_tsss.fif')
    trans = op.join('../../trans',subject+'-trans.fif')
    mne.gui.coregistration(subject=subject,inst = inst,subjects_dir = subj_dir,
                           advanced_rendering = False)
