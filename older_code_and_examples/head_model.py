#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 13:01:34 2020

@author: david
"""
import mne
import os
import os.path as op
from stormdb.process import Freesurfer, MNEPython
from stormdb.base import mkdir_p
from stormdb.cluster import ClusterBatch
from mne.viz import plot_bem
from stormdb.access import Query

# set a few environmental variables to make sure it works properly
proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MINDLABPROJ']= proj_name
#os.environ['MNE_ROOT']='/users/david/miniconda3/envs/mne' # for create_bem_surfaces

#necessary for coreg gui
os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

subj_dir = op.join('/projects',proj_name,'scratch','fs_subjects_dir')
fwd_dir = op.join('/projects',proj_name,'scratch','forward_models')

mkdir_p(subj_dir)
mkdir_p(fwd_dir)

qy = Query(proj_name)
subs = qy.get_subjects()
subno = [11]#,12,13,14,15,16]#[4,9,10,11]#np.concatenate(([1],np.arange(3,13,1)))
subjects = [subs[s-1] for s in subno]
t1_name = '*t1_mpr*'

# segmentation in freesurfer (takes from 6 to 10 hours)

fs = {}
for subject in subjects:
    inst = op.join('/projects',proj_name,'scratch','maxfiltered_data',
               'tsss_st16_corr96/',subject,'loc_raw_tsss.fif')
    trans = op.join('/projects',proj_name,'scratch','trans',subject+'-trans.fif')
    try:
        fs[subject] = Freesurfer(proj_name= proj_name,subjects_dir = subj_dir)
        fs[subject].recon_all(subject = subject, t1_series = t1_name)
        fs[subject].submit(fake=True)
        fs[subject].submit()
    except:
        continue
    
# Create BEM surfaces

subjects = [subs[s-1] for s in subno]
bem_jobs = {}
for subject in subjects:
    bem_jobs[subject] = Freesurfer(proj_name= proj_name,subjects_dir = subj_dir)
    #bem_jobs[subject].create_bem_surfaces?
    bem_jobs[subject].create_bem_surfaces(subject=subject,make_coreg_head =True)
    bem_jobs[subject].submit(fake=True)
    bem_jobs[subject].submit()

# Plot bem solution  
bem_plots = {}  
subjects = [subs[s-1] for s in subno]    
for subject in subjects:    
    bem_plots[subject] = plot_bem(subject = subject,subjects_dir = subj_dir, 
                              brain_surfaces='white',orientation='coronal')

# coregistration
subjects = [subs[s-1] for s in subno]    
for subject in subjects:  
    inst = op.join('/projects',proj_name,'scratch','maxfiltered_data',
               'tsss_st16_corr96/',subject,'loc_raw_tsss.fif')
    trans = op.join('/projects',proj_name,'scratch','trans',subject+'-trans.fif')
    mne.gui.coregistration(subject=subject,inst = inst,subjects_dir = subj_dir)

# Bem model and surface source space
subno = [11]
subjects = [subs[s-1] for s in subno]
bem_jobs = {}
for subject in subjects:
    bem_fn = op.join(subj_dir,subject,'bem',subject + '-1LBEM-sol.fif')
    src_fn = op.join(subj_dir,subject,'bem',subject + '-src.fif')
    conductivity = [0.3]
    mp = MNEPython(proj_name)
    mp.prepare_bem_model(subject,bem_fn,subjects_dir = subj_dir,
                          conductivity=conductivity)
    # mp.setup_source_space(subject, src_fname = src_fn, subjects_dir = subj_dir,
    #                       add_dist=True)
    mp.submit(fake = True)
    mp.submit()

#prepare volumetric source space:
subno = [11]
subjects = [subs[s-1] for s in subno]
cb = ClusterBatch(proj_name)
for subject in subjects:
    src_fn = op.join(subj_dir,subject,'bem',subject + '_vol2-src.fif')
    in_dir = op.join(subj_dir,subject,'bem','inner_skull.surf')
    script = ("import mne; src = mne.setup_volume_source_space(subject='{}',"
              "subjects_dir = '{}', surface= '{}'); mne.write_source_spaces('{}',src=src)")
    cmd = "python -c \""
    cmd += script.format(subject,subj_dir,in_dir,src_fn)
    cmd += "\""
    print(cmd)
    cb.add_job(cmd = cmd,queue='all.q',n_threads = 2,cleanup = False)
cb.submit()

bem_plots = {}  
subjects = [subs[s-1] for s in subno]    
for subject in subjects:
    src_fn = op.join(subj_dir,subject,'bem',subject + '_vol-src.fif')
    bem_plots[subject] = plot_bem(subject = subject,subjects_dir = subj_dir, 
                              brain_surfaces='white',orientation='coronal',src=src_fn)
   
# compute volumetric forward model
subno = [11]
subjects = [subs[s-1] for s in subno]
bem_jobs = {}
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
subno = [11]
subjects = [subs[s-1] for s in subno]
bem_jobs = {}
for subject in subjects:
    bem_fn = op.join(subj_dir,subject,'bem',subject + '-1LBEM-sol.fif')
    src_fn = op.join(subj_dir,subject,'bem',subject + '-src.fif')
    inst = op.join('/projects',proj_name,'scratch','maxfilter_same_init',
                   'tsss_st16_corr96/',subject,'loc_raw_tsss.fif')
    trans = op.join('/projects',proj_name,'scratch','trans',subject+'-trans.fif')
    fwd_fn = op.join(fwd_dir,subject + '-fwd.fif')
    mp_fwd = MNEPython(proj_name)
    mp_fwd.make_forward_solution(inst,trans,bem_fn,src_fn,fwd_fn)
    mp_fwd.submit(fake = True)
    mp_fwd.submit()