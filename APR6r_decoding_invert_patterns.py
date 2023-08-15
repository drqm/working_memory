# Append to path
project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
project_dir = '/projects/' + project
scripts_dir = project_dir + '/scripts/working_memory/'
import sys
sys.path.append(scripts_dir)

import mne
import os
import os.path as op
import numpy as np
import pickle
from warnings import filterwarnings
from sys import argv
import matplotlib.pyplot as plt
from stormdb.access import Query
import pandas as pd
import src.preprocessing as pfun 
import src.decoding_functions as dfun
import importlib
importlib.reload(dfun)
filterwarnings("ignore", category=DeprecationWarning)

##################### Define relevant variables ################################
# Project info

os.environ['MINDLABPROJ']= project
os.environ['MNE_ROOT']='~/miniconda3/envs/mne'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

#Paths
raw_path = project_dir + '/scratch/maxfiltered_data/tsss_st16_corr96'
ica_path = project_dir + '/scratch/working_memory/ICA'
avg_path = project_dir + '/scratch/working_memory/averages'
log_path = project_dir + '/misc/working_memory_logs'

subjects_dir = project_dir + '/scratch/fs_subjects_dir'
fwd_path = project_dir + '/scratch/forward_models'

# Subject info:
qy = Query(project)
subs = qy.get_subjects()

# Block ('localizer' or 'task')
block = 'task'
if len(argv) > 1:
    block = argv[1]

#Subject
scode = 11
if len(argv) > 2:
    scode = int(argv[2])
sub = subs[scode-1] 

ch_type = 'grad' # 'mag', 'all'
if len(argv) > 3:
    ch_type = argv[3]

# Block-specific variables
if block == 'localizer':    
    conds_orig = ['loc'] # MEG block code
    conds = ['localizer'] # New block code
    tmin = -.1 #epoch start
    tmax = 1.8 #epoch end
    events_fun = pfun.default_events_fun
    vmin, vmax = .18, .48
    vlines, hlines = [.9], [.9]
    nrows, ncols = 1, 1
    
elif block == 'task':
    conds_orig = ['main','inv'] # MEG block code
    conds = ['maintenance','manipulation'] # New block code
    lnames = ['recognize','invert']
    tmin = -.1 #epoch start
    tmax = 4 #epoch end
    events_fun = pfun.main_task_decoding_events_fun
    vmin, vmax = 0, 1
    vlines, hlines = [2], [2] 
    nrows, ncols = 2, 2

# Filter?
l_freq = .05
h_freq = None


# Filename
suffix = '_{}_lf_{}_hf_{}_{}'.format(block, l_freq, h_freq, ch_type)
        
print('output will be saved to the following filename:\n\n{}{}'.format(sub,suffix))

# Create subject specific directories if they don't exist
if not os.path.exists(avg_path + '/data/' + sub):
    os.mkdir(avg_path + '/data/' + sub)
    
# output paths
inv_path = avg_path + '/data/{}/{}_decoding_inv{}.p'.format(sub,sub,suffix)

################################ Epoch data #########################################

reject = dict(mag = 4e-12, grad = 4000e-13) # rejection criteria

# Initialize
epochs = {}
print('\n############### EPOCHING #################\n')
for cidx, c in enumerate(conds_orig):
    nc = conds[cidx] # new condition name
    
    # Files to retrieve
    fname = os.path.join(raw_path, sub, c + '_raw_tsss.fif')
    icaname = os.path.join(ica_path, sub, c + '_raw_tsss-ica.fif')
    
    if block == 'task':    
        lfname = op.join(log_path, sub[0:4] + '_' + lnames[cidx] + '_MEG.csv')
        events_fun_kwargs = {'lfname': lfname}
        
    elif block == 'localizer':
        events_fun_kwargs = {}
        
    #Epoching proper:
    epochs[nc] = pfun.WM_epoching(data_path = fname,
                                  ica_path = icaname,
                                  tmin = tmin, tmax = tmax,
                                  l_freq = l_freq, h_freq = None,
                                  resample = 100, bads = [],
                                  baseline = None, notch_filter = 50,
                                  events_fun = events_fun, 
                                  events_fun_kwargs = events_fun_kwargs,
                                  reject=reject)
    
################################# Inverse solution ##################################


# Load forward model 
fwd = mne.read_forward_solution(op.join(fwd_path, sub + '_vol-fwd.fif'))

# Select Channels
if ch_type in ['grad','mag']:
    epochs = {e: epochs[e].load_data().pick_types(ch_type) for e in epochs}

print(epochs)

# Calculate data covariance
data_cov = mne.compute_covariance([epochs[e] for e in epochs], 
                                    tmin= 0, tmax = tmax, rank ='info')

# Calculate noise covariance
noise_cov = mne.make_ad_hoc_cov(epochs[conds[0]].info)

# Calculate inverse solution
inv = mne.beamformer.make_lcmv(epochs[conds[0]].info,
                                fwd, data_cov, 
                                noise_cov=noise_cov,
                                reg=0.05,
                                pick_ori='max-power',
                                weight_norm= 'nai',
                                rank = 'info')

inv_file = open(inv_path,'wb')
pickle.dump(inv, inv_file)
inv_file.close()
