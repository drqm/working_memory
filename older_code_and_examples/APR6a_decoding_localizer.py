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
import src.preprocessing as pfun #import WM_epoching, main_task_events_fun
import src.decoding_functions as dfun
import importlib

filterwarnings("ignore", category=DeprecationWarning)

##################### Define relevant variables ################################
# Project info
project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
project_dir = '/projects/' + project
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
scode = 24
if len(argv) > 1:
    scode = int(argv[1])
sub = subs[scode-1] 

# Mode (source or sensor?)
mode = 'sensor'
if len(argv) > 2:
    mode = argv[2]
    
#Other variables
conds_orig = ['loc'] # MEG block code
conds = ['localizer'] # New block code

# control smoothing:
tstep = .025
twin = .05
if tstep:
    smooth_kwargs = {'tstep': tstep, 'twin': twin}
else:
    smooth_kwargs = None

# Filter?
l_freq = .05
h_freq = None

# Filename
suffix = '_localizer_{}_lf_{}_hf_{}_tstep_{}_twin_{}'.format(mode, l_freq, h_freq, tstep, twin)
print('output will be saved to the following filename:\n\n{}{}'.format(sub,suffix))

# output paths
gen_path = avg_path + '/data/{}_models{}.p'.format(sub,suffix)
patterns_path = avg_path + '/data/{}_patterns{}.p'.format(sub,suffix)
filters_path = avg_path + '/data/{}_filters{}.p'.format(sub,suffix)
scores_path = avg_path + '/data/{}_scores{}.p'.format(sub,suffix)
times_path = avg_path + '/data/{}_times{}.p'.format(sub,suffix)
fig_path = avg_path + '/figures/{}_scores{}.pdf'.format(sub,suffix)

################################ Epoch data #########################################
tmin = -.1 #epoch start
tmax = 1.8 #epoch end
reject = dict(mag = 4e-12, grad = 4000e-13) # rejection criteria

# Initialize
epochs = {}
print('\n############### EPOCHING #################\n')
for cidx, c in enumerate(conds_orig):
    nc = conds[cidx] # new condition name
    
    # Files to retrieve
    fname = os.path.join(raw_path, sub, c + '_raw_tsss.fif')
    icaname = os.path.join(ica_path, sub, c + '_raw_tsss-ica.fif')
    
    #Epoching proper:
    epochs[nc] = pfun.WM_epoching(data_path = fname,
                                  ica_path = icaname,
                                  tmin = tmin, tmax = tmax,
                                  l_freq = l_freq, h_freq = None,
                                  resample = 100, bads = [],
                                  baseline = None, notch_filter = 50,
                                  events_fun = pfun.default_events_fun, 
                                  reject=reject)
    
################################# Inverse solution ##################################
if mode == 'source':
    
    # Load forward model 
    fwd = mne.read_forward_solution(op.join(fwd_path, sub + '_vol-fwd.fif'))
   
    # Select Magnetometers
    epochs = {e: epochs[e].load_data().pick_types('mag') for e in epochs} 
    print(epochs)
    
    # Calculate data covariance
    data_cov = mne.compute_covariance([epochs[e] for e in epochs], 
                                      tmin= 0, tmax = 1.8,rank ='info')

    # Calculate inverse solution
    inv = mne.beamformer.make_lcmv(epochs[conds[0]].info,
                                   fwd, data_cov, reg=0.05,
                                   pick_ori='max-power',
                                   weight_norm= 'nai',
                                   rank = 'info')
    
    n_features = 500
    
elif mode == 'sensor':
    inv = None
    n_features = 'all'

################################# Perform decoding ##################################
gen, patterns, filters, scores, times = dfun.WM_time_gen_classify(epochs, mode = mode,
                                                                  inv = inv, score = True, 
                                                                  n_features = n_features,
                                                                  twindows = [-.1,1.8],
                                                                  l_freq = None,
                                                                  h_freq = h_freq,
                                                                  smooth = smooth_kwargs,
                                                                  save_filters = filters_path,                     
                                                                  save_scores = scores_path,
                                                                  save_patterns = patterns_path,
                                                                  save_gen = gen_path,
                                                                  save_times = times_path)

############################## make figure and save #################################
print('Plotting')
dfun.plot_time_gen_accuracy(scores, times, nrows=1, ncols=1, vlines=[.9],
                            hlines=[.9], savefig=fig_path, vmin=.18, vmax=.48)