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

# Mode (source or sensor?)
mode = 'sensor'
if len(argv) > 3:
    mode = argv[3]

# Define ROIs for masking:
label_names = {'rA1':  'ctx_rh_G_temp_sup-G_T_transv',
           'rThal':'Right-Thalamus-Proper',
           'rPCC': 'ctx_rh_G_and_S_cingul-Mid-Post',
           'rHC':  'Right-Hippocampus',    
           'rPCN': 'ctx_rh_G_precuneus',
           'lA1':  'ctx_lh_G_temp_sup-G_T_transv',
           'lThal':'Left-Thalamus-Proper',
           'lPCC': 'ctx_lh_G_and_S_cingul-Mid-Post',
           'lHC':  'Left-Hippocampus',
           'lPCN': 'ctx_lh_G_precuneus'
          }

# Mask (include / exclude)
mask = {}
mask['include'] = None
mask['exclude'] = None

if len(argv) > 4:
    mask[argv[4]] = argv[5:]
    
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
suffix = '_{}_{}_lf_{}_hf_{}_tstep_{}_twin_{}'.format(block, mode, l_freq, h_freq, tstep, twin)

if mask['include']:   
    suffix += '_include'
    for iii in mask['include']:
        suffix += '_' + iii

if mask['exclude']:   
    suffix += '_exclude'
    for eee in mask['exclude']:
        suffix += '_' + eee
        
print('output will be saved to the following filename:\n\n{}{}'.format(sub,suffix))

# Create subject specific directories if they don't exist
if not os.path.exists(avg_path + '/data/' + sub):
    os.mkdir(avg_path + '/data/' + sub)
if not os.path.exists(avg_path + '/figures/' + sub):
    os.mkdir(avg_path + '/figures/' + sub)
    
# output paths
gen_path = avg_path + '/data/{}/{}_models{}.p'.format(sub,sub,suffix)
patterns_path = avg_path + '/data/{}/{}_patterns{}.p'.format(sub,sub,suffix)
filters_path = avg_path + '/data/{}/{}_filters{}.p'.format(sub,sub,suffix)
scores_path = avg_path + '/data/{}/{}_scores{}.p'.format(sub,sub,suffix)
times_path = avg_path + '/data/{}/{}_times{}.p'.format(sub,sub,suffix)
fig_path = avg_path + '/figures/{}/{}_scores{}.pdf'.format(sub,sub,suffix)

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
if mode == 'source':
    
    # Load forward model 
    fwd = mne.read_forward_solution(op.join(fwd_path, sub + '_vol-fwd.fif'))
   
    # Select Magnetometers
    #epochs = {e: epochs[e].load_data().pick_types('grad') for e in epochs}
    
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
    
    n_features = 500
    lmask = []
    
    # Add mask if required
    if mask['include'] or mask['exclude']:
        label_file = subjects_dir + '/{}/mri/aparc.a2009s+aseg.mgz'.format(sub)
        labels = mne.get_volume_labels_from_aseg(label_file)
        llist = []
        
        for mm in mask:
            if mask[mm]:
                print('{} following labels: \n'.format(mm))
                for lmi in mask[mm]:
                    print(lmi)

                for ll in labels:
                    if ll in [label_names[cll] for cll in mask[mm]]:
                        llist += [int(mm == 'include')]
                    else:
                        llist += [int(mm == 'exclude')]
        
        lmask = mne.labels_to_stc([label_file, labels], np.array(llist), subject=sub, src=fwd['src'], verbose=None).data
        print(lmask.shape)
        print(np.sum(lmask))
        
elif mode == 'sensor':
    inv = None
    n_features = 'all'
    lmask = []
    
################################# Perform decoding ##################################
gen, patterns, filters, scores, times, _ = dfun.WM_time_gen_classify(epochs, mode = mode,
                                                                  inv = inv, lmask = lmask,
                                                                  score = True, 
                                                                  n_features = n_features,
                                                                  twindows = [tmin,tmax],
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
dfun.plot_time_gen_accuracy(scores, times, nrows=nrows, ncols=ncols, vlines=vlines,
                            hlines=hlines, savefig=fig_path, vmin=vmin, vmax=vmax)

#################### Perform decoding of block-flipped labels #######################
# output paths
# inv_flipped = epochs['manipulation'].copy()
# cmodulo = np.sum(np.unique(np.array(inv_flipped.events[:,2])))
# inv_flipped.events[:,2] = [-ee + cmodulo for ee in inv_flipped.events[:,2]] 
# print(inv_flipped.events[:,2])
# epochs2 = {}
# epochs2['flipped'] = mne.concatenate_epochs([epochs['maintenance'],inv_flipped])

# gen_path2 = avg_path + '/data/{}/{}_models{}_flipped.p'.format(sub,sub,suffix)
# patterns_path2 = avg_path + '/data/{}/{}_patterns{}_flipped.p'.format(sub,sub,suffix)
# filters_path2 = avg_path + '/data/{}/{}_filters{}_flipped.p'.format(sub,sub,suffix)
# scores_path2 = avg_path + '/data/{}/{}_scores{}_flipped.p'.format(sub,sub,suffix)
# times_path2 = avg_path + '/data/{}/{}_times{}_flipped.p'.format(sub,sub,suffix)
# fig_path2 = avg_path + '/figures/{}/{}_scores{}_flipped.pdf'.format(sub,sub,suffix)
# gen2, patterns2, filters2, scores2, times2, _ = dfun.WM_time_gen_classify(epochs2,
#                                                                         mode = mode,
#                                                                   inv = inv, lmask = lmask,
#                                                                   score = True, 
#                                                                   n_features = n_features,
#                                                                   twindows = [tmin,tmax],
#                                                                   l_freq = None,
#                                                                   h_freq = h_freq,
#                                                                   smooth = smooth_kwargs,
#                                                                   save_filters = filters_path2,                     
#                                                                   save_scores = scores_path2,
#                                                                   save_patterns = patterns_path2,
#                                                                   save_gen = gen_path2,
#                                                                   save_times = times_path2)

# ############################## make figure and save #################################
# print('Plotting')
# dfun.plot_time_gen_accuracy(scores2, times2, nrows=1, ncols=1, vlines=vlines,
#                             hlines=hlines, savefig=fig_path2, vmin=vmin, vmax=vmax)