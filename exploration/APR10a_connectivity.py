import mne
import mne_connectivity
import os
import os.path as op
import scipy.signal as signal
import numpy as np
import pickle
from warnings import filterwarnings
from sys import argv
import matplotlib.pyplot as plt
from stormdb.access import Query
import pandas as pd
from src.decoding_functions import smooth_data
import src.preprocessing as pfun
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

# Subjects info:
qy = Query(project)
subs = qy.get_subjects()

#Subject
scode = 11
if len(argv) > 1:
    scode = int(argv[1])
sub = subs[scode-1]

suffix = ''
if len(argv) > 2:
    suffix = argv[2]
    
print('output will be saved to the following filename:\n\n{}{}'.format(sub,suffix))

# Create subject specific directories if they don't exist
if not os.path.exists(avg_path + '/data/' + sub):
    os.mkdir(avg_path + '/data/' + sub)
if not os.path.exists(avg_path + '/figures/' + sub):
    os.mkdir(avg_path + '/figures/' + sub)

# Define output paths
conn_path = avg_path + '/data/{}/{}_conn{}.p'.format(sub,sub,suffix)
fig_path = avg_path + '/figures/{}/{}_conn{}.pdf'.format(sub,sub,suffix)

# Define block names (original MEG names, new condition names and logfile names)
conds_orig = ['main','inv'] # MEG block code
conds = ['maintenance','manipulation'] # New block code
lnames = ['recognize','invert']

################################ Epoch data #########################################

# Epoching parameters
reject = dict(mag = 4e-12, grad = 4000e-13) # rejection thresholds
events_fun = pfun.main_task_events_fun # Event function (see src/preprocessing.py)
tmin = -2 #epoch start
tmax = 8 #epoch end
l_freq = .05 #HP filter
h_freq = None #LP filter
baseline = -.2
# Initialize
epochs = {}
print('\n############### EPOCHING #################\n')
for cidx, c in enumerate(conds_orig):
    nc = conds[cidx] # new condition name
    
    # Files to retrieve
    fname = os.path.join(raw_path, sub, c + '_raw_tsss.fif')
    icaname = os.path.join(ica_path, sub, c + '_raw_tsss-ica.fif')
    lfname = op.join(log_path, sub[0:4] + '_' + lnames[cidx] + '_MEG.csv')
    events_fun_kwargs = {'cond': nc,'lfname': lfname} # input to the events function (new condition name and logfile)
               
    #Epoching proper:
    epochs[nc] = pfun.WM_epoching(data_path = fname, #raw data path
                                  ica_path = icaname, #ICA components path
                                  tmin = tmin, tmax = tmax, #Epoch times
                                  l_freq = l_freq, h_freq = None, #Filterning options
                                  resample = 100, bads = [], #Resample and bad channels to reject
                                  baseline = None, notch_filter = 50, # Demean baseline
                                  events_fun = events_fun, #Event function to use for epoching
                                  events_fun_kwargs = events_fun_kwargs, #Arguments for event function
                                  reject=reject) # thresholds to reject artifacts

epochs = mne.concatenate_epochs([epochs[e] for e in epochs])

## Source localization
data_cov = mne.compute_covariance(epochs.load_data().copy().pick_types('mag'),
                                       tmin= 0, tmax = 6.25,rank ='info')
print('\n computing sources \n')
fwd_fn = op.join(fwd_path, sub + '_vol-fwd.fif')
fwd = mne.read_forward_solution(fwd_fn)
inv = mne.beamformer.make_lcmv(epochs['manip'].info,fwd,data_cov, reg=0.05,
                                pick_ori='max-power', #noise_cov=noise_cov,#,depth = 0.95,
                                weight_norm= 'nai', rank = 'info')
src_epochs = mne.beamformer.apply_lcmv_epochs(epochs,inv)

# Obtain parcellation

label_file = subjects_dir + '/{}/mri/aparc.a2009s+aseg.mgz'.format(sub)
labels = mne.get_volume_labels_from_aseg(label_file)
src = mne.read_source_spaces(subjects_dir + '/{}/bem/{}_vol-src.fif'.format(sub,sub))

# Extract time courses
clabels = ['ctx_rh_G_temp_sup-G_T_transv',
           'Right-Thalamus-Proper',
           'ctx_rh_G_and_S_cingul-Mid-Post',
           'Right-Hippocampus',    
           'ctx_rh_G_precuneus',
           'ctx_lh_G_temp_sup-G_T_transv',
           'Left-Thalamus-Proper',
           'ctx_lh_G_and_S_cingul-Mid-Post',
           'Left-Hippocampus',
           'ctx_lh_G_precuneus'
           ]

stc_labels = []
for cidx, c in enumerate(src_epochs):
    print('extracting sources for epoch {}'.format(cidx+1))
    stc_labels += [src_epochs[cidx].extract_label_time_course(labels = [label_file,clabels], src = src, mode = 'auto')]

roi_data = np.array(stc_labels)
roi_data.shape

# Estimate and save connectivity
sfreq = epochs.info['sfreq']
fmin = {'delta': .5, 'theta': 4}
fmax = {'delta': 2, 'theta': 8}
cwt_bands = {'delta': np.array([.5,.75,1,1.25,1.5,1.75,2]), 'theta': np.array([4,5,6,7,8])}
cwt_n_cycles = {'delta': np.array([1,2,2,2,2,2,2]), 'theta': np.array([3,3,3,3,3])}
periods = {'whole': [0,10]}#,'listen': [0,1.75], 'imagine': [2,4]}
conn = {}
for b in cwt_bands:
    conn[b] = {}
    for p in periods:
        conn[b][p] = mne_connectivity.phase_slope_index(
            roi_data, names=clabels, mode='cwt_morlet', cwt_freqs = cwt_bands[b], #method='pli',
            cwt_n_cycles = cwt_n_cycles[b], sfreq=sfreq, fmin=fmin[b], fmax=fmax[b], #faverage=True,
            n_jobs=1,tmin = periods[p][0], tmax = periods[p][1])#c mt_adaptive=True

cfile = open(conn_path,'wb')
pickle.dump(conn,cfile)
cfile.close()

## Make and save a plot
fig, ax = plt.subplots(2,10,figsize = (30,8),sharex=True, sharey=True)
for bix,b in enumerate(conn):
    print(b)
    bdata =  np.squeeze(conn[b]['whole'].get_data()).reshape(10,10,-1)
    for rix, r in enumerate(conn[b]['whole'].names):
        cdata = np.squeeze(bdata[rix,:,:].copy())
        cdata[(rix+1):,:] = bdata[(rix+1):,rix,:]*-1
        axx, axy = rix % 10, rix // 10 + 1*bix
        ix = np.arange(10) + rix*10        
        im = ax[axy, axx].imshow(cdata, aspect='auto',vmin=-.1,vmax=.1,
                  interpolation='nearest',cmap='RdBu_r',extent=[epochs.times[0],epochs.times[-1],len(clabels),0])#origin='lower'
#         if axy != 3:
#             ax[axy, axx].set_xticks([])
        if axy == 0:
            ax[axy, axx].set_title(r)
        
        if axx == 0:
            ax[axy, axx].set_yticks(np.arange(len(clabels)) + .5)
            ax[axy, axx].set_yticklabels(clabels)
fig.colorbar(im)
plt.tight_layout()
plt.savefig(fig_path)