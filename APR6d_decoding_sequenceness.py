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
from src.preprocessing import WM_epoching, main_task_events_fun, default_events_fun
import src.decoding_functions as dfun
from sklearn.linear_model import LinearRegression
import importlib
importlib.reload(dfun)

filterwarnings("ignore", category=DeprecationWarning)

## Define relevant variables
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

#Other variables
sub = subs[scode-1] #'0002_BYG'#'0002_BYG'#'0008_HMD'#'0002_BYG'
fig_dir = avg_path + '/figures/' + sub + '/'
data_dir = avg_path + '/data/' + sub + '/'

conds_orig = ['loc','main','inv'] #  ['mainv2','invv2']
conds = ['localizer','maintenance','manipulation']
lnames = [None,'recognize','invert'] # Log file names to exclude wrong trials
tmin = [0,1.75,1.75] # epoch tmin
tmax = [1.8,4,4] # epoch tmax
event_funs = [default_events_fun,  main_task_events_fun, main_task_events_fun]

#Epoch data
reject = dict(mag = 4e-12, grad = 4000e-13) # rejection criteria

# Initialize
epochs = {}
print('\n epoching \n')
for cidx, c in enumerate(conds_orig):
    nc = conds[cidx] # new condition name
     
    # Files to retrieve
    fname = os.path.join(raw_path, sub, c + '_raw_tsss.fif')
    icaname = os.path.join(ica_path,sub, c + '_raw_tsss-ica.fif')
    if lnames[cidx]:
        lfname = op.join(log_path, sub[0:4] + '_' + lnames[cidx] + '_MEG.csv')
        events_fun_kwargs = {'cond': nc, 'lfname': lfname}
    else:
        events_fun_kwargs = {}
        
    print(tmin[cidx])
    print(tmax[cidx])
    #Epoching proper:
    epochs[nc] = WM_epoching(data_path = fname, ica_path = icaname, tmin = tmin[cidx], tmax = tmax[cidx],
                                l_freq = None, h_freq = None, resample = 100, bads = [],
                                baseline = None, notch_filter = 50,
                                events_fun = event_funs[cidx], 
                                events_fun_kwargs = events_fun_kwargs,
                                reject=reject)

################## Perform localizer decoding ################
smooth_kwargs = {'tstep': .01, 'twin': .05}
gen, lpatterns, lfilters, scores, times = dfun.WM_time_gen_classify({'localizer': epochs['localizer']}, mode='sensor',
                                                                  kind='Sliding', inv = None, score = True, n_features = 'all',
                                                                  twindows = [0,1.8], l_freq=None, h_freq = None,
                                                                 smooth = smooth_kwargs, save_filters=None, save_scores = None,
                                                                 save_gen=None, save_patterns=None, penalty='l1')

dfun.plot_diagonal_accuracy(scores, times, nrows=1, ncols=1,vlines=[.9],hlines=[.9], chance=1/3,savefig=None, ylims = [.25,.75])
plt.savefig(fig_dir + sub + '_localizer_diagonal_accuracy.pdf')

for lp in lpatterns:
    cpf = lpatterns[lp].plot_joint()
    cpf[0].savefig(fig_dir + sub + '_' + lp + '_patterns_mag.pdf')
    cpf[1].savefig(fig_dir + sub + '_' + lp + '_patterns_grad.pdf')

tix = [np.argmax(scores['localizer_from_localizer']*(times['localizer']<.8)), 
       np.argmax(scores['localizer_from_localizer']*(times['localizer']>=.8))]
# tix = [np.argmin(np.abs(times['localizer'] - x)) for x in [0.28,1.8]]
print(np.array(tix), times['localizer'][tix])
probs, times2, events = dfun.get_probs(gen['localizer'], epochs, tix, blocks = ['maintenance','manipulation'])

delays = np.arange(1,101)
betas = {b: dfun.sequence_betas(probs[b], delays) for b in probs}

probs_file = open(data_dir + sub + '_predicted_proba.p','wb')
pickle.dump({'probs': probs, 'times': times2, 'events': events, 'delays': delays, 'betas': betas}, probs_file)
probs_file.close()

fig, ax = plt.subplots(4,3, figsize=(10,12))
for ctx, ct in enumerate(tix):
    for bix, b in enumerate(probs):
        print('block: '+ b)
        event_sel = {m: np.where([a in np.unique(epochs[b][m].events[:,2]) for a in events[b][:,2]]) for m in ['mel1','mel2']}
        event_sel = {e: event_sel[e][0] for e in event_sel}
        for t in range(probs[b].shape[3]):
            cbix = 2*ctx + bix  
            ax[cbix,t].plot(times2[b],np.median(probs[b][ctx,event_sel['mel1'],:,t],0))
            ax[cbix,t].plot(times2[b],np.median(probs[b][ctx,event_sel['mel2'],:,t],0))
            ax[cbix,t].set_ylim([0,1])
            ax[cbix,t].set_title('time: ' + str(np.round(times['localizer'][ct],2)) + ' s - ' + b + ' ' + 'tone ' + str(t+1))
            ax[cbix,t].legend(['mel1','mel2'])
plt.tight_layout()
plt.savefig(fig_dir + sub + '_probs_timecourses.pdf')

patterns = np.array([[[1,1,1],[1,1,1],[1,1,1]],
                     [[1,0,0],[0,1,0],[0,0,1]],
                    [[0,0,0],[1,0,0],[0,1,0]],
                    [[0,1,0],[0,0,1],[0,0,0]]])
patterns = patterns.reshape(patterns.shape[0],-1).T   

perms = {}
for b in betas:
    print('block: '+ b)
    event_sel = {m: np.where([a in np.unique(epochs[b][m].events[:,2]) for a in events[b][:,2]]) for m in ['mel1','mel2']}
    event_sel = {e: event_sel[e][0] for e in event_sel}
    perms[b] = dfun.seq_permutation_within(betas[b], patterns, nperm = 200, event_sel = event_sel)
    
perms_file = open(data_dir + sub + '_sequenceness.p','wb')
pickle.dump({'perms': perms, 'patterns': patterns}, perms_file)
perms_file.close()

######### Plot coefficients ##########
patterns = {'forward': 2,'backward': 3}
colors = {'forward': 'b','backward': 'r'}
blocks = [b for b in perms]
mels = [m for m in perms[blocks[0]]['stat']]
fig, ax = plt.subplots(len(blocks),len(mels), figsize=[12,12])
for bix,b in enumerate(blocks):
    for mix,m in enumerate(mels):
        for p in patterns:
            pix = patterns[p]
            sig_lines = np.percentile(perms[b]['null'][m][p],[2.5,97.5], axis = 0)[:,1,:,pix]            
            ax[bix,mix].plot(delays, perms[b]['stat'][m][1,:,pix],color=colors[p])
            ax[bix,mix].plot(delays, sig_lines.T, color=colors[p], linestyle='--',alpha=.4)
            
        ax[bix,mix].axhline(0,color='k',alpha = .6)
        ax[bix,mix].set_xlabel('delays (s)')
        ax[bix,mix].set_xlim(delays[[0,-1]])
        ax[bix,mix].set_ylim([-.25,.25])
        ax[bix,mix].set_ylabel('sequenceness (a.u.)')
        ax[bix,mix].legend(['forward','forward null 2.5%','forward null 97.5%', 'backward','backward null 2.5%',
                            'backward null 97.5%'],ncol=2)
        ax[bix,mix].set_title(b + ' - ' + m)
plt.tight_layout()
plt.savefig(fig_dir + sub + '_sequenceness.pdf')