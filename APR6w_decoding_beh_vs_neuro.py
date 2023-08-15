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
importlib.reload(pfun)
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

#Subject
scode = 12
if len(argv) > 1:
    scode = int(argv[1])
sub = subs[scode-1] 

# Mode (source or sensor?)
mode = 'sensor'
if len(argv) > 2:
    mode = argv[2]

block = 'task'
conds_orig = ['main','inv'] # MEG block code
conds = ['maintenance','manipulation'] # New block code
lnames = ['recognize','invert']
tmin = -.1 #epoch start
tmax = 4 #epoch end
events_fun = pfun.main_task_decoding_events_fun
#events_fun = pfun.main_task_events_fun

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
table_path = avg_path + '/data/accuracies/{}_accuracy_table.csv'.format(sub,sub)
################################ Epoch data #########################################

reject = dict(mag = 4e-12, grad = 4000e-13) # rejection criteria

# Initialize
epochs = {}
ldfs = {}
print('\n############### EPOCHING #################\n')
for cidx, c in enumerate(conds_orig):
    nc = conds[cidx] # new condition name
    
    # Files to retrieve
    fname = os.path.join(raw_path, sub, c + '_raw_tsss.fif')
    icaname = os.path.join(ica_path, sub, c + '_raw_tsss-ica.fif')
     
    lfname = op.join(log_path, sub[0:4] + '_' + lnames[cidx] + '_MEG.csv')
    ldfs[c] = pd.read_csv(lfname, sep = ',', header=0)
    ldfs[c]['acc'] = ldfs[c]['type'] == ldfs[c]['response']
    events_fun_kwargs = {}#{'lfname': lfname, 'cond': nc}
        
    #Epoching proper:
    epochs[nc] = pfun.WM_epoching(data_path = fname,
                                  ica_path = icaname,
                                  tmin = tmin, tmax = tmax,
                                  l_freq = l_freq, h_freq = None,
                                  resample = 100, bads = [],
                                  baseline = None, notch_filter = 50,
                                  events_fun = events_fun, 
                                  events_fun_kwargs = events_fun_kwargs,
                                  reject=reject,
                                  )

drop_incorrect = {ccc: np.where(ldfs[coc]['acc']==0)[0] for ccc, coc in zip(conds, conds_orig)}
gfile = open(gen_path, 'rb')
gen = pickle.load(gfile)
gfile.close()

inv = None
n_features = 'all'
lmask = []

################################# Perform decoding ##################################
_, _, _, scores, times, _ = dfun.WM_time_gen_classify(epochs, mode = mode,
                                                                inv = inv, lmask = lmask,
                                                                score = True, 
                                                                n_features = n_features,
                                                                twindows = [tmin,tmax],
                                                                l_freq = l_freq,
                                                                h_freq = h_freq,
                                                                smooth = smooth_kwargs,
                                                                scoring_output = 'trial_acc',
                                                                kind = 'Generalizing',
                                                                drop_incorrect=drop_incorrect)
                                                    
pdict = {'sub': [], 'block': [],
         'trial': [], 'test_type': [], 
         'trial_type': [], 'period': [],
         'neur_acc': [], 'time': [],
         'response': [],'beh_acc': [], 'rt': []}

conds_orig = ['main','inv'] # MEG block code
conds = ['maintenance','manipulation'] # New block code
lnames = ['recognize','invert']
new_names = ['recall', 'manipulation']
periods = {'listening': [0,2], 'imagination': [2,4]}
for c1ix,c1 in enumerate(conds):
   for c2ix,c2 in enumerate(conds):
        if c2 == c1:
            test_type = 'within'
        else:
            test_type = 'between'
        cc = c1 + '_from_' + c2
        for tr in range(scores[cc].shape[0]):
                crt = ldfs[conds_orig[c1ix]]['rt'].iloc[tr]
                cacc = ldfs[conds_orig[c1ix]]['acc'].iloc[tr]
                ctype = ldfs[conds_orig[c1ix]]['type'].iloc[tr]
                cresp = ldfs[conds_orig[c1ix]]['response'].iloc[tr]
                crit = int(ldfs[conds_orig[c1ix]]['target'].iloc[tr][2])
                
                if (crit in [1,3]): #or (ctype == 1):
                     ctype += 1
                for cprd in periods:
                    pix = (times[c1] >= periods[cprd][0]) & (times[c1] < periods[cprd][1]) 
                    for tt in np.where(pix)[0]:
                        pdict['sub'] += [sub]
                        pdict['block'] += [new_names[c1ix]]
                        pdict['trial'] += [tr+1]
                        pdict['test_type'] += [test_type]
                        pdict['trial_type'] += [ctype]
                        pdict['period'] += [cprd] 
                        pdict['time'] += [np.round(times[c1][tt],3)]
                        pdict['neur_acc'] += [scores[cc][tr].diagonal()[tt]]
                        #pdict['neur_acc'] += [np.mean(scores[cc][tr,pix])]
                        pdict['response'] += [cresp]
                        pdict['beh_acc'] += [int(cacc)]
                        pdict['rt'] += [np.round(crt)]

pndf = pd.DataFrame(pdict)
pndf.to_csv(table_path,index=False)

print('report of mean accuracy:\n')
for ctt in np.unique(pndf['period']):
    for bb in np.unique(pndf['block']):
        for ct in np.unique(pndf['test_type']):
            cix = np.where((pndf['test_type'] == ct) & (pndf['block'] == bb) & (pndf['period'] == ctt))
            print(ctt,bb,ct, np.round(np.mean(pndf['neur_acc'].iloc[cix]),3))

