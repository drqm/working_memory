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
filterwarnings("ignore", category=DeprecationWarning)

project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
project_dir = '/projects/' + project
os.environ['MINDLABPROJ']= project
#os.environ['MNE_ROOT']='/users/david/miniconda3/envs/mne3d' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

raw_path = project_dir + '/scratch/maxfiltered_data/tsss_st16_corr96'
ica_path = project_dir + '/scratch/working_memory/ICA'
avg_path = project_dir + '/scratch/working_memory/averages'
log_path = project_dir + '/misc/working_memory_logs'

subjects_dir = project_dir + '/scratch/fs_subjects_dir'
fwd_path = project_dir + '/scratch/forward_models'

# Mapping of old triggers to new codes
mappings = {'maintenance': {'maint/mel1/same':   {'prev': [11,111,122], 'new': 11}, 
                            'maint/mel1/diff1':  {'prev': [11,213,222], 'new': 12},
                            'maint/mel1/diff2':  {'prev': [11,211,223], 'new': 13},
                            'maint/mel2/same':   {'prev': [13,113,122], 'new': 21},
                            'maint/mel2/diff1':  {'prev': [13,211,222], 'new': 22},
                            'maint/mel2/diff2':  {'prev': [13,213,221], 'new': 23}},
            
            'manipulation':{'manip/mel1/inv':    {'prev': [11,113,122], 'new': 31}, 
                            'manip/mel1/other1': {'prev': [11,211,222], 'new': 32},
                            'manip/mel1/other2': {'prev': [11,213,221], 'new': 33},
                            'manip/mel2/inv':    {'prev': [13,111,122], 'new': 41},
                            'manip/mel2/other1': {'prev': [13,213,222], 'new': 42},
                            'manip/mel2/other2': {'prev': [13,211,223], 'new': 43}}
           }

# event_ids = [[['same',1],['different1',2],['different2',3]],
#              [['inverted',1],['other1',2],['other2',3]]]

qy = Query(project)
subs = qy.get_subjects()
scode = 24
if len(argv) > 1:
    scode = int(argv[1])

sub = subs[scode-1]#'0002_BYG'#'0002_BYG'#'0008_HMD'#'0002_BYG'
conds_orig = ['main','inv'] #  ['mainv2','invv2']
conds = ['maintenance','manipulation']
lnames = ['recognize','invert']
save_averages = True#False#True
plot_topo = True#True
compute_sources = True # False
plot_sources = False
# sub, conds, save_averages,  = argv[0], argv[1], argv[2]
# plot_topo, compute_sources, plot_sources = argv[3], argv[4]
epochs = {}
evokeds = {}

print('\n epoching \n')
for cidx, c in enumerate(conds_orig):
    nc = conds[cidx]
    ## Load log file
    lfname = op.join(log_path,sub[0:4] + '_' + lnames[cidx] + '_MEG.csv')
    ldf = pd.read_csv(lfname, sep = ',', header=0)
    #print(ldf)
    ldf['acc'] = ldf['type'] == ldf['response']
    #print(ldf['acc'])
    #load and preprocess:
    fname = os.path.join(raw_path, sub, c + '_raw_tsss.fif')
    icaname = os.path.join(ica_path,sub, c + '_raw_tsss-ica.fif')
    raw = mne.io.read_raw_fif(fname, preload = True) # load data
    ica = mne.preprocessing.read_ica(icaname) # load ica solution
    raw = ica.apply(raw) # apply ICA
    raw.notch_filter(50)
    raw.resample(100)
    raw.filter(0.05,40,fir_design = 'firwin')

    # Get, correct and recode triggers:
    events = mne.find_events(raw, shortest_event = 1)
    events = events[np.append(np.diff(events[:,0]) >2,True)] # delete spurious t
    events2 = events.copy()[events[:,2] < 20] # Get first tone of first melody
    events3 = events.copy()[np.isin(events[:,2]//10, [11,21]),:] # get first tone of target melody
    events4 = events.copy()[np.isin(events[:,2]//10, [12,22]),:] # get second tone of target melody
    new_events = events2
        
    for eid in mappings[nc]:
        p1,p4,p5 = mappings[nc][eid]['prev']
        idx = [x and y and z for x,y,z in zip(events2[:,2]==p1,events3[:,2]==p4,events4[:,2]==p5)]
        new_events[idx,2] = mappings[nc][eid]['new']
    new_events = new_events[ldf['acc'], :] # select only successful trials

    # Epoching:
    picks = mne.pick_types(raw.info, meg = True)
    tmin, tmax = -0.5, 6.25 #epoch time
    baseline = None#-0.5,0) # baseline time
    reject = dict(mag = 4e-12, grad = 4000e-13)#eeg = 200e-6, #, eog = 250e-6)
    event_id = {ccc: mappings[nc][ccc]['new'] for ccc in mappings[nc]}
    print(event_id)
    epochs[c] = mne.Epochs(raw, events = new_events, event_id = event_id,
                    tmin = tmin, tmax = tmax, picks = picks,
                    baseline = baseline)

#compute difference between conditions:
epochs = mne.concatenate_epochs([epochs[e] for e in epochs])
data_cov = mne.compute_covariance(epochs.load_data().copy().pick_types('mag'),
                                       tmin= 0, tmax = 6.25,rank ='info')
evokeds = dict((cond,epochs[cond].average()) for
                        cond in sorted(epochs.event_id.keys()))
other_conds = ['maint','manip','maint/mel1','maint/mel2','manip/mel1','manip/mel2']
for oc in other_conds:
    evokeds[oc] = epochs[oc].average()
    evokeds[oc].comment = oc
    
#save output:
if save_averages:
    evkd_fname = op.join(avg_path,'data',sub + '_evoked.p')
    evkd_file = open(evkd_fname,'wb')
    pickle.dump(evokeds,evkd_file)
    evkd_file.close()
    print('evoked file saved')

print('done epoching')

# Plot some figures
fig1, axis = plt.subplots(nrows=1,ncols=1,figsize=(60,30))
mne.viz.plot_evoked_topo([evokeds['maint'].copy().pick_types('mag'),
                          evokeds['manip'].copy().pick_types('mag')],axes = axis)
plt.tight_layout()
plt.savefig(avg_path + '/figures/{}_ERFs_topo_mag.pdf'.format(sub))

fig2, axis = plt.subplots(nrows=1,ncols=1,figsize=(60,30))
mne.viz.plot_evoked_topo([evokeds['maint'].copy().pick_types('grad'),
                          evokeds['manip'].copy().pick_types('grad')],
                         axes = axis,merge_grads=True)
plt.tight_layout()
plt.savefig(avg_path + '/figures/{}_ERFs_topo_grad.pdf'.format(sub))

diff = evokeds['manip'].copy()
diff.data = evokeds['manip'].data - evokeds['maint'].data

fig3, axis = plt.subplots(nrows=1,ncols=1,figsize=(60,30))
mne.viz.plot_evoked_topo(diff.copy().pick_types('grad'),
                         axes = axis,merge_grads=True)
plt.tight_layout()

plt.savefig(avg_path + '/figures/{}_ERFs_diff_topo_grad.pdf'.format(sub))

fig4, axis = plt.subplots(nrows=1,ncols=1,figsize=(60,30))
mne.viz.plot_evoked_topo(diff.copy().pick_types('mag'),
                         axes = axis)
plt.tight_layout()
plt.savefig(avg_path + '/figures/{}_ERFs_diff_topo_mag.pdf'.format(sub))

fig5, axis = plt.subplots(nrows=1,ncols=1,figsize=(60,30))
mne.viz.plot_evoked_topo([evokeds['maint/mel1'].copy().pick_types('mag'),
                          evokeds['maint/mel2'].copy().pick_types('mag')],axes = axis)
plt.tight_layout()
plt.savefig(avg_path + '/figures/{}_ERFs_topo_mag_mels_maint.pdf'.format(sub))

fig6, axis = plt.subplots(nrows=1,ncols=1,figsize=(60,30))
mne.viz.plot_evoked_topo([evokeds['maint/mel1'].copy().pick_types('grad'),
                          evokeds['maint/mel2'].copy().pick_types('grad')],
                         axes = axis,merge_grads=True)
plt.tight_layout()
plt.savefig(avg_path + '/figures/{}_ERFs_topo_grad_mels_maint.pdf'.format(sub))

fig5, axis = plt.subplots(nrows=1,ncols=1,figsize=(60,30))
mne.viz.plot_evoked_topo([evokeds['manip/mel1'].copy().pick_types('mag'),
                          evokeds['manip/mel2'].copy().pick_types('mag')],axes = axis)
plt.tight_layout()
plt.savefig(avg_path + '/figures/{}_ERFs_topo_mag_mels_manip.pdf'.format(sub))

fig6, axis = plt.subplots(nrows=1,ncols=1,figsize=(60,30))
mne.viz.plot_evoked_topo([evokeds['manip/mel1'].copy().pick_types('grad'),
                          evokeds['manip/mel2'].copy().pick_types('grad')],
                         axes = axis,merge_grads=True)
plt.tight_layout()
plt.savefig(avg_path + '/figures/{}_ERFs_topo_grad_mels_maint.pdf'.format(sub))

diff_mels_maint = evokeds['maint/mel1'].copy()
diff_mels_maint.data = evokeds['maint/mel2'].data - evokeds['maint/mel1'].data

diff_mels_manip = evokeds['manip/mel1'].copy()
diff_mels_manip.data = evokeds['manip/mel2'].data - evokeds['manip/mel1'].data

fig7, axis = plt.subplots(nrows=1,ncols=1,figsize=(60,30))
mne.viz.plot_evoked_topo(diff_mels_maint.copy().pick_types('grad'),
                         axes = axis,merge_grads=True)
plt.tight_layout()
plt.savefig(avg_path + '/figures/{}_ERFs_diff_topo_grad_mels_maint.pdf'.format(sub))

fig8, axis = plt.subplots(nrows=1,ncols=1,figsize=(60,30))
mne.viz.plot_evoked_topo(diff_mels_maint.copy().pick_types('mag'),
                         axes = axis)
plt.tight_layout()
plt.savefig(avg_path + '/figures/{}_ERFs_diff_topo_mag_mels_maint.pdf'.format(sub))

fig9, axis = plt.subplots(nrows=1,ncols=1,figsize=(60,30))
mne.viz.plot_evoked_topo(diff_mels_manip.copy().pick_types('grad'),
                         axes = axis,merge_grads=True)
plt.tight_layout()
plt.savefig(avg_path + '/figures/{}_ERFs_diff_topo_grad_mels_manip.pdf'.format(sub))

fig10, axis = plt.subplots(nrows=1,ncols=1,figsize=(60,30))
mne.viz.plot_evoked_topo(diff_mels_manip.copy().pick_types('mag'),
                         axes = axis)
plt.tight_layout()
plt.savefig(avg_path + '/figures/{}_ERFs_diff_topo_mag_mels_manip.pdf'.format(sub))

plt.close('all')
# if plot_topo:
#   for e in evokeds:
#         times = np.arange(0.11,6.11,0.1)
#         #evokeds[e].plot_joint(topomap_args ={'average': 0.5},picks='mag',times = times)
#         evokeds[e].plot_topo()


## Source analysis
if compute_sources:
    print('\n computing sources \n')
    fwd_fn = op.join(fwd_path, sub + '_vol-fwd.fif')
    fwd = mne.read_forward_solution(fwd_fn)
    #compute noise covariance
    noise_cov = mne.compute_covariance(epochs,tmin = -1,
                                       tmax=0, rank='info')
#     data_cov = mne.compute_covariance(epochs.load_data().copy().pick_types('mag'),
#                                        tmin= 0, tmax = 6.25,rank ='info')
    ## mne solution
    inv = mne.beamformer.make_lcmv(epochs['manip'].info,fwd,data_cov, reg=0.05,
                                    pick_ori='max-power', #noise_cov=noise_cov,#,depth = 0.95,
                                    weight_norm= 'nai', rank = 'info')
    # inv = mne.minimum_norm.make_inverse_operator(epochs[conds[0]].info,fwd,
    #                                               noise_cov,loose = 1)
    SNR = 3
    sources = {}
    for e in evokeds:
        # sources[e][c] = mne.minimum_norm.apply_inverse(evokeds[e][c],inv,
        #                                                 lambda2=1/SNR**2)
        sources[e] = mne.beamformer.apply_lcmv(evokeds[e],inv)#,max_ori_out='signed')

        # if plot_sources:
        #     brain = sources[e][c].plot(subjects_dir=subjects_dir,initial_time=-1,hemi = 'split',
        #                                time_viewer=True, views=['lateral','medial'],
        #                                title = 'ERF- {} {} {}'.format(sub,e,c))
        #     # brain.save_movie(avg_path + '/figures/{}_ERF_sources_{}_{}.mov'.format(sub,e,c),
        #     #                 framerate=4,time_dilation = 10)
        #     brain.close()

    src_fname = op.join(avg_path,'data',sub + '_evoked_sources.p')
    src_file = open(src_fname,'wb')
    pickle.dump(sources,src_file)
    src_file.close()

    inv_fname = op.join(avg_path,'data',sub + '_evoked_inverse.p')
    inv_file = open(inv_fname,'wb')
    pickle.dump(inv,inv_file)
    inv_file.close()
    print('\n sources file saved')
