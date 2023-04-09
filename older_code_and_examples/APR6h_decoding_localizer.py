import mne
import os
import os.path as op
import numpy as np
import pickle
from sys import argv
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import GeneralizingEstimator, get_coef, LinearModel,cross_val_multiscore
# from mne.beamformer import make_lcmv, apply_lcmv
# from nilearn.plotting import plot_stat_map
# from nilearn.image import index_img

os.environ['MINDLABPROJ']='MINDLAB2020_MEG-AuditoryPatternRecognition'
#os.environ['MNE_ROOT']='/users/david/miniconda3/envs/mne3d' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

os.environ['ETS_TOOLKIT'] = 'qt4'
# os.environ['QT_API'] = 'pyqt5'

wdir = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/working_memory'
raw_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/maxfiltered_data/tsss_st16_corr96'
ica_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/working_memory/ICA'
avg_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/working_memory/averages'
fwd_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/forward_models'
subjects_dir = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/misc/fs_subjects_dir'
sub = '0011_U7X'
#sub = argv[1]#'0013_NHJ'#'0002_TQ8' #0001_AF5 0003_BHS

blocks = ['main','inv']
types = ['listened','imagined']

srate = 100
get_scores = False#True
plot_scores = False#True
save_movie = False
fit_models = False#True#True
## Epoch localizer:

loc_fname = os.path.join(raw_path, sub, 'loc_raw_tsss.fif')
icaname = os.path.join(ica_path,sub, 'loc_raw_tsss-ica.fif')

raw = mne.io.read_raw_fif(loc_fname, preload = True) # load data
ica = mne.preprocessing.read_ica(icaname) # load ica solution
raw = ica.apply(raw) # apply ICA

try:
    raw.drop_channels(['MISC001','MISC002']) # delete audio channels
except:
    raw.drop_channels(['MISC002','MISC003'])

raw.resample(srate)

raw.filter(0.1,40,fir_design = 'firwin')

## Define epochs for listened and imagined:
events = {}
events['listened'] = mne.find_events(raw, shortest_event = 1)
events['listened'] = events['listened'][np.append(np.diff(events['listened'][:,0]) >2,True)] # delete spurious t
events['listened'] = events['listened'][events['listened'][:,2] < 4] # delete spurious events > 3
# events['imagined'] = events['listened'].copy()
# events['imagined'][:,0] = events['imagined'][:,0] + 1.25*srate

#Epoching and averaging:
event_id = {'1': 1, '2': 2, '3': 3}
picks = mne.pick_types(raw.info, meg = True, eog = False)
tmin, tmax = -0.1,2.5 #epoch time
baseline = (-0.1,0) # baseline time #None#

#epoching:
epochs = {}
for e in events:
    epochs[e] = mne.Epochs(raw, events = events[e], event_id = event_id,
                tmin = tmin, tmax = tmax, picks = picks,
                baseline = None)#,  decim = 2) #reject = reject,


### Analyses on experimental blocks ##
# for b in blocks:
#     block_fname = os.path.join(raw_path, sub, b +'_raw_tsss.fif')
#     icaname = os.path.join(ica_path,sub, b + '_raw_tsss-ica.fif')
#
#     raw = mne.io.read_raw_fif(block_fname, preload = True) # load data
#     ica = mne.preprocessing.read_ica(icaname) # load ica solution
#     raw = ica.apply(raw) # apply ICA
#
#     try:
#         raw.drop_channels(['MISC001','MISC002']) # delete audio channels
#     except:
#         raw.drop_channels(['MISC002','MISC003'])
#     raw.resample(srate)
#
#     raw.filter(0.1,40,fir_design = 'firwin')
#
#     events[b] = mne.find_events(raw, shortest_event = 1)
#     events[b] = events[b][np.append(np.diff(events[b][:,0]) > 1,True)] # delete spurious t
#     events[b + '_delay'] = events[b].copy()
#     events[b + '_imagined'] = events[b].copy()
#     events[b][:,2] = events[b][:,2]%10 # recode events
#     events[b+'_delay'] = events[b + '_delay'][events[b+'_delay'][:,2]<20,:]
#     events[b+'_delay'][:,0] = events[b + '_delay'][:,0] + srate*2
#     events[b+'_delay'][:,2] = events[b + '_delay'][:,2]%10
#     events[b+'_imagined'] = events[b + '_imagined'][events[b + '_imagined'][:,2]<100]
#     events[b+'_imagined'][:,0] = events[b+'_imagined'][:,0] + srate*2
#     events[b+'_imagined'][:,2] = events[b + '_imagined'][:,2]%10
#     if (b=='invv2') or (b== 'inv'):
#         events[b+'_imagined'][:,2] = (events[b+'_imagined'][:,2]-4)*-1
#     events[b+'_imagined2cat'] = events[b+'_imagined'][events[b+'_imagined'][:,2]!=2,:]
#
#     #Epoching and averaging:
#     event_id = {'1': 1, '2': 2, '3': 3}
#     event_id2 = {'1':1,'3':3}
#     picks = mne.pick_types(raw.info, meg = True, eog = False)
#     tmin, tmax = -0.1,0.5 #epoch time
#     baseline = (-0.1,0) # baseline time
#     reject = dict(mag = 4e-12, grad = 4000e-13)#, eog = 250e-6)
#
#     #epoching:
#     epochs[b] = mne.Epochs(raw, events = events[b], event_id = event_id,
#                     tmin = tmin, tmax = tmax, picks = picks,
#                     baseline = None)#,  decim = 1) #reject = reject,
#     epochs[b+'_delay'] = mne.Epochs(raw, events = events[b+'_delay'], event_id = event_id2,
#                     tmin = -0.5, tmax = 2, picks = picks,
#                     baseline = None)
#     epochs[b+'_imagined'] = mne.Epochs(raw, events = events[b+'_imagined'],
#                                        event_id = event_id,tmin = -0.25, tmax = 0.25,
#                                        picks = picks, baseline = None)
#     epochs[b+'_imagined2cat'] = mne.Epochs(raw, events = events[b+'_imagined2cat'],
#                                        event_id = event_id2,tmin = -0.25, tmax = 0.25,
#                                        picks = picks, baseline = None)

# fit classifiers and get patterns and filters:
if fit_models:
    gen ={}
    patterns = {}
    filters = {}
    clf = make_pipeline(StandardScaler(),
                        LinearModel(LogisticRegression(max_iter = 1000,solver='lbfgs')))
    for g in epochs:
        gen[g] = GeneralizingEstimator(clf, n_jobs=4, scoring = 'accuracy') #scoring='roc_auc',
        gen[g].fit(X=epochs[g].get_data(),y=epochs[g].events[:, 2])
        pat = get_coef(gen[g],'patterns_',inverse_transform = True)
        fil = get_coef(gen[g],'filters_',inverse_transform = True)
        patterns[g], filters[g] = {},{}
        if len(pat.shape) < 3:
            patterns[g]['1'] = mne.EvokedArray(pat,epochs[g].info,tmin = epochs[g].times[0])
            filters[g]['1'] = mne.EvokedArray(fil,epochs[g].info,tmin = epochs[g].times[0])
        else:
            for t in np.arange(pat.shape[1]):
                patterns[g][str(t+1)] = mne.EvokedArray(pat[:,t,:],epochs[g].info,
                                                        tmin = epochs[g].times[0])
                filters[g][str(t+1)] = mne.EvokedArray(fil[:,t,:],epochs[g].info,
                                                       tmin = epochs[g].times[0])

    # save clasifiers, patterns and filters:
    filt_fname = op.join(avg_path,'data',sub + '_filters.p')
    filt_file = open(filt_fname,'wb')
    pickle.dump(filters,filt_file)
    filt_file.close()
    print('filters file saved')

    pat_fname = op.join(avg_path,'data',sub + '_patterns.p')
    pat_file = open(pat_fname,'wb')
    pickle.dump(patterns,pat_file)
    pat_file.close()
    print('patterns file saved')

    dec_fname = op.join(avg_path,'data',sub + '_decoders.p')
    dec_file = open(dec_fname,'wb')
    pickle.dump(gen,dec_file)
    dec_file.close()
    print('decoders file saved')

# filt_fname = op.join(avg_path,'data',sub + '_filters.p')
# filt_file = open(filt_fname,'rb')
# filters = pickle.load(filt_file)
# filt_file.close()
#
# pat_fname = op.join(avg_path,'data',sub + '_patterns.p')
# pat_file = open(pat_fname,'rb')
# patterns = pickle.load(pat_file)
# pat_file.close()
#
# dec_fname = op.join(avg_path,'data',sub + '_decoders.p')
# dec_file = open(dec_fname,'rb')
# gen = pickle.load(dec_file)
# dec_file.close()
#
# ## get scores/accuracies:
# if get_scores:
#     scores = {}
#     for g1 in gen:
#         for g2 in gen:
#             nk = '{}_from_{}'.format(g1,g2)
#             if g1 == g2 : #if same data, use cross validation
#                 scores[nk] = cross_val_multiscore(gen[g2],X=epochs[g1].get_data(),
#                                                   y=epochs[g1].events[:, 2],
#                                                   cv = 5,n_jobs = 5)
#                 scores[nk] = np.mean(scores[nk],axis = 0)
#             else: # if different data, test in one, predict in the other
#                 if np.array_equal(np.unique(epochs[g1].events[:,2]),
#                                   np.unique(epochs[g2].events[:,2])):
#                    scores[nk]= gen[g2].score(X=epochs[g1].get_data(),
#                                               y=epochs[g1].events[:, 2])
#
#     acc_fname = op.join(avg_path,'data',sub + '_scores.p')
#     acc_file = open(acc_fname,'wb')
#     pickle.dump(scores,acc_file)
#     acc_file.close()
#     print('scores file saved')
#
# acc_fname = op.join(avg_path,'data',sub + '_scores.p')
# acc_file = open(acc_fname,'rb')
# scores = pickle.load(acc_file)
#
# ### plotting accuracies
# n_dtypes = 5 #number of decoders
# if plot_scores:
#     fig, axes = plt.subplots(ncols=n_dtypes,nrows=n_dtypes,figsize = (40,30))#,gridspec_kw=dict(width_ratios=[1,1,1,1]) )
#     skeys = [key for key in scores.keys()
#              if not any(e in key for e in ['inv_delay', 'main_delay','inv_imagined2cat',
#                                            'main_imagined2cat'])]
#     for sidx,s in enumerate(skeys):
#         f,se = s.split('_from_')
#         ext = [epochs[f].times[0],epochs[f].times[-1],
#                epochs[se].times[0],epochs[se].times[-1]]
#         im = axes[sidx//n_dtypes,sidx%n_dtypes].matshow(scores[s], vmin=0.18, vmax=0.48,
#                                           cmap='RdBu_r', origin='lower', extent=ext)
#         axes[sidx//n_dtypes,sidx%n_dtypes].axhline(0., color='k')
#         axes[sidx//n_dtypes,sidx%n_dtypes].axvline(0., color='k')
#         axes[sidx//n_dtypes,sidx%n_dtypes].xaxis.set_ticks_position('bottom')
#         axes[sidx//n_dtypes,sidx%n_dtypes].set_xlabel('Testing Time (s)')
#         axes[sidx//n_dtypes,sidx%n_dtypes].set_ylabel('Training Time (s)')
#         axes[sidx//n_dtypes,sidx%n_dtypes].set_anchor('W')
#         axes[sidx//n_dtypes,sidx%n_dtypes].set_title('pred. {}'.format(s),{'horizontalalignment': 'center'})
#     cbar_ax = fig.add_axes([0.925,0.15,0.01,0.7])
#     fig.colorbar(im,cax=cbar_ax)
#     fig.suptitle('Decoding accuracy', fontsize =  20)
#     plt.savefig(avg_path + '/figures/{}_accuracies1.pdf'.format(sub),orientation='landscape')
#     plt.show()
#     #plt.savefig(avg_path + '/figures/{}_accuracies1.pdf'.format(sub),orientation='landscape')
#
#     fig2, axes2 = plt.subplots(ncols=4,nrows=4,figsize = (20,8))#,gridspec_kw=dict(width_ratios=[1,1,1,1]) )
#     skeys = [key for key in scores.keys()
#              if any(e in key for e in ['inv_delay', 'main_delay','inv_imagined2cat',
#                                        'main_imagined2cat'])]
#     for sidx,s in enumerate(skeys):
#         f,se = s.split('_from_')
#         ext = [epochs[f].times[0],epochs[f].times[-1],
#                epochs[se].times[0],epochs[se].times[-1]]
#         im = axes2[sidx//4,sidx%4].matshow(scores[s], vmin=0.2, vmax=0.8,
#                                           cmap='RdBu_r', origin='lower', extent=ext)
#         axes2[sidx//4,sidx%4].axhline(0., color='k')
#         axes2[sidx//4,sidx%4].axvline(0., color='k')
#         axes2[sidx//4,sidx%4].xaxis.set_ticks_position('bottom')
#         axes2[sidx//4,sidx%4].set_xlabel('Testing Time (s)')
#         axes2[sidx//4,sidx%4].set_ylabel('Training Time (s)')
#         axes2[sidx//4,sidx%4].set_anchor('W')
#         axes2[sidx//4,sidx%4].set_title('pred. {}'.format(s),{'horizontalalignment': 'center'})
#     cbar_ax = fig2.add_axes([0.925,0.15,0.01,0.7])
#     fig.colorbar(im,cax=cbar_ax)
#     fig.suptitle('Decoding accuracy', fontsize =  20)
#     plt.savefig(avg_path + '/figures/{}_accuracies2.pdf'.format(sub),orientation='landscape')
#     plt.show()
#     #plt.savefig(avg_path + '/figures/{}_accuracies2.pdf'.format(sub),orientation='landscape')
#



### Plotting patterns and filters joint
# for p in patterns:
#     for t in patterns[p]:
#         times = patterns[p][t].times[np.arange(0,patterns[p][t].times.shape[0],2)]
#         ani_tpm = patterns[p][t].animate_topomap(times = times,ch_type='mag',
#                                                  butterfly=True,blit=False,frame_rate=5)
#         ani_tpm[1].save(avg_path + '/figures/{}_patterns_{}_{}_mag.mov'.format(sub,p,t))

#         ani_tpm = filters[p][t].animate_topomap(times = times,ch_type='mag',
#                                                  butterfly=True,blit=False,frame_rate=5)
#         ani_tpm[1].save(avg_path + '/figures/{}_filters_{}_{}_mag.mov'.format(sub,p,t))

#         ani_tpm = patterns[p][t].animate_topomap(times = times,ch_type='grad',
#                                                  butterfly=True,blit=False,frame_rate=5)
#         ani_tpm[1].save(avg_path + '/figures/{}_patterns_{}_{}_grad.mov'.format(sub,p,t))

#         ani_tpm = filters[p][t].animate_topomap(times = times,ch_type='grad',
#                                                  butterfly=True,blit=False,frame_rate=5)
#         ani_tpm[1].save(avg_path + '/figures/{}_filters_{}_{}_grad.mov'.format(sub,p,t))
#         plt.close('all')

# epochs['main']['1'].average().plot_joint()#times = np.arange(0.05,1,0.05))
# epochs['main']['2'].average().plot_joint()#times = np.arange(0.05,1,0.05))
# epochs['main']['3'].average().plot_joint()#times = np.arange(0.05,1,0.05))

# epochs['inv']['1'].average().plot_joint()#times = np.arange(0.05,1,0.05))
# epochs['inv']['2'].average().plot_joint()#times = np.arange(0.05,1,0.05))
# epochs['inv']['3'].average().plot_joint()#times = np.arange(0.05,1,0.05))

### Plotting patterns and filters

# types = ['main_delay','inv_delay','main_delay','inv_delay']
# #tws = [[0.22,0.28],[0.22,0.28],[1.5,1.95],[1.5,1.95]]
# tws = [[2,2.5],[2,2.5],[1.5,1.95],[1.5,1.95]]

# #vlims = [[-50,50],[-50,50],[-50,50],[-50,50]]
# fig2, axes2 = plt.subplots(ncols=3,nrows = 4,figsize = (10,10))
# for tidx, t in enumerate(types):
#     tw = tws[tidx]
#     avg_idx = (patterns[t]['1'].times >= tw[0]) & (patterns[t]['1'].times <= tw[1])
#     for pidx,p in enumerate(patterns[t]):
#         cdata =patterns[t][p].copy()
#         cdata.data = np.mean(cdata.data[:,avg_idx],1)
#         cdata.data = cdata.data.reshape(cdata.data.shape[0],1)
#         cdata.times = np.array([1])
#         time_str = '{}-{} s'.format(tw[0],tw[1])
#         print(time_str)
#         print(tw)
#         cdata.plot_topomap(times = 1, axes = axes2[tidx,pidx],ch_type = 'mag',
#                            time_format = time_str, colorbar = False
#                            #vmin=vlims[tidx][0],vmax = vlims[tidx][1]
#                            )
# fig2.suptitle('Patterns', fontsize =  20)

# types = ['main','inv','main_delay','inv_delay']
# tws = [[0.22,0.28],[0.22,0.28],[1.5,2],[1.5,2]]
# vlims = [[-30,30],[-30,30],[-15,15],[-15,15]]
# fig2, axes2 = plt.subplots(ncols=3,nrows = 4,figsize = (10,10))
# for tidx, t in enumerate(types):
#     tw = tws[tidx]
#     avg_idx = (filters[t]['1'].times >= tw[0]) & (filters[t]['1'].times <= tw[1])
#     for pidx,p in enumerate(filters[t]):
#         cdata =filters[t][p].copy()
#         cdata.data = np.mean(cdata.data[:,avg_idx],1)
#         cdata.data = cdata.data.reshape(cdata.data.shape[0],1)
#         cdata.times = np.array([1])
#         time_str = '{}-{} s'.format(tw[0],tw[1])
#         print(time_str)
#         print(tw)
#         cdata.plot_topomap(times = 1, axes = axes2[tidx,pidx],ch_type = 'mag',
#                            time_format = time_str, colorbar = False
#                            #vmin=vlims[tidx][0],vmax = vlims[tidx][1])
#                            )
# fig2.suptitle('Filters', fontsize =  20)


# # patterns['listened']['1'].plot_topomap(times = np.arange(0.1,0.31,0.05),ch_type='grad',
# #                                        average=0.05)
# # patterns['listened']['2'].plot_topomap(times = np.arange(0.1,0.31,0.05),ch_type='grad',
# #                                        average=0.05)
# # patterns['listened']['3'].plot_topomap(times = np.arange(0.1,0.31,0.05),ch_type='grad',
# #                                        average=0.05)

# # filters['listened']['1'].plot_topomap(times = np.arange(0.1,0.31,0.05),ch_type='mag',
# #                                        average=0.05)
# # filters['listened']['2'].plot_topomap(times = np.arange(0.1,0.31,0.05),ch_type='mag',
# #                                        average=0.05)
# # filters['listened']['3'].plot_topomap(times = np.arange(0.1,0.31,0.05),ch_type='mag',
# #                                        average=0.05)

# # epochs['listened']['1'].average().plot_topomap(times = np.arange(0.1,0.31,0.05),
# #                                                ch_type='mag',average=0.05)
# # epochs['listened']['2'].average().plot_topomap(times = np.arange(0.1,0.31,0.05),
# #                                                ch_type='mag',average=0.05)
# # epochs['listened']['3'].average().plot_topomap(times = np.arange(0.1,0.31,0.05),
# #                                                ch_type='mag',average=0.05)

# # patterns['imagined']['1'].plot_topomap(times = np.arange(-0.1,1,0.05),ch_type='grad',
# #                                        average=0.05)
# # patterns['imagined']['2'].plot_topomap(times = np.arange(-0.1,1,0.05),ch_type='grad',
# #                                        average=0.05)
# # patterns['imagined']['3'].plot_topomap(times = np.arange(-0.1,1,0.05),ch_type='grad',
# #                                        average=0.05)

# mne.viz.plot_evoked_topo([epochs['main_imagined']['1'].copy().average().pick_types('mag'),
#                           epochs['main_imagined']['2'].copy().average().pick_types('mag'),
#                           epochs['main_imagined']['3'].copy().average().pick_types('mag')])

# mne.viz.plot_evoked_topo([epochs['inv_imagined']['1'].copy().average().pick_types('mag'),
#                           epochs['inv_imagined']['2'].copy().average().pick_types('mag'),
#                           epochs['inv_imagined']['3'].copy().average().pick_types('mag')])

# mne.viz.plot_evoked_topo([patterns['listened']['1'].copy().pick_types('mag'),
#                           patterns['listened']['2'].copy().pick_types('mag'),
#                           patterns['listened']['3'].copy().pick_types('mag')])

# vlims = [[-80,80],[-200,300]]
# fig3, axes3 = plt.subplots(ncols=3,nrows = 2,figsize = (10,10))
# for tidx, t in enumerate(types):
#     tw = tws[tidx]
#     avg_idx = (filters[t]['tone1'].times >= tw[0]) & (filters[t]['tone1'].times <= tw[1])
#     for pidx,p in enumerate(tones):
#         cdata =filters[t][p].copy()
#         cdata.data = np.mean(cdata.data[:,avg_idx],1)
#         cdata.data = cdata.data.reshape(cdata.data.shape[0],1)
#         cdata.times = np.array([1])
#         time_str = '{}-{} s'.format(tw[0],tw[1])
#         cdata.plot_topomap(times = 1, axes = axes3[tidx,pidx],
#                            time_format = time_str, colorbar = False,
#                            vmin=vlims[tidx][0],vmax = vlims[tidx][1])

# fig3.suptitle('Filters', fontsize =  20)
# fig2.colorbar(axes2[0,0].collections[1],ax=axes2,location = 'right')
# plt.show()

pat_fname = op.join(avg_path,'data',sub + '_patterns.p')
pat_file = open(pat_fname,'rb')
patterns = pickle.load(pat_file)
pat_file.close()
print('patterns file loaded')

### Source localization
fwd_fn = op.join(fwd_path, sub + '_vol-fwd.fif')
fwd = mne.read_forward_solution(fwd_fn)
src = op.join(subjects_dir,sub,'bem',sub + '_vol-src.fif')
#compute and plot covariance
#data_cov = mne.compute_covariance([epochs['invv2_delay'],epochs['mainv2_delay']])
# noise_cov = mne.compute_covariance([epochs['imagined'],epochs['listened']],
#                                          tmin = -0.1,tmax=0)

## mne solution
SNR = 3
sources = {}
movs = {}
for p in patterns:
    sources[p]={}
    data_cov = mne.compute_covariance([epochs[p].load_data().copy().pick_types('mag')
                                           for e in epochs],tmin= 0,tmax = 6.5,rank ='info')
    ## inverse solution
    inv = mne.beamformer.make_lcmv(epochs[p].info,fwd,data_cov, reg=0.05,
                                 #noise_cov=noise_cov,#pick_ori='max-power',depth = 0.95,
                                 weight_norm= 'nai', rank = 'info')
    for t in patterns[p]:
        sources[p][t] = mne.beamformer.apply_lcmv(patterns[p][t],inv)
        #brain =[]
        # if save_movie:
        # brain = sources[p][t].plot(subjects_dir=subjects_dir,initial_time=-1,hemi = 'split',
        #           time_viewer=False, views=['lateral','medial'],
        #           title = 'patterns - {} {} {}'.format(sub,p,t))
        #     brain.save_movie(avg_path + '/figures/{}_pattern_sources_{}_{}.mov'.format(sub,p,t),
        #                      framerate=4,time_dilation = 10)
        #     brain.close()

src_fname = op.join(avg_path,'data',sub + '_pattern_sources.p')
src_file = open(src_fname,'wb')
pickle.dump(sources,src_file)
src_file.close()
print('sources file saved')

## Epoch inversion

# data_cov = mne.compute_covariance([epochs['listened'].load_data().copy().pick_types('mag')
#                                        for e in epochs],tmin= 0,tmax = 1,rank ='info')
# inv = mne.beamformer.make_lcmv(epochs['listened'].info,fwd,data_cov, reg=0.05,
#                              #noise_cov=noise_cov,#pick_ori='max-power',depth = 0.95,
#                              weight_norm= 'nai', rank = 'info')

# times = epochs['listened'].times
# esource = mne.beamformer.apply_lcmv_epochs(epochs['listened'], inv)
# avg = {}
# avg['1'] = esource[0].copy()
# avg['2'] = esource[0].copy()
# avg['3'] = esource[0].copy()

# avg['1'].data = np.zeros(esource[0].data.shape)
# avg['2'].data = np.zeros(esource[0].data.shape)
# avg['3'].data = np.zeros(esource[0].data.shape)

# scount = {}
# scount['1'], scount['2'], scount['3']  = 0, 0 ,0
# for eidx, e in enumerate(esource):
#     cnote = str(epochs['listened'].events[eidx,2])
#     scount[cnote] = scount[cnote] + 1
#     avg[cnote].data = avg[cnote].data + e.data

# avg['1'].data = avg['1'].data / scount['1']
# avg['2'].data = avg['2'].data / scount['2']
# avg['3'].data = avg['3'].data / scount['3']


# clf = make_pipeline(StandardScaler(),LinearModel(LogisticRegression(max_iter = 1000,solver='lbfgs')))

# egen = GeneralizingEstimator(clf, n_jobs=5, scoring = 'accuracy') #scoring='roc_auc',
# egen.fit(X=sdata,y=epochs['listened'].events[:, 2])
# epat = get_coef(egen,'patterns_',inverse_transform = True)
# efil = get_coef(egen,'filters_',inverse_transform = True)

# epat_s = avg_s1.copy()
# epat_s.data = epat

# escores = cross_val_multiscore(egen,X=sdata,
#                                 y=epochs['main_delay'].events[:, 2],
#                                   cv = 3,n_jobs = 5)
# mescores = np.mean(escores,axis = 0)

# #plot accuracies

# plt.figure()
# plt.matshow(escores, vmin=0.33, vmax=0.66,cmap='RdBu_r', origin='lower')
# plt.axhline(0., color='k')
# plt.axvline(0., color='k')
# plt.colorbar()

# plt.set_ticks_position('bottom')
# plt.set_xlabel('Testing Time (s)')
# plt.set_ylabel('Training Time (s)')
# plt.set_anchor('W')
# plt.colorbar()
