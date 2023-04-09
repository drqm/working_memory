import mne
import os
import os.path as op
import numpy as np
import pickle
from stormdb.access import Query
from sys import argv
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import GeneralizingEstimator, get_coef, LinearModel,cross_val_multiscore


os.environ['MINDLABPROJ']='MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

os.environ['ETS_TOOLKIT'] = 'qt4'

wdir = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/working_memory'
raw_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/maxfiltered_data/tsss_st16_corr96'
ica_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/working_memory/ICA'
avg_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/working_memory/averages'
fwd_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/forward_models'
subjects_dir = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/misc/fs_subjects_dir'
qy = Query('MINDLAB2020_MEG-AuditoryPatternRecognition')
subs = qy.get_subjects()
subno = 49
#plt.savefig(avg_path + '/figures_gemma/{}_main_sensor_accuracies1.pdf'.format(sub),orientation='landscape')
#plt.savefig(avg_path + '/figures_gemma/{}_inv_sensor_accuracies1.pdf'.format(sub),orientation='landscape')

if len(argv) > 1:
    subno = int(argv[1])
sub = subs[subno - 1]#'0011_U7X'#'0090_0MJ'#'0031_WZD'#'0011_U7X'#'0002_TQ8' #0001_AF5 0003_BHS

#sub = argv[1]#'0013_NHJ'#'0002_TQ8' #0001_AF5 0003_BHS

#blocks = ['inv']
blocks = ['main']
types = ['listened','imagined']

srate = 100
get_scores = True
plot_scores = True
save_movie = False
fit_models = True
plot = True

## analyses on experimental blocks:
for b in blocks:
    block_fname = os.path.join(raw_path, sub, b +'_raw_tsss.fif')
    icaname = os.path.join(ica_path,sub, b + '_raw_tsss-ica.fif')

    raw = mne.io.read_raw_fif(block_fname, preload = True) # load data
    ica = mne.preprocessing.read_ica(icaname) # load ica solution
    raw = ica.apply(raw) # apply ICA

    try:
        raw.drop_channels(['MISC001','MISC002']) # delete audio channels
    except:
        raw.drop_channels(['MISC002','MISC003'])
    raw.resample(srate)

    raw.filter(0.1,40,fir_design = 'firwin')
    
    events = {}
    events[b] = mne.find_events(raw, shortest_event = 1)
    events[b] = events[b][np.append(np.diff(events[b][:,0]) > 1,True)] # delete spurious t
    events[b + '_delay'] = events[b].copy()
    events[b + '_imagined'] = events[b].copy()
    events[b][:,2] = events[b][:,2]%10 # recode events
    events[b+'_delay'] = events[b + '_delay'][events[b+'_delay'][:,2]<20,:]
    events[b+'_delay'][:,0] = events[b + '_delay'][:,0] + srate*2
    events[b+'_delay'][:,2] = events[b + '_delay'][:,2]%10
    events[b+'_imagined'] = events[b + '_imagined'][events[b + '_imagined'][:,2]<100]
    events[b+'_imagined'][:,0] = events[b+'_imagined'][:,0] + srate*2
    events[b+'_imagined'][:,2] = events[b + '_imagined'][:,2]%10
    if (b=='invv2') or (b== 'inv'):
        events[b+'_imagined'][:,2] = (events[b+'_imagined'][:,2]-4)*-1
    events[b+'_imagined2cat'] = events[b+'_imagined'][events[b+'_imagined'][:,2]!=2,:]

    #Epoching and averaging:
    event_id = {'1': 1, '2': 2, '3': 3}
    event_id2 = {'1':1,'3':3}
    picks = mne.pick_types(raw.info, meg = True, eog = False) #channels to include
    tmin, tmax = -0.1,0.5 #epoch time
    baseline = (-0.1,0) # baseline time
    reject = dict(mag = 4e-12, grad = 4000e-13)#, eog = 250e-6)

    #epoching:
    epochs = {}
    epochs[b] = mne.Epochs(raw, events = events[b], event_id = event_id,
                    tmin = tmin, tmax = tmax, picks = picks,
                    baseline = None)#,  decim = 1) #reject = reject,
    epochs[b+'_delay'] = mne.Epochs(raw, events = events[b+'_delay'], event_id = event_id2,
                    tmin = -0.5, tmax = 2, picks = picks,
                    baseline = None)
    epochs[b+'_imagined'] = mne.Epochs(raw, events = events[b+'_imagined'],
                                        event_id = event_id,tmin = -0.25, tmax = 0.25,
                                        picks = picks, baseline = None)
    epochs[b+'_imagined2cat'] = mne.Epochs(raw, events = events[b+'_imagined2cat'],
                                        event_id = event_id2,tmin = -0.25, tmax = 0.25,
                                        picks = picks, baseline = None)

# fit classifiers and get patterns and filters:
if fit_models:
    gen ={}
    patterns = {}
    filters = {}
    clf = make_pipeline(StandardScaler(),
                        LinearModel(LogisticRegression(max_iter = 1000,solver='lbfgs')))
    for g in epochs:
        print(g)
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

## get scores/accuracies:
if get_scores:
     scores = {}
     for g1 in gen:
         for g2 in gen:
             nk = '{}_from_{}'.format(g1,g2)
             if g1 == g2 : #if same data, use cross validation
                 scores[nk] = cross_val_multiscore(gen[g2],X=epochs[g1].get_data(),
                                                   y=epochs[g1].events[:, 2],
                                                   cv = 5,n_jobs = 5)
                 scores[nk] = np.mean(scores[nk],axis = 0)
             else: # if different data, test in one, predict in the other
                 if np.array_equal(np.unique(epochs[g1].events[:,2]),
                                   np.unique(epochs[g2].events[:,2])):
                    scores[nk]= gen[g2].score(X=epochs[g1].get_data(),
                                               y=epochs[g1].events[:, 2])

     acc_fname = op.join(avg_path,'data',sub + '_scores_loc.p')
     acc_file = open(acc_fname,'wb')
     pickle.dump(scores,acc_file)
     acc_file.close()
     print('scores file saved')

acc_fname = op.join(avg_path,'data',sub + '_scores_loc.p')
acc_file = open(acc_fname,'rb')
scores = pickle.load(acc_file)

## plotting accuracies
n_dtypes = 2 #number of decoders
if plot_scores:
     fig, axes = plt.subplots(ncols=n_dtypes,nrows=n_dtypes,figsize = (20,15))#,gridspec_kw=dict(width_ratios=[1,1,1,1]) )
     for sidx,s in enumerate(scores):
         f,se = s.split('_from_')
         ext = [epochs[f].times[0],epochs[f].times[-1],
                epochs[se].times[0],epochs[se].times[-1]]
         
         im = axes[sidx//n_dtypes,sidx%n_dtypes].matshow(scores[s], vmin = 0, vmax = 1,#vmin=0.18, vmax=0.48,
                                           cmap='RdBu_r', origin='lower', extent=ext)
         axes[sidx//n_dtypes,sidx%n_dtypes].axhline(0., color='k')
         axes[sidx//n_dtypes,sidx%n_dtypes].axvline(0., color='k')
         axes[sidx//n_dtypes,sidx%n_dtypes].xaxis.set_ticks_position('bottom')
         axes[sidx//n_dtypes,sidx%n_dtypes].set_xlabel('Testing Time (s)')
         axes[sidx//n_dtypes,sidx%n_dtypes].set_ylabel('Training Time (s)')
         axes[sidx//n_dtypes,sidx%n_dtypes].set_anchor('W')
         axes[sidx//n_dtypes,sidx%n_dtypes].set_title('pred. {}'.format(s),{'horizontalalignment': 'center'})
     cbar_ax = fig.add_axes([0.925,0.15,0.01,0.7])
     fig.colorbar(im,cax=cbar_ax)
     fig.suptitle('Decoding accuracy', fontsize =  20)
     # plt.savefig('/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/working_memory/averages/figures_gemma/{}_main_sensor_accuracies1.pdf'.format(sub),orientation='landscape')
     
if plot:
    plt.savefig(avg_path + '/figures_gemma/{}_main_sensor_accuracies1.pdf'.format(sub),orientation='landscape')
