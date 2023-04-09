import mne
import os
import os.path as op
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sys import argv
from stormdb.access import Query
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from mne.decoding import GeneralizingEstimator, get_coef, LinearModel,cross_val_multiscore

os.environ['MINDLABPROJ']='MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'
os.environ['ETS_TOOLKIT'] = 'qt4'

wdir = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/working_memory'
raw_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/maxfiltered_data/tsss_st16_corr96'
ica_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/working_memory/ICA'
avg_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/working_memory/averages'
fwd_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/forward_models'
subjects_dir = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/fs_subjects_dir'
#sub = '0011_U7X'

qy = Query('MINDLAB2020_MEG-AuditoryPatternRecognition')
subs = qy.get_subjects()
subno = 12

if len(argv) > 1:
    subno = int(argv[1])
sub = subs[subno - 1]#'0011_U7X'#'0090_0MJ'#'0031_WZD'#'0011_U7X'#'0002_TQ8' #0001_AF5 0003_BHS

types = ['listened','imagined']

srate = 100
get_scores = True
plot_scores = True
save_movie = False
fit_models = True

## Epoching:
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

#define epochs for listened and imagined:
events = {}
events['listened'] = mne.find_events(raw, shortest_event = 1)
events['listened'] = events['listened'][np.append(np.diff(events['listened'][:,0]) >2,True)] # delete spurious t
events['listened'] = events['listened'][events['listened'][:,2] < 4] # delete spurious events > 3
events['imagined'] = events['listened'].copy()
events['imagined'][:,0] = events['imagined'][:,0] + 0.9*srate

#epoching and averaging:
event_id = {'1': 1, '2': 2, '3': 3} #one for each tone
picks = mne.pick_types(raw.info, meg = True, eog = False) #channels to include
tmin, tmax = -0.1,0.9 #epoch time (in seconds)
baseline = (-0.1,0) # baseline time #None#

#epoching:
epochs = {}
for e in events:
    epochs[e] = mne.Epochs(raw, events = events[e], event_id = event_id,
                tmin = tmin, tmax = tmax, picks = picks,
                baseline = None)#,  decim = 2) #reject = reject,

## Source localization:
fwd_fn = op.join(fwd_path, sub + '_vol-fwd.fif')
fwd = mne.read_forward_solution(fwd_fn)
src_file = op.join(subjects_dir,sub,'bem',sub + '_vol-src.fif')
src = mne.read_source_spaces(src_file)
source_lpass = None

#SNR = 3
sources = {}
data_cov = mne.compute_covariance([epochs[e].load_data().copy().pick_types('mag') for e in epochs],
                                  tmin= 0,tmax = 2,rank ='info')
#inverse solution
inv = mne.beamformer.make_lcmv(epochs[e].info,fwd,data_cov, reg=0.05, pick_ori='max-power',
                             #noise_cov=noise_cov,#pick_ori='max-power',depth = 0.95,
                             weight_norm= 'nai', rank = 'info')

for e in epochs:
  sources[e]={}
  csource = mne.beamformer.apply_lcmv_epochs(epochs[e],inv)
  sources[e] = np.array([cs.data for cs in csource])
  if source_lpass:
      sources[e] = mne.filter.filter_data(sources[e], srate, None,source_lpass, n_jobs = 8)
  

## Fit classifiers and get patterns and filters:
if fit_models:
    gen ={}
    patterns = {}
    filters = {}
    clf = make_pipeline(StandardScaler(),SelectKBest(f_classif, k=500),
                        LinearModel(LogisticRegression(max_iter = 1000,solver='lbfgs')))
    for g in epochs:
        print(g)
        gen[g] = GeneralizingEstimator(clf, n_jobs=4, scoring = 'accuracy') #scoring='roc_auc',
        gen[g].fit(X=sources[g],y=epochs[g].events[:, 2])
        pat = get_coef(gen[g],'patterns_',inverse_transform = True)
        fil = get_coef(gen[g],'filters_',inverse_transform = True)
        patterns[g], filters[g] = {},{}
        for t in np.arange(pat.shape[1]):
            # patterns[g][str(t+1)] = mne.EvokedArray(pat[:,t,:],epochs[g].info,
            #                                         tmin = epochs[g].times[0])
            # filters[g][str(t+1)] = mne.EvokedArray(fil[:,t,:],epochs[g].info,
            #                                        tmin = epochs[g].times[0])
            patterns[g][str(t+1)], filters[g][str(t+1)] = csource[0].copy(), csource[0].copy()
            patterns[g][str(t+1)].data = pat[:,t,:]
            filters[g][str(t+1)].data = fil[:,t,:]
            
    # save clasifiers, patterns and filters:
    filt_fname = op.join(avg_path,'data',sub + '_filters_loc_source.p')
    filt_file = open(filt_fname,'wb')
    pickle.dump(filters,filt_file)
    filt_file.close()
    print('filters file saved')

    pat_fname = op.join(avg_path,'data',sub + '_patterns_loc_source.p')
    pat_file = open(pat_fname,'wb')
    pickle.dump(patterns,pat_file)
    pat_file.close()
    print('patterns file saved')

    dec_fname = op.join(avg_path,'data',sub + '_decoders_loc_source.p')
    dec_file = open(dec_fname,'wb')
    pickle.dump(gen,dec_file)
    dec_file.close()
    print('decoders file saved')

## Get scores/accuracies:
if get_scores:
     scores = {}
     for g1 in gen:
         for g2 in gen:
             nk = '{}_from_{}'.format(g1,g2)
             if g1 == g2 : #if same data, use cross validation
                 scores[nk] = cross_val_multiscore(gen[g2],X=sources[g1],
                                                   y=epochs[g1].events[:, 2],
                                                   cv = 5,n_jobs = 5)
                 scores[nk] = np.mean(scores[nk],axis = 0)
             else: # if different data, test in one, predict in the other
                 if np.array_equal(np.unique(epochs[g1].events[:,2]),
                                   np.unique(epochs[g2].events[:,2])):
                    scores[nk]= gen[g2].score(X=sources[g1],
                                               y=epochs[g1].events[:, 2])

     acc_fname = op.join(avg_path,'data',sub + '_scores_loc_source.p')
     acc_file = open(acc_fname,'wb')
     pickle.dump(scores,acc_file)
     acc_file.close()
     print('scores file saved')

acc_fname = op.join(avg_path,'data',sub + '_scores_loc_source.p')
acc_file = open(acc_fname,'rb')
scores = pickle.load(acc_file)

## Plotting accuracies:
n_dtypes = 2 #number of decoders
if plot_scores:
     fig, axes = plt.subplots(ncols=n_dtypes,nrows=n_dtypes,figsize = (20,15))#,gridspec_kw=dict(width_ratios=[1,1,1,1]) )
     for sidx,s in enumerate(scores):
         f,se = s.split('_from_')
         ext = [epochs[f].times[0],epochs[f].times[-1],
                epochs[se].times[0],epochs[se].times[-1]]
         
         im = axes[sidx//n_dtypes,sidx%n_dtypes].matshow(scores[s], vmin=0.18, vmax=0.48,
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
     plt.savefig(avg_path + '/figures_gemma/loc_source/{}_loc_source_accuracies1.pdf'.format(sub),orientation='landscape')
     plt.show()