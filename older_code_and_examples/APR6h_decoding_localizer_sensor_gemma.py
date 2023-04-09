import mne
import os
import os.path as op
import numpy as np
import pickle
from sys import argv
from stormdb.access import Query
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import GeneralizingEstimator, get_coef, LinearModel,cross_val_multiscore
from mne.beamformer import make_lcmv, apply_lcmv
from nilearn.plotting import plot_stat_map
from nilearn.image import index_img

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
#sub = '0092_8LG'
qy = Query('MINDLAB2020_MEG-AuditoryPatternRecognition')
subs = qy.get_subjects()
subno = 92

if len(argv) > 1:
    subno = int(argv[1])
sub = subs[subno - 1]#'0011_U7X'#'0090_0MJ'#'0031_WZD'#'0011_U7X'#'0002_TQ8' #0001_AF5 0003_BHS

srate = 100
get_scores = True
plot_scores = True
save_movie = False
fit_models = True

## epoch localizer:
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
events['imagined'] = events['listened'].copy()
events['imagined'][:,0] = events['imagined'][:,0] + 0.9*srate

## Epoching and averaging:
event_id = {'1': 1, '2': 2, '3': 3} #one for each tone
picks = mne.pick_types(raw.info, meg = True, eog = False) #channels to include
tmin, tmax = -0.1,0.9 #epoch time (in seconds)
baseline = (-0.1,0) # baseline time #None#

## Epoching:
epochs = {}
for e in events:
    epochs[e] = mne.Epochs(raw, events = events[e], event_id = event_id,
                tmin = tmin, tmax = tmax, picks = picks,
                baseline = None)#,  decim = 2) #reject = reject,

## Fit classifiers and get patterns and filters:
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
        for t in np.arange(pat.shape[1]):
            patterns[g][str(t+1)] = mne.EvokedArray(pat[:,t,:],epochs[g].info,
                                                    tmin = epochs[g].times[0])
            filters[g][str(t+1)] = mne.EvokedArray(fil[:,t,:],epochs[g].info,
                                                   tmin = epochs[g].times[0])

    # save clasifiers, patterns and filters:
    filt_fname = op.join(avg_path,'data',sub + '_filters_loc_sensor.p')
    filt_file = open(filt_fname,'wb')
    pickle.dump(filters,filt_file)
    filt_file.close()
    print('filters file saved')

    pat_fname = op.join(avg_path,'data',sub + '_patterns_loc_sensor.p')
    pat_file = open(pat_fname,'wb')
    pickle.dump(patterns,pat_file)
    pat_file.close()
    print('patterns file saved')

    dec_fname = op.join(avg_path,'data',sub + '_decoders_loc_sensor.p')
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
                 scores[nk] = cross_val_multiscore(gen[g2],X=epochs[g1].get_data(),
                                                   y=epochs[g1].events[:, 2],
                                                   cv = 5,n_jobs = 5)
                 scores[nk] = np.mean(scores[nk],axis = 0)
             else: # if different data, test in one, predict in the other
                 if np.array_equal(np.unique(epochs[g1].events[:,2]),
                                   np.unique(epochs[g2].events[:,2])):
                    scores[nk]= gen[g2].score(X=epochs[g1].get_data(),
                                               y=epochs[g1].events[:, 2])

     acc_fname = op.join(avg_path,'data',sub + '_scores_loc_sensor.p')
     acc_file = open(acc_fname,'wb')
     pickle.dump(scores,acc_file)
     acc_file.close()
     print('scores file saved')

acc_fname = op.join(avg_path,'data',sub + '_scores_loc_sensor.p')
acc_file = open(acc_fname,'rb')
scores = pickle.load(acc_file)

## Plotting accuracies
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
     plt.show()
     plt.savefig(avg_path + '/figures_gemma/loc_sensor_g/{}_loc_sensor_accuracies1.pdf'.format(sub),orientation='landscape')
     
