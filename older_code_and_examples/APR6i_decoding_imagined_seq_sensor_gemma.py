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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, f_classif
from mne.decoding import GeneralizingEstimator, get_coef, LinearModel,cross_val_multiscore,SlidingEstimator
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MINDLABPROJ']=project
os.environ['MNE_ROOT']='~/miniconda3/envs/mne' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

wdir = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/working_memory'
raw_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/maxfiltered_data/tsss_st16_corr96'
ica_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/working_memory/ICA'
avg_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/working_memory/averages'
fwd_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/forward_models'
subjects_dir = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/fs_subjects_dir'
qy = Query(project)
subs = qy.get_subjects()
subno = 2

if len(argv) > 1:
    subno = int(argv[1])
sub = subs[subno - 1]#'0011_U7X'#'0090_0MJ'#'0031_WZD'#'0011_U7X'#'0002_TQ8' #0001_AF5 0003_BHS

blocks = ['main','inv']
types = ['listened','imagined']

srate = 40
source_lpass = 4
suffix = '_s40_lp15_No_hp'

## Define epochs for listened and imagined:        
events = {}
epochs = {}
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
    #raw.filter(0.1,15,fir_design = 'firwin')
    raw.filter(None,15,fir_design = 'firwin')
    raw.resample(srate)
            
    events[b] = mne.find_events(raw, shortest_event = 1)
    events[b] = events[b][np.append(np.diff(events[b][:,0]) > 1,True)] # delete spurious t
    events[b + '_delay'] = events[b].copy()
    events[b][:,2] = events[b][:,2]%10 # recode events
    events[b+'_delay'] = events[b + '_delay'][events[b+'_delay'][:,2]<20,:]
    events[b+'_delay'][:,2] = events[b + '_delay'][:,2]%10
    events[b+'_fmel'] = events[b + '_delay'].copy()
    events[b+'_delay'][:,0] = events[b + '_delay'][:,0] + srate*2
    
    picks = mne.pick_types(raw.info, meg = True, eog = False)
        
    event_id2 = {'1':1,'3':3}
    epochs[b+'_delay'] = mne.Epochs(raw, events = events[b+'_delay'], event_id = event_id2,
                    tmin = -0.5, tmax = 2, picks = picks, 
                    baseline = None)
    epochs[b+'_fmel'] = mne.Epochs(raw, events = events[b+'_fmel'], event_id = event_id2,
                tmin = -0.5, tmax = 2, picks = picks, 
                baseline = None)
    
# fwd_fn = op.join(fwd_path, sub + '_vol-fwd.fif')
# fwd = mne.read_forward_solution(fwd_fn)
# src_file = op.join(subjects_dir,sub,'bem',sub + '_vol-src.fif')
# src = mne.read_source_spaces(src_file)

## Fit classifiers and get patterns and filters:
gen_cnd = ['main_delay','inv_delay','main_fmel','inv_fmel']
gen ={}
patterns = {}
filters = {}
clf = make_pipeline(StandardScaler(),SelectKBest(f_classif, k='all'),
                    LinearModel(LogisticRegression(max_iter = 1000,solver='lbfgs')))

crop_times = {'main_delay': [-.25,2],'inv_delay': [-.25,2],'main_fmel': [-.25,1.5], 'inv_fmel': [-.25,1.5]}

for g in gen_cnd:
    print(g)
    gen[g] = GeneralizingEstimator(clf, n_jobs=4, scoring = 'accuracy')
    epochs[g].load_data() #preload epochs
    gen[g].fit(X=epochs[g].crop(crop_times[g][0],crop_times[g][1]).get_data(),y=epochs[g].events[:, 2])
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
filt_fname = op.join(avg_path,'data',sub + '_filters_sensor_seq_ imagined{}.p'.format(suffix))             
filt_file = open(filt_fname,'wb')
pickle.dump(filters,filt_file)
filt_file.close()
print('filters file saved')

pat_fname = op.join(avg_path,'data',sub + '_patterns_sensor_seq_imagined{}.p'.format(suffix))             
pat_file = open(pat_fname,'wb')
pickle.dump(patterns,pat_file)
pat_file.close()
print('patterns file saved')

dec_fname = op.join(avg_path,'data',sub + '_decoders_sensor_seq_imagined{}.p'.format(suffix))             
dec_file = open(dec_fname,'wb')
pickle.dump(gen,dec_file)
dec_file.close()
print('decoders file saved')

## Get scores/accuracies:
#gen_cnd = ['main_delay','inv_delay']
scores = {}
for g1 in gen_cnd:
    for g2 in gen_cnd:
        nk = '{}_from_{}'.format(g1,g2)
        print(nk)
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

acc_fname = op.join(avg_path,'data',sub + '_scores_sensor_seq_imagine_seq.p')
acc_file = open(acc_fname,'wb')
pickle.dump(scores,acc_file)
acc_file.close()
print('scores file saved')

## Plotting accuracies:
ncols = 4
fig, axes = plt.subplots(ncols=ncols,nrows=4, figsize = (20,10)) #,gridspec_kw=dict(width_ratios=[1,1,1,1]) )
for sidx,s in enumerate(scores):
    f,se = s.split('_from_')
    ext = [epochs[f].times[0],epochs[f].times[-1],
            epochs[se].times[0],epochs[se].times[-1]]
    rix, cix = sidx//ncols,sidx%ncols
    im = axes[rix, cix].matshow(scores[s], vmin = 0, vmax = 1,#vmin=0.18, vmax=0.48,
                                      cmap='RdBu_r', origin='lower', extent=ext)
    axes[rix, cix].axhline(0., color='k')
    axes[rix, cix].axvline(0., color='k')
    axes[rix, cix].xaxis.set_ticks_position('bottom')
    axes[rix, cix].set_xlabel('Testing Time (s)')
    axes[rix, cix].set_ylabel('Training Time (s)')
    axes[rix, cix].set_anchor('W')
    axes[rix, cix].set_title('pred. {}'.format(s),{'horizontalalignment': 'center'})
cbar_ax = fig.add_axes([0.925,0.15,0.01,0.7])
fig.colorbar(im,cax=cbar_ax)
fig.suptitle('Decoding accuracy (ROC - AUC)', fontsize =  20)
plt.savefig(avg_path + '/figures_gemma/seq_sensor/{}_seq_sensor_accuracies_imagined{}.pdf'.format(sub,suffix),orientation='landscape')
plt.tight_layout()
#plt.show()