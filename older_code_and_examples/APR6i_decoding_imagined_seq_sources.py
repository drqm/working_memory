import mne
from meegkit.detrend import create_masked_weight, detrend
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
from src.decoding_functions import smooth_data
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
subno = 14

if len(argv) > 1:
    subno = int(argv[1])
sub = subs[subno - 1]#'0011_U7X'#'0090_0MJ'#'0031_WZD'#'0011_U7X'#'0002_TQ8' #0001_AF5 0003_BHS

blocks = ['main','inv']
types = ['listened','imagined']

# smoothing parameters:
smooth_tstep = 0.025
smooth_twin = 0.08
srate = 100
source_lpass = None
suffix = '_smoothing25_80_hp001'
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
    raw.filter(0.01,40,fir_design = 'firwin')
    raw.resample(srate)
            
    events[b] = mne.find_events(raw, shortest_event = 1)
    events[b] = events[b][np.append(np.diff(events[b][:,0]) > 1,True)] # delete spurious t
    events[b + '_delay'] = events[b].copy()
    events[b][:,2] = events[b][:,2]%10 # recode events
    events[b+'_delay'] = events[b + '_delay'][events[b+'_delay'][:,2]<20,:]
    events[b+'_delay'][:,2] = events[b + '_delay'][:,2]%10
    events[b+'_fmel'] = events[b + '_delay'].copy()
    events[b+'_delay'][:,0] = events[b + '_delay'][:,0] + srate*2
    
#    print('\nTrial masked robust detrending')
#    sfreq = raw.info['sfreq']
#     data = raw.get_data().T
#     det_mask = create_masked_weight(data,np.array(events[b+'_fmel'][:,0]),tmin=0, tmax = 6.5, sfreq=sfreq)
#     data,_,_ = detrend(data,order=3,w=det_mask)
#     raw = mne.io.RawArray(data=data.T, info = raw.info.copy())
    #raw.filter(0.1, None,fir_design = 'firwin')
    
    picks = mne.pick_types(raw.info, meg = True, eog = False)
        
    event_id2 = {'1':1,'3':3}
    epochs[b+'_delay'] = mne.Epochs(raw, events = events[b+'_delay'], event_id = event_id2,
                    tmin = -0.5, tmax = 2, picks = picks, 
                    baseline = None)
    epochs[b+'_fmel'] = mne.Epochs(raw, events = events[b+'_fmel'], event_id = event_id2,
                tmin = -0.5, tmax = 2, picks = picks, 
                baseline = None)
    
fwd_fn = op.join(fwd_path, sub + '_vol-fwd.fif')
fwd = mne.read_forward_solution(fwd_fn)
src_file = op.join(subjects_dir,sub,'bem',sub + '_vol-src.fif')
src = mne.read_source_spaces(src_file)

SNR = 3
sources = {}
new_times = {}
data_cov = mne.compute_covariance([epochs[e].load_data().copy().pick_types('mag') for e in epochs],
                                  tmin= 0,tmax = 2,rank ='info')
## inverse solution
inv = mne.beamformer.make_lcmv(epochs['main_delay'].info,fwd,data_cov, reg=0.05, pick_ori='max-power',
                             #noise_cov=noise_cov,#pick_ori='max-power',depth = 0.95,
                             weight_norm= 'nai', rank = 'info')

crop_times = {'main_delay': [-.25,2],'inv_delay': [-.25,2],'main_fmel': [-.25,1.5], 'inv_fmel': [-.25,1.5]}
for e in epochs:
    csource = mne.beamformer.apply_lcmv_epochs(epochs[e].copy().crop(crop_times[e][0],crop_times[e][1]),inv)
    sources[e] = np.array([cs.data for cs in csource])
    if source_lpass:
        sources[e] = mne.filter.filter_data(sources[e], srate, None,source_lpass, n_jobs = 8)
    if smooth_tstep:
        sources[e], new_times[e] = smooth_data(sources[e], tstart=csource[0].tmin,
                                               tstep=smooth_tstep, twin=smooth_twin, Fs=srate, taxis=2)
    else:
        times[e] = csource.times.copy()
        
gen_cnd = ['main_delay','inv_delay','main_fmel','inv_fmel']
gen ={}
patterns = {}
filters = {}
clf = make_pipeline(StandardScaler(),SelectKBest(f_classif, k=500),
                    LinearModel(LogisticRegression(max_iter = 1000,solver='lbfgs')))
                    #LinearModel(LDA()))
for g in gen_cnd:
    gen[g] = GeneralizingEstimator(clf, n_jobs=4, scoring = 'roc_auc') #scoring='roc_auc',
    #gen[g] =SlidingEstimator(clf, n_jobs=-1, scoring = 'accuracy') #scoring='roc_auc',
    gen[g].fit(X=sources[g],y=epochs[g].events[:, 2])
    pat = get_coef(gen[g],'patterns_',inverse_transform = True)
    fil = get_coef(gen[g],'filters_',inverse_transform = True)
    csource[0].tmin = new_times[g][0]
    if smooth_tstep:
        csource[0].tstep = smooth_tstep
    patterns[g], filters[g] = csource[0].copy(), csource[0].copy()
    patterns[g].data = pat
    filters[g].data = fil

# save clasifiers, patterns and filters:
filt_fname = op.join(avg_path,'data',sub + '_filters_imagined{}.p'.format(suffix))             
filt_file = open(filt_fname,'wb')
pickle.dump(filters,filt_file)
filt_file.close()
print('filters file saved')

pat_fname = op.join(avg_path,'data',sub + '_patterns_imagined{}.p'.format(suffix))             
pat_file = open(pat_fname,'wb')
pickle.dump(patterns,pat_file)
pat_file.close()
print('patterns file saved')

dec_fname = op.join(avg_path,'data',sub + '_decoders_imagined{}.p'.format(suffix))             
dec_file = open(dec_fname,'wb')
pickle.dump(gen,dec_file)
dec_file.close()
print('decoders file saved')

## get scores/accuracies:
#gen_cnd = ['main_delay','inv_delay']
scores = {}
for g1 in gen_cnd:
    for g2 in gen_cnd:
        nk = '{}_from_{}'.format(g1,g2)
        print(nk)
        if g1 == g2 : #if same data, use cross validation
            None
            scores[nk] = cross_val_multiscore(gen[g2], X=sources[g1],
                                              y=epochs[g1].events[:, 2],
                                              cv = 5,n_jobs = 4)
            scores[nk] = np.mean(scores[nk], axis = 0)
        else: # if different data, test in one, predict in the other
            if np.array_equal(np.unique(epochs[g1].events[:,2]),
                              np.unique(epochs[g2].events[:,2])):
               scores[nk]= gen[g2].score(X=sources[g1],
                                        y=epochs[g1].events[:, 2])

acc_fname = op.join(avg_path,'data',sub + '_scores_imagined{}.p'.format(suffix))             
acc_file = open(acc_fname,'wb')
pickle.dump(scores,acc_file)
acc_file.close()
print('scores file saved')

ncols = 4
fig, axes = plt.subplots(ncols=ncols,nrows=4, figsize = (20,10)) #,gridspec_kw=dict(width_ratios=[1,1,1,1]) )
for sidx,s in enumerate(scores):
    f,se = s.split('_from_')
    ext = [new_times[f][0],new_times[f][-1],
           new_times[se][0],new_times[se][-1]]
#     ext = [epochs[f].times[0],epochs[f].times[-1],
#            epochs[se].times[0],epochs[se].times[-1]]
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
plt.savefig(avg_path + '/figures/{}_accuracies_imagined{}.pdf'.format(sub,suffix),orientation='landscape')
plt.tight_layout()
plt.show()