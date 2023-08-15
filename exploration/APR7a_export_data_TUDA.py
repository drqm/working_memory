import mne
import os
import numpy as np
import scipy.io as sio

os.environ['MINDLABPROJ']='MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MNE_ROOT']='/users/david/miniconda3/envs/mne2' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

os.environ['ETS_TOOLKIT'] = 'qt4'
# os.environ['QT_API'] = 'pyqt5'

wdir = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/invmain_analyses'
raw_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/maxfiltered_data/tsss_st16_corr96'
ica_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/invmain_analyses/ICA'
avg_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/invmain_analyses/averages'
fwd_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/forward_models'
subjects_dir = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/fs_subjects_dir'
TUDA_dir = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/invmain_analyses/TUDA_data' 
subjects = ['0031_WZD']#'0021_LZW']#,'0011_U7X','0031_WZD']
#sub = argv[1]#'0013_NHJ'#'0002_TQ8' #0001_AF5 0003_BHS

blocks = ['main','inv']
block_names = ['maintenance','manipulation']
srate = 400
### Analyses on experimental blocks ##
for sub in subjects:
    for bidx, b in enumerate(blocks): 
        bname = block_names[bidx]
        data = {};
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
                    
        events, events2 = [],[]
        events = mne.find_events(raw, shortest_event = 1)
        events = events[np.append(np.diff(events[:,0]) > 1,True)] # delete spurious t
        events2 = events.copy()
        events = events[events[:,2]<20,:]
        events[:,2] = events[:,2]%10
        events[events[:,2]==3,2] = 2 
        
        events2 = events2[events2[:,2] >= 100,:]
        events2[events2[:,2] == 113,2] = 2
        events2[events2[:,2] == 111,2] = 1
        events2[np.where(events2[:,2] == 221)[0] - 1,2] = 3
        events2[np.where(events2[:,2] == 223)[0] - 1,2] = 4
        events2[events2[:,2] == 213,2] = 2
        events2[events2[:,2] == 211,2] = 1
        events2 = events2[events2[:,2]<10,:]
        events2[:,0] = events[:,0].copy()
        
        #Epoching and averaging:
        event_id = {'1': 1, '2': 2}
        picks = mne.pick_types(raw.info, meg = True, eog = False)
        tmin, tmax = -2,6.5 #epoch time
        reject = dict(mag = 4e-12, grad = 4000e-13)#, eog = 250e-6)
        
        #epoching:
        epochs = []
        epochs = mne.Epochs(raw, events = events, event_id = event_id,
                        tmin = tmin, tmax = tmax, picks = picks, 
                        baseline = None)#,reject = reject)
    
        event_id2 = {'1': 1, '2': 2, '3': 3, '4': 4}
        
        epochs2 = []
        epochs2 = mne.Epochs(raw, events = events2, event_id = event_id2,
                        tmin = tmin, tmax = tmax, picks = picks, 
                        baseline = None)#,reject = reject)
        
        data['X'] = epochs.get_data()
        epochs2.get_data()
        data['y1'] = np.array(epochs.events[:,2])
        data['y2'] = np.array(epochs2.events[:,2])
        data['time'] = np.array(epochs.times)
        data['channels'] = epochs.info['ch_names']
        data['srate'] = srate
        sio.savemat('{}/{}_{}_TUDA.mat'.format(TUDA_dir,sub,bname),{'data': data})
        