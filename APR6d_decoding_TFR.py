import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import GeneralizingEstimator, get_coef, LinearModel
from mne.time_frequency import tfr_morlet

wdir = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/invmain_analyses'
raw_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/maxfiltered_data/tsss_st16_corr96'
ica_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/invmain_analyses/ICA'
avg_path = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/invmain_analyses/averages'

blocks = ['main','inv']
sub = '0002_BYG'#'0002_TQ8' #0001_AF5 0003_BHS
#bads = ['EEG028']

loc_fname = os.path.join(raw_path, sub, 'loc_raw_tsss.fif')
icaname = os.path.join(ica_path,sub, 'loc_raw_tsss-ica.fif')

raw = mne.io.read_raw_fif(loc_fname, preload = True) # load data
ica = mne.preprocessing.read_ica(icaname) # load ica solution
raw = ica.apply(raw) # apply ICA

raw.drop_channels(['MISC002','MISC003']) # delete audio channels
raw.resample(200)

#raw.filter(0.1,50,fir_design = 'firwin')
        
events = mne.find_events(raw, shortest_event = 1)
events = events[np.append(np.diff(events[:,0]) >2,True)] # delete spurious t
events = events[events[:,2] < 4] # delete spurious events > 55
#events2 = events[events[:,2] == 12] # recode de events:
#events2[:,2] = events[events[:,2]>50,2] - 50   

#Epoching and averaging:
event_id = {'1': 1, '2': 2, '3': 3}
picks = mne.pick_types(raw.info, meg = True, eog = False)
tmin, tmax = -0.1,0.5 #epoch time
baseline = (-0.1,0) # baseline time
reject = dict(mag = 4e-12, grad = 4000e-13)#, eog = 250e-6)

#epoching:
epochs = mne.Epochs(raw, events = events, event_id = event_id,
                tmin = tmin, tmax = tmax, picks = picks, 
                baseline = baseline,  decim = 2) #reject = reject,

freqs = [10]#np.arange(8,15,1)
#freqs = np.logspace(*np.log10([2,100]),20)
n_cycles = 2

power = tfr_morlet(epochs, freqs = freqs, n_cycles= n_cycles,average=False, 
                   use_fft = True,return_itc = False, decim = 1, n_jobs=4)

clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(max_iter = 1000,solver='lbfgs')))
time_gen = GeneralizingEstimator(clf, n_jobs=1, scoring = 'accuracy') #scoring='roc_auc',
#time_slide = mne.decoding.SlidingEstimator(clf, n_jobs = 1, scoring  = 'accuracy')
# time_gen.fit(X=epochs.get_data(),
#              y=epochs.events[:, 2])
time_gen.fit(X=np.mean(power.data,2),
             y=power.events[:, 2])

###load block and preprocessing#########
accuracies = {}
for b in blocks:
   
    block_fname = os.path.join(raw_path, sub, b +'_raw_tsss.fif')
    icaname = os.path.join(ica_path,sub, b + '_raw_tsss-ica.fif')
    
    raw = mne.io.read_raw_fif(block_fname, preload = True) # load data
    ica = mne.preprocessing.read_ica(icaname) # load ica solution
    raw = ica.apply(raw) # apply ICA
    
    raw.drop_channels(['MISC002','MISC003']) # delete audio channels
    raw.resample(200)
    
    raw.filter(0.1,50,fir_design = 'firwin')
            
    events = mne.find_events(raw, shortest_event = 1)
    events = events[np.append(np.diff(events[:,0]) >1,True)] # delete spurious t
    events2 = events.copy()
    events2 = events2[events[:,2] < 20] 
    events2[:,2] = events2[:,2]%10 # recode events
    #events[:,2] = events[:,2]%10 # recode events

    #Epoching and averaging:
    event_id = {'1': 1, '2': 2, '3': 3}
    picks = mne.pick_types(raw.info, meg = True, eog = False)
    tmin, tmax = -0.1,0.5 #epoch time
    baseline = (-0.1,0) # baseline time
    reject = dict(mag = 4e-12, grad = 4000e-13)#, eog = 250e-6)
    
    #epoching:
    epochs2 = mne.Epochs(raw, events = events, event_id = event_id,
                    tmin = tmin, tmax = tmax, picks = picks, 
                    baseline = baseline,  decim = 2) #reject = reject,
                    
    power2 = tfr_morlet(epochs2, freqs = freqs, n_cycles= n_cycles,average=False, 
                   use_fft = True,return_itc = False, decim = 1, n_jobs=4)                
         
    # time_slide.fit(X=epochs.get_data(),
    #              y=epochs.events[:, 2])             
    #scores = time_gen.score(X=epochs[450:600].get_data(),
      #                       y=epochs[450:600].events[:, 2])
    # scores = time_gen.score(X=epochs2.get_data(),
    #                         y=epochs2.events[:, 2])
    scores = time_gen.score(X=np.mean(power2.data,2),
                            y=power2.events[:, 2])
    accuracies[b] = scores.copy()
    
    #plotting                                                                
    fig, ax = plt.subplots(1)
    im = ax.matshow(scores, vmin=0.13, vmax=0.53, cmap='RdBu_r', origin='lower',
                    extent=epochs.times[[0, -1, 0, -1]])
    ax.axhline(0., color='k')
    ax.axvline(0., color='k')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Generalization across time and condition')
    plt.colorbar(im, ax=ax)
    plt.show()
    #fig.savefig(wdir + '/results/figures/scores_' + b + '.pdf')
    
    fig, ax = plt.subplots(1)
    ax.plot(epochs.times,np.diag(scores))
    ax.axhline(0.33, color='k')
    ax.axvline(0., color='k')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Generalization across time and condition')
    plt.colorbar(im, ax=ax)
    plt.show()
    
    event_id2 = {'1': 1, '3': 3}
    picks = mne.pick_types(raw.info, meg = True, eog = False)
    tmin, tmax = -3,6.5 #epoch time
    baseline = (-3,0) # baseline time
    reject = dict(mag = 4e-12, grad = 4000e-13)#, eog = 250e-6)
    
    #epoching:
    epochs3 = mne.Epochs(raw, events = events2, event_id = event_id2,
                    tmin = tmin, tmax = tmax, picks = picks, 
                    baseline = baseline,  decim = 2) #reject = reject,
    
    #fig.savefig(wdir + '/results/figures/scores_diag_' + b + '.pdf')
    
    # event_id2 = {'1': 1}
    # tmin, tmax = -3,6.5 #epoch time
    # baseline = (-3,0) # baseline time
    # reject = dict(mag = 4e-12, grad = 4000e-13)#, eog = 250e-6)
    
    # #epoching:
    # epochs3 = mne.Epochs(raw, events = events2, event_id = event_id2,
    #                 tmin = tmin, tmax = tmax, picks = picks, 
    #                 baseline = baseline,  decim = 2)
    power3 = tfr_morlet(epochs3, freqs = freqs, n_cycles= n_cycles,average=False, 
                        use_fft = True,return_itc = False, decim = 1, n_jobs=4) 
    pred = time_gen.predict_proba(power3.data[:,:,0,:])
    predidx1 = np.where((epochs2.times >= 0.12) & (epochs2.times <= 0.27))
   # predidx2 = np.where(epochs3.times == 0.33)
    
    pred1 = np.mean(np.squeeze(pred[:,predidx1,:,:]),1)
    pred1 =  np.transpose(pred1,(0,2,1))
    plt.matshow(pred1[59],extent = (-475,476,-300,300))
    plt.yticks(ticks = np.arange(-200,201,200),labels = [3,2,1])
    plt.xticks(ticks = np.arange(-475,476,50),labels = epochs3.times[np.arange(0,951,50)])
    
    scores32 = time_gen.score(X=np.mean(power3.data,2),
                             y=[2]*len(power3.events[:, 2]))
    scores3x = time_gen.score(X=np.mean(power3.data,2),
                             y=power3.events[:, 2])
    
    fig, ax = plt.subplots(1)
    im = ax.matshow(scores3x, vmin=0.13, vmax=0.53, cmap='RdBu_r', origin='lower',
                    extent=np.concatenate((epochs3.times[[0, -1]],epochs.times[[0, -1]])))
    ax.axhline(0., color='k')
    ax.axvline(0., color='k')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Generalization across time and condition')
    plt.colorbar(im, ax=ax)
    plt.show()
    
    # pred2 = np.squeeze(pred[:,predidx2,:,:])
    # pred2 =  np.transpose(pred2,(0,2,1))
    # plt.matshow(pred2[59],extent = (-475,476,-300,300))
    # plt.yticks(ticks = np.arange(-200,201,200),labels = [3,2,1])
    # plt.xticks(ticks = np.arange(-475,476,50),labels = epochs3.times[np.arange(0,951,50)])
    
    #fig, ax = plt.subplots(1)
    # ax.plot(epochs.times,np.diag(pred[0,:,:,0]), color = 'b')
    # ax.plot(epochs.times,np.diag(pred[0,:,:,1]), color = 'r')
    # ax.plot(epochs.times,np.diag(pred[0,:,:,2]), color = 'k')

coef = get_coef(time_gen,'patterns_',inverse_transform = True)
tone1 = mne.EvokedArray(coef[:,0,:],epochs.info,tmin = epochs.times[0])
tone2 = mne.EvokedArray(coef[:,1,:],epochs.info,tmin = epochs.times[0])
tone3 = mne.EvokedArray(coef[:,2,:],epochs.info,tmin = epochs.times[0])

mne.viz.plot_evoked_topo([tone1,tone2,tone3])
mne.viz.plot_evoked_topo([epochs2['1'].average(),epochs2['2'].average(),
                          epochs2['3'].average()])

tone1a = mne.EvokedArray(np.squeeze(power2['1'].average().data),
                         epochs2.info,tmin = epochs2.times[0])
tone2a = mne.EvokedArray(np.squeeze(power2['2'].average().data),
                         epochs2.info,tmin = epochs2.times[0])
tone3a = mne.EvokedArray(np.squeeze(power2['3'].average().data),
                         epochs2.info,tmin = epochs2.times[0])
mne.viz.plot_evoked_topo([tone1a,tone2a,tone3a])
