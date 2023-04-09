import numpy as np
import mne
import matplotlib.pyplot as plt
from pickle import dump
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_selection import SelectKBest, f_classif
from mne.decoding import GeneralizingEstimator, get_coef, LinearModel,cross_val_multiscore,SlidingEstimator

def smooth_data(data, tstart, tstep, twin, Fs, taxis=2):
    
    # get data shape
    old_dims = data.shape
    
    # Arrange dimensions in standard form
    new_dimord = np.array([taxis] + [d for d in range(len(old_dims)) if d != taxis])
    old_dimord = np.argsort(new_dimord)
    data = np.transpose(data,new_dimord)
    new_dims = data.shape
    
    # Calculate old and new time vectors
    tend = tstart + new_dims[0] / Fs
    ctime = np.arange(tstart, tend + 1/Fs, 1/Fs)
    ntime = np.arange(tstart + twin/2, tend-twin/2 + 1/Fs, tstep)
    
    # Initialize output data
    new_data = np.ones((tuple([len(ntime)]) + new_dims[1:])) * np.nan
    
    # Loop over timesteps and smooth
    for ntix, nt in enumerate(ntime):
        lims = np.array([nt - twin / 2, nt + twin / 2]) # Current interval to average
        cix = [np.argmin(np.abs(l - ctime)) for l in lims] # Limit indices
        new_data[ntix] = np.mean(data[cix[0]:(cix[1]+1)],0) # Average interval and store
    
    # Reorder dimesions and return
    new_data = np.transpose(new_data, old_dimord)
    return new_data, ntime

def WM_time_gen_classify(epochs, mode='sensor', kind = 'Generalizing', inv = None, lmask=[], score = True, n_features = 'all',
                         twindows = None, l_freq=None, h_freq =None, smooth=None, save_filters=None,
                         save_scores = None, save_gen=None, save_patterns=None, save_times=None, penalty='l2'):
        
        ## Prepare the data and get timepoints
        if isinstance(epochs, mne.Epochs):
            epochs = {'epochs': epochs}
        
        if isinstance(twindows, list):
            cwin = twindows.copy()
            twindows = {e: cwin for e in epochs} 
        
        if twindows: 
            for e in epochs:
                epochs[e].crop(twindows[e][0], twindows[e][1])   
        
        # Get time vector per condition
        times = {e: epochs[e].times for e in epochs}
        
        # Get data depending on mode (source vs sensor)
        data, labels, containers = {}, {}, {}
        for e in epochs:
            if mode == 'source':
                csource = mne.beamformer.apply_lcmv_epochs(epochs[e],inv)
                containers[e] = csource[0].copy()
                data[e] = np.array([cs.data for cs in csource])
                if len(lmask)>0:
                    data[e] = data[e] * lmask
                    
            elif mode == 'sensor':
                data[e] = epochs[e].get_data()
                containers[e] = epochs[e].average().copy()
            labels[e] = epochs[e].events[:,2]
            
        # Perform filtering if required
        if l_freq or h_freq:
            for e in data:
                data[e] = mne.filter.filter_data(data[e], epochs[e].info['sfreq'],
                                                 l_freq, h_freq, n_jobs = 8)
                    
        # Perform smoothing if requried
        if smooth:
            for e in data:
                data[e], times[e] = smooth_data(data[e], tstart = times[e][0],
                                                **smooth, Fs = epochs[e].info['sfreq'])
                
        # Initialize output:
        gen, patterns, filters, scores = {}, {}, {}, {}
        
        # Start classifier object:
        clf = make_pipeline(StandardScaler(),#SelectKBest(f_classif, k=n_features),
                            LinearModel(LogisticRegression(penalty=penalty,solver='liblinear')))
        
        # Loop over conditions
        for e in data:
            # Start and fit generalizing estimator
            print('fitting ', e)
             
            if kind == 'Generalizing':
                gen[e] = GeneralizingEstimator(clf, n_jobs=2, scoring = 'balanced_accuracy', verbose = True)
            elif kind == 'Sliding':
                gen[e] = SlidingEstimator(clf, n_jobs=2, scoring = 'balanced_accuracy', verbose = True)
            
            gen[e].fit(X = data[e], y = labels[e])   
                
            ## Get patterns and filters
            print('extractig patterns and filters')
            cpatterns = get_coef(gen[e],'patterns_',inverse_transform=True)
            cfilters = get_coef(gen[e],'filters_', inverse_transform=True)

            # If only two classes (one set of patterns), expand dimensions for coherence with loops below:
            if len(cpatterns.shape) < 3:
                cpatterns = np.expand_dims(cpatterns,1)
                cfilters = np.expand_dims(cfilters,1)

            # Export patterns and filters to evoked or sources
            if mode == 'source':
                # Loop over classes:
                for cl in range(cpatterns.shape[1]):
                    cname = e + str(cl+1)
                    patterns[cname] = containers[e]
                    patterns[cname].tstart = times[e][0]
                    patterns[cname].tstep = np.diff(times[e])[0]
                    filters[cname] = patterns[cname].copy()
                    patterns[cname].data = cpatterns[:,cl,:].copy()
                    filters[cname].data = cfilters[:,cl,:].copy()
                
            elif mode == 'sensor':
                cinfo = containers[e].info.copy()
                cinfo['srate'] = 1 / np.diff(times[e])[0]
                # Loop over classes:
                for cl in range(cpatterns.shape[1]):
                    cname = e + str(cl+1)
                    patterns[cname] = mne.EvokedArray(cpatterns[:,cl,:].copy(), info = cinfo, baseline = None,
                                                      tmin=times[e][0], comment=e)
                    filters[cname] = patterns[cname].copy()
                    filters[cname].data = cfilters[:,cl,:].copy()
            
            # Obtain scores looping over conditions to test:
            for e2 in data:
                scond = e2 + '_from_' + e
                print('scoring ', scond)
                if e != e2:
                    # If different conditions, perform test
                    scores[scond] = gen[e].score(data[e2], labels[e2])
                else:
                    # If same condition, cross-validate
                    scores[scond] = cross_val_multiscore(gen[e], data[e],labels[e], cv = 5,n_jobs = 5).mean(0)
        
        ## Save output if requried
        
        if save_gen:
            print('saving models')
            gen_file = open(save_gen,'wb')
            dump(gen,gen_file)
            gen_file.close()

        if save_patterns:
            print('saving patterns')
            pat_file = open(save_patterns,'wb')
            dump(patterns,pat_file)
            pat_file.close()

        if save_filters:
            print('saving filters')
            fil_file = open(save_filters,'wb')
            dump(filters,fil_file)
            fil_file.close()
        
        if save_scores:
            print('saving scores')
            score_file = open(save_scores,'wb')
            dump(scores,score_file)
            score_file.close()
        
        if save_times:
            print('saving times')
            times_file = open(save_times,'wb')
            dump(times, times_file)
            times_file.close()
            
        return gen, patterns, filters, scores, times

def plot_time_gen_accuracy(scores, times, masks = None, nrows=2,
                           ncols=2,vlines=[],hlines=[],
                           savefig=None, vmin=None,vmax=None):

    #Create figure:
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows*8,ncols*8))
    
    # Loop over conditions:
    for six, s in enumerate(scores):
        
        # Get plot locations:
        x, y = six // ncols, six % ncols
        
        # Select axis:
        if nrows == 1:
            if ncols == 1:
                cax = axes
            else:
                cax = axes[y]
        else:
            cax = axes[x,y]
        
        # Get axes tick labels from condition times:
        #tr,te = s.split('_from_')
        strs = s.split('_')
        te, tr = strs[0], strs[2]
        tx1, tx2 = times[te][[0,-1]]
        ty1, ty2 = times[tr][[0,-1]]
        extent=[tx1, tx2, ty1, ty2]
                   
        # Make a plot
        im = cax.matshow(scores[s], vmin=vmin, vmax=vmax, cmap='RdBu_r', origin='lower',
                               extent=extent, aspect='auto',alpha=1)
        
        # Add mask contour if required (useful to display significant values)
        if masks:
            cax.contour(masks[s].copy().astype('float'), levels=[-.1,1], colors='k',
                        extent=extent, origin='lower',corner_mask=False)
            
        # Axis lines:
        cax.axhline(0., color='k')
        cax.axvline(0., color='k')
        
        # Additional lines:
        for vl in vlines:
            cax.axvline(vl, color='k')
        
        for hl in hlines:
            cax.axhline(hl, color='k')            
        
        # Customize
        cax.xaxis.set_ticks_position('bottom')
        cax.set_xlabel('Testing Time (s)')
        cax.set_ylabel('Training Time (s)')
        cax.set_title(s)        
        plt.colorbar(im, ax=cax)
    
    plt.tight_layout()
    
    if savefig:
        plt.savefig(savefig)

def plot_diagonal_accuracy(scores, times, CI = None, masks = None, nrows=2,
                           ncols=2,vlines=[],hlines=[],chance=1/2,
                           savefig=None, ylims = None, color = 'k'):
    
    #Create figure:
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows*6,ncols*6))
    for six, s in enumerate(scores):
        
        # Get plot locations:
        x, y = six // ncols, six % ncols
        
        # Select axis:
        if nrows == 1:
            if ncols == 1:
                cax = axes
            else:
                cax = axes[y]
        else:
            cax = axes[x,y]
        
        # Get axes tick labels from condition times:
        strsp = s.split('_from_')
        tr = strsp[0]
        te = strsp[1]
        # Setup masks
        if masks:
            cmask = masks[s].copy().astype('float')
        else:
            cmask = np.zeros(scores[s].shape,dtype='float')
        
        cmask[cmask == 0] = np.nan
        # Manage two-dimensional arrays
        if len(scores[s].shape) > 1:
            tseries = np.diagonal(scores[s])
            tseries_mask = np.diagonal(scores[s]*cmask)
        else: 
            tseries = scores[s]
            tseries_mask = scores[s]*cmask
        
        # Plot
        cax.plot(times[tr], tseries,color=color) #[:,range(scores[s].shape[0])].)
        cax.plot(times[tr], tseries_mask, linewidth=4,color=color)
        if CI:
            cax.fill_between(times[tr],  
                             np.diagonal(CI[s]['lower']),
                             np.diagonal(CI[s]['upper']),
                             alpha=.2,
                             color=color)
            
        cax.axvline(0., color='k')
        cax.axhline(chance, color='k')
        
        # Additional lines:
        for vl in vlines:
            cax.axvline(vl, color='k')
        
        for hl in hlines:
            cax.axhline(hl, color='k') 
        
        cax.set_xlabel('time')
        cax.set_ylabel('accuracy')
        cax.set_xlim((times[tr][0], times[tr][-1]))
        if ylims:
            cax.set_ylim(ylims)
    plt.tight_layout()

def get_probs(gen, epochs, train_times, blocks = None):
    
    ntt = len(train_times) # number of training time points
    
    # Select all blocks if not given
    if blocks == None:
        blocks = [s for s in epochs]
    
    # Initialize output
    times, probs, events = {}, {}, {}     
    
    # Loop over blocks and extract probabilities
    for b in blocks:
        
        times[b] = epochs[b].times # Get time courses
        events[b] = epochs[b].events # Get events info
        probs[b] = np.zeros((ntt,epochs[b].get_data().shape[0],len(times[b]),3)) # Initialize probabilities
        
        # Loop over timepoints models
        for t in range(len(times[b])):
            
            # Loop over training time points and estimate probabilities
            for te in range(ntt):
                probs[b][te,:,t,:] = gen.estimators_[train_times[te]].predict_proba(epochs[b].get_data()[:,:,t])
                
    return probs, times, events
    
    
def sequence_betas(probs, delays):
    
    #Initialize output
    betas = []
    
    #Loop over training times
    for m in range(probs.shape[0]):
        
        cbetas = [] #training time specific betas        
        
        #Loop over trials
        for e in range(probs.shape[1]):
            print('Model ', m+1, ' trial ', e+1)
            dbetas = [] # trial specific betas
            
            # Loop over delays
            for dt in delays:
                cbeta = np.zeros([probs.shape[-1],probs.shape[-1]]) # Initialize beta-specific delay
                ptc = np.squeeze(probs[m,e,:,:]) # Get current probability time courses
                X = np.zeros(ptc.shape) # Initialize design matrix
                X[dt:,:] = ptc[0:(ptc.shape[0]-dt),:] # Apply delay and padding
                
                # Loop over states
                for i in range(probs.shape[-1]): # 
                    y_i = ptc[:,i] # Get current state timecourse
                    reg = LinearRegression().fit(X,y_i) #Regress
                    cbeta[i] = reg.coef_ #Get coefficients
                    
                dbetas += [cbeta] # Append current delay
            cbetas += [dbetas] # Append current trial
        betas += [cbetas] # Append current training time
    betas = np.array(betas) # Convert to numeric array
    
    return betas
       
def sequenceness(betas, patterns):
    # Initialize output
    S = np.zeros(list(betas.shape[0:3])+[patterns.shape[1]])
    
    # Loop over training times
    for m in range(S.shape[0]):
        
        # Loop over trials
        for e in range(S.shape[1]):
            
            #loop over delays
            for d in range(S.shape[2]):
                # Fit model
                S[m,e,d,:] = np.linalg.inv((patterns.T@patterns))@patterns.T@betas.reshape(list(betas.shape[0:3])+[-1])[m,e,d]
    return S

def seq_permutation_within(betas, patterns, reg_ix= {'forward': 2, 'backward': 3}, nperm = 100, event_sel = None):
    
    if not event_sel:
        event_sel = {'all': np.arange(betas.shape[1])}
    
    # Initialize output
    stat, pval, qval, null = {}, {}, {}, {}
    
    # Loop over event selections
    for es in event_sel:
        
        print('testing events: '+ es)

        # Select events:
        cbetas = betas[:,event_sel[es],:,:]

        # Calculate current sequenceness:
        stat[es] = np.mean(sequenceness(cbetas, patterns),1,keepdims=False)

        
        # Initialize null dist and pvals for current event selection
        null[es], pval[es], qval[es] = {}, {}, {}
        
        # Loop over regressors to test
        for rname in reg_ix:
            
            cpatterns = patterns.copy()

            print('testing regressor: ' + rname)
            null[es][rname] = np.zeros((nperm, betas.shape[0],betas.shape[2],patterns.shape[1]))
            pval[es][rname + '_pos'] = np.zeros((betas.shape[0],betas.shape[2]))
            pval[es][rname + '_neg'] = np.zeros((betas.shape[0],betas.shape[2]))
            qval[es][rname + '_pos'] = np.zeros((betas.shape[0],betas.shape[2]))
            qval[es][rname + '_neg'] = np.zeros((betas.shape[0],betas.shape[2]))
            
            r = reg_ix[rname]
            r_alt = np.array([reg_ix[rn] for rn in reg_ix if rn != rname]) # Find index of the other regressors

            # Loop over permutations
            for n in np.arange(nperm):
                print('permutation ',n+1,' / ', nperm)
                # Permute patterns and make sure the pattern is not the same as the original
                probe = None
                while probe == None:
                    cpat = patterns[:,r].copy()
                    np.random.shuffle(cpat)
                    #print(cpat)
                    # Check permutated pattern does not overlap with intercept, diagonal or alternative patterns
                    if ((np.sum(cpat * patterns[:,r]) < 2) and (np.sum([cpat*patterns[:,pp] for pp in r_alt]) < 2) 
                    and (np.sum(cpat * patterns[:,1]) < 3)):                        
                        cpatterns[:,r] = cpat
                        probe = 1
                null[es][rname][n] = np.mean(sequenceness(cbetas, cpatterns),1,keepdims=False)
            
            # Loop over training times
            print('calculating pvals')
            for e in range(betas.shape[0]):
                
                # Loop over delays
                for d in range(betas.shape[2]):
                    # Calculate pvalues
                    pval[es][rname + '_pos'][e,d] = np.sum(null[es][rname][:,e,d,r] >= stat[es][e,d,r]) / nperm
                    pval[es][rname + '_neg'][e,d] = np.sum(null[es][rname][:,e,d,r] <= stat[es][e,d,r]) / nperm
                    
            qval[es][rname + '_pos'] = mne.stats.fdr_correction(pval[es][rname + '_pos'], .025)
            qval[es][rname + '_neg'] = mne.stats.fdr_correction(pval[es][rname + '_neg'], .025)
            
    return {'stat': stat, 'pval': pval, 'qval': qval, 'null': null}  
    