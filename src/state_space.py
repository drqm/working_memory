import numpy as np
from dPCA import dPCA

def crossval_transform(trls, labels, protect = 't', n_components=10,  regularizer=0., n_folds = 5, exclude_nan=True):

    """This function takes a set of trials in sensor space and projects them to 
    demixed principal component (dPCA) space using leave-one-out cross-validation.
    
    Parameters
    ----------
    trls: ndarray
        The shape of the array is (ntrials, nchannels, ncond1, ncond2,...,ntimes)
        and corresponds to the same shape of trial data for dPCA analyses e.g. (30,306,2,2,501) 
    labels: str
        Name of the different factors included in dPCA. Each letter corresponds to a factor.
        For example bmt (block, melody, time)
    protect: str
        Axes to protect from shuffling, typically time (t), which is the default.
    n_components: int
        number of dPCA components for each combination of factors. Defaults to 10.
    regularizer: float
        A float number indicating the strenght of the regularization parameter "lambda"
        Typically, this would be a number obtained from a previous dPCA fit.
    exclude_nan: bool
        Whether to exclude rows with nan values from cross-validation. Recomended (default). Note that NaNs
        are excluded from the test set but not from the training set, to maximize the information used for
        training. 

    Returns
    ----------
    cvZ: dict
         Dictionary containing the test data projected into dPCA space (Z). Each dictionary key corresponds to
         a combination of the factors indicated in "labels", for example ('t','m','b','bm','bt','mt','bmt').
         Each entry in the dictionary is and ndarray of shape (ntrials, n_components, ncond1, ncond2,...,ntimes)
    mZ: dict
         Dictionary containing the training data projected into dPCA space. Each dictionary key corresponds to
         a combination of the factors indicated in "labels", for example ('t','m','b','bm','bt','mt','bmt').
         Each entry in the dictionary is and ndarray of shape (ntrials, n_components, ncond1, ncond2,...,ntimes). Note that,
         for each trial in the data, a slighlty different training set is used in leave-one-out cross-validation.
         By calculating the trial-wise distance between cvZ and mZ, test trials can be used to decode stimuli and conditions.
         See classify_trials for details.
    """

    cvZ = {} # initialize projected test data
    mZ = {} # Initialize projected train data
    nanix = np.unique(np.where(np.isnan(trls))[0]) # Index of nan trials
    
    # Get index of trials with no NaN (no_nanix):
    no_nanix = np.arange(trls.shape[0]) # if exclude_nan != True, then use all trials
    if (exclude_nan) & (len(nanix > 0)): # If exclude_nan == True and there are NaNs in data, then get index of not NaN trials
        no_nanix = np.arange(np.min(nanix))
    
    if n_folds == 'loo':
        n_folds = len(no_nanix)

    rand_ix = np.random.choice(no_nanix, no_nanix.shape[0], replace=False)
    orig_ix = np.argsort(rand_ix)
    n_test = np.round(len(rand_ix)/n_folds).astype(int)
    tcount = 0
    print('\ncross-validation\n')
    for cf in range(n_folds):
        
        tstart, tend = cf*n_test, (cf+1)*n_test
        if cf  == (n_folds-1):
            tend = len(rand_ix)
        cfold_ix = rand_ix[tstart:tend] 
        curdata = trls.copy()
        curtrls = trls.copy()[cfold_ix]
        curdata[cfold_ix] = np.nan # Exclude test trials
        curmean = np.nanmean(curdata,axis=0) # Mean of trial data
        curmean -= np.nanmean(curmean.reshape((curmean.shape[0],-1)),1)[:,None,None,None] # Center mean data
        curtrls -= np.nanmean(curtrls.reshape((curtrls.shape[0],curtrls.shape[1],-1)),axis=(0,2))[None,:,None,None,None] # Center test trls
        curtrls[np.isnan(curtrls)] = 0 # If exclude_nan != True, replace test nan with 0
        dpca_cv = dPCA.dPCA(labels = labels, regularizer=regularizer,n_components=n_components)
        dpca_cv.protect = [protect] # Set protected axis (usually t)
        meanZ = dpca_cv.fit_transform(curmean)#, curdata) # fit and transform train data
        for trlix in range(curtrls.shape[0]):
            tcount += 1
            print(f'testing fold {cf + 1} - trial {tcount} ({cfold_ix[trlix]+1})')
            curtrl = curtrls.copy()[trlix] 
            cZ = dpca_cv.transform(curtrl)
            for cz in cZ:
            # Set defaults if first trial
                cvZ.setdefault(cz,[])
                mZ.setdefault(cz, [])

                # Append
                cvZ[cz] += [cZ[cz]]
                mZ[cz] += [meanZ[cz]]

    cvZ = {cz: np.array(cvZ[cz])[orig_ix] for cz in cvZ}
    mZ = {cz: np.array(mZ[cz])[orig_ix] for cz in mZ}
              
    return cvZ, mZ

def classify_trials(cvZ0,mZ0,stat_comps=None):

    """This function takes a set of test trials projected into dPCA space, calculates the euclidian distance to their
    corresponding projected mean train data, and classifies them according to minimum distance.
    Different factors and components within each factor can be used for classification. Currently, this function is
    optimized to obtain accuracy of dPCA trial projections of shape (ntrials, n_components, 2, 2, ntimes), with
    two factors labeled "b" and "m" with 2 conditions each.
    
    Parameters
    ----------
    cvZ0: dict
        Dictionary containing the test data projected into dPCA space (Z). Each dictionary key corresponds to
        a combination of the factors indicated in "labels", for example ('t','m','b','bm','bt','mt','bmt').
        Each entry in the dictionary is and ndarray of shape (ntrials, n_components, ncond1, ncond2,...,ntimes)
    mZ0: dict
        Dictionary containing the training data projected into dPCA space. Each dictionary key corresponds to
        a combination of the factors indicated in "labels", for example ('t','m','b','bm','bt','mt','bmt').
        Each entry in the dictionary is and ndarray of shape (ntrials, n_components, ncond1, ncond2,...,ntimes). Note that,
        for each trial in the data, a slighlty different training set is used in leave-one-out cross-validation.
        By calculating the trial-wise distance between cvZ and mZ, test trials can be used to decode stimuli and conditions.
    stat_comps: dict
        Dictionary indicating which factors and components within each factor to use for classification. For example,
        stat_comps={'bt': [0,1,2], 'mt': [0,3]} will use the first three components of factor bt and components 1 and 4
        of factor mt. If None (default), it will use all factors and components present in cvZ0. 

    Returns
    ----------
    bacc: ndarray (ntrials, nconds1, nconds2, ntimes)
         trial level accuracy for binary classification of first factor (e.g., b or block). 1 = correct, 0 = incorrect. (Chance = 0.5)
    macc: ndarray (ntrials, nconds1, nconds2, ntimes) 
         trial level accuracy for binary classification of second factor (e.g., m or melody). 1 = correct, 0 = incorrect. (Chance = 0.5)
    intacc: ndarray (ntrials, nconds1, nconds2, ntimes)
        trial level accuracy for 4-way classification of the two factors (e.g., b and m). 1 = correct, 0 = incorrect. (chance = 0.25)     
    """

    # Set default components to use:
    if not stat_comps:
        # stat_comps = {cz: np.array([0,1,2,3]) for cz in cvZ0}

        stat_comps = {key: np.array([0, 1, 2]) for key in cvZ0.keys()}

    # Concatenate trials from the indicated components
        
    trial_stack = [] # Initialize test trials
    mean_stack = [] # Initialize train trials

    # Loop and append components
    for sc in stat_comps:
        trial_stack += [cvZ0[sc][:,stat_comps[sc]]]
        mean_stack += [mZ0[sc][:,stat_comps[sc]]]

    # Concatenate to ndarray
    trial_stack = np.concatenate(trial_stack,axis=1)
    mean_stack = np.concatenate(mean_stack,axis=1)

    # Initialize distances
    distances = []

    # Loop over conditions within each factor
    for b in range(mean_stack.shape[2]):
        for m in range(mean_stack.shape[3]):

            # Train data mean for current b and m
            ccm = mean_stack[:,:,b,m]

            # Euclidian distance between test trial and train mean
            cdist = np.sqrt(np.sum((trial_stack - ccm[:,:,None,None])**2,axis=1))
            distances += [cdist] # Store current distance

    distances = np.array(distances) # Convert distances to array
    min_dist = np.argmin(distances, axis=0) # Find minimum distance across the four (2X2) conditions

    # Initialize NaN ndarrays to store accuracies
    bacc = np.zeros(min_dist.shape)*np.nan
    macc = np.zeros(min_dist.shape)*np.nan
    intacc = np.zeros(min_dist.shape)*np.nan

    # Loop over conditions for each factor and calculate accuracy
    for b in range(mean_stack.shape[2]):
        for m in range(mean_stack.shape[3]):

            bacc[:,b,m] = min_dist[:,b,m] // 2 == b # Factor 1 (block) accuracy
            macc[:,b,m] = min_dist[:,b,m] % 2 == m # Factor 2 (melody) accuracy
            intacc[:,b,m] = min_dist[:,b,m] == b*2 + m # 4-way Factor1 X Factor2 (block and melody) accuracy

    return bacc, macc, intacc