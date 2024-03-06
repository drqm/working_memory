#%matplotlib inline
import sys
sys.path.append('src')
sys.path.append('py_scripts_for_dPCA_by_xxy')

from dpca_calculation import dpca_fit
import pickle
import basic_data_reshape
import plots
import state_space as ss
import decoding_functions as sd
import importlib
import warnings
importlib.reload(basic_data_reshape)
importlib.reload(plots)
importlib.reload(ss)
importlib.reload(sd)
from basic_data_reshape import *
from plots import *
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import matplotlib.pyplot as plt

ls = ['0011_U7X']#, '0012_VK2', '0013_NHJ', '0014_BKO', '0016_HJF', '0017_G8O']
#ls = ['0011_U7X']#,'0012_VK2']

if len(sys.argv) > 1:
    ls = sys.argv[1:]

host = 'aarhus' #'china'

if host == 'aarhus':
    wdir = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/'
    mf_dir = wdir + 'scratch/maxfiltered_data/tsss_st16_corr96/'
    ica_dir = wdir + 'scratch/working_memory/ICA/'
    log_dir = wdir + 'misc/working_memory_logs/'
    fig_dir = wdir + 'scratch/working_memory/dPCA/figures/'
    out_dir = wdir + 'scratch/working_memory/dPCA/data/'
elif host == 'china':
    wdir = '/Users/xiangxingyu/Downloads/毕业设计/UCB线上科研/data/'
    mf_dir = wdir + '01_raw_maxfiltered/'
    ica_dir = wdir + '02_ica_solution/'
    log_dir = wdir + 'working_memory_logs/'
    fig_dir = './'
    out_dir = './'
array_ls = []
tmin, tmax = -1,4
for i in ls:
    recall_epochs, man_epochs = reshape_to_epochs(tmin, tmax, _id_ = i, 
                                                  mf_dir=mf_dir,
                                                  ica_dir=ica_dir,
                                                  log_dir=log_dir,
                                                  baseline=(-1, 0),
                                                  notch_filter=50,
                                                  h_freq=20,
                                                  l_freq=0.1,
                                                  events_fun = main_task_decoding_events_fun,
                                                  resample=100)
    array1 = reshape_epoch_to_array(recall_epochs)
    array2 = reshape_epoch_to_array(man_epochs)
    tmp_array = concat_different_array(array1, array2)
    tmp_array, times = sd.smooth_data(tmp_array,-1, 0.025, 0.05, 100, len(tmp_array.shape)-1)
    array_ls.append(tmp_array)

# Get sampling frequency
sfreq = 1/np.diff(times)[0]

final_array_for_dpca, original_indices = stack_different_subject_arrays(array_ls)

print(final_array_for_dpca.shape)

# Clean the Nan arrays
#final_array_for_dpca = remove_nan_matrices(final_array_for_dpca)
print((final_array_for_dpca.shape))

# Fit dPCA
Z, dpca = dpca_fit(final_array_for_dpca, 'bst')
Z_dic = make_dic(Z, dpca, final_array_for_dpca)

#Inspect
print(Z.keys, type(dpca))
print(list(Z_dic.keys())[-10:])
print(Z_dic['average'].keys())
print(Z_dic['average']['t'].shape)

if len(ls) == 1:
    prefix = ls[0]
else:
    prefix = str(len(ls))

# Save array
with open(out_dir + prefix + '_dPCA.p', 'wb') as file:
    pickle.dump({'Z_dic': Z_dic, 'original_indices': original_indices,
                 'ls': ls, 'dpca': dpca, 'sfreq': sfreq, 'tmin': tmin}, file)

# Plot
plot_component_coordinate(Z_dic, times)
plt.savefig(fig_dir + prefix + '_subject_stack_average_component.pdf')
plt.close()

component_keys = ['b','s','bs']
three_d_plot_component_coordinate(Z_dic, component_keys)
plt.savefig(fig_dir +  prefix + '_subject_bs.pdf')

component_keys = ['bt','st', 'bst']
three_d_plot_component_coordinate(Z_dic, component_keys)
plt.savefig(fig_dir + prefix + '_subject_bst.pdf')

topo_PlotBasis(dpca, recall_epochs, fig_dir=fig_dir, sid=prefix, ch_type='mag')
topo_PlotBasis(dpca, recall_epochs, fig_dir=fig_dir, sid=prefix, ch_type='grad')

#### loo decoding #####

cvZ, mZ = ss.crossval_transform(final_array_for_dpca, 'bst', regularizer = dpca.regularizer,n_folds=3)
stat_comps_dic = {'st': [0,1,2],'bst':[0,1,2],'bt': [0,1,2], 'b': [0,1,2],'s': [0,1,2], 't': [0,1,2]}
#stat_comps_dic = {'st': [1]}#,'bst':[0,1,2],'bt': [0,1,2], 'b': [0,1,2],'s': [0,1,2], 't': [0,1,2]}
bacc, macc, intacc = ss.classify_trials('cvZ','mZ', stat_comps=stat_comps_dic)

with open(out_dir + prefix + '_dPCA_crossval.p', 'wb') as file:
    pickle.dump({'cvZ': cvZ, 'mZ': mZ, 'sfreq': sfreq, 'tmin': tmin}, file)

## Plot accuracy
#times = recall_epochs.times
accs = {'block': bacc, 'melody': macc, 'interaction': intacc}
ncols = 3
nrows = np.ceil(len(accs)/ncols).astype(int)
pcount = 0
fig = plt.figure(figsize=(ncols*6,nrows*8))
for cacc in accs:
    pcount += 1
    cax = plt.subplot(nrows,ncols,pcount)
    cax.plot(times, np.mean(accs[cacc],axis=(0,1,2)), color='k')
    cax.set_ylim(0,1)
    cax.set_xlim(times[0],times[-1])
    cax.axhline(0.5,color='k',alpha=.5)
    cax.axhline(0.25,color='k',alpha=.5)
    cax.axvline(0,color='k',alpha=.5)
    cax.axvline(2,color='k',alpha=.5)
    cax.set_title(cacc)
plt.tight_layout()
plt.savefig(fig_dir + prefix + '_accuracies.pdf')
