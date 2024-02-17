#%matplotlib inline
import sys
sys.path.append('src')
sys.path.append('py_scripts_for_dPCA_by_xxy')

from dpca_calculation import dpca_fit
import pickle
import plots
import importlib
import mne
import matplotlib.pyplot as plt
import basic_data_reshape
importlib.reload(basic_data_reshape)
from basic_data_reshape import *

#mne.ioff()

# Suppose that we've already get the Z_dic    

'''Z_dic: {'average' : {'b': [10,2,2,501],
                        's': [10,2,2,501],
                        ...
                        'bst': [10,2,2,501]
                        },
            'trial0': {'b': [10,2,2,501],
                        's': [10,2,2,501],
                        ...
                        'bst': [10,2,2,501]
                        },
            'trial1':  {'b': [10,2,2,501],
                        's': [10,2,2,501],
                        ...
                        'bst': [10,2,2,501]
                        },
             ...
             
             'trialn': ...
                    
                    }'''

# Suffix
# # ls is the subject list
# ls = ['0011_U7X', '0012_VK2', '0013_NHJ', '0014_BKO', '0016_HJF', '0017_G8O'] 
input_subject = '71'#'71'
if len(sys.argv) > 1:
    input_subject = sys.argv[1]
    
host = 'aarhus'
if host == 'aarhus':
    wdir = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/'
    fig_dir = wdir + 'scratch/working_memory/dPCA/figures/'
    out_dir = wdir + 'scratch/working_memory/dPCA/data/'
    inv_dir = wdir + 'scratch/working_memory/averages/data/'
    fwd_dir = wdir + 'scratch/forward_models/'
    subjects_dir = wdir + 'scratch/fs_subjects_dir/'
    
elif host == 'china':
    wdir = '/Users/xiangxingyu/Downloads/毕业设计/UCB线上科研/data/'
    fig_dir = './'
    out_dir = './'
    inv_dir = '04_inverse_operator/'
    fwd_dir = '03_forward_model/'
    subjects_dir = wdir + 'fs_subjects_dir/'
    
tmp_dir = wdir
with open(out_dir + input_subject + '_dPCA.p', 'rb') as Z_file:
    allZ = pickle.load(Z_file)

Z_dic = allZ['Z_dic']
original_indices = allZ['original_indices']
ls = allZ['ls']
dpca = allZ['dpca']

SBJ_sel = ls #['0011_U7X']
if len(sys.argv) > 2:
    SBJ_sel = sys.argv[2:]
    
# Group by subject:
dic_sorted_by_subject = sort_trial_data_to_subject(Z_dic, original_indices, ls)

inv_type = 'grad' #'all','mag','grad'  # Default type 'grad', see row 65

for subject in SBJ_sel:
    
    # Select file to retrive info (depending on where the analysis is run)
    if host == 'aarhus':
        info_file = f'{inv_dir}{subject}/{subject}_patterns_task_sensor_lf_0.05_hf_None_tstep_0.025_twin_0.05.p'
        with open(info_file, 'rb') as file:
            info_data = pickle.load(file)
        info_data = info_data['maintenance1']
        
    elif host == 'china':
        info_file = f'pkls/{subject}recall_epochs.pkl'
        with open(info_file, 'rb') as file:
            info_data = pickle.load(file)

    #load inverse operator
    inv_op_file_path = f'{inv_dir}{subject}/{subject}_decoding_inv_task_lf_0.05_hf_None_{inv_type}.p'
    with open(inv_op_file_path, 'rb') as file:
        lcmv_beamformer = pickle.load(file)
    
    # Prepare trials
    trials = dic_sorted_by_subject[subject]
    average_trial = {key: np.zeros_like(value) for key, value in trials[next(iter(trials))].items()}
    
    # Sum Up
    for trial in trials.values():
        for key in trial:
            average_trial[key] += trial[key]
    # Average
    for key in average_trial:
        average_trial[key] /= len(trials)

    # Inverse transform
    original_dic = {} #original_dic['b'].shape (10, 306, 2, 2, 501)
    for component in average_trial:

        print(trial[component].shape) # (10, 2, 2, 501)
        data_shape = trial[component].shape

        tmp_array = np.zeros([data_shape[0], 306, data_shape[1], data_shape[2], data_shape[3]])

        for i in range(data_shape[0]):
            component_for_inverse = np.zeros([data_shape[0], data_shape[1], data_shape[2], data_shape[3]])
            component_for_inverse[i] = trial[component][i]
            tmp_array[i] = dpca.inverse_transform(component_for_inverse,component)

        original_dic[component] = tmp_array # (10, 306, 2, 2, 501)

    # Transformed to evoked arrays for source localization
    evoked_dict = {}

    info = info_data.info # Retrieve info from corresponding file
    picked_channels = mne.pick_types(info, meg=True, eeg=False, stim=False, eog=False, exclude='bads')
    info_picked = mne.pick_info(info, picked_channels)
    
    # Transform selected components
    for component in ['st','bt','bst']:
        for comp_idx in range(3):  # iterate 10 components
            data = original_dic[component][comp_idx, :, :, :, :] 

            # Difference of Behavior
            data_diff_dim1 = data[:, 0, :, :].mean(axis=1) - data[:, 1, :, :].mean(axis=1)
            evoked_dim1 = mne.EvokedArray(data_diff_dim1, info_picked)
            evoked_dict[f'{component}_comp{comp_idx}'] = {} 
            evoked_dict[f'{component}_comp{comp_idx}']['diff_behavior'] = evoked_dim1

            # Difference of Stimuli
            data_diff_dim2 = data[:, :, 0, :].mean(axis=1) - data[:, :, 1, :].mean(axis=1)
            evoked_dim2 = mne.EvokedArray(data_diff_dim2, info_picked)
            evoked_dict[f'{component}_comp{comp_idx}']['diff_stimuli'] = evoked_dim2

            # Difference of Cross Melodies
            data_combined_1 = (data[:, 0, 0, :] + data[:, 1, 1, :]) / 2
            data_combined_2 = (data[:, 0, 1, :] + data[:, 1, 0, :]) / 2
            data_diff_combined = data_combined_1 - data_combined_2
            evoked_combined = mne.EvokedArray(data_diff_combined, info_picked)
            evoked_dict[f'{component}_comp{comp_idx}']['diff_cross'] = evoked_combined
    
    # Save sensor space
    print(f'saving topomaps for subject {subject}')   
    with open(f'{out_dir}/{subject}_component_sensors_dPCA_{input_subject}.p','wb') as tf:
        pickle.dump(evoked_dict,tf)
    
    # Do source localization
    sources = {}
    for comp_key in evoked_dict:
        sources[comp_key] = {}
        for key in evoked_dict[comp_key]:
            evoke = evoked_dict[comp_key][key]
            sources[comp_key][key] = mne.beamformer.apply_lcmv(evoke, lcmv_beamformer)
        
    # Save source space
    print(f'saving sources for subject {subject}')        
    with open(f'{out_dir}/{subject}_component_sources_dPCA_{input_subject}.p','wb') as sf:
        pickle.dump(sources,sf)
