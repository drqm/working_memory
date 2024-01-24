from basic_data_reshape import *
from dpca_calculation import dpca_fit
import pickle
import plots
import importlib

import basic_data_reshape

importlib.reload(basic_data_reshape)

from basic_data_reshape import *

import matplotlib.pyplot as plt

import mne
mne.ioff()


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

# ls is the subject list
ls = ['0011_U7X', '0012_VK2', '0013_NHJ', '0014_BKO', '0016_HJF', '0017_G8O'] 

# "sort_trial_data_to_subject" is a new function in basic_data_reshape.py to sort trials with subjects,
# "original_indices" is another output from "stack_different_subject_arrays"
dic_sorted_by_subject = sort_trial_data_to_subject(Z_dic, original_indices, ls)


tmp_dir = '/Users/xiangxingyu/Downloads/毕业设计/UCB线上科研/data/'
inv_types = ['all', 'grad', 'mag']  # Default type 'all', see row 65

for subject in dic_sorted_by_subject:
    
    #subject = '0011_U7X'
    
    print('Processing', subject)
    # recall_epochs, man_epochs = reshape_to_epochs(-1, 4, _id_ = i, baseline=(-1, 0), notch_filter=50, h_freq=20, l_freq=0.1, events_fun = main_task_decoding_events_fun, resample=100)

    with open(f'pkls/{subject}recall_epochs.pkl', 'rb') as file:
        recall_epochs = pickle.load(file)
    with open(f'pkls/{subject}man_epochs.pkl', 'rb') as file:
        man_epochs = pickle.load(file)

    inv_op_file_path = f'{tmp_dir}04_inverse_operator/{subject}_decoding_inv_task_lf_0.05_hf_None_{inv_types[0]}.p'
    fwd_file_path = tmp_dir +f'03_forward_model/{subject}_vol-fwd.fif'

    with open(inv_op_file_path, 'rb') as file:
        lcmv_beamformer = pickle.load(file)

    fwd = mne.read_forward_solution(fwd_file_path)
    src = fwd['src']

    if not os.path.exists(f'figs/source_fig/{subject}'):
            os.mkdir(f'figs/source_fig/{subject}')

    trials = dic_sorted_by_subject[subject]

    average_trial = {key: np.zeros_like(value) for key, value in trials[next(iter(trials))].items()}
    # Sum Up
    for trial in trials.values():
        for key in trial:
            average_trial[key] += trial[key]
    # Average
    for key in average_trial:
        average_trial[key] /= len(trials)

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

    evoked_dict = {}

    # CAUTION: HERE I ONLY TAKE RECALL INFO TO BUILD EVOKED
    info = recall_epochs.info
    picked_channels = mne.pick_types(info, meg=True, eeg=False, stim=False, eog=False, exclude='bads')
    info_picked = mne.pick_info(info, picked_channels)

    for component in original_dic:
        for comp_idx in range(10):  # iterate 10 components
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

    # Now each key in evoked_dict is of the form 'component_compIdx' and each key corresponds to the result of a component
    # Next is the plotting part, which again needs to loop over the components
    for comp_key in evoked_dict:
        for key in evoked_dict[comp_key]:
            evoke = evoked_dict[comp_key][key]
            stc = mne.beamformer.apply_lcmv(evoke, lcmv_beamformer)
            subjects_dir = tmp_dir + 'fs_subjects_dir/'

            component_dir = f'figs/source_fig/{subject}/{comp_key}'  # comp_key include component information
            if not os.path.exists(component_dir):
                os.makedirs(component_dir)

            # Plot and save topomap
            
            time_interval = 1 # second
            
            min_time, max_time = evoke.times[0], evoke.times[-1]
            time_points = np.arange(min_time, max_time, time_interval)
            n_maps = len(time_points)
            fig, axes = plt.subplots(1, n_maps, figsize=(7 * n_maps, 7), dpi = 100)
            if n_maps == 1:
                axes = [axes]
            for ax, time_point in zip(axes, time_points):
                evoke.plot_topomap(times=time_point, axes=ax, show=False, colorbar=False)
            fig.tight_layout()
            fig_path_glass = f'{component_dir}/topomap_{key}.png'
            fig.savefig(fig_path_glass)

            # Plot and save the glass brain image
            brain_glass = stc.plot(src=src, subjects_dir=subjects_dir, subject=subject, mode='glass_brain')
            fig_path_glass = f'{component_dir}/glass_brain_{key}.png'  # Updated path
            brain_glass.savefig(fig_path_glass)

            # Plot and save the stat map image
            brain_stat = stc.plot(src=src, subjects_dir=subjects_dir, subject=subject, mode='stat_map')
            fig_path_stat = f'{component_dir}/stat_map_{key}.png'  # Updated path
            brain_stat.savefig(fig_path_stat)
