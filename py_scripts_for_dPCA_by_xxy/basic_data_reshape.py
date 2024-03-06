# Reshape MEG data to nparray for dPCA

#wdir = '/Users/xiangxingyu/Downloads/毕业设计/UCB线上科研/data/'

import sys
# sys.path.append('../src')
sys.path.append('/Users/xiangxingyu/Desktop/working_memory/src')
from preprocessing import *
import mne

from numpy import *
import os
import pickle

mne.set_log_level('CRITICAL')

def reshape_to_epochs(*args, _id_: str, mf_dir, ica_dir, log_dir, **kwargs):
    _id_number = _id_[:4]

    inv_fname = mf_dir + _id_ + '/inv_raw_tsss.fif'
    inv_ica_fname = ica_dir + _id_ + '/inv_raw_tsss-ica.fif'
    inv_lfname = log_dir + _id_number + '_invert_MEG.csv'

    main_fname = mf_dir + _id_ + '/main_raw_tsss.fif'
    main_ica_fname = ica_dir + _id_ + '/main_raw_tsss-ica.fif'
    main_lfname = log_dir + _id_number + '_recognize_MEG.csv'

    recall_epochs = WM_epoching(inv_fname, inv_ica_fname, *args,
                                events_fun_kwargs={'lfname': inv_lfname}, **kwargs)
    manipulate_epochs = WM_epoching(main_fname, main_ica_fname, *args,
                                    events_fun_kwargs={'lfname': main_lfname},**kwargs)

    return recall_epochs, manipulate_epochs


def reshape_epoch_to_array(epochs):
    event_types = list(epochs.event_id.keys())
    print(event_types)
    n_samples = {event: len(epochs[event]) for event in event_types}

    max_samples = max(n_samples.values())

    final_data = np.full((306, max_samples, len(event_types), 501), np.nan)  # [C, max_samples, 2(melody0,melody1)), T]

    for i, event in enumerate(event_types):
        event_data = epochs[event].get_data().transpose(1, 0, 2)
        n_current_samples = event_data.shape[1]
        # fill with nan
        if n_current_samples < max_samples:
            shortfall = max_samples - n_current_samples
            filler = np.full((306, shortfall, 501), np.nan)
            event_data = np.concatenate([event_data, filler], axis=1)
        final_data[:, :max_samples, i, :] = event_data # [C, max_samples, 2(melody0,melody1), T]

    return [max_samples, final_data]

def concat_different_array(array1, array2):

    max_samples = max(array1[0], array2[0])

    array_shape = array1[1].shape

    final_data = np.full((array_shape[0], max_samples, 2, array_shape[2], array_shape[3]), np.nan)

    final_data[:, :array1[0], 0, :, :]  = array1[1]
    final_data[:, :array2[0], 1, :, :] = array2[1]

    final_data = final_data.transpose(1, 0, 2, 3, 4) # (max_samples, 306, 2(recall/mainpulate), 2(melody0,melody1), 501)

    return final_data

def stack_different_subject_arrays(ls: list):
    
    # stack different subjcets' trials data
    
    stacked_array = np.vstack(ls)

    original_indices = np.concatenate([[i] * arr.shape[0] for i, arr in enumerate(ls)])

    nan_indices = [i for i in range(stacked_array.shape[0]) if np.isnan(stacked_array[i]).any()]

    non_nan_matrices = np.delete(stacked_array, nan_indices, axis=0)
    non_nan_original_indices = np.delete(original_indices, nan_indices)

    return non_nan_matrices, non_nan_original_indices

def remove_nan_matrices(array):    
    """
    remove Nan arrays

    input:
    array -- output from stack_different_subject_arrays

    return:
    a new array same shape as the original, but with no nan arrays
    """
    
    # It seems that this function is useless now

    nan_indices = find_nan_matrices(array)

    return array[~np.isin(np.arange(array.shape[0]), nan_indices)]

def find_nan_matrices(array):
    """
    find array with nan

    input:
    array -- output from stack_different_subject_arrays

    returns:
    nan indecies
    """
    
    # It seems that this function is useless now
    
    axes = tuple(range(1, array.ndim))

    nan_matrices = np.any(np.isnan(array), axis=axes)

    return np.where(nan_matrices)[0]

def make_dic(Z, dpca, final_data, *args, **kwargs): #final_data is by stack_different_subject_arrays; Z, dpca by dpca_calculation.dpca_fit
    dic = {}

    for i in range(len(final_data)):
        tmp_trial = dpca.transform(final_data[i])
#         for j in tmp_trial :
#             tmp_trial [j] = moving_average(tmp_trial[j], *args, **kwargs)

        dic[f'trial{i}'] = tmp_trial
    dic['average'] = Z

    return dic

def moving_average(arr, k = 40, padding='zero'):
    """
    Compute the moving average along the last dimension of the array with padding options.

    Parameters:
    arr -- input NumPy array
    k -- size of the averaging window
    padding -- strategy for padding when there are fewer than k elements (options: 'zero', 'None', 'mean')

    Returns:
    Array of moving averages computed along the last dimension
    """

    if k <= 0:
        raise ValueError("k shall be greater than 0")

    # Decide the padding strategy
    if padding not in ['zero', 'None', 'mean']:
        raise ValueError("padding must be 'zero', 'None', or 'mean'")

    # Initialize an output array of the same shape as input
    result = np.zeros_like(arr, dtype=float)

    # Iterate over the last dimension to compute the moving average
    for i in range(arr.shape[-1]):
        start = max(0, i - k + 1)
        end = i + 1

        # Apply different padding strategies
        if padding == 'None' or start != 0:
            result[..., i] = np.mean(arr[..., start:end], axis=-1)
        elif padding == 'zero':
            window_shape = list(arr.shape)
            window_shape[-1] = k
            window = np.zeros(window_shape, dtype=arr.dtype)
            slice_len = min(end, k)
            window[..., -slice_len:] = arr[..., start:end]
            result[..., i] = np.mean(window, axis=-1)
        elif padding == 'mean':
            window_shape = list(arr.shape)
            window_shape[-1] = k
            window = np.full(window_shape, np.mean(arr), dtype=arr.dtype)
            slice_len = min(end, k)
            window[..., -slice_len:] = arr[..., start:end]
            result[..., i] = np.mean(window, axis=-1)

    return result


def reshape_dpca_base_to_evoked_dic(dpca, num_components = 3, info = None):
    """
    Return:
    dic{ 'b':{ 'componentNo_1', 'componentNo_2', 'component' ...},
         's':{ 'componentNo_1', 'componentNo_2', 'component' ...},
         ...
         'bs':{ 'componentNo_1', 'componentNo_2', 'component' ...},
         'bst':{ 'componentNo_1', 'componentNo_2', 'component' ...}
    }

    Every component is an evoked array.
    """
    if info == None:
        with open('recall_epochs_0011.pkl', 'rb') as file:
            epochs = pickle.load(file)
        info = epochs.info

    picked_channels = mne.pick_types(info, meg=True, eeg=False, stim=False, eog=False,
                                     exclude='bads')
    info_picked = mne.pick_info(info, picked_channels)

    evoked_dic = {}

    component_keys = list(dpca.P.keys())


    for cz in component_keys:

        evoked_dic[cz] = {}
        components = dpca.P[cz]

        for i in range(num_components):

            component = components[:, i]
            evoked_array = mne.EvokedArray(component[:, np.newaxis], info_picked, tmin=0)

            evoked_dic[cz][f'componentNo_{i}'] = evoked_array

    return evoked_dic

def sort_trial_data_to_subject(Z_dic, original_indices, ls):
    # Initialize the sorted dictionary
    dic_sorted_by_subject = {subject: {} for subject in ls}

    # Iterate over each trial in the original dictionary
    for trial_index, subject_index in enumerate(original_indices):
        trial_key = f'trial{trial_index}'
        subject = ls[subject_index]

        # Assign the trial data to the corresponding subject
        dic_sorted_by_subject[subject][trial_key] = Z_dic[trial_key]

    return dic_sorted_by_subject

'''# Running example'''
# recall_epochs, man_epochs = reshape_to_epochs(-1, 4, _id_ = '0011_U7X', baseline=(-1, 0), notch_filter=50, h_freq=20, l_freq=0.1, events_fun=main_task_decoding_events_fun, resample=100)
# array1 = reshape_epoch_to_array(recall_epochs)
# array2 = reshape_epoch_to_array(man_epochs)
# sub1_array = concat_different_array(array1, array2)
#
# recall_epochs, man_epochs = reshape_to_epochs(-1, 4, _id_ = '0012_VK2', baseline=(-1, 0), notch_filter=50, h_freq=20, l_freq=0.1, events_fun=main_task_decoding_events_fun, resample=100)
# array1 = reshape_epoch_to_array(recall_epochs)
# array2 = reshape_epoch_to_array(man_epochs)
# sub2_array = concat_different_array(array1, array2)
#
# print(sub1_array.shape, sub2_array.shape) #(30, 306, 2, 2, 501) (30, 306, 2, 2, 501)
#
# final_array_for_dpca = stack_different_subject_arrays([sub1_array, sub2_array])
#
# print(final_array_for_dpca.shape) #(60, 306, 2, 2, 501)