import pickle
import numpy as np
import math
import sys
sys.path.append('../src')
from preprocessing import *
import torch

host = 'aarhus' #'china'
if host == 'aarhus':
    wdir = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/'
    mf_dir = wdir + 'scratch/maxfiltered_data/tsss_st16_corr96/'
    ica_dir = wdir + 'scratch/working_memory/ICA/'
    log_dir = wdir + 'misc/working_memory_logs/'
    fig_dir = wdir + 'scratch/working_memory/deep_learning/figures/'
    acc_dir = wdir + 'scratch/working_memory/deep_learning/data/'
elif host == 'china':
    wdir = '/Users/xiangxingyu/Downloads/毕业设计/UCB线上科研/data/'
    mf_dir = wdir + '01_raw_maxfiltered/'
    ica_dir = wdir + '02_ica_solution/'
    log_dir = wdir + 'working_memory_logs/'
    fig_dir = './'
    acc_dir = './'

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

def get_array_from_epoch(epochs):
    array_ls = []
    event_types = list(epochs.event_id.keys())
    n_samples = {event: len(epochs[event]) for event in event_types}
    for i, event in enumerate(event_types):
        event_data = epochs[event].get_data()#.transpose(1, 0, 2)
        array_ls.append(event_data)
    return array_ls

def get_subject_array_ls(subject,crop=None):
    recall_epochs, man_epochs = reshape_to_epochs(-2, 4, _id_=subject,
                                                  mf_dir=mf_dir,
                                                  ica_dir=ica_dir,
                                                  log_dir=log_dir,
                                                  baseline=(-2, 0),
                                                  notch_filter=50,
                                                  h_freq=40,
                                                  l_freq=0.1,
                                                  events_fun=main_task_decoding_events_fun,
                                                  resample=100)
    if crop:
        recall_epochs.crop(crop[0],crop[1])
        man_epochs.crop(crop[0],crop[1])

    recall_array_ls = get_array_from_epoch(recall_epochs)  # [(30, 306, 501), (29, 306, 501)]
    manual_array_ls = get_array_from_epoch(man_epochs)  # [(30, 306, 501), (30, 306, 501)]
    # Let's also return the time variable and frequency of sampling
    times, Fs = recall_epochs.times, recall_epochs.info['sfreq']
    return recall_array_ls, manual_array_ls, times, Fs

def calc_diff_entropy_gpu(signal):
    var = torch.var(signal, dim=-1, keepdim=True)
    diff_entropy = 0.5 * torch.log(2 * math.pi * math.e * var)
    return diff_entropy.squeeze()
def DElize(data, window_size, stride = 100):
    # Convert numpy array to PyTorch tensor and move it to GPU(if available else CPU)
    data = torch.tensor(data, dtype=torch.float32).cpu()

    # Create sliding windows
    windows = data.unfold(-1, window_size, stride)

    # Reshape windows to separate window index and time index
    windows = windows.reshape(*windows.shape[:-2], -1, windows.shape[-1])

    batch = windows.reshape(-1, windows.shape[-1])
    diff_entropy_batch = calc_diff_entropy_gpu(batch)
    diff_entropy_seq = diff_entropy_batch.reshape(*windows.shape[:-1])

    # Convert the result back to numpy
    diff_entropy_seq = diff_entropy_seq.cpu().numpy()

    return diff_entropy_seq

def save_accuracy(cacc,csubject,suffix=''):
    f = f'{acc_dir}{csubject}_accuracy{suffix}.p'
    with open(f,'wb') as cf:
        pickle.dump(cacc,cf)
# recall_array_ls, manual_array_ls = get_subject_array_ls('0011_U7X')