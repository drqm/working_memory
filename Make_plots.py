wdir = '/Users/xiangxingyu/Downloads/UCB线上科研/data/'

import matplotlib
matplotlib.use('Qt5Agg')
from numpy import *
import os
import mne
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tempfile
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import rand, randn, randint
from dPCA import dPCA

import sys
sys.path.append('./src')
from preprocessing import *

mne.set_log_level('CRITICAL')

directory = wdir + "01_raw_maxfiltered/"
ls = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
ls = ls[2:]
print(ls)

for _id_ in ls:

    if not os.path.exists('figs/' + _id_):
        os.makedirs('figs/' + _id_)

    _id_number = _id_[:4]

    inv_fname = wdir + '01_raw_maxfiltered/' + _id_ + '/inv_raw_tsss.fif'
    inv_ica_fname = wdir + '02_ica_solution/' + _id_ + '/inv_raw_tsss-ica.fif'
    inv_lfname = wdir + 'working_memory_logs/' + _id_number + '_invert_MEG.csv'

    main_fname = wdir + '01_raw_maxfiltered/' + _id_ + '/main_raw_tsss.fif'
    main_ica_fname = wdir + '02_ica_solution/' + _id_ + '/main_raw_tsss-ica.fif'
    main_lfname = wdir + 'working_memory_logs/' + _id_number + '_recognize_MEG.csv'

    recall_epochs = WM_epoching(inv_fname, inv_ica_fname, -1, 4,
                                baseline=(-1, 0), notch_filter=50, h_freq=20, l_freq=0.1,
                                events_fun = main_task_decoding_events_fun,
                                events_fun_kwargs={'lfname': inv_lfname},
                                resample=100)

    manipulate_epochs = WM_epoching(main_fname, main_ica_fname, -1, 4,
                                    baseline=(-1, 0), notch_filter=50, h_freq=20, l_freq=0.1,
                                    events_fun = main_task_decoding_events_fun,
                                    events_fun_kwargs={'lfname': main_lfname},
                                    resample=100)

    # Load Data
    event_types_recall = list(recall_epochs.event_id.keys())
    event_types_manipulate = list(manipulate_epochs.event_id.keys())

    #
    n_samples_recall = {event: len(recall_epochs[event]) for event in event_types_recall}
    n_samples_manipulate = {event: len(manipulate_epochs[event]) for event in event_types_manipulate}

    # max event number
    max_samples = max(max(n_samples_recall.values()), max(n_samples_manipulate.values()))

    # [C, max_samples, 2, 2, T]
    final_data = np.full((306, max_samples, 2, len(event_types_recall), 501), np.nan)

    for i, event in enumerate(event_types_recall):
        event_data = recall_epochs[event].get_data().transpose(1, 0, 2)
        n_current_samples = event_data.shape[1]
        # fill with nan
        if n_current_samples < max_samples:
            shortfall = max_samples - n_current_samples
            filler = np.full((306, shortfall, 501), np.nan)
            event_data = np.concatenate([event_data, filler], axis=1)
        final_data[:, :max_samples, 0, i, :] = event_data

    for i, event in enumerate(event_types_manipulate):
        event_data = manipulate_epochs[event].get_data().transpose(1, 0, 2)
        n_current_samples = event_data.shape[1]
        # fill with nan
        if n_current_samples < max_samples:
            shortfall = max_samples - n_current_samples
            filler = np.full((306, shortfall, 501), np.nan)
            event_data = np.concatenate([event_data, filler], axis=1)
        final_data[:, :max_samples, 1, i, :] = event_data

    final_data = final_data.transpose(1, 0, 2, 3, 4)  # [:,:,:,:,-200:]
    print(final_data.shape)  # (max_samples, 306, 2(recall/mainpulate), 2(melody0,melody1), 501)


    # dPCA

    # trial-average data
    mR = np.nanmean(final_data, 0)  # skip the Nans

    dims = mR.shape

    # center data
    mR -= np.mean(mR.reshape((-1, np.prod(dims[1:]))), axis=1)[:, None, None, None]
    mdpca = dPCA.dPCA(labels='bst', regularizer='auto')
    mdpca.protect = ['t']
    Z = mdpca.fit_transform(mR, final_data)

    ################################
    ################################

    # 3D Plots for Components

    def three_d_plot_component( perfix = '', time_slice_length = 50 ):

        total_time_points = Z['bt'].shape[3]  # time_length
        num_time_slices = total_time_points // time_slice_length  # cut slice

        colors = ['blue', 'green', 'orange', 'red']
        labels = ['maintenance/mel1', 'maintenance/mel2', 'manipulation/mel1', 'manipulation/mel2']

        fig = plt.figure(figsize=(15, 5 * num_time_slices))

        for t in range(num_time_slices):
            start = t * time_slice_length
            end = (t + 1) * time_slice_length

            b_data = Z['b' + perfix][0, :, :, start:end]
            s_data = Z['s' + perfix][0, :, :, start:end]
            bs_data = Z['bs' + perfix][0, :, :, start:end]

            ax1 = fig.add_subplot(num_time_slices, 4, t * 4 + 1, projection='3d')
            ax2 = fig.add_subplot(num_time_slices, 4, t * 4 + 2, projection='3d')
            ax3 = fig.add_subplot(num_time_slices, 4, t * 4 + 3, projection='3d')
            ax4 = fig.add_subplot(num_time_slices, 4, t * 4 + 4, projection='3d')

            for idx, ax in enumerate([ax1, ax2, ax3, ax4]):
                ax.set_title(f'Time slice: {start}-{end}')  # theme
                ax.text2D(0.2, 0.85, f"Subtitle {idx + 1}", transform=ax.transAxes)

                for i in range(2):
                    for j in range(2):
                        ax.plot(b_data[i, j], s_data[i, j], bs_data[i, j], color=colors[i + j * 2], label=labels[i + j * 2])
                ax.set_xlabel('b' + perfix)
                ax.set_ylabel('s' + perfix)
                ax.set_zlabel('bs'+ perfix)
                ax.legend()

            ax1.view_init(0, 0)
            ax2.view_init(0, 90)
            ax3.view_init(90, 90)
            ax4.view_init(30, 30)

        plt.tight_layout()
        if perfix == '':
            plt.savefig(f'figs/{_id_}/latent_space_slices{time_slice_length}.png')
        else:
            plt.savefig(f'figs/{_id_}/latent_space_slices(t){time_slice_length}.png')
        plt.close(fig)


    three_d_plot_component()
    three_d_plot_component(perfix = 't')

    ################################
    ################################

    # Plot Components

    def plot_component():
        time = np.linspace(0, 5, 501)
        x = 3  # Increase the number of behavior components, for example, set to 3
        S = 2  # Number of melody types
        melody_colors = ['r', 'b']

        components = list(Z.keys())  # Get the list of all component names
        num_components = len(components)  # Get the number of components

        # Dynamically set the size of the figure based on the number of components
        plt.figure(figsize=(7 * num_components, 7 * x), dpi=100)

        # Create a counter for subplot indexing
        subplot_counter = 1

        # Loop through all behavior components
        for i in range(x):
            # Iterate through all components
            for index, data_label in enumerate(components):
                data = Z[data_label]  # Get the data for the current component
                plt.subplot(x, num_components, subplot_counter)
                subplot_counter += 1  # Update the subplot counter

                for s in range(S):
                    for j in range(data.shape[-2]):
                        color = melody_colors[s]
                        alpha = 0.5 if j == 0 else 1
                        label = f"{'Recall' if j == 0 else 'Manipulate'} Melody{s}"
                        plt.plot(time, data[i, s, j], color=color, alpha=alpha, label=label)

                plt.title(f'{i + 1} behavior component for {data_label}')
                plt.legend()

        # Adjust the space between subplots
        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        plt.tight_layout()
        plt.savefig(f'figs/{_id_}/component.png')
        plt.close()


    plot_component()

    ################################
    ################################

    # Plot Basis
    def plot_basis(absolute = True):
        num_components = len(Z.keys())  # number of components

        # Create a large figure to hold all subplots
        # 设定大图的尺寸，你可能需要根据实际情况来调整这里的数值
        plt.figure(figsize=(10 * num_components, 40))

        # Initialize subplot counter
        subplot_counter = 1

        for cz in Z:
            components = np.abs(mdpca.P[cz]) if absolute == True else mdpca.P[cz]
            # Create subplot
            ax = plt.subplot(1, num_components, subplot_counter)  # 放在一行中，subplot_counter列

            # Create heatmap using seaborn
            sns.heatmap(components, cmap="YlGnBu", cbar=True, ax=ax)

            ax.set_xlabel('Principal Components')
            ax.set_ylabel('Neurons')
            ax.set_title(f'dPCA Component Weights for {cz}')

            subplot_counter += 1  # 更新子图计数器

        # Adjust layout and save figure
        plt.tight_layout()
        plt.savefig(f'figs/{_id_}/dPCA_Component_Weights_horizontal.png')
        plt.close()


    plot_basis()

    ################################
    ################################

    # topo_Plot Basis
    def topo_PlotBasis(ch_tpye = 'grad'):
        info = recall_epochs.info
        picked_channels = mne.pick_types(info, meg=True, eeg=False, stim=False, eog=False,
                                         exclude='bads')
        info_picked = mne.pick_info(info, picked_channels)

        for cz in Z:
            components = mdpca.P[cz]

            fig, main_axes = plt.subplots(2, 5, figsize=(16, 6))

            for idx, ax in enumerate(main_axes.ravel()):
                component = components[:, idx]
                evoked_array = mne.EvokedArray(component[:, np.newaxis], info_picked, tmin=0)

                # 设置色标的范围从0到components的最大值
                evoked_array.plot_topomap(times=[0], axes=ax, colorbar=False, show=False,
                                          outlines='head', sensors=True
                                          , ch_type=ch_tpye)
                ax.set_title(f'Component {idx + 1}')

            # 为色标创建一个新的轴
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(ax.images[0], cax=cbar_ax)

            fig.suptitle(f'Topomaps for {cz}', fontsize=16, y=1.05)
            plt.tight_layout(rect=[0, 0, 0.9, 1])
            plt.savefig(f'figs/{_id_}/Topomaps_for_{cz}_CHTYPE{ch_tpye}.png')
            plt.close(fig)



    topo_PlotBasis(ch_tpye='grad')
    topo_PlotBasis(ch_tpye=None)

