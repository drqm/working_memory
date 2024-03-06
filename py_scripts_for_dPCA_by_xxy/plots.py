import matplotlib.pyplot as plt
import numpy as np
import mne
import pickle

def three_d_plot_component_coordinate(Z_dic, component_keys, only_average = True,subject = 'average', time_slice_length = 10): #Z_dic: dic by basic_data_resahape.make_dic; component_keys: ls(len = 3)
    total_time_points = Z_dic[subject]['t'].shape[-1]  # time_length
    num_time_slices = total_time_points // time_slice_length  # cut slice

    colors = ['blue', 'green', 'orange', 'red']
    labels = ['maintenance/mel1', 'maintenance/mel2', 'manipulation/mel1', 'manipulation/mel2']

    fig = plt.figure(figsize=(15, 5 * num_time_slices))

    if only_average == True:
        for t in range(num_time_slices):
            start = t * time_slice_length
            end = (t + 1) * time_slice_length

            data_array = np.stack([Z_dic[subject][key][0, :, :, start:end] for key in component_keys], axis=0) #(3, 2, 2, 50)

            ax1 = fig.add_subplot(num_time_slices, 4, t * 4 + 1, projection='3d')
            ax2 = fig.add_subplot(num_time_slices, 4, t * 4 + 2, projection='3d')
            ax3 = fig.add_subplot(num_time_slices, 4, t * 4 + 3, projection='3d')
            ax4 = fig.add_subplot(num_time_slices, 4, t * 4 + 4, projection='3d')

            for idx, ax in enumerate([ax1, ax2, ax3, ax4]):
                ax.set_title(f'Time slice: {start}-{end}')  # theme
                ax.text2D(0.2, 0.85, f"Subtitle {idx + 1}", transform=ax.transAxes)

                for i in range(2):
                    for j in range(2):

                        x = data_array[0][i, j]
                        y = data_array[1][i, j]
                        z = data_array[2][i, j]
                        # print(x.ndim, y.ndim, z.ndim)

                        ax.plot(x, y, z, color=colors[i + j * 2], label=labels[i + j * 2])
                ax.set_xlabel(component_keys[0])
                ax.set_ylabel(component_keys[1])
                ax.set_zlabel(component_keys[2])
                ax.legend()

            ax1.view_init(0, 0)
            ax2.view_init(0, 90)
            ax3.view_init(90, 90)
            ax4.view_init(30, 30)

        plt.tight_layout()

    else:
        for subject in Z_dic:

            alpha = 1 if subject == 'average' else 0.5/(len(Z_dic)-1)

            for t in range(num_time_slices):
                start = t * time_slice_length
                end = (t + 1) * time_slice_length

                data_array = np.stack([Z_dic[subject][key][0, :, :, start:end] for key in component_keys],
                                      axis=0)  # (3, 2, 2, 50)

                ax1 = fig.add_subplot(num_time_slices, 4, t * 4 + 1, projection='3d')
                ax2 = fig.add_subplot(num_time_slices, 4, t * 4 + 2, projection='3d')
                ax3 = fig.add_subplot(num_time_slices, 4, t * 4 + 3, projection='3d')
                ax4 = fig.add_subplot(num_time_slices, 4, t * 4 + 4, projection='3d')

                for idx, ax in enumerate([ax1, ax2, ax3, ax4]):
                    ax.set_title(f'Time slice: {start}-{end}')  # theme
                    ax.text2D(0.2, 0.85, f"Subtitle {idx + 1}", transform=ax.transAxes)

                    for i in range(2):
                        for j in range(2):
                            x = data_array[0][i, j]
                            y = data_array[1][i, j]
                            z = data_array[2][i, j]
                            # print(x.ndim, y.ndim, z.ndim)

                            ax.plot(x, y, z, color=colors[i + j * 2], label=labels[i + j * 2], alpha = alpha)
                    ax.set_xlabel(component_keys[0])
                    ax.set_ylabel(component_keys[1])
                    ax.set_zlabel(component_keys[2])
                    ax.legend()

                ax1.view_init(0, 0)
                ax2.view_init(0, 90)
                ax3.view_init(90, 90)
                ax4.view_init(30, 30)

            plt.tight_layout()

    # plt.savefig(f'figs/{subject}/latent_space_slices{component_keys}{time_slice_length}.png')


def plot_component_coordinate(Z_dic, time, only_average = True, component_keys = None, subject = 'average', num_components = 3, main_compare = 'Stimuli'):

#    time = np.linspace(0, Z_dic[subject]['t'].shape[-1] // 100, Z_dic[subject]['t'].shape[-1])

    S = 2  # Number of melody types

    components = list(Z_dic[subject].keys()) if component_keys == None else component_keys # Get the list of all(required) component names
    num_type_components = len(components)  # Get the number of components

    # Dynamically set the size of the figure based on the number of components
    plt.figure(figsize=(7 * num_type_components, 7 * num_components), dpi=100)

    if only_average == True:
        subject = 'average'

        if main_compare == 'Stimuli':
            colors = [['r', 'b'], ['r', 'b']]
            alphas = [[1, 1], [0.5, 0.5]]

        else:
            colors = [['r', 'r'], ['b', 'b']]
            alphas = [[1, 0.5], [1, 0.5]]

        for index, data_label in enumerate(components):
            # Create a counter for subplot indexing
            subplot_counter = 1 + index
            for i in range(num_components):
                # Iterate through all components
                data = Z_dic[subject][data_label]  # Get the data for the current component
                plt.subplot(num_components, num_type_components, subplot_counter)
                subplot_counter += num_type_components  # Update the subplot counter

                for s in range(S):
                    for j in range(data.shape[-2]):
                        color = colors[j][s]
                        alpha = alphas[j][s]
                        alpha *= (0.1 if subject != 'average' else 1)
                        label = f"{'Recall' if j == 0 else 'Manipulate'} Melody{s}" if subject == 'average' else None
                        plt.plot(time, data[i, j, s], color=color, alpha=alpha, label=label)

                plt.title(f'{i + 1} behavior component for {data_label}')
                plt.legend()

    else:
        for subject in Z_dic:

            if main_compare == 'Stimuli':
                colors = [['r', 'b'], ['r', 'b']] if subject != 'average' else [['m', 'c'], ['m', 'c']]
                alphas = [[1, 1], [0.5, 0.5]]

            else:
                colors = [['r', 'r'], ['b', 'b']] if subject != 'average' else [['m', 'm'], ['c', 'c']]
                alphas = [[1, 0.5], [1, 0.5]]

            for index, data_label in enumerate(components):
                # Create a counter for subplot indexing
                subplot_counter = 1 + index
                for i in range(num_components):
                    # Iterate through all components
                    data = Z_dic[subject][data_label]  # Get the data for the current component
                    plt.subplot(num_components, num_type_components, subplot_counter)
                    subplot_counter += num_type_components  # Update the subplot counter

                    for s in range(S):
                        for j in range(data.shape[-2]):
                            color = colors[j][s]
                            alpha = alphas[j][s]
                            alpha *= (0.1 if subject != 'average' else 1)
                            label = f"{'Recall' if j == 0 else 'Manipulate'} Melody{s}" if subject == 'average' else None
                            plt.plot(time, data[i, j, s], color=color, alpha=alpha, label=label)

                    plt.title(f'{i + 1} behavior component for {data_label}')
                    if subject == 'average':
                        plt.legend()

                    # Adjust the space between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.tight_layout()

    # plt.savefig(f'figs/{subject}/component_plots{components}.png')

def topo_PlotBasis(dpca, epochs, fig_dir='.', sid='all', component_keys = None, num_components = 3, ch_type = 'mag'):
    # with open('recall_epochs_0011.pkl', 'rb') as file:
    #     epochs = pickle.load(file)

    info = epochs.info
    picked_channels = mne.pick_types(info, meg=True, eeg=False, stim=False, eog=False,
                                     exclude='bads')
    info_picked = mne.pick_info(info, picked_channels)


    if component_keys == None:
        component_keys = list(dpca.P.keys())

    for cz in component_keys:
        components = dpca.P[cz]

        fig, main_axes = plt.subplots(1, num_components, figsize=(16, 6))

        for idx, ax in enumerate(main_axes.ravel()):
            component = components[:, idx]
            evoked_array = mne.EvokedArray(component[:, np.newaxis], info_picked, tmin=0)

            # 设置色标的范围从0到components的最大值
            evoked_array.plot_topomap(times=[0], axes=ax, colorbar=False, show=False,
                                      outlines='head', sensors=True
                                      , ch_type=ch_type)
            ax.set_title(f'Component {idx + 1}')

        # 为色标创建一个新的轴
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(ax.images[0], cax=cbar_ax)

        fig.suptitle(f'Topomaps for {cz}', fontsize=16, y=1.05)
        plt.tight_layout(rect=[0, 0, 0.9, 1])

        plt.savefig(f'{fig_dir}{sid}_Topomaps_{cz}_{ch_type}.pdf')
        plt.show()
        plt.close()



