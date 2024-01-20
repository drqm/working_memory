'''To get Z_array from Z_dic'''

# from basic_data_reshape import *

# Z_array = np.zeros([len(Z_dic)-1, n_comp, 10, 2, 2, 501])
# for i in Z_dic:
#     if i != 'average':
#         ncount = 0
#         for comp in ['b', 's', 't', 'bs', 'bt', 'st', 'bst']:
#             Z_array[int(i[5:])][ncount] = moving_average(Z_dic[i][comp], 5)
#             ncount += 1
# print(Z_array.shape)  (n_trials, 7, 10, 2, 2, 501)


from significance_analysis import *
import matplotlib.pyplot as plt

final_accuracies = significance_analysis(Z_array)

pc_names = ['b', 's', 't', 'bs', 'bt', 'st', 'bst']

'''paste the [for k in range(3):] if would like to make individual figures for each type of classification'''
plt.figure(figsize=(35, 15), dpi = 300)

num_component = 3
# iterate over the PC
for i in range(7):  # 7 PC
    for j in range(num_component):  # 3 component
        plt.subplot(num_component, 7, i + 1 + j * 7)
        for k in range(3):  # 3 types of classification
            plt.plot(final_accuracies[i, j, k, :], label=f'Class Type {k+1}')

        # titles and labels
        plt.title(f'PC: {pc_names[i]}, Component {j+1}')
        plt.xlabel('Time Points')
        plt.ylabel('Accuracy')

        plt.legend()

plt.tight_layout()
# plt.savefig()

