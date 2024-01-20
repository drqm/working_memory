import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def split_data(data, test_size=0.25):
    """
    Splits the data into training and testing sets.

    Parameters:
    - data (numpy array): The data to split, expected shape (num_trials, 2, 2, num_time_points).
    - test_size (float or int): If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
                               If int, represents the absolute number of test samples.

    Returns:
    - train_data (numpy array): Training set.
    - test_data (numpy array): Testing set.
    """
    num_trials = data.shape[0]
    indices = np.arange(num_trials)
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=None)

    train_data = data[train_indices]
    test_data = data[test_indices]

    return train_data, test_data

# Example usage
# Assuming Z_array is your data array with shape (156, 7, 10, 2, 2, 501)
# Z_array = np.random.rand(156, 7, 10, 2, 2, 501) # Replace this with your actual data

# Splitting a single component, for example, the first principal component and the first component
# component_data = Z_array[:, 0, 0, :, :, :]
# train_data, test_data = split_data(component_data, test_size=0.2)

# The train_data and test_data will now have shapes corresponding to the split
# train_data.shape, test_data.shape



def calculate_accuracy(data, mean_data, classification_type):
    '''
    data[n_trials, 2, 2, num_time_points]
    mean_data [2, 2, num_time_points]
    '''
    num_trials, _, _, num_time_points = data.shape
    accuracy = np.zeros(num_time_points)

    for time_point in range(num_time_points):
        correct = 0
        for trial in range(num_trials):
            if classification_type == 1:
                # 1st type Behaviour
                distances = [np.linalg.norm(data[trial, idx, :, time_point] - mean_data[idx, :, time_point]) for idx in range(2)]
                predicted = np.argmin(distances)
                actual = 0 if data[trial, 0, :, time_point].sum() > data[trial, 1, :, time_point].sum() else 1
            elif classification_type == 2:
                # 2nd type Stimuli
                distances = [np.linalg.norm(data[trial, :, idx, time_point] - mean_data[:, idx, time_point]) for idx in range(2)]
                predicted = np.argmin(distances)
                actual = 0 if data[trial, :, 0, time_point].sum() > data[trial, :, 1, time_point].sum() else 1
            elif classification_type == 3:
                        # 3rd type Cross_Melody
                        class_1_distances = [np.linalg.norm(data[trial, i, j, time_point] - mean_data[i, j, time_point]) for i, j in [(0, 0), (1, 1)]]
                        class_2_distances = [np.linalg.norm(data[trial, i, j, time_point] - mean_data[i, j, time_point]) for i, j in [(0, 1), (1, 0)]]
                        predicted = 0 if min(class_1_distances) < min(class_2_distances) else 1

                        actual = 0 if (np.array([data[trial, 0, 0, time_point], data[trial, 1, 1, time_point]]).sum() <
                                             np.array([data[trial, 0, 1, time_point], data[trial, 1, 0, time_point]]).sum()) else 1

            correct += (predicted == actual)

        accuracy[time_point] = correct / num_trials

    return accuracy


def significance_analysis(Z_array, num_repetitions = 100, num_component = 3, test_size = 0.2):
    num_repetitions = num_repetitions # Number of times to repeat the training/testing split and accuracy calculation
    num_component = num_component
    final_accuracies = np.zeros((7, num_component, 3, 501))  # Initialize an array to store the final accuracies

    for i in range(7):  # Iterate over principal components
        for j in range(num_component):  # Iterate over components
            for k in range(1, 4):  # Iterate over classification types
                component_accuracies = np.zeros((num_repetitions, 501))  # Store the accuracies for each random grouping

                for n in tqdm(range(num_repetitions),
                              desc = f"Processing PC {i + 1}, Component {j + 1}, Class Type {k}"):  # Repeat training-testing split and accuracy calculation
                    # Split the data
                    train_data, test_data = split_data(Z_array[:, i, j, :, :, :], test_size = test_size)

                    # Calculate the mean of the training data
                    mean_data = np.mean(train_data, axis = 0)

                    # Calculate accuracy only on the test set
                    test_accuracy = calculate_accuracy(test_data, mean_data, classification_type = k)

                    # Store the accuracy
                    component_accuracies[n, :] = test_accuracy

                # Compute the average accuracy across all repetitions
                final_accuracies[i, j, k - 1, :] = np.mean(component_accuracies, axis = 0)

    print(final_accuracies.shape)

    return final_accuracies