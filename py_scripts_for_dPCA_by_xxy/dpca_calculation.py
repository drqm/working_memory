from dPCA import dPCA
import numpy as np

def dpca_fit(stack_array: np.array, dim: str):
    if stack_array.ndim - 3 != len(dim) - 1:
        raise Exception(f'The shape of stack array is not compatible with dim for dpca, {stack_array.shape} and {len(dim) - 1}')

    # Suppose that R.shape is (306, 2, 2, 501)
    R = np.nanmean(stack_array, 0)
    dims = R.shape

    # calculate mean
    reshaped_mR = R.reshape(dims[0], -1)
    mean_values = np.mean(reshaped_mR, axis=1)

    mean_values_expanded = mean_values[:, np.newaxis, np.newaxis, np.newaxis]

    R -= mean_values_expanded

    dpca = dPCA.dPCA(labels = dim, regularizer='auto')
    dpca.protect = ['t']
    Z = dpca.fit_transform(R, stack_array)

    return Z, dpca

