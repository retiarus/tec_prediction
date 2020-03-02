import pdb

from scipy.ndimage import gaussian_filter


def get_input_targets(np_batch, window_train):
    """Separate input and target from sequence."""
    np_inputs = np_batch[:window_train]
    np_targets = np_batch[window_train:]
    return np_inputs, np_targets


def get_periodic(inputs, prediction_len):
    """Get the part of the input corresponding to the prediction length."""

    return inputs[-prediction_len:]


def blur_array(array):
    """Apply blur on sequence and compute difference (for residual learning)."""
    array_blur = array.copy()
    for i in range(array_blur.shape[0]):
        for j in range(array_blur.shape[1]):
            array_blur[i, j, 0] = gaussian_filter(array_blur[i, j, 0], sigma=3)
    return array_blur
