import pdb

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.autograd import Variable
from tqdm import tqdm

# data normalization
TEC_MEAN = 19
TEC_MIN = 0
TEC_MAX = 150


def preprocess(x):
    """Normalize TEC data."""
    return (x - TEC_MEAN) / (TEC_MAX - TEC_MIN)


def unprocess(x):
    """Unnormalize TEC data."""
    return x * (TEC_MAX - TEC_MIN) + TEC_MEAN


def get_input_targets(np_batch, window_train):
    """Separate input and target from sequence."""
    np_inputs = np_batch[:window_train]
    np_targets = np_batch[window_train:]
    return np_inputs, np_targets


def get_periodic(inputs, prediction_len):
    """Get the part of the input corresponding to the prediction length."""
    return inputs[-prediction_len:]
    # if prediction_len>24:
    #     raise Exception("Error prediction > 48h, TODO")
    # if prediction_len == 24:
    #     return inputs[-prediction_len:]
    # else:
    #     return inputs[-24:-24+prediction_len]


def get_periodic_blur_targets_diff(periodic, targets):
    """Apply blur on sequence and compute difference (for residual learning)."""
    periodic_blur = periodic.copy()
    for i in range(periodic_blur.shape[0]):
        for j in range(periodic_blur.shape[1]):
            periodic_blur[i, j, 0] = gaussian_filter(periodic_blur[i, j, 0],
                                                     sigma=3)
    targets_diff = targets - periodic_blur
    return periodic_blur, targets_diff


def blur_array(array):
    """Apply blur on sequence and compute difference (for residual learning)."""
    array_blur = array.copy()
    for i in range(array_blur.shape[0]):
        for j in range(array_blur.shape[1]):
            array_blur[i, j, 0] = gaussian_filter(array_blur[i, j, 0], sigma=3)
    return array_blur


def rms(data, axis=None):
    """Compute RMS."""
    if (axis is None):
        return np.sqrt((data**2).mean())
    else:
        return np.sqrt((data**2).mean(axis=axis))


def process_data(net,
                 optimizer,
                 criterion,
                 train_loader,
                 test_loader,
                 window_train,
                 window_predict,
                 diff,
                 cuda,
                 training=False):
    weights = np.arange(-36, 36).reshape((72, 1))
    weights = np.repeat(weights, 72, 1)
    weights = np.abs(weights)
    weights[:36] -= 1
    weights = np.cos((weights.astype(float) / 36) * np.pi / 2)
    weights /= weights.sum()

    if training:
        # training mode
        net.train()
        #iterate on the train dataset
        t = tqdm(train_loader, ncols=150)
    else:
        net.eval()
        #iterate on the train dataset
        t = tqdm(test_loader, ncols=150)

    loss = 0
    rms_ = 0  # mean rms oversequence
    rms_periodic = 0  # mean rms over sequence
    rms_per_frame = [0 for i in range(window_predict)]
    rms_periodic_per_frame = [0 for i in range(window_predict)]
    rms_per_sequence = []
    rms_per_sequence_periodic = []
    count = 0
    rms_lattitude = np.zeros(72)

    rms_global_mean = []

    for batch in t:
        count += batch[0].size(
            0) * window_predict  # count number of prediction images

        # preprocess the batch (TODO: go pytorch)
        # 1. disable preprocess, start to use relu or elu
        # batch_np = preprocess(batch[0].numpy().transpose((1,0,2,3,4)))
        np_batch = batch[0].numpy().transpose((1, 0, 2, 3, 4))

        # create inputs and targets for network
        np_inputs, np_targets = get_input_targets(np_batch, window_train)

        # select window_predict elements from the end of np_input
        # this last elements will have some information about the periodic
        # struct from data
        np_periodic = get_periodic(np_inputs, window_predict)

        # smooth, blur np_periodic and generate the pytorch tensor
        np_periodic_blur = blur_array(np_periodic)
        periodic_blur = torch.from_numpy(np_periodic_blur).float()
        #if cuda:
        #    periodic_blur = periodic_blur.cuda()

        # test for residual train
        if diff:  # use residual
            np_network_targets = np_targets - np_periodic_blur
        else:
            np_targets_network = np_targets.copy()

        # create pytorch tensors for inputs and targets
        inputs = torch.from_numpy(np_inputs).float()
        targets = torch.from_numpy(np_targets_network).float()
        #if cuda:
        #    inputs = inputs.cuda()
        #    targets = targets.cuda()

        # code for training and testing fase
        if training:
            # set gradients to zero
            optimizer.zero_grad()
            # forward pass in the network
            outputs = net.forward(inputs,
                                  window_predict,
                                  diff=diff,
                                  predict_diff_data=periodic_blur)
            # compute error and backprocj
            error = criterion(outputs, targets)
            error.backward()
            optimizer.step()
        else:  # testing
            # forward pass in the network
            outputs = net.forward(inputs,
                                  window_predict,
                                  diff=diff,
                                  predict_diff_data=periodic_blur)
            error = criterion(outputs, targets)  # compute loss for comparison

        # outputs
        np_outputs = outputs.cpu().data.numpy()
        if diff:
            # remove code associated with preprocess
            # outputs_complete = unprocess(outputs_np + np_periodic_blur)
            outputs_complete = np_outputs + np_periodic_blur
        else:
            # remove code associated with preprocess
            # outputs_complete = unprocess(outputs_np)
            outputs_complete = np_outputs
        # remove code associated with preprocess
        # periodic_complete = unprocess(periodic_np)
        # targets_complete = unprocess(targets_np)
        periodic_complete = np_periodic
        targets_complete = np_targets

        # update loss
        loss += float(error.cpu().item())

        # compute the rms for each image
        rms_tec_images = rms(outputs_complete - targets_complete,
                             axis=(2, 3, 4))
        rms_tec_images_periodic = rms(periodic_complete - targets_complete,
                                      axis=(2, 3, 4))

        rms_tec_images_lattitude = rms(outputs_complete - targets_complete,
                                       axis=(2, 4))
        rms_lattitude += rms_tec_images_lattitude.sum(axis=(0, 1))

        #rms_gm = outputs_complete.mean(axis=(2,3,4))-targets_complete.mean(axis=(2,3,4))
        rms_gm = (outputs_complete * weights[None, None, None, :, :]).sum(
            axis=(2, 3, 4)) - (targets_complete *
                               weights[None, None, None, :, :]).sum(axis=(2, 3,
                                                                          4))
        rms_gm = rms_gm.transpose(1, 0)
        for i in range(rms_gm.shape[0]):
            rms_global_mean.append(rms_gm[i])

        # update global rms
        rms_ += rms_tec_images.sum()
        rms_periodic += rms_tec_images_periodic.sum()

        # update rms per seq frame
        for frame_id in range(window_predict):
            rms_per_frame[frame_id] += rms_tec_images[frame_id].sum()
            rms_periodic_per_frame[frame_id] += rms_tec_images_periodic[
                frame_id].sum()

        for seq_id in range(rms_tec_images.shape[1]):
            rms_per_sequence.append(rms_tec_images[:, seq_id].mean())
            rms_per_sequence_periodic.append(
                rms_tec_images_periodic[:, seq_id].mean())

        # update TQDM
        t.set_postfix(Loss=loss / count,
                      RMS=rms_ / count,
                      RMS_P=rms_periodic / count)

    rms_global_mean = np.array(rms_global_mean)

    print("RMS GLOBAL MEAN", rms_global_mean.shape,
          rms(rms_global_mean, axis=1).mean())

    loss = loss / count
    rms_ = rms_ / count
    rms_lattitude = rms_lattitude / count
    rms_periodic = rms_periodic / count
    for frame_id in range(window_predict):
        rms_per_frame[frame_id] /= count / window_predict
        rms_periodic_per_frame[frame_id] /= count / window_predict

    return loss, rms_, rms_periodic, rms_per_frame, rms_periodic_per_frame, rms_per_sequence, rms_per_sequence_periodic, rms_lattitude
