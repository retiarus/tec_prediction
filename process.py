import pdb
from time import time

import numpy as np

from calc_errors import CalcErrors
from metrics import rms
from pre_processing import blur_array, get_input_targets, get_periodic
from tqdm import tqdm


def process_data(net,
                 optimizer,
                 criterion,
                 loader,
                 window_train,
                 window_predict,
                 diff,
                 cuda,
                 pytorch,
                 training=False):
    calc_errors = CalcErrors(window_predict, diff)

    if pytorch:
        import torch
        from torch.autograd import Variable

        if training:
            # training mode
            net.train()
        else:
            net.eval()

    t = tqdm(loader, ncols=100)
    it_t = iter(t)

    size = t.total
    print(size)
    count = 0
    while count < size:
        count += 1
        start = time()
        batch = next(it_t)
        end = time()
        time_load = end-start

        if pytorch:
            start = time()
            # preprocess the batch (TODO: go pytorch)
            # 1. disable preprocess, start to use relu or elu
            # batch_np = preprocess(batch[0].numpy().transpose((1,0,2,3,4)))
            np_batch = batch[0].numpy().transpose((1, 0, 2, 3, 4))

            # create inputs and targets for network
            np_inputs, np_targets = get_input_targets(np_batch, window_train)

            # count number of prediction images
            calc_errors.update_count(batch[0].size(0) * window_predict)

            # residual train
            if diff:  # use residual
                # select window_predict elements from the end of np_input
                # this last elements will have some information about the periodic
                # struct from data
                np_periodic = get_periodic(np_inputs, window_predict)
                # smooth, blur np_periodic and generate the pytorch tensor
                np_periodic_blur = blur_array(np_periodic)
                np_periodic_blur = np_periodic_blur.transpose((1, 0, 2, 3, 4))
                periodic_blur = torch.from_numpy(np_periodic_blur).float()
                np_targets_network = np_targets - np_periodic_blur
            else:
                np_targets_network = np_targets.copy()
                periodic_blur = None

            # create pytorch tensors for inputs and targets
            np_inputs = np_inputs.transpose((1, 0, 2, 3, 4))
            np_targets_network = np_targets_network.transpose((1, 0, 2, 3, 4))
            inputs = torch.from_numpy(np_inputs).float()
            targets = torch.from_numpy(np_targets_network).float()
            end = time()
            time_preprocessing = end-start

            if cuda:
                start = time()
                inputs = inputs.cuda()
                targets = targets.cuda()
                if diff:
                    periodic_blur = periodic_blur.cuda()
                end = time()
                time_load_to_gpu = end-start


            # code for training and testing fase
            if training:
                start = time()
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
                end = time()
                time_training = end-start
            else:  # testing
                # forward pass in the network
                outputs = net.forward(inputs,
                                      window_predict,
                                      diff=diff,
                                      predict_diff_data=periodic_blur)
                error = criterion(outputs,
                                  targets)  # compute loss for comparison

            # outputs
            np_outputs = outputs.cpu().data.numpy()
            np_outputs = np_outputs.transpose((1, 0, 2, 3, 4))

            calc_errors.update_loss(float(error.cpu().detach().numpy()))

        else:
            import tensorflow as tf

            @tf.function
            def train_step(net, criterion, optimizer, np_inputs):
                with tf.GradientTape() as tape:
                    # forward pass in the network
                    np_outputs = net(np_inputs, training=True)
                    error = criterion(np_outputs, np_targets)
                    gradients = tape.gradient(error, net.trainable_variables)
                    optimizer.apply_gradients(
                        zip(gradients, net.trainable_variables))

                    return error, np_outputs

            # count number of prediction images
            calc_errors.update_count(batch[1].shape[0] * window_predict)
            np_inputs = batch[0]
            np_targets = batch[1].transpose((1, 0, 2, 3, 4))
            np_periodic_blur = batch[0]['blur']
            np_periodic = get_periodic(np_inputs['x'], window_predict)

            # code for training and testing fase
            if training:
                error, np_outputs = train_step(net, criterion, optimizer,
                                               np_inputs)
                error = float(error.cpu().numpy())
                np_outputs = np_outputs.cpu().numpy()
            else:
                np_outputs = net(np_inputs)
                error = criterion(np_outputs, np_targets)

            # update loss
            calc_errors.update_loss(error)

        calc_errors(np_outputs, None, np_periodic, np_targets)

        # upd#ate TQDM
        t.set_postfix(Loss=calc_errors.get_loss(),
                      RMS=calc_errors.get_rms(),
                      RMS_P=calc_errors.get_rms_periodic(),
                      time_load=time_load,
                      time_preprocessing=time_preprocessing,
                      time_training=time_training)

    dict_loss = calc_errors.calc_errors()
    return dict_loss
