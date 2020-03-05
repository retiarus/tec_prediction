import pdb

import numpy as np

from calc_errors import CalcErrors
from metrics import rms
from pre_processing import blur_array, get_input_targets, get_periodic
from tqdm import tqdm


def process_data(net,
                 optimizer,
                 criterion,
                 train_loader,
                 test_loader,
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

    if training:
        # training mode
        # iterate on the train dataset
        t = tqdm(train_loader, ncols=100)
    else:
        t = tqdm(test_loader, ncols=100)

    for batch in t:
        if pytorch:
            # count number of prediction images
            calc_errors.update_count(batch[0].size(0) * window_predict)

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

            if cuda:
                periodic_blur = periodic_blur.cuda()

            # test for residual train
            if diff:  # use residual
                np_targets_network = np_targets - np_periodic_blur
            else:
                np_targets_network = np_targets.copy()

            # create pytorch tensors for inputs and targets
            inputs = torch.from_numpy(np_inputs).float()
            targets = torch.from_numpy(np_targets_network).float()

            if cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

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
                error_np = error.detach().numpy()
                error.backward()
                optimizer.step()
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

            # update loss
            calc_errors.update_loss(float(error.cpu().item()))

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

        calc_errors(np_outputs, np_periodic_blur, np_periodic, np_targets)

        # upd#ate TQDM
        t.set_postfix(Loss=calc_errors.get_loss(),
                      RMS=calc_errors.get_rms(),
                      RMS_P=calc_errors.get_rms_periodic())

    dict_loss = calc_errors.calc_errors()
    return dict_loss
