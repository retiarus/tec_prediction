"""
main file

License is from https://github.com/aboulch/tec_prediction
"""
import argparse
import logging
import os
import pdb
import subprocess
import uuid

from colors import print_blue, print_green, print_red
from log_loss import log_loss
from process_torch import process_data
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_length_min",
                        type=int,
                        default=7200,
                        help="train network or not")
    parser.add_argument("--step_min", type=int, default=10)
    parser.add_argument("--window_train", type=int, default=432)
    parser.add_argument("--window_predict", type=int, default=288)
    parser.add_argument("--work_loader", type=int, default=40)
    parser.add_argument("--batch_size",
                        type=int,
                        default=50,
                        help="train network or not")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--cuda", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--model", type=str, default="simple")
    parser.add_argument("--diff", type=bool, default=False)
    parser.add_argument("--pytorch", type=bool, default=False)
    parser.add_argument("--data", type=str, default='tec')
    parser.add_argument("--target",
                        type=str,
                        default="./results",
                        help="target directory")
    parser.add_argument("--source", type=str, help="source directory")
    parser.add_argument("--station", type=str, default="all",
                        help="select one station of year to train the model, or alls")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PATH"] = "/var/lib/jupyterhub/anaconda/envs/dscience/bin" + ":"  + env["PATH"]
    #env["PATH"] = "/scratch/ampemi/pedro.santos2/anaconda3/envs/dscience/bin" + ":"  + env["PATH"]
    subprocess.Popen(["redis-server", "./redis.conf"], env=env)

    # each execution has a unique uuid, for the log file and for save the model
    # here denote by uid (unique-id)
    uid = uuid.uuid1()

    if args.pytorch:
        tool = 'pytorch'
    else:
        tool = 'keras'

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(
        f'./log/{args.model}_{args.window_train}_{args.window_predict}_{tool}_{args.data}_{args.station}_{uid}.log'
    )

    format = logging.Formatter(f'%(levelname)s:%(message)s')
    handler.setFormatter(format)
    logger.addHandler(handler)

    tb = SummaryWriter()

    epoch_max = args.epochs

    seq_length = int(args.seq_length_min / args.step_min)

    if args.pytorch:
        print_green("Using Pytorch")
    else:
        print_green("Using Keras")
    print_green("Sequence length:", seq_length)

    # create the result directory
    if not os.path.exists(args.target):
        os.makedirs(args.target)

    # define optimization parameters
    root_dir = args.source

    if args.pytorch:
        import torch
        from data_loader_torch import SequenceLoader

        torch.set_default_dtype(torch.float32)

        # Setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        print()

        # Additional Info when using cuda
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:',
                  round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
            print('Cached:   ',
                  round(torch.cuda.memory_cached(0) / 1024**3, 1), 'GB')

        # CUDA
        if args.cuda:
            torch.backends.cudnn.benchmark = True

        print_blue("Creating data loader...")
        ds = SequenceLoader('train', root_dir, args.seq_length_min,
                            args.step_min, args.window_train,
                            args.window_predict, args.data,
                            args.station)
        ds_val = SequenceLoader('validation', root_dir, args.seq_length_min,
                                args.step_min, args.window_train,
                                args.window_predict, args.data,
                                args.station)
        seq_train = torch.utils.data.DataLoader(ds,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.work_loader,
                                                pin_memory=True)
        seq_test = torch.utils.data.DataLoader(ds_val,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.work_loader,
                                               pin_memory=True)

        if args.data == "tec+scin":
            num_channels = 2
        else:
            num_channels = 1

        print_blue("Creating network...")
        if args.model == "simple":
            from network_simple4 import SimpleConvRecurrent
            net = SimpleConvRecurrent(num_channels, act_cuda=args.cuda)
        elif args.model == "unet":
            from network_unet import UnetConvRecurrent
            net = UnetConvRecurrent(num_channels, act_cuda=args.cuda)
        elif args.model == "dilation121":
            from network_dilation_121 import DilationConvRecurrent
            net = DilationConvRecurrent(num_channels, act_cuda=args.cuda)
        else:
            print_red("Error bad network")
            exit()


        if args.cuda:
            net = DataParallel(net)
            net.cuda()

#        X = next(iter(seq_train))
#        tb.add_graph(net, X)

        print("PARAMTERS")

        #    summary(net, (args.window_train, args.batch, 1, 72, 72))

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters()
                       if p.requires_grad)

        print(count_parameters(net))

        print_blue("Setting up the optimizer...")
        #optimizer = torch.optim.Adadelta(net.parameters())
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        print_blue("Setting up the criterion...")
#        criterion_1 = torch.nn.L1Loss(reduction='sum')
        criterion = torch.nn.MSELoss()
#        def criterion_function(y, y_predict):
#            return  criterion_1(y, y_predict) + criterion_2(y, y_predict)

#        criterion = criterion_function

    else:
        import tensorflow as tf
        run_opts = tf.compat.v1.RunOptions(
            report_tensor_allocations_upon_oom=True)
        from data_generator import DataGenerator
        from tensorflow.keras import losses, optimizers

        print_blue("Creating data loader...")
        seq_train = DataGenerator(name='train',
                                  path_files=root_dir,
                                  seq_length_min=args.seq_length_min,
                                  step_min=args.step_min,
                                  window_train=args.window_train,
                                  window_predict=args.window_predict,
                                  data=args.data,
                                  batch_size=args.batch_size,
                                  to_fit=True,
                                  diff=args.diff)
        seq_test = DataGenerator(name='test',
                                 path_files=root_dir,
                                 seq_length_min=args.seq_length_min,
                                 step_min=args.step_min,
                                 window_train=args.window_train,
                                 window_predict=args.window_predict,
                                 data=args.data,
                                 batch_size=args.batch_size,
                                 to_fit=True,
                                 diff=args.diff)
        seq_test = DataGenerator(name='validation',
                                 path_files=root_dir,
                                 seq_length_min=args.seq_length_min,
                                 step_min=args.step_min,
                                 window_train=args.window_train,
                                 window_predict=args.window_predict,
                                 data=args.data,
                                 batch_size=args.batch_size,
                                 to_fit=True,
                                 diff=args.diff)

        print_blue("Creating network...")
        if args.model == "simple":
            from network_simple_keras2 import SimpleConvRecurrent
            net = SimpleConvRecurrent(input_nbr=1)
        elif args.model == "unet":
            pass
            # from network_unet import UnetConvRecurrent
            # net = UnetConvRecurrent(1)
        elif args.model == "dilation121":
            pass
            # from network_dilation_121 import UnetConvRecurrent
            # net = UnetConvRecurrent(1)
        else:
            print_red("Error bad network")
            exit()

        print_blue("Setting up the optimizer...")
        optimizer = optimizers.Adam(1e-4)

        print_blue("Setting up the criterion...")
        criterion = losses.MeanSquaredError()

        net.compile(loss=criterion, optimizer=optimizer, options=run_opts)

    if not args.test:
        print_blue("TRAINING")

        # iterate on epochs
        for epoch in range(epoch_max):

            print_green("Epoch", epoch)

            # train
            dict_loss = process_data(net=net,
                                     optimizer=optimizer,
                                     criterion=criterion,
                                     loader=seq_train,
                                     window_train=args.window_train,
                                     window_predict=args.window_predict,
                                     diff=args.diff,
                                     cuda=args.cuda,
                                     pytorch=args.pytorch,
                                     training=True,
                                     tb=tb)

            logger.info(f"{epoch}:train:{dict_loss['loss'],dict_loss['rms_']}")
            loss_train = dict_loss['loss']

            for i in net.named_parameters():
                tb.add_histogram(i[0], i[1], epoch)
                tb.add_histogram(f'{i[0]}.grad', i[1].grad, epoch)

            with torch.no_grad():
                dict_loss = process_data(net=net,
                                         optimizer=optimizer,
                                         criterion=criterion,
                                         loader=seq_test,
                                         window_train=args.window_train,
                                         window_predict=args.window_predict,
                                         diff=args.diff,
                                         cuda=args.cuda,
                                         pytorch=args.pytorch,
                                         tb=tb)
                logger.info(
                    f"{epoch}:test:{dict_loss['loss'],dict_loss['rms_']}")
            loss_test = dict_loss['loss']

            tb.add_scalars('loss',
                           {'train': loss_train,
                            'test': loss_test},
                           epoch)

            # save the model in the end of each epoch
            if args.pytorch:
                torch.save(
                    net.state_dict(),
                    os.path.join(
                        args.target,
                        f"state_dict_{args.model}_{args.window_train}_{args.window_predict}_{tool}_{args.data}_{args.station}_{uid}.pth"
                    ))

    # Test mode
    print_blue("TESTING")

    print_blue("Loading model")
    if args.pytorch:
        net.load_from_filename(
            os.path.join(
                args.target,
                f"state_dict_{args.model}_{args.window_train}_{args.window_predict}_{tool}_{args.data}_{args.station}_{uid}.pth"
            ))

        with torch.no_grad():
            dict_loss = process_data(net=net,
                                     optimizer=optimizer,
                                     criterion=criterion,
                                     train_loader=seq_train,
                                     test_loader=seq_test,
                                     window_train=args.window_train,
                                     window_predict=args.window_predict,
                                     diff=args.diff,
                                     cuda=args.cuda,
                                     pytorch=args.pytorch)
    else:
        dict_loss = process_data(net=net,
                                 optimizer=optimizer,
                                 criterion=criterion,
                                 train_loader=seq_train,
                                 test_loader=seq_test,
                                 window_train=args.window_train,
                                 window_predict=args.window_predict,
                                 diff=args.diff,
                                 training=False)

    log_loss(dict_loss, args.target, seq_length)


if __name__ == '__main__':
    main()
