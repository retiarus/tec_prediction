"""
main file

License is from https://github.com/aboulch/tec_prediction
"""

import argparse
import os

import torch
from colors import print_blue, print_green, print_red
from data_loader_1 import SequenceLoader
from log_loss import log_loss
from process import process_data

#from torchsummary import summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_length_min",
                        type=int,
                        default=7200,
                        help="train network or not")
    parser.add_argument("--step_min", type=int, default=10)
    parser.add_argument("--window_train", type=int, default=432)
    parser.add_argument("--window_predict", type=int, default=288)
    parser.add_argument("--batch_size",
                        type=int,
                        default=50,
                        help="train network or not")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--cuda", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--model", type=str, default="simple")
    parser.add_argument("--diff", type=bool, default=False)
    parser.add_argument("--target",
                        type=str,
                        default="./results",
                        help="target directory")
    parser.add_argument("--source", type=str, help="source directory")
    args = parser.parse_args()
    epoch_max = args.epochs

    seq_length = int(args.seq_length_min / args.step_min)

    print_green("Sequence length:", seq_length)

    # create the result directory
    if not os.path.exists(args.target):
        os.makedirs(args.target)

    # define optimization parameters
    root_dir = args.source

    # CUDA
    # if args.cuda:
    # torch.backends.cudnn.benchmark = True

    print_blue("Creating data loader...")
    ds = SequenceLoader('train', root_dir, args.seq_length_min, args.step_min,
                        args.window_train, args.window_predict)
    ds_val = SequenceLoader('validation', root_dir, args.seq_length_min,
                            args.step_min, args.window_train,
                            args.window_predict)
    train_loader = torch.utils.data.DataLoader(ds,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(ds_val,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=2)

    print_blue("Creating network...")
    if args.model == "simple":
        from network_simple import SimpleConvRecurrent
        net = SimpleConvRecurrent(1)
    elif args.model == "unet":
        from network_unet import UnetConvRecurrent
        net = UnetConvRecurrent(1)
    elif args.model == "dilation121":
        from network_dilation_121 import UnetConvRecurrent
        net = UnetConvRecurrent(1)
    else:
        print_red("Error bad network")
        exit()

    # if args.cuda:
    #     net.cuda()

    print("PARAMTERS")

    #    summary(net, (args.window_train, args.batch, 1, 72, 72))

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(net))
    # exit()

    print_blue("Setting up the optimizer...")
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    print_blue("Setting up the criterion...")
    criterion = torch.nn.L1Loss()

    if not args.test:
        print_blue("TRAINING")

        # iterate on epochs
        for epoch in range(epoch_max):

            print_green("Epoch", epoch)

            # train
            dict_loss = process_data(net=net,
                                     optimizer=optimizer,
                                     criterion=criterion,
                                     train_loader=train_loader,
                                     test_loader=test_loader,
                                     window_train=args.window_train,
                                     window_predict=args.window_predict,
                                     diff=args.diff,
                                     cuda=args.cuda,
                                     training=True)

            # save the model
            torch.save(net.state_dict(),
                       os.path.join(args.target, "state_dict.pth"))

    # Test mode
    print_blue("TESTING")

    print_blue("Loading model")
    net.load_from_filename(os.path.join(args.target, "state_dict.pth"))

    with torch.no_grad():
        dict_loss = process_data(net=net,
                                 optimizer=optimizer,
                                 criterion=criterion,
                                 train_loader=train_loader,
                                 test_loader=test_loader,
                                 window_train=args.window_train,
                                 window_predict=args.window_predict,
                                 diff=args.diff,
                                 cuda=args.cuda)

    log_loss(dict_loss, args.target, seq_length)


if __name__ == '__main__':
    main()
