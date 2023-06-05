#!/usr/bin/env python3
""" Training for E2E-Spot """

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import logging
import torch
import tabulate
from util.conf import get_config
from model.e2e import get_model

from dataset.soccernet_ball import get_dataloader

from util.log import set_logger


def main():
    torch.backends.cudnn.benchmark = True

    # CONFIG
    args = get_config()
    args.is_train = False
    args.resume_epoch = 100
    args.batch_size = 16
    args.overlap_len = 99

    args.dilate_len = 2
    args.stride = 2
    args.feature_arch = 'rny008_gsm'
    args.swa = False
    args.swa_start_epoch = 40
    args.save_predict_result = True

    # INIT
    param_str = '_f_arch_{}_t_arch_{}_dilate_len_{}_stride_{}'.format(
        args.feature_arch,
        args.temporal_arch,
        args.dilate_len,
        args.stride
    )

    save_path = os.path.join(args.save_dir, args.dataset, param_str)
    set_logger(log_path=os.path.join(save_path, 'eval.log'))

    # DATA
    train_loader, val_loader, classes = get_dataloader(args)

    # MODEL
    model = get_model(args, classes)

    resume_epoch_set = [35]
    for i in range(len(resume_epoch_set)):
        args.resume_epoch = resume_epoch_set[i]
        if args.swa:
            checkpoint = torch.load(os.path.join(save_path, 'checkpoint_{}_swa.pt'.format(args.resume_epoch)))
        else:
            checkpoint = torch.load(os.path.join(save_path, 'checkpoint_{}.pt'.format(args.resume_epoch)))
        model.load(checkpoint['state_dict'])

        # TEST
        time_ep = time.time()
        mAP = model.evaluate(args, val_loader, classes, save_path)
        time_ep = time.time() - time_ep

        columns = ["resume_epoch", "overlap_len", "mAP", "cost_time"]
        values = [args.resume_epoch, args.overlap_len, mAP, time_ep]
        table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
        if i % 50 == 0:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        logging.info(table)


if __name__ == '__main__':
    main()
