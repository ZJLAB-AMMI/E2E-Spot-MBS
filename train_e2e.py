#!/usr/bin/env python3
""" Training for E2E-Spot """
import copy
import os

from util.log import set_logger
from util.utils import adjust_learning_rate, moving_average, update_bn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import tabulate
import random
import torch
from util.conf import get_config
from model.e2e import get_model

from torch.optim.lr_scheduler import (ChainedScheduler, LinearLR, CosineAnnealingLR)
from dataset.soccernet_ball import get_dataloader
from util.io import store_json
import logging


def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(args.warm_up_epochs, cosine_epochs))
    return ChainedScheduler(
        [
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                     total_iters=args.warm_up_epochs * num_steps_per_epoch),
            CosineAnnealingLR(optimizer, num_steps_per_epoch * cosine_epochs)
        ]
    )


def main():
    def worker_init_fn(id):
        random.seed(id + epoch * 100)

    torch.backends.cudnn.benchmark = True

    # CONFIG
    args = get_config()

    args.dilate_len = 2
    args.stride = 2
    args.feature_arch = 'rny008_gsm'
    args.swa = False
    args.swa_start_epoch = 40

    # INIT
    param_str = '_f_arch_{}_t_arch_{}_dilate_len_{}_stride_{}'.format(
        args.feature_arch,
        args.temporal_arch,
        args.dilate_len,
        args.stride
    )
    save_path = os.path.join(args.save_dir, args.dataset, param_str)
    os.makedirs(save_path, exist_ok=True)
    store_json(os.path.join(save_path, 'config.json'), vars(args))
    set_logger(log_path=os.path.join(save_path, 'train.log'))

    # DATA
    train_loader, val_loader, classes = get_dataloader(args, worker_init_fn)

    # MODEL
    model = get_model(args, classes)

    checkpoint = None
    if args.resume and args.resume_epoch >= 0:
        checkpoint = torch.load(os.path.join(save_path, 'checkpoint_{}.pt'.format(args.resume_epoch)))

    if checkpoint is not None:
        model.load(checkpoint['state_dict'])

    swa_model = copy.deepcopy(model) if args.swa else None
    swa_n = 0

    # TRAIN
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})
    num_steps_per_epoch = len(train_loader) // args.acc_grad_iter
    lr_scheduler = get_lr_scheduler(args, optimizer, num_steps_per_epoch)
    start_epoch = 0
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
    losses = []

    for epoch in range(start_epoch, args.num_epochs):
        time_ep = time.time()

        if swa_model is not None and epoch > args.swa_start_epoch:
            adjust_learning_rate(optimizer, 0.0005)

        train_loss = model.epoch(
            train_loader,
            optimizer,
            scaler,
            lr_scheduler=lr_scheduler if swa_model is None or epoch <= args.swa_start_epoch else None,
            acc_grad_iter=args.acc_grad_iter
        )

        if swa_model is not None and epoch > args.swa_start_epoch:
            moving_average(swa_model._model, model._model, 1.0 / (swa_n + 1.0))
            update_bn(train_loader, swa_model._model, swa_model.device)
            swa_n += 1

        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.num_epochs:
            losses.append({'epoch': epoch, 'train_loss': train_loss})
            store_json(os.path.join(save_path, 'loss.json'), losses, pretty=True)
            torch.save(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict()
                },
                os.path.join(save_path, 'checkpoint_{}.pt'.format(epoch + 1))
            )

            if swa_model is not None and epoch > args.swa_start_epoch:
                torch.save(
                    {
                        'epoch': epoch + 1,
                        'state_dict': swa_model.state_dict()
                    },
                    os.path.join(save_path, 'checkpoint_{}_swa.pt'.format(epoch + 1))
                )

        time_ep = time.time() - time_ep
        columns = ["epoch", "learning_rate", "train_loss", "cost_time"]
        values = [epoch + 1, optimizer.param_groups[0]['lr'], train_loss, time_ep]
        table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
        if epoch % 50 == 0:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        logging.info(table)


if __name__ == '__main__':
    main()
