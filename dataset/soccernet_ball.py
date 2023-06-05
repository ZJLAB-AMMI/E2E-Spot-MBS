import os

from torch.utils.data import DataLoader

from dataset.frame import ActionSpotDataset, ActionSpotVideoDataset


def get_datasets(args):
    dataset_kwargs = {
        'crop_dim': args.crop_dim,
        'dilate_len': args.dilate_len,
        'mixup': args.mixup
    }

    if args.fg_upsample is not None:
        assert args.fg_upsample > 0
        dataset_kwargs['fg_upsample'] = args.fg_upsample

    dataset_len = args.epoch_num_frames // args.clip_len

    classes = {'DRIVE': 1, 'PASS': 2}

    train_data = None
    val_data = None

    if args.is_train:
        train_data = ActionSpotDataset(
            classes,
            os.path.join(args.data_dir, args.dataset, 'train.json'),
            os.path.join(args.data_dir, 'match-jpg'),
            args.modality,
            args.clip_len,
            dataset_len,
            is_eval=False,
            stride=args.stride,
            **dataset_kwargs
        )
        train_data.print_info()
    else:
        val_data = ActionSpotVideoDataset(
            classes,
            os.path.join(args.data_dir, args.dataset, 'test.json'),
            frame_dir=os.path.join(args.data_dir, 'match-jpg'),
            pad_len=args.pad_len,
            clip_len=args.clip_len,
            crop_dim=args.crop_dim,
            stride=args.stride,
            overlap_len=args.overlap_len
        )
        val_data.print_info()

    return classes, train_data, val_data


def get_dataloader(args, worker_init_fn=None):
    classes, train_data, val_data = get_datasets(args)
    train_loader = None
    val_loader = None
    if args.is_train:
        train_loader = DataLoader(
            train_data,
            shuffle=False,
            batch_size=args.batch_size // args.acc_grad_iter,
            pin_memory=True,
            num_workers=min(os.cpu_count(), args.num_workers),
            prefetch_factor=1,
            worker_init_fn=worker_init_fn
        )
    else:
        val_loader = DataLoader(
            val_data,
            num_workers=min(os.cpu_count(), args.num_workers),
            pin_memory=True,
            batch_size=args.batch_size
        )
    return train_loader, val_loader, classes
