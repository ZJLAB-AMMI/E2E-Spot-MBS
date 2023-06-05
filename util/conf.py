def data_args(parser):
    group = parser.add_argument_group('Data', '')
    group.add_argument("--dataset", type=str, default="soccernet_ballpValid", help=" ")
    group.add_argument('--data_dir', type=str, default='/SoccerNet-Ball')

    group.add_argument('--epoch_num_frames', type=int, default=500000)
    group.add_argument('--dilate_len', type=int, default=5)
    group.add_argument('--pad_len', type=int, default=5)
    group.add_argument('--stride', type=int, default=1)
    group.add_argument('--clip_len', type=int, default=100)
    group.add_argument('--overlap_len', type=int, default=99)
    group.add_argument('--fg_upsample', type=float)

    group.add_argument('--batch_size', type=int, default=8)
    group.add_argument('--acc_grad_iter', type=int, default=1)
    group.add_argument('--modality', type=str, default='rgb')
    group.add_argument('--crop_dim', type=int, default=224)
    group.add_argument('--mixup', type=bool, default=True)
    group.add_argument('--num_workers', type=int, default=8)


def model_args(parser):
    group = parser.add_argument_group('Model', '')
    group.add_argument('--feature_arch', type=str, choices=[
        # From torchvision
        'rn18',
        'rn18_tsm',
        'rn18_gsm',
        'rn50',
        'rn50_tsm',
        'rn50_gsm',

        # From timm (following its naming conventions)
        'rny002',
        'rny002_tsm',
        'rny002_gsm',
        'rny008',
        'rny008_tsm',
        'rny008_gsm',

        # From timm
        'convnextt',
        'convnextt_tsm',
        'convnextt_gsm'
        
        # From timm
        'efficientnet',
        'efficientnet_gsm'
    ], default='rny008_gsm', help='CNN architecture for feature extraction')
    group.add_argument('--temporal_arch', type=str, choices=['', 'gru', 'deeper_gru', 'mstcn', 'asformer'],
                       default='gru', help='Spotting architecture, after spatial pooling')
    group.add_argument('--gpu_parallel', action='store_true', default=False)
    group.add_argument('--pretrained', action='store_true', default=False)


def train_args(parser):
    group = parser.add_argument_group('Train', '')
    group.add_argument('--is_train', action='store_true', default=True)
    group.add_argument('--resume', action='store_true', default=False)
    group.add_argument('--swa', action='store_true', default=False)
    group.add_argument('--swa_start_epoch', type=int, default=40)
    group.add_argument('--resume_epoch', type=int, default=-1)
    group.add_argument('--num_epochs', type=int, default=100)
    group.add_argument('--warm_up_epochs', type=int, default=3)
    group.add_argument('--learning_rate', type=float, default=0.001)
    group.add_argument('--eval_sample_rate', type=float, default=1.0)
    group.add_argument('--eval_freq', type=int, default=5)
    group.add_argument('--save_predict_result', action='store_true', default=True)
    group.add_argument('--save_freq', type=int, default=5)
    group.add_argument('--save_dir', type=str, default='/competition')


def get_config():
    import argparse
    parser = argparse.ArgumentParser()

    data_args(parser)
    model_args(parser)
    train_args(parser)

    return parser.parse_args()
