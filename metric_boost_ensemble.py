import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from glob import glob

import numpy as np
from natsort import natsorted

from util.io import load_gz_json
from util.utils import select_frames, get_targets, compute_mAP
from util.log import set_logger
import logging

data_dir = '/SoccerNet-Ball'
dataset = 'soccernet_ballpValid'
dilate_stride_list = [(5, 2), (4, 2), (2, 2), (5, 1), (4, 1)]
t_arch = 'gru'

set_logger(log_path=os.path.join('/competition', dataset, 'metric_boost_ensemble.log'))

# DATA
sub_model_root_dir = '/competition'
f_arch_list = ['rny008_gsm', 'efficientnet_gsm']
pred_file_list = []

for f_arch in f_arch_list:
    for dilate_stride in dilate_stride_list:
        dilate_len = dilate_stride[0]
        stride = dilate_stride[1]
        files1 = natsorted(
            glob(
                os.path.join(
                    sub_model_root_dir,
                    dataset,
                    '_f_arch_{}_t_arch_{}_dilate_len_{}_stride_{}'.format(f_arch, t_arch, dilate_len, stride),
                    'pred_test.*_99.score.json.gz'
                )
            )
        )

        files2 = natsorted(
            glob(
                os.path.join(
                    sub_model_root_dir,
                    dataset,
                    '_f_arch_{}_t_arch_{}_dilate_len_{}_stride_{}_swa'.format(f_arch, t_arch, dilate_len, stride),
                    'pred_test.*_95.score.json.gz'
                )
            )
        )

        pred_file_list += files1 + files2

print("total_files:{}".format(len(pred_file_list)))

# MAIN

_, targets = get_targets(data_dir=data_dir)

pred_file_score_list = []
for pred_file in pred_file_list:
    pred = load_gz_json(pred_file)
    predictions = []
    for k, v in pred.items():
        vs = np.stack(v, axis=0)
        predictions.append(select_frames(vs))
    mAP = compute_mAP(targets, predictions)
    pred_file_score_list.append((pred_file, mAP))
pred_file_score_list.sort(key=lambda x: x[1], reverse=True)

# the first
best_ensemble_pred = {}
pred = load_gz_json(pred_file_score_list[0][0])
for k, v in pred.items():
    best_ensemble_pred[k] = np.stack(v, axis=0)
best_mAP = pred_file_score_list[0][1]
selected_pred_files = [pred_file_score_list[0][0]]
selected_pred_files_weight = [1.0]

# others
weight_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
num_candidates = len(pred_file_score_list) // 3
while True:
    cur_bst_ens_pred = {}
    cur_bst_mAP = best_mAP
    cur_selected_pred_file = None
    cur_selected_pred_file_weight = None

    # select sub model
    for pred_file_score in pred_file_score_list[:num_candidates]:
        pred_file = pred_file_score[0]
        if pred_file in selected_pred_files:
            continue

        pred = load_gz_json(pred_file)

        ct = 0
        for wt in weight_list:

            predictions = []
            ens_tmp = {}

            for k, v in pred.items():
                vs = np.stack(v, axis=0)
                last_ens_pred = best_ensemble_pred[k]

                nf = min(last_ens_pred.shape[0], vs.shape[0])
                vs = vs[:nf, :]
                last_ens_pred = last_ens_pred[:nf, :]

                vs_sum = np.sum(vs, axis=-1, keepdims=True)

                cur_ens_pred = np.where(
                    vs_sum <= 0.001,
                    last_ens_pred,
                    (1.0 - wt) * last_ens_pred + wt * vs
                )

                predictions.append(select_frames(cur_ens_pred))
                ens_tmp[k] = cur_ens_pred

            mAP = compute_mAP(targets, predictions)

            print(cur_bst_mAP, mAP)

            if mAP > cur_bst_mAP:
                cur_bst_ens_pred = ens_tmp
                cur_selected_pred_file = pred_file
                cur_selected_pred_file_weight = wt
                cur_bst_mAP = mAP
                ct = 0
            else:
                ct += 1

            if ct >= 3:
                break

    # store sub model
    if cur_selected_pred_file is None:
        break
    else:
        best_ensemble_pred = cur_bst_ens_pred
        best_mAP = cur_bst_mAP
        selected_pred_files.append(cur_selected_pred_file)
        selected_pred_files_weight = [
                                         sw * (1.0 - cur_selected_pred_file_weight) for sw in selected_pred_files_weight
                                     ] + [cur_selected_pred_file_weight]

    logging.info(selected_pred_files)
    logging.info(selected_pred_files_weight)
    logging.info(best_mAP)
