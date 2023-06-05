import json
import os
from collections import defaultdict

import numpy as np
import torch

from util.io import store_json


def get_targets(num_classes=2, framerate=25, data_dir=None):
    SoccerNet_path = os.path.join(data_dir, 'match')
    test_game_list = [
        'england_efl/2019-2020/2019-10-01 - Reading - Fulham',
        'england_efl/2019-2020/2019-10-01 - Stoke City - Huddersfield Town'
    ]
    label_files = "Labels-ball.json"
    targets = []
    videos = []
    for game in test_game_list:
        labels = json.load(open(os.path.join(SoccerNet_path, game, label_files)))
        vector_size = 90 * 60 * framerate
        label_half1 = np.zeros((vector_size, num_classes))
        label_half2 = np.zeros((vector_size, num_classes))
        for annotation in labels["annotations"]:
            event = annotation["label"]
            if event not in {'PASS', 'DRIVE'}:
                continue
            half = int(annotation["gameTime"][0])
            frame = int(framerate * (int(annotation["position"]) / 1000))
            label = 0 if event == 'PASS' else 1
            if half == 1:
                frame = min(frame, vector_size - 1)
                label_half1[frame][label] = 1
            if half == 2:
                frame = min(frame, vector_size - 1)
                label_half2[frame][label] = 1
        targets.append(label_half1)
        targets.append(label_half2)
        videos.append(os.path.join(game, '1'))
        videos.append(os.path.join(game, '2'))
    return videos, targets


def get_predictions(num_classes=2, framerate=25):
    Predictions_path = '/AMMI_DATA_01/WLP/competition/soccernet_ball/eval_result'
    prediction_file = "results_spotting.json"
    test_game_list = [
        'england_efl/2019-2020/2019-10-01 - Reading - Fulham',
        'england_efl/2019-2020/2019-10-01 - Stoke City - Huddersfield Town'
    ]

    predictions = []
    for game in test_game_list:
        prediction = json.load(open(os.path.join(Predictions_path, game, prediction_file)))
        vector_size = 90 * 60 * framerate
        prediction_half1 = np.zeros((vector_size, num_classes)) - 1
        prediction_half2 = np.zeros((vector_size, num_classes)) - 1
        for annotation in prediction["predictions"]:
            event = annotation["label"]
            if event not in {'PASS', 'DRIVE'}:
                continue
            half = int(annotation["half"])
            frame = int(framerate * (int(annotation["position"]) / 1000))
            label = 0 if event == 'PASS' else 1
            value = annotation["confidence"]
            if half == 1:
                frame = min(frame, vector_size - 1)
                prediction_half1[frame][label] = value
            if half == 2:
                frame = min(frame, vector_size - 1)
                prediction_half2[frame][label] = value
        predictions.append(prediction_half1)
        predictions.append(prediction_half2)
    return predictions


def compute_class_scores(target, detection, delta):
    gt_indexes = np.where(target != 0)[0]
    pred_indexes = np.where(detection > 0)[0]
    pred_scores = detection[pred_indexes]

    game_detections = np.zeros((len(pred_indexes), 2))
    game_detections[:, 0] = np.copy(pred_scores)

    remove_indexes = list()

    for gt_index in gt_indexes:
        max_score = -1
        max_index = None
        game_index = 0
        selected_game_index = 0

        for pred_index, pred_score in zip(pred_indexes, pred_scores):

            if pred_index < gt_index - delta / 2:
                game_index += 1
                continue
            if pred_index > gt_index + delta / 2:
                break

            if abs(pred_index - gt_index) <= delta / 2 and pred_score > max_score and pred_index not in remove_indexes:
                max_score = pred_score
                max_index = pred_index
                selected_game_index = game_index
            game_index += 1

        if max_index is not None:
            game_detections[selected_game_index, 1] = 1
            remove_indexes.append(max_index)

    return game_detections, len(gt_indexes)


def compute_precision_recall_curve(targets, detections, delta):
    # Store the number of classes
    num_classes = targets[0].shape[-1]
    # 200 confidence thresholds between [0,1]
    thresholds = np.linspace(0, 1, 200)
    # Store the precision and recall points
    precision = list()
    recall = list()

    for c in np.arange(num_classes):
        total_detections = np.zeros((1, 2))
        total_detections[0, 0] = -1
        n_gt_labels = 0
        # Get the confidence scores and their corresponding TP or FP characteristics for each game
        for target, detection in zip(targets, detections):
            tmp_detections, tmp_n_gt_labels = compute_class_scores(
                target[:, c],
                detection[:, c],
                delta
            )
            total_detections = np.append(total_detections, tmp_detections, axis=0)
            n_gt_labels += tmp_n_gt_labels
        precision.append(list())
        recall.append(list())
        for threshold in thresholds:
            pred_indexes = np.where(total_detections[:, 0] >= threshold)[0]
            TP = np.sum(total_detections[pred_indexes, 1])
            p = TP / len(pred_indexes) if len(pred_indexes) > 0 else 0
            r = TP / n_gt_labels if n_gt_labels > 0 else 0
            precision[-1].append(p)
            recall[-1].append(r)
    precision = np.array(precision).transpose()
    recall = np.array(recall).transpose()
    for i in np.arange(num_classes):
        index_sort = np.argsort(recall[:, i])
        precision[:, i] = precision[index_sort, i]
        recall[:, i] = recall[index_sort, i]
    return precision, recall


def compute_mAP(targets, detections, delta=25):
    precision, recall = compute_precision_recall_curve(targets, detections, delta)
    AP = np.array([0.0] * precision.shape[-1])
    for i in np.arange(precision.shape[-1]):

        # 11 point interpolation
        for j in np.arange(11) / 10:

            index_recall = np.where(recall[:, i] >= j)[0]

            possible_value_precision = precision[index_recall, i]
            max_value_precision = 0

            if possible_value_precision.shape[0] != 0:
                max_value_precision = np.max(possible_value_precision)

            AP[i] += max_value_precision
    return np.mean(AP / 11.0)


def select_frames(score=None, window=10, framerate=25, threshold=0.01):
    """
    score: [num_frame, 3], 0-NO, 1-DRIVE, 2-PASS
    output: [num_frame, 2], 0-PASS, 1-DRIVE, val=confidence
    """
    output_len = 5400 * framerate
    output = np.zeros((output_len, 2))
    if score is None:
        return output

    for lidx in range(2):
        for i in range(score.shape[0]):
            start = max(0, i - window)
            end = min(score.shape[0], i + window)
            for j in range(start, end):
                if score[j, lidx + 1] > score[i, lidx + 1]:
                    break
            else:
                if score[i, lidx + 1] >= threshold and i < output_len:
                    output[i, 1 - lidx] = score[i, lidx + 1]
    return output


def store_eval_files(raw_pred, eval_dir):
    game_pred = defaultdict(list)
    for obj in raw_pred:
        game, half = obj['video'].rsplit('/', 1)
        half = int(half)
        for event in obj['events']:
            ss = event['frame'] / obj['fps']
            position = int(ss * 1000)

            mm = int(ss / 60)
            ss = int(ss - mm * 60)
            game_pred[game].append({
                'gameTime': '{} - {}:{:02d}'.format(half, mm, ss),
                'label': event['label'],
                'half': str(half),
                'position': str(position),
                'confidence': str(event['score'])
            })

    for game, pred in game_pred.items():
        game_out_dir = os.path.join(eval_dir, game)
        os.makedirs(game_out_dir, exist_ok=True)
        store_json(os.path.join(game_out_dir, 'results_spotting.json'), {
            'UrlLocal': game, 'predictions': pred
        }, pretty=True)


def store_submit_files(videos, predictions, save_dir, fps=25):
    class_dict = {0: 'PASS', 1: 'DRIVE'}
    game_pred = defaultdict(list)
    for i in range(len(videos)):
        video_name = videos[i]
        game, half = video_name.rsplit('/', 1)
        prediction = predictions[i]
        for j in range(prediction.shape[0]):
            for k in range(2):
                if prediction[j, k] <= 0:
                    continue
                ss = j / fps
                position = int(ss * 1000)
                mm = int(ss / 60)
                ss = int(ss - mm * 60)
                game_pred[game].append({
                    'gameTime': '{} - {}:{:02d}'.format(half, mm, ss),
                    'label': class_dict[k],
                    'half': str(half),
                    'position': str(position),
                    'confidence': str(prediction[j, k])
                })

    for game, pred in game_pred.items():
        game_out_dir = os.path.join(save_dir, game)
        os.makedirs(game_out_dir, exist_ok=True)
        store_json(
            os.path.join(game_out_dir, 'results_spotting.json'),
            {
                'UrlLocal': game,
                'predictions': pred
            },
            pretty=True
        )


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 1.0 - alpha
        param1.data += param2.data * alpha


def update_bn(loader, model, device=None):
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for index, input in enumerate(loader):
        if isinstance(input, (list, tuple)):
            input = input[0]

        frame = loader.dataset.load_frame_gpu(input, device)
        model(frame)
        if index > 0.3 * len(loader):
            break

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)
