import numpy as np

from util.io import load_gz_json
from util.utils import select_frames, get_targets, compute_mAP, store_submit_files

"""
todo:
    add sub-model score results in file_path_list
    and, corresponding weight in wt_list
"""
file_path_list = []
wt_list = []

challenge_phase = False
# MAIN
ensemble_pred = {}
for i in range(len(file_path_list)):
    print(i)
    file_path = file_path_list[i]
    pred = load_gz_json(file_path)

    for k, v in pred.items():
        vs = np.stack(v, axis=0)
        if k in ensemble_pred:
            nf = min(ensemble_pred[k].shape[0], vs.shape[0])
            ensemble_pred[k] = ensemble_pred[k][:nf, :] + wt_list[i] * vs[:nf, :]
        else:
            ensemble_pred[k] = wt_list[i] * vs

predictions = []
videos = []
for k, v in ensemble_pred.items():
    predictions.append(select_frames(v))
    videos.append(k)

if challenge_phase:
    store_submit_files(videos, predictions, '/submit')
else:
    _, targets = get_targets(data_dir='/SoccerNet-Ball')
    mAP = compute_mAP(
        targets,
        predictions,
        delta=25
    )
    print(mAP)
