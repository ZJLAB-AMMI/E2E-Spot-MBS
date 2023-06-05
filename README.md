# Metric Boosting Strategy of E2E-Spot for Ball Action Spotting: One State-of-the-Art to the SoccerNet Challenge 2023
This is the Python code used to implement the E2E-Spot-MBS method as described in the technical report

[//]: # ([**A Metric Boosting Strategy for Model Ensembling for Action Spotting in Videos**  )

[//]: # (Luping Wang, Hao Guo, Bin Liu]&#40;https:xx&#41;)

## Abstract
This technical report presents our solution to the Ball Action Spotting task submitted to the CVPR'23 SoccerNet Challenge. Details of this challenge can be found at https://www.soccer- net.org/tasks/ball-action-spotting. Our solution achieved a mean Average Precision (mAP@1) of 86.37% in the test phase and 83.47% in the challenge phase. Our approach is developed based on a baseline model termed E2E-Spot [3], which was provided by the organizers of this challenge. We first generated several variants of the E2E-Spot model, resulting in a candidate model set. We then proposed a strategy for selecting appropriate model members from this set and assigning an appropriate weight to each model. The aim of this strategy is to boost the performance metric of the resulting model ensemble. Therefore, we call our approach the metric boosting strategy (MBS). 
## Dataset
The dataset download can follow [here](https://www.soccer-net.org/data).

## Requirements

- moviepy==1.0.3
- natsort==8.3.1
- numpy==1.21.5
- opencv_contrib_python==4.2.0.32
- SoccerNet==0.1.51
- tabulate==0.8.10
- timm==0.4.12
- torch==1.12.1
- torchvision==0.13.1
- tqdm==4.64.0

## Setup
Download the dataset based on the description of the link above.

Framing all the video data by running ```frames_as_jpg_soccernet_ball.py``` and storing corresponding results in ```./match-jpg```.

Install all the required packages.

## Sub-model candidates preparasion
**sub-model training:**
execute ```python train_e2e.py``` after setting following specific hyperparameters of each sub-model ```dilate_len, stride, feature_arch, swa, swa_start_epoch```.

**sub-model inference:**
execute ```python test_e2e.py``` for each trained sub-model.

## Metric Boosting Strategy
- execute ```python metric_boost_ensemble.py``` to achieve the information of selecting sub-model and corresponding weight.

- execute ```python ensemble.py``` to achieve the ensembled result based on above achieved information.

## Disclaimer
Please only use the code and dataset for research purposes.

## Contact
Luping Wang</br>
Zhejiang Lab, Research Center for Applied Mathematics and Machine Intelligence</br>
wangluping@zhejianglab.com

Hao Guo</br>
Zhejiang Lab, Research Center for Applied Mathematics and Machine Intelligence</br>
guoh@zhejianglab.com
