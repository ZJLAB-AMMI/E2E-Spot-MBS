# Metric Boosting Strategy of E2E-Spot for Ball Action Spotting: One State-of-the-Art to the SoccerNet Challenge 2023
This is the Python code used to implement the E2E-Spot-MBS method as described in the technical report

[//]: # ([**Metric Boosting Strategy of E2E-Spot for Ball Action Spotting: One State-of-the-Art to the SoccerNet Challenge 2023**  )

[//]: # (Luping Wang, Hao Guo, Bin Liu]&#40;https:xx&#41;)

## Abstract
This technical report presents our solution, which was submitted to the Ball Action Spotting task in SoccerNet Challenge 2023, featured at CVPR'23. Details regarding the challenge are available at https://www.soccer-net.org/tasks/ball-action-spotting. Our solution achieved 86.37 in test phase and 83.47 in challenge phase in terms of mean of Average Precision (mAP@1). According to our understanding of the key difficulties in this challenge, we proposed a Metric Boosting Strategy (MBS) to tackle all the possible difficulties those a single E2E-Spot model may faced. MBS is a strategy that used to determine which sub-model to be selected as well as its weight based on a objective function which is related to the metric concerned in the challenge. And, each sub-model owns part of abilities of solving the possible difficulties. Specifically, each sub-model is trained by varying the training samples, the feature architecture and the optimizer. Efficient combination of these sub-model candidates made our solution performs excellent in the challenge. Our code is available at https://github.com/ZJLAB-AMMI/E2E-Spot-MBS.
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