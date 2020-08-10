# Anomaly-Detector
This repository contains the source code for deep learning based hierarchial anomaly detector developed as a part of CitySCENE grand challenge organized at ACM MM, 2020. Our team MonIIT ranked second in general anomaly detection task(ROC AUC 87.94) and fourth in specific anomaly detection task(Macro F1 45.52).

Link to the challenge website: https://cityscene.github.io/#/challenge

Link to the report will be added soon.

# Installation
**Note: this repository requires Python 3.7+ to work**
1. Follow installation instructions from this repository to install detectoron2: https://github.com/airsplay/py-bottom-up-attention, this will also setup PyTorch and OpenCV.
2. Install tqdm and skvideo

Pretrained weights can be downloaded from here: https://drive.google.com/drive/folders/1iYjCekm571GgJbd4m_kT7jNEW05nJkjm?usp=sharing

# Running
1. python preprocess.py       
2. python icfeatures.py
3. python train_anomaly.py/train_binary.py
