# Anomaly-Detector
This repository contains the source code for deep learning based hierarchial anomaly detector developed as a part of CitySCENE grand challenge organized at ACM MM, 2020. Our team MonIIT ranked second in general anomaly detection task(ROC AUC 87.94) and fourth in specific anomaly detection task(Macro F1 45.52).

Link to the challenge website: https://cityscene.github.io/#/challenge

Report: https://dl.acm.org/doi/10.1145/3394171.3416302

# Installation
**Note: this repository requires Python 3.7+ to work**
1. Follow installation instructions from this repository to install detectoron2: https://github.com/airsplay/py-bottom-up-attention, this will also setup PyTorch and OpenCV.
2. Install tqdm and skvideo

Pretrained weights can be downloaded from here: https://drive.google.com/drive/folders/1iYjCekm571GgJbd4m_kT7jNEW05nJkjm?usp=sharing

# Running
Assuming data is located in the expected path as mentioned in [dataset/README.md](https://github.com/soumilkanwal80/Anomaly-Detector/blob/master/dataset/README.md)

### Data Preprocessing 
```
python preprocess.py  # extracts and reshapes each video frame into 171x128 and stores them as a folder of frames, also creates our train-val split.     
python icfeatures.py  # extract and stores mean pooled object representations 
```

### Training
```
python train_anomaly.py or python train_binary.py # training models from scratch
```

# Citation
If you find this useful, please cite this work as follows:
```

@inproceedings{10.1145/3394171.3416302,
author = {Kanwal, Soumil and Mehta, Vineet and Dhall, Abhinav},
title = {Large Scale Hierarchical Anomaly Detection and Temporal Localization},
year = {2020},
isbn = {9781450379885},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3394171.3416302},
doi = {10.1145/3394171.3416302},
abstract = {Abnormal event detection is a non-trivial task in machine learning. The primary reason behind this is that the abnormal class occurs sparsely, and its temporal location may not be available. In this paper, we propose a multiple feature-based approach for CitySCENE challenge-based anomaly detection. For motion and context information, Res3D and Res101 architectures are used. Object-level information is extracted by object detection feature-based pooling. Fusion of three channels above gives relatively high performance on the challenge Test set for the general anomaly task. We also show how our method can be used for temporal localisation of the abnormal activity event in a video.},
booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
pages = {4674–4678},
numpages = {5},
keywords = {anomaly detection, convolutional neural networks, CitySCENE},
location = {Seattle, WA, USA},
series = {MM '20}
}

```
