# FSPMU

**Feature Space-Preserving Machine Unlearning for Robust Image Classification with Noisy Labels**  


## Overview

This repository contains the implementation of **FSPMU**, a novel noisy-label learning framework leveraging machine unlearning. By utilizing class centroids in the feature space to perform selective unlearning, our method successfully removes noisy samples from pre-trained models, achieving significant improvements in accuracy.

## Requirements
- Python==3.10.12
- CUDA==12.4
- PyTorch==2.1.0a0


## Datasets
We utilized the CIFAR-10 and CIFAR-100. 

## Model Training

The model training process consists of two main steps: **Pre-trained Model Training** and **Model Unlearning**. A framework for unlearning for forgetting noisy labels. The proposed method enables post-hoc model improvement by forgetting incorrect information through feature-based unlearning and recovering correct knowledge using the same feature representations. 

---

### Step 1: Pre-trained Model Training
You indicate dataeset and noise_mode, noise-rate.
For example, to run an experiment for cifar-10, with 50% symmetric noise, run:
```shell
python main.py --dataset cifar10 --noise_rate 0.5 --noise_mode sym --save True --tsne True
```
--save: When set to True, the trained model weights will be saved in the weight/ directory, located two levels above the execution directory.

--tsne: When set to True, a 2D visualization of the feature space using t-SNE will be generated and saved in the current directory.

Note: This feature is exclusively available for the CIFAR-10 dataset.

#### Expected Directory Structure
```text
.
├── weight/
│   ├── net/             # Saved models (.pth, etc.) --save True
│   └── TSNE/            # t-SNE visualization images --tsne True
└── your_project/
    └── scripts/         # Execute your script here
```

### Step 2: Model Unlearning
You indicate pre-trained method, dataset and noise_mode to select model to do unlearning.
For example, to run an experiment for cifar-10, with 50% symmetric noise, run:
```shell
python unlearning.py --dataset cifar10 --noise_rate 0.5 --noise_mode sym --method pro  --pred gmm --pretrain_method None --tsne 0 --save True
```
--method: Select `scrub` or `pro`. `scrub` uses the SCRUB method, while `pro` uses the proposed method.

--pred: Specifies the prediction method. Set this to `GMM` to use a Gaussian Mixture Model. To use known noisy labels instead, set this to `None`.

--tsne: Choose from (0, 1, 2).  
'0' disables TSNE image generation.  
'1' generates TSNE images after the `e_n` and `e_r` epochs.  
'2' generates the same TSNE images as '1', plus an additional image from the epoch with the best accuracy.

--save: When set to `True`, the trained model weights will be saved in the `weight/` directory located two levels above the execution directory.

--pretrain_method: Choice None, DivideMix, ProMix, or LongReMix. None corresponds to standard CE learning (Original). To use models trained with methods other than Original, please refer to the following repository:

-DivideMix:
https://github.com/LiJunnan1992/DivideMix

-ProMix:
https://github.com/Justherozen/ProMix

-LongReMix:
https://github.com/filipe-research/LongReMix


### Step 3: Feature Space Visualization
To create TSNE visualizations for the unlearned model, move the model generated in Step 2 to the current directory (`./`).

For example, to run an experiment on CIFAR-10 with 50% symmetric noise, execute:
```shell
python visualize.py --dataset cifar10 --noise_rate 0.5 --noise_mode sym --method pro  --pred gmm --pretrain_method None
```

## Parameters
Original and Noisy Label Learning(NLL) use each following parameter.
| Method | $\delta$ | $\zeta$ | $\gamma$ | $t$ | batch size (forget set) | batch size (retain set) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Original | 500 | 0.5 | 1 | 0.25 | 512 | 128 |
| NLL | 1 | 1 | 5 | 0.25 | 512 | 128 |
## Results
CIFAR10
## 1枚目

### CIFAR-10

| Method | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | Asym. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 89.04 | 83.13 | 76.46 | 67.51 | 57.83 | 47.93 | 35.91 | 25.57 | 16.41 | 77.50 |
| SCRUB (Last) | 91.82 | 90.27 | 89.03 | 87.08 | 84.83 | 82.55 | 78.24 | 68.67 | 38.23 | 91.22 |
| SCRUB (Best) | 91.92 | 90.35 | 89.08 | 87.14 | 85.15 | 82.65 | 78.48 | 69.14 | 48.62 | 91.29 |
| Proposed (Last) | 92.36 | 91.85 | 91.10 | 90.52 | 89.01 | 88.22 | 86.05 | 83.30 | 74.67 | 91.86 |
| Proposed (Best) | 92.50 | 92.09 | 91.36 | 90.53 | 89.42 | 88.22 | 86.16 | 83.30 | 74.67 | 91.93 |
| DivideMix | 95.32 | 95.13 | 93.17 | 94.00 | 94.13 | 94.21 | 93.67 | 92.25 | 62.67 | 92.24 |
| DivideMix+proposed | 95.93 | 95.86 | 95.88 | 95.50 | 95.59 | 95.36 | 94.43 | 93.56 | 83.09 | 95.30 |
| ProMix | 97.15 | 96.87 | 96.89 | 96.76 | 96.43 | 96.16 | 96.21 | 91.66 | 80.04 | 96.02 |
| ProMix+proposed | 96.90 | 96.89 | 96.52 | 96.36 | 96.41 | 96.20 | 96.33 | 91.85 | 83.35 | 96.11 |
| LongReMix | 95.59 | 95.34 | 93.37 | 94.14 | 94.33 | 94.33 | 93.99 | 92.68 | 77.58 | 91.71 |
| LongReMix+proposed | 95.96 | 95.80 | 95.27 | 95.67 | 95.47 | 95.49 | 94.39 | 93.54 | 86.93 | 95.14 |
||||||||||||

### CIFAR-100

| Method | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | Asym. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 67.96 | 60.65 | 54.41 | 45.84 | 37.66 | 27.84 | 18.96 | 10.12 | 3.86 | 44.54 |
| SCRUB (Last) | 71.67 | 68.63 | 65.18 | 61.90 | 58.32 | 52.65 | 42.64 | 28.30 | 12.18 | 53.24 |
| SCRUB (Best) | 71.84 | 68.78 | 65.38 | 62.19 | 58.55 | 52.88 | 42.83 | 28.49 | 12.40 | 53.49 |
| Proposed (Last) | 73.12 | 71.13 | 69.70 | 66.75 | 62.89 | 58.58 | 51.20 | 36.37 | 22.59 | 70.69 |
| Proposed (Best) | 73.22 | 71.31 | 69.70 | 67.15 | 63.63 | 58.60 | 51.22 | 36.68 | 22.76 | 70.95 |
| DivideMix | 74.45 | 74.89 | 72.72 | 72.64 | 72.18 | 69.88 | 65.86 | 56.34 | 27.11 | 51.07 |
| DivideMix+proposed | 79.10 | 78.10 | 77.92 | 76.90 | 75.01 | 72.68 | 69.38 | 61.21 | 38.74 | 75.00 |
| ProMix | 80.41 | 79.83 | 79.01 | 78.18 | 76.41 | 73.57 | 60.37 | 43.02 | 20.66 | 74.09 |
| ProMix+proposed | 81.06 | 80.37 | 79.15 | 78.23 | 76.45 | 73.76 | 61.58 | 43.70 | 30.09 | 77.69 |
| LongReMix | 75.84 | 75.03 | 74.44 | 73.69 | 71.41 | 68.73 | 62.89 | 52.44 | 31.53 | 52.15 |
| LongReMix+proposed | 79.24 | 77.94 | 77.27 | 75.86 | 74.22 | 75.13 | 67.27 | 58.13 | 45.36 | 75.01 |
||||||||||||

CIFAR100
## 2枚目

### CIFAR-10

| Method | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | Asym. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 89.04 | 83.13 | 76.46 | 67.51 | 57.83 | 47.93 | 35.91 | 25.57 | 16.41 | 77.50 |
| SCRUB (Last) | 89.13 | 83.80 | 77.57 | 68.29 | 53.85 | 47.91 | 36.41 | 25.86 | 16.86 | 77.23 |
| SCRUB (Best) | 89.16 | 83.53 | 77.57 | 68.17 | 59.50 | 48.81 | 36.77 | 26.36 | 17.15 | 77.55 |
| Proposed (Last) | 92.54 | 91.34 | 89.84 | 87.94 | 83.58 | 75.34 | 62.85 | 46.81 | 33.67 | 77.85 |
| Proposed (Best) | 92.54 | 91.41 | 90.17 | 87.96 | 85.21 | 79.82 | 72.21 | 58.76 | 34.78 | 79.03 |
| DivideMix | 95.32 | 95.13 | 93.17 | 94.00 | 94.13 | 94.21 | 93.67 | 92.25 | 62.67 | 92.24 |
| DivideMix+proposed | 95.55 | 95.57 | 94.04 | 94.69 | 94.78 | 94.80 | 94.13 | 92.99 | 62.40 | 92.16 |
| ProMix | 97.15 | 96.87 | 96.89 | 96.76 | 96.43 | 96.16 | 96.21 | 91.66 | 80.04 | 96.02 |
| ProMix+proposed | 97.07 | 96.82 | 96.49 | 96.78 | 96.50 | 96.05 | 96.17 | 91.74 | 80.51 | 95.68 |
| LongReMix | 95.59 | 95.34 | 93.37 | 94.14 | 94.33 | 94.33 | 93.99 | 92.68 | 77.58 | 91.71 |
| LongReMix+proposed | 95.86 | 95.45 | 93.87 | 94.81 | 94.61 | 95.14 | 94.56 | 93.06 | 77.60 | 92.29 |
||||||||||||

### CIFAR-100

| Method | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | Asym. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 67.96 | 60.65 | 54.41 | 45.84 | 37.66 | 27.84 | 18.96 | 10.12 | 3.86 | 44.54 |
| SCRUB (Last) | 67.97 | 60.79 | 54.54 | 45.92 | 37.64 | 28.18 | 18.93 | 10.02 | 4.08 | 44.22 |
| SCRUB (Best) | 67.97 | 60.79 | 54.52 | 46.22 | 37.92 | 28.42 | 19.27 | 10.27 | 4.21 | 44.66 |
| Proposed (Last) | 71.95 | 67.09 | 59.62 | 51.29 | 42.23 | 31.36 | 22.35 | 12.22 | 4.85 | 47.63 |
| Proposed (Best) | 72.00 | 67.59 | 61.26 | 52.83 | 43.99 | 34.24 | 23.87 | 12.40 | 4.85 | 48.22 |
| DivideMix | 74.45 | 74.89 | 72.72 | 72.64 | 72.18 | 69.88 | 65.86 | 56.34 | 27.11 | 51.07 |
| DivideMix+proposed | 78.34 | 77.72 | 75.85 | 76.09 | 75.03 | 72.45 | 68.37 | 59.45 | 28.15 | 52.54 |
| ProMix | 80.41 | 79.83 | 79.01 | 78.18 | 76.41 | 73.57 | 60.37 | 43.02 | 20.66 | 74.09 |
| ProMix+proposed | 81.69 | 80.09 | 80.51 | 79.22 | 77.85 | 74.62 | 61.58 | 43.70 | 22.05 | 74.81 |
| LongReMix | 75.84 | 75.03 | 74.44 | 73.69 | 71.41 | 68.73 | 62.89 | 52.44 | 31.53 | 52.15 |
| LongReMix+proposed | 78.61 | 78.17 | 76.04 | 77.17 | 75.11 | 73.75 | 66.02 | 56.31 | 34.06 | 53.01 |
||||||||||||
## Paper Status
The paper is currently under review at IEEE Access.

Note:
As the manuscript is under peer review, the repository is currently in a limited-release state. Some details, including datasets, trained models, and complete documentation, will be provided after the review process concludes.


## Citation

A BibTeX entry will be provided here upon acceptance.
