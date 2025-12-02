# FSPMU

**Feature Space-Preserving Machine Unlearning for Robust Image Classification with Noisy Labels**  


## Overview

This repository contains the implementation of **FSPMU**, 

## Requirements
- Python==3.10.12
- CUDA==12.4
- PyTorch==2.1.0a0
- tqdm==4.64.0
- torchsummary==1.5.1
- loguru
- sentence-transformer==2.2.2

## Datasets
We utilized the CIFAR-10 and CIFAR-100. 

## Model Training

The model training process consists of two main steps: **Pre-trained Model Training** and **Model Unlearning**.

---

### Step 1: Pre-trained Model Training

You indicate dataeset and noise_mode, noise-rate and run by follows:

```shell
# python main.py --dataset cifar100 noise_mode sym --noise_rate 0.5
```

### Step 2: Model Unlearning





## Results


## Paper Status



## Citation

A BibTeX entry will be provided here upon acceptance.
