# FSPMU

**Feature Space-Preserving Machine Unlearning for Robust Image Classification with Noisy Labels**  


## Overview

This repository contains the implementation of **FSPMU**, 

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

You indicate dataeset and noise_mode, noise-rate and run by follows:

```shell

```

### Step 2: Model Unlearning
You indicate pre-trained method, dataset and noise_mode to select model to do unlearning. Run by follows:
```shell

```




## Results
![CIFAR result table](images/Result_CIFAR10.png)
![CIFAR result table](images/Result_CIFAR100.png)

## Paper Status



## Citation

A BibTeX entry will be provided here upon acceptance.
