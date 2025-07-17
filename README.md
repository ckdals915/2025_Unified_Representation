# Unified Representation for Components of Deep Neural Network as Message Passing

**Date:** 2025.07.17

**Laboratory:** Handong Global University Industrial Intelligence Lab

**Author:** Chang-Min Ahn & Young-Keun Kim*

**Github:** https://github.com/ckdals915/2025_Unified_Representation.git



## I. Introduction

This study proposes an unified representation that integrates the components of deep neural networks(MLP, CNN, Transformer, GNN) as message passing. Unified representation provides a mathematical formulation that facilitates structural analysis and comparative evaluation across different representations. The results validate that the mean absolute error (MAE) remains below $10^{-6}$ in the evaluation.

**Figure #1. Unified Representation Overview**

![](https://github.com/user-attachments/assets/8b69d380-a165-482e-a9d6-0ddf117357fe)



## II. Requirement

### Hardware

* NVIDIA GeForce RTX 4080 Laptop GPU
* NVIDIA Ampere A30 x 4

### Software

* Python 3.9.21
* CUDA == 11.8
* PyTorch == 2.1.2
* PyTorch Geometric == 2.5.3



## III. Installation

```bash
# Install Anaconda from website

# Update CONDA in Base
conda update -n base -c defaults conda

# Create myEnv=PyG_py39
conda create -n PyG_py39 python=3.9.21

# Activate myEnv
conda activate PyG_py39

# Install Numpy, OpenCV
conda install numpy==1.26.4
pip install opencv-python==4.11.0.86

# Install Scikit-Learn, tqdm
pip install scikit-learn==1.6.1 tqdm==4.67.1

# Install PyTorch
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torchsummary==1.5.1 torchinfo==1.8.0

# Install PyTorch Geometric (Windows)
pip install torch_geometric==2.5.3
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.2+cu118.html

# Check Installed Packaged in myENV
conda list
```



## IV. Program Tree

```bash
SOFTWARE
├─MNIST_MLP_Train.py                              # Train MLP/MLPAsGNN Model for MNIST
├─CWRU_1DCNN_Train.py                             # Train 1DCNN/1DCNNAsGNN Model for CWRU
├─CIFAR100_VGG11_Train.py                         # Train VGG11/VGG11AsGNN Model for CIFAR100
├─CIFAR100_ResNet18_Train.py                      # Train ResNet18/ResNet18AsGNN Model for CIFAR100
├─CIFAR100_ViT_Train.py                           # Train ViT/ViTAsGNN Model for CIFAR100
├─Memory_Inference_Analysis.py                    # Analysis GPU Memory/Inference Time 
│
├─modules
│  │  __init__.py
│  │  MLPAsGNN.py                                 # Linear to Message-Passing
│  │  ConvAsGNN.py                                # Conv1D/Conv2D to Message-Passing
│  │  PoolingAsGNN.py                             # Avg/MaxPooling to Message-Passing
│  └─ MultiHeadSelfAttentionAsGNN.py              # Transformer to Message-Passing
│
├─models
│  │  __init__.py
│  │  MNIST_MLP.py                                # MLP Custom Model for MNIST
│  │  CWRU_1DCNN.py                               # 1D-CNN Custom Model for CWRU
│  │  CIFAR100_VGG.py                             # VGG11 Model for CIFAR100
│  │  CIFAR100_ResNet18.py                        # ResNet18 Model for CIFAR100
│  └─ CIFAR100_ViT.py                             # ViT Model for CIFAR100
│
├─utils
│   │  __init__.py
│   │  CWRU_Data_Preprocessing.py                 # Convert CWRU Raw Data(*.mat) to NumPy(*.npy)
│   │  DataLoader_GNN.py                          # MNIST, CWRU, CIFAR100 DataLoader
│   │  Log_GNN.py                                 # Logging Function about Train
│   └─ Train_Test.py                              # Train/Evaluation/Test Function
│
├─data                                            # Dataset(MNIST,CWRU,CIFAR100)
│  ├─MNIST
│  ├─CWRU
│  └─cifar-100-python
│
├─checkpoint                                      # Checkpoint Models
│  │  SEED0_CNN1DCWRUAsGNN_CWRU_lr5e-05_best.pth
│  └─ ...
│
└─log                                             # Log Train Result(*.txt)
   │  SEED0_MLPAsGNNMNIST_bs128.txt 
   └─ ...
   
```



## V. Usage

### 5.1 Unified Representation Validation

To validate the unified representation of DNN components, we verified the equivalence between operations implemented with the standard and unified representations.

#### 5.1.1 Experiment Setup

* 10,000 random samples ($$x_i \sim \mathcal{N}(0,1)$$)

* Use identical model parameter and hyper parameter

* Compute the Mean Absolute Error (MAE) between the output-node vectors of the standard and unified representations

#### 5.1.2 Execution Procedure

##### 0) Environment Activation

```bash
$ cd {Local_Path}\2025_Unified_Representation
$ conda activate PyG_py39
```

##### 1) MLP Component

```bash
$ python -m modules.MLPAsGNN
```

##### 2) CNN Component

```bash
# 1D/2D Convolution
$ python -m modules.ConvAsGNN

# 1D/2D Pooling (Average, Max)
$ python -m modules.PoolingAsGNN
```

##### 3) Transformer Component

```bash
$ python -m modules.MultiHeadSelfAttentionAsGNN
```



**Unified Representation Validation Result**


<div align="center">
  <img src="https://github.com/user-attachments/assets/eaebc0dc-fd3d-437b-9885-df0b22bb2774"
       style="width:700px; height:auto;" />
</div>

### 5.2 Training Unified Representation-based Model

We designed an unified representation-based model. Comparative performance analyses were conducted using the MNIST, CWRU, and CIFAR-100 datasets.

#### 5.1.1 Experiment Setup

* Use identical training environment (batch size, data preprocessing, etc.)
* Conduct training of each model using 10 different random seeds

#### 5.1.2 Execution Procedure

##### 0) Environment Activation

```bash
$ cd {Local_Path}\2025_Unified_Representation
$ conda activate PyG_py39
```

##### 1) MNIST MLP Model Training

```bash
$ python -m MNIST_MLP_Train
```

##### 2) CWRU 1D-CNN Model Training

```bash
# CWRU Data Pre-Processing
$ python -m utils.CWRU_Data_Preprocessing

# CWRU 1D-CNN Model Training
$ python -m CWRU_1DCNN_Train
```

##### 3) CIFAR100 VGG11 Model Training

```bash
$ python -m CIFAR100_VGG11_Train
```

##### 4) CIFAR100 ResNet18 Model Training

```bash
$ python -m CIFAR100_ResNet18_Train
```

##### 5) CIFAR100 ViT Model Training

```bash
$ python -m CIFAR100_ViT_Train
```



**Unified Representation Model Accuracy Result**
![](https://github.com/user-attachments/assets/68c3ccb7-b460-45d6-bfdf-eec3a8b171de)


#### 5.1.3 GPU Memory & Inference Time Analysis

##### 0) Environment Activation

```bash
$ cd {Local_Path}\2025_Unified_Representation
$ conda activate PyG_py39
```

##### 1) GPU Memory & Inference Time Check

```bash
$ python -m Memory_Inference_Analysis
```

**Analysis Result**

![](https://github.com/user-attachments/assets/09774e6c-11c1-4bf4-9ad1-3c4a38c04cd9)

