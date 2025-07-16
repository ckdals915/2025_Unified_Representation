# Unified Representation for Components of Deep Neural Network as Message Passing

**Date:** 2025.07.17

**Laboratory:** Handong Global University Industrial Intelligence Lab

**Author:** Chang-Min Ahn & Young-Keun Kim*



## I. Introduction

This study proposes an unified representation that integrates the components of deep neural networks(MLP, CNN, Transformer, GNN) as message passing. Unified representation provides a mathematical formulation that facilitates structural analysis and comparative evaluation across different representations. The results validate that the mean absolute error (MAE) remains below $10^{-6}$ in the evaluation.

**Figure #1. Unified Representation Overview**

![](https://github.com/user-attachments/assets/8b69d380-a165-482e-a9d6-0ddf117357fe)



## II. Requirement

### Hardware

* NVIDIA GeForce RTX 4080 Laptop GPU

### Software

* Python 3.9.21
* CUDA == 11.8
* PyTorch == 2.1.2
* PyTorch Geometric == 2.5.3
* Torch_Scatter == 2.1.2+pt21cu118



## III. Installation

```bash
# Install Anaconda from website

# Update CONDA in Base
conda update -n base -c defaults conda

# Create myEnv=PyG_py39
conda create -n PyG_py39 python=3.9.21

# Activate myEnv
conda activate PyG_py39

# Install Numpy 1.26.4
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



