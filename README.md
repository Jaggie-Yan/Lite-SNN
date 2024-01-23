# IJCAI2024 Submission #2679: LitE-SNN: Designing Lightweight and Efficient Spiking Neural Network through Spatial-Temporal Compressive Network Search and Joint Optimization

This code is the PyTorch implementation code for [LitE-SNN: Designing Lightweight and Efficient Spiking Neural Network through
Spatial-Temporal Compressive Network Search and Joint Optimization].

## Requirements
* Python 3.9.12
* CUDA 11.6
* PyTorch 1.13.0
* TorchVision 0.14.0
* fitlog 0.9.15

# Dataset Preparation
To proceed, please download the CIFAR10/100 dataset on your own.

# Cofe for LiteSNN
We provide search, decode and retrain code for CIFAR10/100.

## Search
For search procedure, execute: \
  `bash search.sh`

Once we have conducted a search, the next step is to decode the results in order to retrieve the searched architecture.

## Decode
For decode, execute: \
  `bash decode.sh`

## Retrain
Searched Architecture:
```bash
network_path = [0,0,1,1,1,2,2,2] # default
cell_arch = [[1, 0],
                  [0, 0],
                  [2, 1],
                  [3, 1],
                  [7, 1],
                  [5, 1]]
bits_arch = [0, 2, 1, 1, 1, 1, 0, 2]
```
Replace the searched architecture and the searched quantization bit indexes of cells in `LEAStereo.py`.\
For retrain procedure, execute: \
  `bash train.sh`

## Evaluation
You can evaluate the performance of our models or models trained by yourself.\
The medium model of our work is avaliable at './pretrain_model', and you can also put the checkpoints of the models trained by yourself at here to evaluate.\
For evaluation procedure, execute: \
  `bash eval.sh`

# Code Reference
Our code is developed based on the code from [Che, K.; Leng, L.; Zhang, K.; Zhang, J.; Meng, Q.; Cheng, J.; Guo, Q.; and Liao, J. 2022. Differentiable hierarchical and surrogate gradient search for spiking neural networks. NeurIPS, 35: 24975–24990].



