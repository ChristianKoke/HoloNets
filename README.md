# HoloNets: Spectral Convolutions do extend to Directed Graphs

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/holonets-spectral-convolutions-do-extend-to/node-classification-on-squirrel)](https://paperswithcode.com/sota/node-classification-on-squirrel?p=holonets-spectral-convolutions-do-extend-to)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/holonets-spectral-convolutions-do-extend-to/node-classification-on-chameleon)](https://paperswithcode.com/sota/node-classification-on-chameleon?p=holonets-spectral-convolutions-do-extend-to)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/holonets-spectral-convolutions-do-extend-to/node-classification-on-arxiv-year)](https://paperswithcode.com/sota/node-classification-on-arxiv-year?p=holonets-spectral-convolutions-do-extend-to)[![PWC]([https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/holonets-spectral-convolutions-do-extend-to](https://paperswithcode.com/badge/holonets-spectral-convolutions-do-extend-to/node-classification-on-snap-patents)](https://paperswithcode.com/sota/node-classification-on-snap-patents?p=holonets-spectral-convolutions-do-extend-to)

HolonNets are machine learning models that extend spectral convolutions to directed graphs. This repository contains the official implementation of the paper ["HoloNets: Spectral Convolutions do extend to Directed Graphs"](https://arxiv.org/abs/2310.02232), where we introduce -among other things - the FaberNet architecture which conforms to our general developed theory of HoloNets and achieves improved performance for node classification on heterophilic graphs.

- [HoloNets: Spectral Convolutions do extend to Directed Graphs](#holonets-spectral-convolutions-do-extend-to-directed-graphs)
  - [Overview](#overview)
  - [Getting Started](#getting-started)
    - [Setting Up the Environment](#setting-up-the-environment)
    - [Installing Dependencies](#installing-dependencies)
    - [Code Structure](#code-structure)
  - [Running Experiments](#running-experiments)
    - [Node Classification Experiments](#node-classification-experiments)
  - [Dataset Fix](#dataset-fix)
  - [Command Line Arguments](#command-line-arguments)
    - [Dataset Arguments](#dataset-arguments)
    - [Preprocessing Arguments](#preprocessing-arguments)
    - [Model Arguments](#model-arguments)
    - [Training Args](#training-args)
    - [System Args](#system-args)
  - [Citation](#citation)
  - [Contact](#contact)

## Overview

Within the graph learning community, conventional wisdom dictates that spectral convolutional networks may only be deployed on undirected graphs: Only there could the existence of a well-defined graph Fourier transform be guaranteed, so that information may be translated between spatial- and spectral domains. Here we show this traditional reliance on the graph Fourier transform to be superfluous and -- making use of certain advanced tools from complex analysis and spectral theory -- extend spectral convolutions to directed graphs. We provide a frequency-response interpretation of newly developed filters, investigate the influence of the basis used to express filters and discuss the interplay with characteristic operators on which networks are based. In order to thoroughly test the developed theory, we conduct experiments in real world settings, showcasing that directed spectral convolutional networks provide new state of the art results for heterophilic node classification on many datasets and -- as opposed to baselines -- may be rendered stable to resolution-scale varying topological perturbations. 

## Getting Started

To get up and running with the project, you need to first set up your environment and install the necessary dependencies. This guide will walk you through the process step by step.

### Setting Up the Environment

The project is designed to run on Python 3.10. We recommend using [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to set up the environment as follows:

```bash
conda create -n holonet python=3.10
conda activate holonet
```

### Installing Dependencies

Once the environment is activated, install the required packages:

```bash
conda install pytorch==2.0.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg pytorch-sparse -c pyg
pip install ogb==1.3.6
pip install pytorch_lightning==2.0.2
pip install gdown==4.7.1
```

Please ensure that the version of `pytorch-cuda` matches your CUDA version. If your system does not have a GPU, use the following command to install PyTorch:

```bash
conda install pytorch==2.0.1 -c pytorch
```

For M1/M2/M3 Mac users, `pyg` (PyTorch Geometric) needs to be installed from source. Detailed instructions for this process can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-from-source).

### Code Structure

* `run.py`: This script is used to run the models.

* `model.py`: Contains the definition of the directed convolutional network called HoloNet in the code and FaberNet in the Paper.


## Running Experiments

This section provides instructions on how to reproduce the experiments outlined in the paper. Note that some of the results may not be reproduced *exactly*, given that some of the operations used are intrinsically non-deterministic on the GPU, as explained [here](https://github.com/pyg-team/pytorch_geometric/issues/92). However, you should obtain results very close to those in the paper.

### FaberNet-Experiments

To reproduce the results of Table 1 in the paper, use the following command:

```bash
python -m src.run --dataset chameleon --use_best_hyperparams --num_runs 10
```

The `--dataset` parameter specifies the dataset to be used. Replace `chameleon` with the name of the dataset you want to use. 

## Command Line Arguments

The following command line arguments can be used with the code:

### Dataset Arguments

| Argument               | Type | Default Value | Description                   |
| ---------------------- | ---- | ------------- | ----------------------------- |
| --dataset              | str  | "chameleon"   | Name of the dataset           |
| --dataset_directory    | str  | "dataset"     | Directory to save datasets    |
| --checkpoint_directory | str  | "checkpoint"  | Directory to save checkpoints |

### Preprocessing Arguments

| Argument     | Action     | Description                     |
| ------------ | ---------- | ------------------------------- |
| --undirected | store_true | Use undirected version of graph |
| --self_loops | store_true | Add self-loops to the graph     |
| --transpose  | store_true | Use transpose of the graph      |

### Model Arguments

| Argument         | Type   | Default Value | Description                         |
| ---------------- | ------ | ------------- | ----------------------------------- |
| --model          | str    | "gnn"         | Model type                          |
| --hidden_dim     | int    | 64            | Hidden dimension of model           |
| --num_layers     | int    | 3             | Number of GNN layers                |
| --dropout        | float  | 0.0           | Feature dropout                     |
| --alpha          | float  | 0.5           | Direction convex combination params |
| --learn_alpha    | action | -             | If set, learn alpha                 |
| --conv_type      | str    | "fabernet"    | Model                               |
| --normalize      | action | -             | If set, normalize                   |
| --jk             | str    | "max"         | Either "max", "cat" or None         |
| --weight_penalty | str    | "exp"         | Either "exp", "line" or None        |
| --k_plus         | int    | 3             | Polynomial Order                    |
| --exponent       | float  | -0.25         | normalization in adj-matrix         |
| --lrelu_slope    | float  | -1            | LeakyRelu slope                     |
| --zero_order     | bool   | False         | Whether to use zero-order term      |




### Training Args

| Argument            | Type  | Default Value | Description                                        |
| ------------------- | ----- | ------------- | -------------------------------------------------- |
| --lr                | float | 0.001         | Learning Rate                                      |
| --weight_decay      | float | 0.0           | Weight decay (if only real weights are used)       |
| --imag_weight_decay | float | 0.0           | Weight decay for imaginary part of weight matrices |
| --real_weight_decay | float | 0.0           | Weight decay for real      part of weight matrices |
| --num_epochs        | int   | 10000         | Max number of epochs                               |
| --patience          | int   | 10            | Patience for early stopping                        |
| --num_runs          | int   | 1             | Max number of runs                                 |

### System Args

| Argument               | Type | Default Value | Description                                      |
| ---------------------- | ---- | ------------- | ------------------------------------------------ |
| --use_best_hyperparams | flag |               | If specified, use the best hyperparameters       |
| --gpu_idx              | int  | 0             | Indexes of GPU to run the program on             |
| --num_workers          | int  | 0             | Number of workers for the dataloader             |
| --log                  | str  | "INFO"        | Log Level. Choices: ["DEBUG", "INFO", "WARNING"] |
| --profiler             | flag |               | If specified, enable profiler                    |



## Citation

```bibtex
@inproceedings{
koke2024holonets,
title={HoloNets: Spectral Convolutions do extend to Directed Graphs},
author={Christian Koke and Daniel Cremers},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=EhmEwfavOW}
}
```

## Contact
If you have any questions, issues or feedback, feel free to reach out to Christian Koke at `christian.koke@tum.de`.

