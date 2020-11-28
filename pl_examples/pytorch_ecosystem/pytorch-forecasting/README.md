![# PyTorch Forecasting](./logo.svg)

## Introduction

[Pytorch Forecasting](https://pytorch-forecasting.readthedocs.io/) aims to ease time series forecasting with neural networks for both real-world cases and research alike. The package is built on PyTorch Lightning to allow training on CPUs, single and multiple GPUs out-of-the-box. The models all inherit from a `BaseModel` class that is `LightningModule` with pre-defined hooks. The Lightning `Trainer` can directly be used to train these models.

Specifically, the package provides

- A timeseries dataset class which abstracts handling variable transformations, missing values, randomized subsampling, multiple history lengths, etc.
- A base model class which provides basic training of timeseries models along with logging in tensorboard and generic visualizations such actual vs predictions and dependency plots
- Multiple neural network architectures for timeseries forecasting that have been enhanced for real-world deployment and come with in-built interpretation capabilities
- Multi-horizon timeseries metrics
- Ranger optimizer for faster model training
- Hyperparameter tuning with [optuna](https://optuna.readthedocs.io/)

## Documentation

For detailed tutorials and documentation, visit the [PyTorch Forecasting documentation](https://pytorch-forecasting.readthedocs.io/) or read
about it on [Towards Data Science](https://towardsdatascience.com/introducing-pytorch-forecasting-64de99b9ef46).

## Example

### Setup

If you are working windows, you need to first install PyTorch with

`pip install torch -f https://download.pytorch.org/whl/torch_stable.html`.

Otherwise, you can proceed with

`pip install pytorch-forecasting`

Alternatively, you can install the package via conda

`conda install pytorch-forecasting pytorch -c pytorch>=1.7 -c conda-forge`

PyTorch Forecasting is now installed from the conda-forge channel while PyTorch is install from the pytorch channel. PyTorch Lightning will be installed as a dependency of PyTorch Forecasting.

### Tutorial

[Demand forecasting tutorial](./stallion.ipynb): This tutorial introduces forecasting with the [Temporal Fusion Transformer](https://arxiv.org/pdf/1912.09363.pdf), an architecture that outperforms DeepAR by Amazon by 36-69% in benchmarks.
