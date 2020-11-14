#  [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric) examples with Lighting

### Introduction

PyTorch Geometric (PyG) is a geometric deep learning extension library for PyTorch. It relies on lower level libraries such as

* PyTorch Cluster: A package consists of a small extension library of highly optimized graph cluster algorithms in Pytorch
* PyTorch Sparse: A package consists of a small extension library of optimized sparse matrix operations with autograd support in Pytorch
* PyTorch Scatter: A package consists of a small extension library of highly optimized sparse update (scatter and segment) operations for the use in PyTorch

## Setup

```
pyenv install 3.7.8
pyenv local 3.7.8
python -m venv
source .venv/bin/activate
poetry install
```

Run example

```
python cora_dna.py
```

## Current example lists

| `DATASET` | `MODEL` | `TASK` | DATASET DESCRIPTION | MODEL DESCRIPTION                                                                                                                                                                   |                                                                                                                                                                     |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Cora | DNA | Node Classification | The citation network datasets "Cora", "CiteSeer" and "PubMed" from the "Revisiting Semi-Supervised Learning with Graph Embeddings" <https://arxiv.org/abs/1603.08861> | The dynamic neighborhood aggregation operator from the "Just Jump: Towards Dynamic Neighborhood Aggregation in Graph Neural Networks"


## DATASET SIZES

```
 16M    ./cora
```
