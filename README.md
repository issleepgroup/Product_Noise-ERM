# Product Noise ERM

## Overview

**Product Noise ERM** is a research-oriented project that focuses on protecting user privacy during machine learning model training. By introducing a custom-designed product noise mechanism, this project enables privacy-preserving empirical risk minimization (ERM) beyond traditional differential privacy approaches.

## Key Features

- Implements a novel product noise mechanism to perturb gradients or model parameters.
- Enhances privacy protection during local training on client-side data.
- Built with PyTorch, and easy to integrate into existing deep learning pipelines.
- Supports multiple datasets (e.g., MNIST, CIFAR-10) with modular code design.

## Installation

You can install all required Python packages using the `pip install` command:

```bash
pip install opacus torch torchvision numpy tqdm
```
## How to Run

### Usage

Use the following command to run the training script with the corresponding dataset:

```bash
python examples/mnist.py        # For MNIST
python examples/cifar10.py      # For CIFAR-10
python examples/adult.py        # For Adult dataset
python examples/imdb.py         # For IMDB sentiment dataset
python examples/movielens.py    # For MovieLens dataset
```

### Steps



Dependencies
- This project is based on the PyTorch framework and depends on the following Python packages:
- torch – Core deep learning library
- torchvision – Utilities for common vision datasets
- opacus – Differential privacy library
- numpy – Numerical computing library
- tqdm – Command-line progress bar utility
- math – Python built-in module (no need to install)
