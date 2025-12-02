# Product-Noise-ERM

[![Conference](https://img.shields.io/badge/SIGMOD-2026-blue)](https://2026.sigmod.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the paper: **"Privacy Loss of Noise Perturbation via Concentration Analysis of A Product Measure"** (SIGMOD 2026).

## Overview

This repository contains the code for our proposed **Product Noise** mechanism. Unlike classic Gaussian mechanisms, our approach leverages the concentration analysis of a product measure to achieve a superior **privacy-utility trade-off**, especially in **high-dimensional** settings.

We provide implementations for both:
* **Convex ERM**: Logistic Regression, SVM (via Output & Objective Perturbation).
* **Non-convex ERM**: Deep Neural Networks (via Gradient Perturbation / DPSGD).

> **Note**: If you find this code useful for your research, please cite our paper. See [Citation](#-citation) below.

## Key Features

- Implements a novel product noise mechanism to perturb gradients or model parameters.
- Enhances privacy protection for sensitive training data.
- Built with PyTorch, designed for easy integration into existing deep learning pipelines.
- Supports multiple datasets (e.g., MNIST, CIFAR-10, Adult, RCV1) with modular code design.

---

## OUTPUT PERTURBATION

### Installation
1. Navigate to the `datasets` directory.
2. Run the following command to download and preprocess all benchmark datasets automatically:

```bash
python main_preprocess.py all
```
If you want to download one of the datasets, just replace ''all'' with the name of the dataset. All available datasets are listed as following.

```bash
Adult, KDDCup99, MNIST, Synthetic-H, Real-sim, RCV1
```

### How to Run
Navigate to this repository. 

Run the python file named after the dataset.

```bash
python kddcup99.py
```


## OBJECTIVE_PERTURBATION

### Installation
Navigate to the ''datasets'' directory.

Run the following command line to download and preprocess all the benchmark datasets automatically.
```bash
python main_preprocess.py all
```
If you want to download one of the datasets, just replace ''all'' with the name of the dataset. All available datasets are listed as following.

```bash
MNIST, Synthetic-H, Real-sim, RCV1
```
### How to Run
Navigate to this repository.

Run algorithms on one dataset using the following command.

```bash
python gridsearch.py [ALG_NAME] [DATASET_NAME] [LR/SVM]
```

## DPSGD Section
### Installation

Install all required Python packages using the `pip install` command:

```bash
pip install opacus, torch, torchvision, numpy, tqdm
```
### How to Run
Download the Opacus library and replace the optimizer.py and privacy_engine.py files within it

Use the following command to run the training script with the corresponding dataset:

```bash
python examples/mnist.py        # For MNIST
python examples/cifar10.py      # For CIFAR-10
python examples/adult.py        # For Adult dataset
python examples/imdb.py         # For IMDB sentiment dataset
python examples/movielens.py    # For MovieLens dataset
```

### Dependencies
- This project is based on the PyTorch framework and depends on the following Python packages:
- torch – Core deep learning library
- torchvision – Utilities for common vision datasets
- opacus – Differential privacy library
- numpy – Numerical computing library
- tqdm – Command-line progress bar utility
- math – Python built-in module (no need to install)

## 📌 Citation

If you use this code or our results in your research, please cite our paper:

```bibtex
@inproceedings{product_noise_dp,
  title={Privacy Loss of Noise Perturbation via Concentration Analysis of A Product Measure},
  author={Liu, Shuainan and Ji, Tianxi and Fang, Zhongshuo and Wei, Lu and Li, Pan},
  booktitle={Proceedings of the 2026 International Conference on Management of Data (SIGMOD '26)},
  year={2026},
  publisher={ACM}
}







