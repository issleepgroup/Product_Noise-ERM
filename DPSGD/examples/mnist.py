#!/usr/bin/env python3
# This file is prepared for anonymous submission or public release.
# Original license, authorship, and institutional references have been removed for anonymity.
# Annotated for clarity and educational understanding.
# No changes were made to the algorithmic logic.

# Import necessary libraries

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import math

from sympy import false

from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# Precomputed mean and std for MNIST normalization
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# Optional name function for model identification
def name():
    return "SampleConvNet"

# Define a simple CNN for MNIST
class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layer 1: input channel=1, output channel=16
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        # Convolutional layer 2: input channel=16, output channel=32
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        # Fully connected layer 1
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        # Fully connected layer 2 (output logits for 10 classes)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.max_pool2d(x, 2, 1)
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, 2, 1)
        x = x.view(-1, 32 * 4 * 4)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# tighter composition
def compute_tighter_epsilon_delta(epsilon, epoch, batch_size, datasize):

    delta = 1.9779675188471893e-07
    delta_tilde = 1e-8

    q = batch_size / datasize
    epsilon_amp = math.log(1 + q * (math.exp(epsilon) - 1))
    delta_amp = q * delta
    epsilons = []
    deltas = []
    num_steps_per_epoch = math.ceil(datasize // batch_size)

    for epoch in range(1, epoch + 1):
        T = num_steps_per_epoch * epoch
        epsilon_prime_term1 = T * ((math.exp(epsilon_amp) - 1) * epsilon_amp) / (math.exp(epsilon_amp) + 1)
        epsilon_prime_term2 = math.sqrt(2 * math.log(1 / delta_tilde) * T * epsilon_amp**2)
        epsilon_prime = round(epsilon_prime_term1, 8) + round(epsilon_prime_term2, 8)

        delta_double_prime_radio = epsilon_prime + T * epsilon_amp
        delta_double_prime_fraction = T * epsilon_amp - epsilon_prime

        delta_double_prime_term1 = math.exp(-delta_double_prime_radio / 2)

        try:
            base_val = (1 / (1 + math.exp(epsilon_amp))) * (2 * T * epsilon_amp / delta_double_prime_fraction)
            MAX_VAL = 1e300
            result = base_val ** T
            result = min(result, MAX_VAL)
            delta_double_prime_term2 = round(result, 10)
        except OverflowError:
            delta_double_prime_term2 = round(1e300, 10)

        delta_double_prime_term3 = ((T * epsilon_amp + epsilon_prime) / delta_double_prime_fraction) ** (-delta_double_prime_radio / (2 * epsilon_amp))

        delta_double_prime = delta_double_prime_term1 * delta_double_prime_term2 * delta_double_prime_term3

        delta_prime_power_term = math.ceil(epsilon_prime / epsilon_amp)
        delta_prime_subterm = delta_amp / (1 + math.exp(epsilon_amp))

        delta_prime_term1 = (1 - math.exp(epsilon_amp) * delta_prime_subterm) ** delta_prime_power_term
        delta_prime_term2 = (1 - delta_prime_subterm) ** (T - delta_prime_power_term)
        delta_prime_term3 = (1 - delta_prime_subterm) ** T

        delta_prime = 2 - delta_prime_term1 * delta_prime_term2 - delta_prime_term3 + delta_double_prime

        epsilons.append(epsilon_prime)
        deltas.append(delta_prime)

    return epsilons, deltas


# Training loop for one epoch
def train(args, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    total = 0
    losses = []

    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    accuracy = correct / total

    if not args.disable_dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        print(f"Train Epoch: {epoch} \t ε = {epsilon:.2f} \t Accuracy: {accuracy:.2f}%")
    else:
        print(f"Train Epoch: {epoch} \t Accuracy: {accuracy:.2f}%")

    return accuracy

# Testing loop (inference only)
def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = len(test_loader.dataset)

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = correct / total
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n")
    return accuracy

# Main training loop and argument handling
def main():
    parser = argparse.ArgumentParser(description="Opacus MNIST Example",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--test-batch-size", type=int, default=1024, help="Testing batch size")
    parser.add_argument("-n", "--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("-r", "--n-runs", type=int, default=1, help="Number of runs (for averaging)")
    parser.add_argument("--lr", type=float, default=0.15, help="Learning rate")
    parser.add_argument("--sigma", type=float, default=1.3, help="Noise multiplier for DP")
    parser.add_argument("-c", "--max-per-sample-grad-norm", type=float, default=1.0, help="Clipping bound for per-sample gradients")
    parser.add_argument("--delta", type=float, default=1e-5, help="Target delta for DP")
    parser.add_argument("-e", "--epsilon", type=float, default=0.3, help="Initial epsilon for adaptive accountant")
    parser.add_argument("-k", type=int, default=40000, help="Number of samples used for privacy accounting")
    parser.add_argument("--device", type=str, default="cpu", help="Device for training (e.g., 'cpu', 'cuda')")
    parser.add_argument("--save-model", action="store_true", help="Option to save trained model")
    parser.add_argument("--disable-dp", action="store_true", default=false, help="Disable DP and use vanilla SGD")
    parser.add_argument("--secure-rng", action="store_true", default=false, help="Use cryptographic RNG (slower but more secure)")
    parser.add_argument("--data-root", type=str, default="../mnist", help="Path to store/load MNIST data")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load training data with normalization
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
                       ])),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )

    # Load test data
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_root, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
                       ])),
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    train_acc = []
    test_acc = []

    for _ in range(args.n_runs):
        model = SampleConvNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)

        privacy_engine = None
        if not args.disable_dp:
            # Enable differential privacy
            privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.sigma,
                initial_epsilon=args.epsilon,
                k=args.k,
                max_grad_norm=args.max_per_sample_grad_norm,
            )

        for epoch in range(1, args.epochs + 1):
            train_acc.append(train(args, model, device, train_loader, optimizer, privacy_engine, epoch))
            test_acc.append(test(model, device, test_loader))

    # calculate total epsilon for each epoch
    epsilons, deltas = compute_tighter_epsilon_delta(args.epsilon, args.epochs, args.batch_size, len(train_loader.dataset))

    now = datetime.now()
    formatted_time = now.strftime("%d-%m-%Y_%H-%M-%S")

    # Save training results
    file_name = (
        f"mnist_{args.lr}_{args.sigma}_{args.max_per_sample_grad_norm}_{args.batch_size}_{args.epochs}_{args.epsilon}_{args.k}_{formatted_time}.csv")
    df = pd.DataFrame({
        "Epoch": range(1, len(train_acc) + 1),
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "Epsilon": epsilons,
        "Delta": deltas,
    })
    df.to_csv(file_name, index=False)
    print(f"CSV saved {file_name}")


if __name__ == "__main__":
    main()
