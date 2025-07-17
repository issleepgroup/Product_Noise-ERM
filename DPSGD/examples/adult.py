#!/usr/bin/env python3
# This file is prepared for anonymous submission or public release.
# Original license, authorship, and institutional references have been removed for anonymity.
# Annotated for clarity and educational understanding.
# No changes were made to the algorithmic logic.

# Import necessary libraries
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine  # For differential privacy training
from tqdm import tqdm
import os
import math
from datetime import datetime

# Define a simple two-layer fully connected neural network for binary classification
class TwoLayerNN(nn.Module):
    def __init__(self, input_dim):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)  # First hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)  # Output layer (binary classification)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Download and cache the UCI Adult dataset if not already present
def download_adult_data(data_path):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    if not os.path.exists(data_path):
        df = pd.read_csv(url, header=None, names=[
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
        ])
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
    return df

# Load, preprocess, encode and normalize the Adult dataset
def load_adult_data(data_path):
    data = download_adult_data(data_path)
    data.dropna(inplace=True)
    label_encoder = LabelEncoder()
    data['income'] = label_encoder.fit_transform(data['income'])
    categorical_columns = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_columns)
    X = data.drop('income', axis=1).values
    y = data['income'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Compute total epsilon using privacy amplification by subsampling (approximate)
def compute_total_epsilon(epsilon, datasize, batch_size, epochs):
    delta_c = 1e-10
    q = batch_size / datasize
    subsampled_epsilon = math.log(1 + q * (math.exp(epsilon) - 1))
    num_steps_per_epoch = datasize // batch_size + 1
    epsilons = []
    for epoch in range(1, epochs + 1):
        k = num_steps_per_epoch * epoch
        term1 = math.sqrt(2 * k * np.log(1 / delta_c)) * subsampled_epsilon
        term2 = k * subsampled_epsilon * (math.exp(subsampled_epsilon) - 1)
        epsilon_total = term1 + term2
        epsilons.append(epsilon_total)
    return epsilons

# Compute tighter epsilon and delta using more accurate accounting
def compute_tighter_epsilon_delta(epsilon, epoch, batch_size, datasize):
    delta = 2.7224847860651043e-07
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

# Perform one epoch of training with or without DP
def train(args, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    correct = 0
    total = 0

    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
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

    training_accuracy = correct / total
    if not args.disable_dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.8f}, δ = {args.delta}), "
            f"Training Accuracy: {training_accuracy:.4f}"
        )
    return training_accuracy

# Evaluate model performance on the test set
def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f})\n".format(
            test_loss, correct, len(test_loader.dataset), test_accuracy,
        )
    )
    return test_accuracy

# Main training script entry point
def main():
    parser = argparse.ArgumentParser(description="Opacus Adult Data Example")
    parser.add_argument("-N", "--traning-data", type=int, default=29305)
    parser.add_argument("-b", "--batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("-n", "--epochs", type=int, default=40)
    parser.add_argument("-r", "--n-runs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--sigma", type=float, default=0.55)
    parser.add_argument("-e", "--epsilon", type=float, default=0.8)
    parser.add_argument("-k", type=int, default=150000)
    parser.add_argument("-c", "--max-per-sample-grad_norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-model", action="store_true", default=False)
    parser.add_argument("--disable-dp", action="store_true", default=False)
    parser.add_argument("--secure-rng", action="store_true", default=False)
    parser.add_argument("--data-path", type=str, default="../adult.data")
    args = parser.parse_args()

    device = torch.device(args.device)
    X_train, X_test, y_train, y_test = load_adult_data(args.data_path)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    train_acc = []
    test_acc = []
    input_dim = X_train.shape[1]

    for _ in range(args.n_runs):
        model = TwoLayerNN(input_dim).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        privacy_engine = None

        if not args.disable_dp:
            privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
                initial_epsilon=args.epsilon,
                k=args.k,
            )

        for epoch in range(1, args.epochs + 1):
            train_acc.append(train(args, model, device, train_loader, optimizer, privacy_engine, epoch))
            test_acc.append(test(model, device, test_loader))

    epsilons, deltas = compute_tighter_epsilon_delta(args.epsilon, args.epochs, args.batch_size, len(X_train))

    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = (
        f"adult_{args.lr}_{args.sigma}_{args.max_per_sample_grad_norm}_{args.batch_size}_{args.epochs}_{args.epsilon}_{args.k}_{formatted_time}.csv"
    )
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
