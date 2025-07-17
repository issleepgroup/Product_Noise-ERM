#!/usr/bin/env python3
# This file is prepared for anonymous submission or public release.
# Original license, authorship, and institutional references have been removed for anonymity.
# Annotated for clarity and educational understanding.
# No changes were made to the algorithmic logic.
# Import necessary libraries

import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error

from opacus import PrivacyEngine
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import urllib.request
import zipfile

from datetime import datetime

# Constants for the MovieLens dataset
NUM_USERS = 6040
NUM_MOVIES = 3706
NUM_CLASSES = 5  # Ratings are from 1 to 5, so we treat this as a 5-class classification task

# Custom Dataset class for MovieLens data
class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.data = data  # numpy array of (user, movie, rating)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx, 0]
        movie = self.data[idx, 1]
        rating = self.data[idx, 2] - 1  # Convert rating from [1-5] to [0-4]
        return torch.tensor(user, dtype=torch.long), torch.tensor(movie, dtype=torch.long), torch.tensor(rating, dtype=torch.long)

# Neural network model using embeddings
class MovieLensModel(nn.Module):
    def __init__(self, num_users, num_movies, latent_dim_user=10, latent_dim_movie=10, latent_dim_mf=5):
        super(MovieLensModel, self).__init__()
        # GMF (Generalized Matrix Factorization) part
        self.user_embedding_mf = nn.Embedding(num_users, latent_dim_mf)
        self.movie_embedding_mf = nn.Embedding(num_movies, latent_dim_mf)

        # MLP part
        self.user_embedding = nn.Embedding(num_users, latent_dim_user)
        self.movie_embedding = nn.Embedding(num_movies, latent_dim_movie)

        # Final classification layer
        self.fc = nn.Linear(latent_dim_user + latent_dim_movie + latent_dim_mf, NUM_CLASSES)

    def forward(self, user, movie):
        # GMF embeddings and interaction
        user_embedding_mf = self.user_embedding_mf(user)
        movie_embedding_mf = self.movie_embedding_mf(movie)
        mf_vector = user_embedding_mf * movie_embedding_mf

        # MLP embeddings and concatenation
        user_embedding = self.user_embedding(user)
        movie_embedding = self.movie_embedding(movie)
        mlp_vector = torch.cat([user_embedding, movie_embedding], dim=-1)

        # Final prediction vector
        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        output = self.fc(predict_vector)
        return output

# Download and extract MovieLens 1M dataset
def Download_Extract_MovieLens(data_root):
    url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = os.path.join(data_root, "ml-1m.zip")
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    if not os.path.exists(zip_path):
        print("Downloading MovieLens 1M dataset...")
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_root)

# Load and preprocess MovieLens data
def load_MovieLens(data_root):
    Download_Extract_MovieLens(data_root)
    data_path = os.path.join(data_root, 'ml-1m', 'ratings.dat')
    ratings = pd.read_csv(data_path, sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

    ratings = ratings[['UserID', 'MovieID', 'Rating']].values
    ratings[:, 0] -= 1  # Shift user IDs to 0-indexed
    ratings[:, 1] -= 1  # Shift movie IDs to 0-indexed

    print(f"Max UserID: {ratings[:, 0].max()}, Max MovieID: {ratings[:, 1].max()}")
    ratings = ratings[(ratings[:, 0] < NUM_USERS) & (ratings[:, 1] < NUM_MOVIES)]

    np.random.shuffle(ratings)
    train_data = ratings[:800000]
    test_data = ratings[800000:]
    return train_data, test_data

# Estimate total epsilon across all epochs
# tighter composition
def compute_tighter_epsilon_delta(epsilon, epoch, batch_size, datasize):

    delta = 2.9674255156360687e-07
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

# Training loop
def train(args, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    all_preds = []
    all_targets = []

    for data in tqdm(train_loader):
        user, movie, rating = data
        user, movie, rating = user.to(device), movie.to(device), rating.to(device)

        optimizer.zero_grad()
        output = model(user, movie)
        loss = criterion(output, rating)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        all_preds.extend(torch.argmax(output, dim=1).cpu().detach().numpy())
        all_targets.extend(rating.cpu().detach().numpy())

    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

    if not args.disable_dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"RMSE: {rmse:.4f} "
            f"(ε = {epsilon:.8f}, δ = {args.delta})"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f} \t Train_RMSE: {rmse:.4f}")

    return rmse
# Evaluation on test set
def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for users, movies, ratings in tqdm(test_loader):
            users, movies, ratings = users.to(device), movies.to(device), ratings.to(device)
            output = model(users, movies)
            test_loss += criterion(output, ratings).item()
            all_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
            all_targets.extend(ratings.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    print(f"\nTest set: Average loss: {test_loss:.4f}, Test_RMSE: {rmse:.4f}\n")
    return rmse

# Main function
def main():
    parser = argparse.ArgumentParser(description="Opacus MovieLens Example")
    parser.add_argument("-N", "--traning-data", type=int, default=800167)
    parser.add_argument("-b", "--batch-size", type=int, default=10000)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("-n", "--epochs", type=int, default=20)
    parser.add_argument("-r", "--n-runs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument("-e", "--epsilon", type=float, default=0.8)
    parser.add_argument("-k", type=int, default=20000)
    parser.add_argument("-c", "--max-per-sample-grad_norm", type=float, default=5.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-model", action="store_true", default=False)
    parser.add_argument("--disable-dp", action="store_true", default=False)
    parser.add_argument("--secure-rng", action="store_true", default=False)
    parser.add_argument("--data-root", type=str, default="../data")
    args = parser.parse_args()
    device = torch.device(args.device)

    # Load dataset
    train_data, test_data = load_MovieLens(args.data_root)
    train_loader = DataLoader(MovieLensDataset(train_data), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(MovieLensDataset(test_data), batch_size=args.test_batch_size, shuffle=False)

    rmse_list = []
    test_rsme_list = []

    for _ in range(args.n_runs):
        model = MovieLensModel(NUM_USERS, NUM_MOVIES).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        privacy_engine = None

        # Attach Opacus PrivacyEngine
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
            rmse_list.append(train(args, model, device, train_loader, optimizer, privacy_engine, epoch))
            test_rsme_list.append(test(model, device, test_loader))

    epsilons, deltas = compute_tighter_epsilon_delta(args.epsilon, args.epochs, args.batch_sizew, len(train_data))

    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Save training results
    filename = (
        f"MovieLens_{args.lr}_{args.sigma}_{args.max_per_sample_grad_norm}_{args.epochs}_{args.epsilon}_{args.k}_{formatted_time}"
    )
    df = pd.DataFrame({
        "Epoch": range(1, len(rmse_list) + 1),
        "Train RMSE": rmse_list,
        "Test RMSE": test_rsme_list,
        "Epsilon": epsilons,
        "Delta": deltas,
    })
    df.to_csv(filename, index=False)
    print(f"CSV saved {filename}")

# Run main function
if __name__ == "__main__":
    main()
