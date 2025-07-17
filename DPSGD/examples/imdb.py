import argparse
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from opacus import PrivacyEngine
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast
import math

from datetime import datetime

class SampleNet(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 16)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.emb(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def name(self):
        return "SampleNet"


def binary_accuracy(preds, y):
    correct = (y.long() == torch.argmax(preds, dim=1)).float()
    acc = correct.sum() / len(correct)
    return acc


def padded_collate(batch, padding_idx=0):
    x = pad_sequence(
        [elem["input_ids"] for elem in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    y = torch.stack([elem["label"] for elem in batch]).long()
    return x, y


def train(args, model, train_loader, optimizer, privacy_engine, epoch):
    eps = None
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    device = torch.device(args.device)
    model = model.train().to(device)

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        predictions = model(data).squeeze(1)
        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accuracies.append(acc.item())

    if not args.disable_dp:
        eps = privacy_engine.accountant.get_epsilon(delta=args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Train Loss: {np.mean(losses):.6f} "
            f"Train Accuracy: {np.mean(accuracies):.6f} "
            f"(ε = {eps:.2f}, δ = {args.delta})"
        )
    else:
        print(
            f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f} ] \t Accuracy: {np.mean(accuracies):.6f}"
        )
    return np.mean(accuracies), eps


def evaluate(args, model, test_loader):
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    device = torch.device(args.device)
    model = model.eval().to(device)

    with torch.no_grad():
        for data, label in tqdm(test_loader):
            data = data.to(device)
            label = label.to(device)

            predictions = model(data).squeeze(1)

            loss = criterion(predictions, label)
            acc = binary_accuracy(predictions, label)

            losses.append(loss.item())
            accuracies.append(acc.item())

    mean_accuracy = np.mean(accuracies)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(
            np.mean(losses), mean_accuracy * 100
        )
    )
    return mean_accuracy

# tighter composition
def compute_tighter_epsilon_delta(epsilon, epoch, batch_size, datasize):

    delta = 1.1192002027896169e-07
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

def main():
    parser = argparse.ArgumentParser(description="Opacus IMDB Example",formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("-b", "--batch-size", type=int, default=512, metavar="B", help="input batch size for test")
    parser.add_argument("-n", "--epochs", type=int, default=60, metavar="N", help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.02, metavar="LR", help="learning rate")
    parser.add_argument("--sigma", type=float, default=0.56, metavar="S", help="Noise multiplier")
    parser.add_argument("-c", "--max-per-sample-grad_norm", type=float, default=1.0, metavar="C",
                        help="Clip per-sample gradients to this norm")
    parser.add_argument("-k", default=40000, type=int, metavar="N", help="number of k",)
    parser.add_argument("--epsilon", default=1, type=float, metavar="N", help="the number of initial privacy budget",)
    parser.add_argument("--delta", type=float, default=1e-5, metavar="D", help="Target delta (default: 1e-5)")
    parser.add_argument("--max-sequence-length", type=int, default=256, metavar="SL",
                        help="Longer sequences will be cut to this length")
    parser.add_argument("--device", type=str, default="cpu", help="GPU ID for this process")
    parser.add_argument("--save-model", action="store_true", default=False, help="Save the trained model")
    parser.add_argument("--disable-dp", action="store_true", default=False,
                        help="Disable privacy training and just train with vanilla optimizer")
    parser.add_argument("--secure-rng", action="store_true", default=False,
                        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost")
    parser.add_argument("--data-root", type=str, default="../imdb", help="Where IMDB is/will be stored")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers")

    args = parser.parse_args()
    device = torch.device(args.device)

    raw_dataset = load_dataset("imdb", cache_dir=args.data_root)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    dataset = raw_dataset.map(
        lambda x: tokenizer(
            x["text"], truncation=True, max_length=args.max_sequence_length
        ),
        batched=True,
    )
    dataset.set_format(type="torch", columns=["input_ids", "label"])

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_loader = DataLoader(
        train_dataset,
        num_workers=args.workers,
        batch_size=args.batch_size,
        collate_fn=padded_collate,
        pin_memory=True,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=padded_collate,
        pin_memory=True,
    )

    model = SampleNet(vocab_size=len(tokenizer)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    privacy_engine = None
    if not args.disable_dp:
        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng, accountant="rdp")

        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.sigma,
            max_grad_norm=args.max_per_sample_grad_norm,
            poisson_sampling=False,
            initial_epsilon=args.epsilon,
            k=args.k
        )

    training_accs = []
    test_accs = []

    for epoch in range(1, args.epochs + 1):
        train_acc, epsilon = train(args, model, train_loader, optimizer, privacy_engine, epoch)
        test_acc = evaluate(args, model, test_loader)
        training_accs.append(train_acc)
        test_accs.append(test_acc)

    epsilons, deltas = compute_tighter_epsilon_delta(args.epsilon, args.epochs, args.batch_size, len(train_dataset))

    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")

    file_name = (
        f"IMDB_{args.lr}_{args.sigma}_{args.max_per_sample_grad_norm}_{args.batch_size}_{args.epochs}_{args.epsilon}_{args.k}_{formatted_time}.csv"
    )
    # create DataFrame
    df = pd.DataFrame({
        "Epoch": range(1, len(training_accs) + 1),
        "Train Accuracy": training_accs,
        "Test Accuracy": test_accs,
        "Epsilon": epsilons,
        "Delta": deltas,
    })

    df.to_csv(file_name, index=False)

    print(f"f{file_name}")



if __name__ == "__main__":
    main()
