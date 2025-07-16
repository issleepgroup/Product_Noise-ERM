import argparse
import os
import shutil
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.grad_sample.functorch import make_functional
from torch.func import grad_and_value, vmap
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import math


def setup(args):
    if not torch.cuda.is_available():
        raise NotImplementedError(
            "DistributedDataParallel device_ids and output_device arguments \
            only work with single-device GPU modules"
        )

    if sys.platform == "win32":
        raise NotImplementedError("Windows version of multi-GPU is not supported yet.")

    # Initialize the process group on a Slurm cluster
    if os.environ.get("SLURM_NTASKS") is not None:
        rank = int(os.environ.get("SLURM_PROCID"))
        local_rank = int(os.environ.get("SLURM_LOCALID"))
        world_size = int(os.environ.get("SLURM_NTASKS"))
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "7440"

        torch.distributed.init_process_group(
            args.dist_backend, rank=rank, world_size=world_size
        )

        return (rank, local_rank, world_size)

    # Initialize the process group through the environment variables
    elif args.local_rank >= 0:
        torch.distributed.init_process_group(
            init_method="env://",
            backend=args.dist_backend,
        )
        rank = torch.distributed.get_rank()
        local_rank = args.local_rank
        world_size = torch.distributed.get_world_size()


        return (rank, local_rank, world_size)

    else:
        # logger.debug(f"Running on a single GPU.")
        return (0, 0, 1)

def cleanup():
    torch.distributed.destroy_process_group()

def convnet(num_classes):
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(128, num_classes, bias=True),
    )

def save_checkpoint(state, is_best, filename="checkpoint.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")

def accuracy(preds, labels):
    return (preds == labels).mean()

def train(args, model, train_loader, optimizer, privacy_engine, epoch, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    if args.grad_sample_mode == "no_op":
        # Functorch prepare
        fmodel, _fparams = make_functional(model)

        def compute_loss_stateless_model(params, sample, target):
            batch = sample.unsqueeze(0)
            targets = target.unsqueeze(0)

            predictions = fmodel(params, batch)
            loss = criterion(predictions, targets)
            return loss

        ft_compute_grad = grad_and_value(compute_loss_stateless_model)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))
        # Using model.parameters() instead of fparams
        # as fparams seems to not point to the dynamically updated parameters
        params = list(model.parameters())

    for i, (images, target) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        if args.grad_sample_mode == "no_op":
            per_sample_grads, per_sample_losses = ft_compute_sample_grad(
                params, images, target
            )
            per_sample_grads = [g.detach() for g in per_sample_grads]
            loss = torch.mean(per_sample_losses)
            for p, g in zip(params, per_sample_grads):
                p.grad_sample = g
        else:
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc1 = accuracy(preds, labels)
            top1_acc.append(acc1)

            # compute gradient and do SGD step
            loss.backward()

        losses.append(loss.item())

        # make sure we take a step after processing the last mini-batch in the
        # epoch to ensure we start the next epoch with a clean state
        optimizer.step()
        optimizer.zero_grad()

        if i % args.print_freq == 0:
            if not args.disable_dp:
                epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc):.6f} "
                    f"(ε = {epsilon:.2f}, δ = {args.delta})"
                )
            else:
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc):.6f} "
                )

    return np.mean(top1_acc) # return training accuracy

def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in tqdm(test_loader):
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc1 = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc1)

    top1_avg = np.mean(top1_acc)

    print(f"\tTest set:" f"Loss: {np.mean(losses):.6f} " f"Acc@1: {top1_avg :.6f} ")
    return np.mean(top1_acc)

def compute_tighter_epsilon_delta(epsilon, epoch, batch_size, datasize):

    delta = 1.0157754509132549e-07
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
    args = parse_args()

    # Sets `world_size = 1` if you run on a single GPU with `args.local_rank = -1`
    if args.local_rank != -1 or args.device != "cpu":
        rank, local_rank, world_size = setup(args)
        device = local_rank
    else:
        device = "cpu"
        rank = 0
        world_size = 1

    if args.secure_rng:
        try:
            import torchcsprng as prng
        except ImportError as e:
            msg = (
                "To use secure RNG, you must install the torchcsprng package! "
                "Check out the instructions here: https://github.com/pytorch/csprng#installation"
            )
            raise ImportError(msg) from e

        generator = prng.create_random_device_generator("/dev/urandom")

    else:
        generator = None

    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    train_transform = transforms.Compose(
        augmentations + normalize if args.disable_dp else normalize
    )

    test_transform = transforms.Compose(normalize)

    train_dataset = CIFAR10(
        root=args.data_root, train=True, download=True, transform=train_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        generator=generator,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_dataset = CIFAR10(
        root=args.data_root, train=False, download=True, transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=args.workers,
    )


    model = convnet(num_classes=10)
    model = model.to(device)

    # Use the right distributed module wrapper if distributed training is enabled
    if world_size > 1:
        if not args.disable_dp:
            if args.clip_per_layer:
                model = DDP(model, device_ids=[device])
            else:
                model = DPDDP(model)
        else:
            model = DDP(model, device_ids=[device])

    if args.optim == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError("Optimizer not recognized. Please check spelling")

    privacy_engine = None
    if not args.disable_dp:
        if args.clip_per_layer:
            # Each layer has the same clipping threshold. The total grad norm is still bounded by `args.max_per_sample_grad_norm`.
            n_layers = len(
                [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            )
            max_grad_norm = [
                args.max_per_sample_grad_norm / np.sqrt(n_layers)
            ] * n_layers
        else:
            max_grad_norm = args.max_per_sample_grad_norm

        privacy_engine = PrivacyEngine(
            secure_mode=args.secure_rng,
            accountant='rdp'
        )
        clipping = "per_layer" if args.clip_per_layer else "flat"
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.sigma,
            max_grad_norm=max_grad_norm,
            clipping=clipping,
            grad_sample_mode=args.grad_sample_mode,
            k=args.k,
            initial_epsilon=args.epsilon
        )

    training_accuracy_per_epoch = []  # training accuracy per epoch

    accuracy_per_epoch = [] # test accuracy per epoch

    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.lr_schedule == "cos":
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / (args.epochs + 1)))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        training_acc = train(
            args, model, train_loader, optimizer, privacy_engine, epoch, device
        )
        test_acc = test(model, test_loader, device)

        accuracy_per_epoch.append(float(test_acc))
        training_accuracy_per_epoch.append(float(training_acc))

    if world_size > 1:
        cleanup()

    epsilons, deltas = compute_tighter_epsilon_delta(args.epsilon, args.epochs, args.batch_size, len(train_loader.dataset))


    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    file_name = (
        f"cifar10_{args.lr}_{args.sigma}_"
        f"{args.max_per_sample_grad_norm}_{args.batch_size}_{args.epochs}_{args.epsilon}_{args.k}_{formatted_time}.csv"
    )

    df = pd.DataFrame({
        "Epoch": range(1, len(training_accuracy_per_epoch) + 1),
        "Train Accuracy": training_accuracy_per_epoch,
        "Test Accuracy": accuracy_per_epoch,
        "Epsilon": epsilons,
        "Delta": deltas,
    })

    df.to_csv(file_name, index=False)

    print(f"CSV saved {file_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 DP Training")

    parser.add_argument("--batch-size", default=512, type=int, metavar="N", help="approximate batch size")
    parser.add_argument("--lr", "--learning-rate", default=0.25, type=float, metavar="LR", help="initial learning rate",dest="lr")
    parser.add_argument("-n", "--epochs", default=5, type=int, metavar="N", help="number of total epochs to run",)
    parser.add_argument("--sigma", type=float, default=0.5, metavar="S", help="Noise multiplier (default 1.0)")
    parser.add_argument("-c", "--max-per-sample-grad_norm", type=float, default=1.5, metavar="C", help="Clip per-sample gradients to this norm (default 1.0)")
    parser.add_argument("--delta", type=float, default=1e-5, metavar="D", help="Target delta (default: 1e-5)")
    parser.add_argument("-k", default=300000, type=int, metavar="N", help="number of k",)
    parser.add_argument("-epsilon", default=1, type=float, metavar="N", help="the number of initial privacy budget",)

    parser.add_argument("--grad_sample_mode", type=str, default="hooks")
    parser.add_argument("-j", "--workers", default=2, type=int, metavar="N", help="number of data loading workers (default: 2)",)
    parser.add_argument("--start-epoch", default=1, type=int, metavar="N", help="manual epoch number (useful on restarts)",)
    parser.add_argument("-b", "--batch-size-test", default=256, type=int, metavar="N", help="mini-batch size for test dataset (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="SGD momentum")
    parser.add_argument("--wd", "--weight-decay", default=0, type=float, metavar="W", help="SGD weight decay",dest="weight_decay")
    parser.add_argument("-p", "--print-freq", default=10, type=int, metavar="N", help="print frequency (default: 10)")
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
    parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training.")
    parser.add_argument("--disable-dp", action="store_true", default=False, help="Disable privacy training and just train with vanilla SGD")
    parser.add_argument("--secure-rng", action="store_true", default=False, help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost. Opacus will emit a warning if secure RNG is off, indicating that for production use it's recommended to turn it on.")
    parser.add_argument("--checkpoint-file", type=str, default="checkpoint", help="path to save checkpoints")
    parser.add_argument("--data-root", type=str, default="../cifar10", help="Where CIFAR10 is/will be stored")
    parser.add_argument("--log-dir", type=str, default="/tmp/stat/tensorboard", help="Where Tensorboard log will be stored")
    parser.add_argument("--optim", type=str, default="SGD", help="Optimizer to use (Adam, RMSprop, SGD)")
    parser.add_argument("--lr-schedule", type=str, choices=["constant", "cos"], default="cos")
    parser.add_argument("--device", type=str, default="cpu", help="Device on which to run the code.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank if multi-GPU training, -1 for single GPU training. Will be overridden by the environment variables if running on a Slurm cluster.")
    parser.add_argument("--dist_backend", type=str, default="gloo", help="Choose the backend for torch distributed from: gloo, nccl, mpi")
    parser.add_argument("--clip_per_layer", action="store_true", default=False, help="Use static per-layer clipping with the same clipping threshold for each layer. Necessary for DDP. If `False` (default), uses flat clipping.")
    parser.add_argument("--debug", type=int, default=0, help="debug level (default: 0)")

    return parser.parse_args()


if __name__ == "__main__":
    main()
