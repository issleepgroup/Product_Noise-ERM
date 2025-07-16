import math
import numpy as np

def compute_total_epsilon(epsilon, datasize, batch_size, epochs):
    """
    Compute the total privacy budget (epsilon) over multiple training epochs
    under privacy amplification by subsampling and strong composition theorem.

    Args:
        epsilon (float): The per-iteration (per-step) privacy budget.
        datasize (int): The total number of data samples.
        batch_size (int): The mini-batch size used during training.
        epochs (int): The total number of training epochs.

    Returns:
        list: A list of total epsilon values after each epoch, computed by
              applying privacy amplification by subsampling and strong composition.
              Each element corresponds to the accumulated epsilon after that epoch.
    """
    delta_c = 1e-10  # Fixed delta for composition

    q = batch_size / datasize  # Sampling probability

    # Compute privacy amplification by subsampling
    sampled_epsilon = math.log(1 + q * (math.exp(epsilon) - 1))

    num_steps_per_epoch = datasize // batch_size + 1  # Steps per epoch

    epsilons = []

    for epoch in range(1, epochs + 1):
        k = num_steps_per_epoch * epoch  # Total steps up to this epoch
        term1 = math.sqrt(2 * k * np.log(1 / delta_c)) * sampled_epsilon
        term2 = k * sampled_epsilon * (math.exp(sampled_epsilon) - 1)
        epsilon_total = term1 + term2
        epsilons.append(round(epsilon_total, 10))

    return epsilons
