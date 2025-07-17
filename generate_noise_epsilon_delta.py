import math
import numpy as np
import pandas as pd
from scipy.special import hyp1f1

def generate_noise_and_delta(L, n_samples, lambda_param, m, epsilon, k=1000):
    """
    Generates noise for differential privacy and computes the corresponding delta value.

    Args:
        L (float): Lipschitz constant of the function.
        n_samples (int): Number of training samples.
        lambda_param (float): Regularization term (lambda).
        m (int): Model dimension (number of parameters).
        epsilon (float): Privacy budget (ε).
        k (float): Constant used in the noise scaling term, typically > 1.

    Returns:
        - noise (np.ndarray): A vector of shape (m,) representing the generated noise.
        - delta (float): The corresponding delta value for (ε, δ)-differential privacy.
    """
    # Compute the L2 sensitivity of the function
    sensitivity = (2 * L) / (n_samples * lambda_param)

    # Calculate t² based on the bound from the theorem
    t_squared = 2 * (k ** (4 / m)) * (((m / 4) + 1.5) ** (1 + 4 / m)) / (math.e ** (1 + 2 / m))

    # Derive the standard deviation for the noise
    std_dev_out = sensitivity * math.sqrt(t_squared) / epsilon

    # Generate a random direction on the unit sphere
    h = np.random.randn(m)
    h = h / np.linalg.norm(h)

    # Sample a non-negative scalar from the absolute standard normal distribution
    z = np.abs(np.random.normal(0, 1))

    # Construct the final noise vector using the direction and scaled magnitude
    noise = std_dev_out * z * h

    # Compute λ as the ratio of sensitivity to the output std deviation
    lambda_val = sensitivity / std_dev_out

    # Compute delta according to the formula in Theorem 3
    part1 = math.exp(-lambda_val**2 / 2) / (k * math.sqrt(math.pi))
    term1 = hyp1f1(m / 4 + 0.5, 0.5, lambda_val**2 / 2)
    term2 = math.sqrt(2) * lambda_val * hyp1f1(m / 4 + 1, 1.5, lambda_val**2 / 2)

    # Avoid division by zero in the ratio computation
    if m > 3:
        denominator = math.sqrt(m / 2 - 1.5) * math.sqrt(m / 2 + 0.75)
    else:
        denominator = 1e-10  # fallback small value if m too small

    # Compute the remaining multiplier term in delta expression
    ratio = math.sqrt(m - 1) / denominator

    # Final delta value
    delta = part1 * (term1 + term2) * ratio

    return noise, delta

def compute_tighter_epsilon_delta(epsilon, epoch, batch_size, datasize, delta=2.9674255156360687e-07, delta_tilde=1e-8):
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

    #Return the composed privacy parameters (epsilon and delta).

    return epsilons, deltas

if __name__ == "__main__":
    epsilon = 0.8
    epoch = 20
    batch_size = 10000
    datasize = 800167

    epsilons, deltas = compute_tighter_epsilon_delta(epsilon, epoch, batch_size, datasize)

    print("Epsilons:", epsilons)
    print("Deltas:", deltas)

    filename = f"epsilon_delta_eps{epsilon}.csv"
    df = pd.DataFrame({
        "Epoch": list(range(1, epoch + 1)),
        "Epsilon": epsilons,
        "Delta": deltas
    })
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")

