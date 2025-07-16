import numpy as np
import math
from scipy.special import hyp1f1

L = 1  # Lipschitz constant
lambda_param = 0.01  # Regularization parameter
n_samples = 26010
sensitivity = (2 * L) / (n_samples * lambda_param)
def generate_alternative_noise(m, sensitivity, epsilon, k=1000):
    """
    Generate alternative noise for differential privacy.

    Args:
        m (int): Dimension of the model (number of features).
        sensitivity (float): Sensitivity value (pre-computed outside).
        epsilon (float): Privacy budget.
        k (int): Constant parameter (default=1000).

    Returns:
        numpy.ndarray: Noise vector of shape (m,)
    """
    t_squared = 2 * (k ** (4 / m)) * (((m / 4) + (3 / 2)) ** (1 + 4 / m) / (math.e ** (1 + 2 / m)))
    std_dev_out = sensitivity * math.sqrt(t_squared) / epsilon
    h = np.random.randn(m)
    h = h / np.linalg.norm(h)
    z = np.abs(np.random.normal(0, 1))
    noise = std_dev_out * z * h
    return noise

def generate_alternative_noise_2(L, n_samples, lambda_param, m, epsilon, k=1000):
    """
    Generate alternative noise for differential privacy, with sensitivity computed internally.

    Args:
        L (float): Lipschitz constant.
        n_samples (int): Number of training samples.
        lambda_param (float): Regularization parameter.
        m (int): Dimension of the model (number of features).
        epsilon (float): Privacy budget.
        k (int): Constant parameter (default=1000).

    Returns:
        numpy.ndarray: Noise vector of shape (m,)
    """
    sensitivity = (2 * L) / (n_samples * lambda_param)
    t_squared = 2 * (k ** (4 / m)) * (((m / 4) + (3 / 2)) ** (1 + 4 / m) / (math.e ** (1 + 2 / m)))
    std_dev_out = sensitivity * math.sqrt(t_squared) / epsilon
    h = np.random.randn(m)
    h = h / np.linalg.norm(h)
    z = np.abs(np.random.normal(0, 1))
    noise = std_dev_out * z * h
    return noise


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


