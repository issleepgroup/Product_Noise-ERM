import math
import numpy as np
import pandas as pd

def compute_strong_epsilon_delta(epsilon, datasize, batch_size, epochs):
    delta_c = 1e-10
    q = batch_size / datasize

    subsampled_epsilon = math.log(1 + q * (math.exp(epsilon) - 1))
    subsampled_delta = q * delta_c

    num_steps_per_epoch = math.ceil(datasize // batch_size)

    epsilons = []
    deltas = []

    for epoch in range(1, epochs + 1):
        k = num_steps_per_epoch * epoch

        term1 = math.sqrt(2 * k * np.log(1 / delta_c)) * subsampled_epsilon
        term2 = k * subsampled_epsilon * (math.exp(subsampled_epsilon) - 1)
        epsilon_total = term1 + term2
        delta_total = k * subsampled_delta + delta_c

        epsilons.append(epsilon_total)
        deltas.append(delta_total)

    return epsilons, deltas

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

    return epsilons, deltas

if __name__ == "__main__":
    epsilon = 0.8
    epoch = 20
    batch_size = 10000
    datasize = 800167

    epsilons, deltas = compute_tighter_epsilon_delta(epsilon, epoch, batch_size, datasize)

    print("Epsilons:", epsilons)
    print("Deltas:", deltas)

    filename = f"movie_epsilon_delta_eps{epsilon}.csv"
    df = pd.DataFrame({
        "Epoch": list(range(1, epoch + 1)),
        "Epsilon": epsilons,
        "Delta": deltas
    })
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")

