import math
import pandas as pd
import numpy as np

def compute_total_epsilon(epsilon, datasize, batch_size, epochs, delta):
    delta_c = 1e-10

    q = batch_size / datasize

    sampled_epsilon = math.log(1 + q * (math.exp(epsilon) - 1))
    sampled_delta = q * delta

    num_steps_per_epoch = math.ceil(datasize // batch_size)

    epsilons = []
    deltas = []

    for epoch in range(1, epochs + 1):
        k = num_steps_per_epoch * epoch
        term1 = math.sqrt(2 * k * math.log(1 / delta_c)) * sampled_epsilon
        term2 = k * sampled_epsilon * (math.exp(sampled_epsilon) - 1)
        epsilon_total = term1 + term2
        epsilons.append(round(epsilon_total, 10))

        delta_total = k * sampled_delta + delta_c
        deltas.append(round(delta_total, 10))

    return epsilons,deltas


if __name__ == '__main__':
    # calculate initial epsilon
    initial_epsilon = 0.3
    delta = 6.79326045400711e-08

    batch_size = 256
    datasize = 60000
    epochs = 100
    total_epsilon,total_delta = compute_total_epsilon(initial_epsilon, datasize, batch_size, epochs, delta)
    print(total_epsilon)
    print(total_delta)

    df = pd.DataFrame({
        'Epsilon': total_epsilon
    })
    df.to_csv(f'mnist_epsilon_list.csv', index=False)
