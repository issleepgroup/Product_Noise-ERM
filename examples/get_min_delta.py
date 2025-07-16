# import numpy as np
# from scipy.special import hyp1f1
#
# # Constants
# M = 131466
# epsilon = 1.5
# k = 200000
# q_values = np.linspace(1, M - 3/2, 1000)  # Avoid too small q values
#
# # Define lambda as a function of q
# def calculate_lambda(q, epsilon, k):
#     exponent = 1/2 + 1/(2*q)
#     if exponent > 700:  # Cap to avoid overflow
#         exponent = 700
#     return (epsilon * np.exp(exponent)) / (np.sqrt(2) * k**(1/q) * ((q+3)/2)**(1/2 + 1/q))
#
# # Define delta_1 and delta_2 as functions of q and lambda
#
# def delta_1(q, lambda_val, k):
#     if (M - q - 3/2) <= 0 or (q + 3/4) <= 0:
#         return np.nan
#     term_1 = hyp1f1((q + 1) / 2, 1 / 2, lambda_val**2 / 2)
#     term_2 = np.sqrt(2) * lambda_val * hyp1f1((q + 2) / 2, 3 / 2, lambda_val**2 / 2)
#     return (np.exp(-lambda_val**2 / 2) / (k * np.sqrt(np.pi))) * (term_1 + term_2) * np.sqrt(M - 1) / np.sqrt((M - q - 3/2) * (q + 3/4))
#
# def delta_2(q, lambda_val, k):
#     if (M - q - 3/2) <= 0 or (q + 3/2) <= 0:
#         return np.nan
#     return (2 * np.exp(-lambda_val**2 / 2) * np.sqrt(M - 1)) / (k * np.sqrt(np.pi)) * (1 + lambda_val**2 * (q + 1)) / np.sqrt((M - q - 3/2) * (q + 3/4))
#
# # Calculate lambda, delta_1, and delta_2 for each q
# lambda_values = np.array([calculate_lambda(q, epsilon, k) for q in q_values])
# delta_1_values = np.array([delta_1(q, lambda_val, k) for q, lambda_val in zip(q_values, lambda_values)])
# delta_2_values = np.array([delta_2(q, lambda_val, k) for q, lambda_val in zip(q_values, lambda_values)])
#
# # Set negative or zero values to NaN for log scale
# delta_1_values[delta_1_values <= 0] = np.nan
# delta_2_values[delta_2_values <= 0] = np.nan
#
# # Find the q value for the minimum delta_1 and delta_2
# min_delta_1_index = np.nanargmin(delta_1_values)
# min_delta_2_index = np.nanargmin(delta_2_values)
#
# min_q_delta_1 = q_values[min_delta_1_index]
# min_q_delta_2 = q_values[min_delta_2_index]
# min_delta_1 = delta_1_values[min_delta_1_index]
# min_delta_2 = delta_2_values[min_delta_2_index]
#
# print(f"The value of q for the minimum delta_1 is: {min_q_delta_1}, with delta_1 value: {min_delta_1}")
# print(f"The value of q for the minimum delta_2 is: {min_q_delta_2}, with delta_2 value: {min_delta_2}")


import numpy as np
from scipy.special import hyp1f1

def find_optimal_q(M=146320, epsilon=0.8, k=20000, num_points=1000):
    # Generate q values
    q_values = np.linspace(1, M - 3/2, num_points)

    def calculate_lambda(q):
        exponent = 1/2 + 1/(2*q)
        exponent = min(exponent, 700)  # Avoid overflow
        return (epsilon * np.exp(exponent)) / (np.sqrt(2) * k**(1/q) * ((q + 3)/2)**(1/2 + 1/q))

    def delta_1(q, lambda_val):
        if (M - q - 3/2) <= 0 or (q + 3/4) <= 0:
            return np.nan
        term_1 = hyp1f1((q + 1) / 2, 1 / 2, lambda_val**2 / 2)
        term_2 = np.sqrt(2) * lambda_val * hyp1f1((q + 2) / 2, 3 / 2, lambda_val**2 / 2)
        return (np.exp(-lambda_val**2 / 2) / (k * np.sqrt(np.pi))) * \
               (term_1 + term_2) * np.sqrt(M - 1) / np.sqrt((M - q - 3/2) * (q + 3/4))

    def delta_2(q, lambda_val):
        if (M - q - 3/2) <= 0 or (q + 3/2) <= 0:
            return np.nan
        return (2 * np.exp(-lambda_val**2 / 2) * np.sqrt(M - 1)) / (k * np.sqrt(np.pi)) * \
               (1 + lambda_val**2 * (q + 1)) / np.sqrt((M - q - 3/2) * (q + 3/4))

    # Compute all lambda, delta_1, and delta_2 values
    lambda_values = np.array([calculate_lambda(q) for q in q_values])
    delta_1_values = np.array([delta_1(q, lam) for q, lam in zip(q_values, lambda_values)])
    delta_2_values = np.array([delta_2(q, lam) for q, lam in zip(q_values, lambda_values)])

    # Clean invalid values for log-scale safety
    delta_1_values[delta_1_values <= 0] = np.nan
    delta_2_values[delta_2_values <= 0] = np.nan

    # Find minimums
    min_delta_1_index = np.nanargmin(delta_1_values)
    min_delta_2_index = np.nanargmin(delta_2_values)

    return {
        "min_q_delta_1": q_values[min_delta_1_index],
        "min_delta_1": delta_1_values[min_delta_1_index],
        "min_q_delta_2": q_values[min_delta_2_index],
        "min_delta_2": delta_2_values[min_delta_2_index]
    }

# Example usage
result = find_optimal_q()
print(f"The value of q for the minimum delta_1 is: {result['min_q_delta_1']}, with delta_1 value: {result['min_delta_1']}")
print(f"The value of q for the minimum delta_2 is: {result['min_q_delta_2']}, with delta_2 value: {result['min_delta_2']}")
