# Import necessary libraries
import os
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from sklearn.linear_model import LogisticRegression
import math
import copy
import pandas as pd
from sklearn.metrics import accuracy_score
from datetime import datetime

# ------------------------
# Define the shape of the sparse feature matrix for the rcv1 dataset
# ------------------------
data2shape = {'rcv1': (50000, 47236)}

# ------------------------
# Load sparse features and labels
# ------------------------
dataset_location = '../datasets/data'

# Load CSR components: data, indices, and indptr
data = np.load(os.path.join(dataset_location, 'rcv1_processed_d.npy'))
indices = np.load(os.path.join(dataset_location, 'rcv1_processed_indices.npy'))
indptr = np.load(os.path.join(dataset_location, 'rcv1_processed_indptr.npy'))

# Construct sparse matrix using loaded data
features = csr_matrix((data, indices, indptr), shape=data2shape['rcv1'])

# Load binary labels
labels = np.load(os.path.join(dataset_location, 'rcv1_processed_y.npy'))
labels = labels.astype(float)

# ------------------------
# Split into training and test sets
# ------------------------
training_size = int(features.shape[0] * 0.8)
training_labels = labels[:training_size]
testing_labels = labels[training_size:]
training_features = features[:training_size]
testing_features = features[training_size:]

# ------------------------
# Train baseline (non-private) model
# ------------------------
print("Training scikit-learn classifier on un-normalized data")
classifier = LogisticRegression()
classifier.fit(training_features, training_labels)
y_pred = classifier.predict(testing_features)
accuracy = accuracy_score(testing_labels, y_pred)
print("Accuracy before output perturbation: {:.8f}".format(accuracy))

# Save coefficients for reference
theta = classifier.coef_.squeeze()

# ------------------------
# Set differential privacy parameters
# ------------------------
lambda_param = 0.0001  # Regularization parameter
number_of_samples = training_features.shape[0]
sensitivity = 2 / (number_of_samples * lambda_param)  # Sensitivity for output perturbation
delta = 1e-5  # δ in (ε, δ)-DP

# Define ε values for experimentation
all_eps_list = [0.0001, 0.000316227766016838, 0.001, 0.00316227766016838, 0.01, 0.0316227766016838, 0.1]

# ------------------------
# Prepare result containers
# ------------------------
epsilons = []
gauss_acc_list = []
gauss_std_list = []
alt_acc_list = []
alt_std_list = []

# ------------------------
# Experiment 1: Gaussian output perturbation
# ------------------------
print("\nExperiment 1: Gaussian noise perturbation")
for epsilon in all_eps_list:
    acc_list = []
    for i in range(10):  # Repeat for averaging
        C = 1 / lambda_param
        # Train a fresh logistic regression model
        logreg = LogisticRegression(penalty='l2', C=C, fit_intercept=False)
        logreg.fit(training_features, training_labels)
        f_non_priv = logreg.coef_[0]

        # Add Gaussian noise
        sigma = (np.sqrt(2 * np.log(1.25 / delta)) / epsilon) * sensitivity
        f_priv = f_non_priv + np.random.normal(0, sigma, len(f_non_priv))

        # Evaluate noisy model
        classifier_copy = copy.deepcopy(classifier)
        classifier_copy.coef_ = np.array([f_priv])
        y_pred_priv = classifier_copy.predict(testing_features)
        accuracy_priv = accuracy_score(testing_labels, y_pred_priv)
        acc_list.append(accuracy_priv)

    avg_accuracy = np.mean(acc_list)
    std_accuracy = np.std(acc_list)
    print(f"Epsilon: {epsilon:.8f}, Gaussian Average Accuracy: {avg_accuracy:.8f}, Std: {std_accuracy:.8f}")
    epsilons.append(epsilon)
    gauss_acc_list.append(avg_accuracy)
    gauss_std_list.append(std_accuracy)

# ------------------------
# Experiment 2: Product (alternative) noise perturbation
# ------------------------
print("\nExperiment 2: Our alternative noise")
for idx, epsilon in enumerate(all_eps_list):
    acc_list = []
    for i in range(10):
        C = 1 / lambda_param
        logreg = LogisticRegression(penalty='l2', C=C, fit_intercept=False)
        logreg.fit(training_features, training_labels)
        f_non_priv = logreg.coef_[0]

        # Compute noise scaling factor based on dimensionality
        k = 1000
        m = training_features.shape[1]
        t_squared = 2 * (k ** (4 / m)) * (((m / 4) + (3 / 2)) ** (1 + 4 / m) / (math.e ** (1 + 2 / m)))
        std_dev_out = sensitivity * math.sqrt(t_squared) / epsilon

        # Generate noise in random unit direction
        h = np.random.randn(m)
        h = h / np.linalg.norm(h)
        z = np.abs(np.random.normal(0, 1))  # magnitude
        noise = std_dev_out * z * h

        # Add product noise to weights
        f_priv = f_non_priv + noise
        classifier_copy = copy.deepcopy(classifier)
        classifier_copy.coef_ = np.array([f_priv])

        # Evaluate noisy model
        y_pred_priv = classifier_copy.predict(testing_features)
        accuracy_priv = accuracy_score(testing_labels, y_pred_priv)
        acc_list.append(accuracy_priv)

    avg_accuracy = np.mean(acc_list)
    std_accuracy = np.std(acc_list)
    print(f"Epsilon: {epsilon:.8f}, Alternative Average Accuracy: {avg_accuracy:.8f}, Std: {std_accuracy:.8f}")
    alt_acc_list.append(avg_accuracy)
    alt_std_list.append(std_accuracy)

# ------------------------
# Save results to CSV
# ------------------------
df = pd.DataFrame({
    "Gaussian_Acc": gauss_acc_list,
    "Gaussian_Std": gauss_std_list,
    "Alt_Acc": alt_acc_list,
    "Alt_Std": alt_std_list,
    "Baseline": [accuracy] * len(epsilons)
}, index=epsilons)
df.index.name = "epsilon"

# Add timestamp to filename
timestamp = datetime.now().strftime("%H-%M-%S")
output_csv = f"../op_results/rcv1_{timestamp}_{lambda_param}.csv"
df.to_csv(output_csv)
print(f"\nResults saved to {output_csv}")
