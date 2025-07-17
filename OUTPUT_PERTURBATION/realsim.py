# Import required libraries
import os
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for environments without GUI
from sklearn.svm import SVC  # Not used in current script, but imported
import math
import copy
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score
from datetime import datetime

# -------------------------
# Define the sparse matrix shape for the realsim dataset
# -------------------------
data2shape = {'realsim': (72309, 20958)}

# -------------------------
# Load preprocessed sparse features and labels
# -------------------------
dataset_location = '../datasets/data'
data = np.load(os.path.join(dataset_location, 'realsim_processed_d.npy'))
indices = np.load(os.path.join(dataset_location, 'realsim_processed_indices.npy'))
indptr = np.load(os.path.join(dataset_location, 'realsim_processed_indptr.npy'))

features = csr_matrix((data, indices, indptr), shape=data2shape['realsim'])

labels = np.load(os.path.join(dataset_location, 'realsim_processed_y.npy'))
labels = labels.astype(float)

# -------------------------
# Split data into training and test sets (80% / 20%)
# -------------------------
training_size = int(features.shape[0] * 0.8)
training_labels = labels[:training_size]
testing_labels = labels[training_size:]
training_features = features[:training_size]
testing_features = features[training_size:]

# -------------------------
# Train baseline logistic regression model (non-private)
# -------------------------
print("Training scikit-learn classifier on un-normalized data")
classifier = LogisticRegression()
classifier.fit(training_features, training_labels)
y_pred = classifier.predict(testing_features)
accuracy = accuracy_score(testing_labels, y_pred)
print("Accuracy before output perturbation: {:.8f}".format(accuracy))

theta = classifier.coef_.squeeze()  # Save learned coefficients

# -------------------------
# Set differential privacy parameters
# -------------------------
lambda_param = 0.0001
number_of_samples = training_features.shape[0]
sensitivity = 2 / (number_of_samples * lambda_param)
delta = 1e-5  # δ in (ε, δ)-DP

# List of ε (privacy budget) values to evaluate
all_eps_list = [0.0001, 0.000316227766016838, 0.001, 0.00316227766016838, 0.01, 0.0316227766016838, 0.1]

# -------------------------
# Initialize result containers
# -------------------------
epsilons = []
gauss_acc_list = []
gauss_std_list = []
alt_acc_list = []
alt_std_list = []

# -------------------------
# Experiment 1: Gaussian noise perturbation
# -------------------------
print("\nExperiment 1: Gaussian noise perturbation")
for epsilon in all_eps_list:
    acc_list = []
    for i in range(10):  # Repeat 10 runs for statistical stability
        C = 1 / lambda_param
        logreg = LogisticRegression(penalty='l2', C=C, fit_intercept=False)
        logreg.fit(training_features, training_labels)
        f_non_priv = logreg.coef_[0]  # Get original coefficients

        # Compute Gaussian noise std deviation
        sigma = (np.sqrt(2 * np.log(1.25 / delta)) / epsilon) * sensitivity
        f_priv = f_non_priv + np.random.normal(0, sigma, len(f_non_priv))  # Add noise

        # Copy classifier and assign perturbed coefficients
        classifier_copy = copy.deepcopy(classifier)
        classifier_copy.coef_ = np.array([f_priv])

        # Predict and evaluate
        y_pred_priv = classifier_copy.predict(testing_features)
        accuracy_priv = accuracy_score(testing_labels, y_pred_priv)
        acc_list.append(accuracy_priv)

    # Record average and standard deviation of accuracy
    avg_accuracy = np.mean(acc_list)
    std_accuracy = np.std(acc_list)
    print(f"Epsilon: {epsilon:.8f}, Gaussian Average Accuracy: {avg_accuracy:.8f}, Std: {std_accuracy:.8f}")
    epsilons.append(round(epsilon, 8))
    gauss_acc_list.append(round(avg_accuracy, 8))
    gauss_std_list.append(round(std_accuracy, 8))

# -------------------------
# Experiment 2: Alternative (product) noise perturbation
# -------------------------
print("\nExperiment 2: Our alternative noise")
for idx, epsilon in enumerate(all_eps_list):
    acc_list = []
    for i in range(10):
        C = 1 / lambda_param
        logreg = LogisticRegression(penalty='l2', C=C, fit_intercept=False)
        logreg.fit(training_features, training_labels)
        f_non_priv = logreg.coef_[0]

        # Compute product noise scale
        k = 1000
        m = training_features.shape[1]
        t_squared = 2 * (k ** (4 / m)) * (((m / 4) + 1.5) ** (1 + 4 / m) / (math.e ** (1 + 2 / m)))
        std_dev_out = sensitivity * math.sqrt(t_squared) / epsilon

        # Generate noise direction and magnitude
        h = np.random.randn(m)
        h = h / np.linalg.norm(h)  # Normalize to unit vector
        z = np.abs(np.random.normal(0, 1))  # Random magnitude
        noise = std_dev_out * z * h

        # Add noise to coefficients
        f_priv = f_non_priv + noise
        classifier_copy = copy.deepcopy(classifier)
        classifier_copy.coef_ = np.array([f_priv])

        # Predict and evaluate
        y_pred_priv = classifier_copy.predict(testing_features)
        accuracy_priv = accuracy_score(testing_labels, y_pred_priv)
        acc_list.append(accuracy_priv)

    # Record average and standard deviation
    avg_accuracy = np.mean(acc_list)
    std_accuracy = np.std(acc_list)
    print(f"Epsilon: {epsilon:.8f}, Alternative Average Accuracy: {avg_accuracy:.8f}, Std: {std_accuracy:.8f}")
    alt_acc_list.append(avg_accuracy)
    alt_std_list.append(std_accuracy)

# -------------------------
# Save all results to CSV file
# -------------------------
df = pd.DataFrame({
    "Gaussian_Acc": gauss_acc_list,
    "Gaussian_Std": gauss_std_list,
    "Alt_Acc": alt_acc_list,
    "Alt_Std": alt_std_list,
    "Baseline": [accuracy] * len(epsilons)
}, index=epsilons)
df.index.name = "epsilon"

# Add timestamp to output filename
timestamp = datetime.now().strftime("%H-%M-%S")

# (Note: The file name says mnist_svm which may not match the current dataset; may want to rename.)
output_csv = f"../op_results/mnist_svm_{timestamp}_{lambda_param}.csv"

df.to_csv(output_csv)
print(f"\nResults saved to {output_csv}")
