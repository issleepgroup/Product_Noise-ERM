# Import required libraries
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for safe script execution
from sklearn.linear_model import LogisticRegression
import math
import copy
import pandas as pd
from sklearn.metrics import accuracy_score
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# ------------------------
# Load MNIST dataset
# ------------------------
dataset_location = '../datasets/data'

# Load preprocessed features
features = np.load(os.path.join(dataset_location, 'mnist_processed_x.npy'))
features = features.astype(float)

# Load one-hot encoded labels
labels = np.load(os.path.join(dataset_location, 'mnist_processed_y.npy'))
labels = labels.astype(float)

# ------------------------
# Prepare training and test sets
# ------------------------
training_size = int(features.shape[0] * 0.8)

# Convert one-hot encoded labels to integer class labels
labels_ = []
for row in labels:
    for i in range(len(row)):
        if row[i] == 1:
            labels_.append(i)

training_labels = labels_[:training_size]
testing_labels = labels_[training_size:]
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

# ------------------------
# Set differential privacy hyperparameters
# ------------------------
lambda_param = 0.0001  # Regularization strength
number_of_samples = training_features.shape[0]
L = 1  # Lipschitz constant
sensitivity = (2 * L) / (number_of_samples * lambda_param)
delta = 1e-5  # δ parameter in (ε, δ)-DP

# Define ε values to test
all_eps_list = [0.0001, 0.000316227766016838, 0.001, 0.00316227766016838, 0.01, 0.0316227766016838, 0.1]

# Initialize result containers
epsilons = []
gauss_acc_list = []
gauss_std_list = []
alt_acc_list = []
alt_std_list = []

# ------------------------
# Experiment 1: Gaussian noise output perturbation
# ------------------------
print("\nExperiment 1: Gaussian noise perturbation")
for epsilon in all_eps_list:
    acc_list = []
    for i in range(10):  # Repeat for statistical reliability
        # Train model on training data
        logreg = LogisticRegression(solver='lbfgs', max_iter=10000)
        logreg.fit(training_features, training_labels)
        f_non_priv = logreg.coef_[0]  # Extract learned coefficients

        # Compute Gaussian noise scale (σ)
        sigma = (np.sqrt(2 * np.log(1.25 / delta)) / epsilon) * sensitivity
        noise = np.random.normal(0, sigma, len(f_non_priv))
        f_priv = f_non_priv + noise  # Add Gaussian noise

        # Replace original model weights with perturbed version
        classifier_copy = copy.deepcopy(classifier)
        classifier_copy.coef_ = np.array([f_priv])

        # Predict and evaluate accuracy
        y_pred_priv = classifier_copy.predict(testing_features)
        accuracy_priv = accuracy_score(testing_labels, y_pred_priv)
        acc_list.append(accuracy_priv)

    # Record average and std deviation
    avg_accuracy = np.mean(acc_list)
    std_accuracy = np.std(acc_list)
    print(f"Epsilon: {epsilon:.8f}, Gaussian Average Accuracy: {avg_accuracy:.8f}, Std: {std_accuracy:.8f}")
    epsilons.append(epsilon)
    gauss_acc_list.append(avg_accuracy)
    gauss_std_list.append(std_accuracy)

# ------------------------
# Experiment 2: Alternative product noise perturbation
# ------------------------
print("\nExperiment 2: Our alternative noise")
for idx, epsilon in enumerate(all_eps_list):
    acc_list = []
    for i in range(10):
        # Train model
        logreg = LogisticRegression(solver='lbfgs', max_iter=10000)
        logreg.fit(training_features, training_labels)
        f_non_priv = logreg.coef_[0]

        # Compute theoretical bound for product noise
        k = 1000
        m = training_features.shape[1]  # Number of features
        t_squared = 2 * (k ** (4 / m)) * (((m / 4) + (3 / 2)) ** (1 + 4 / m) / (math.e ** (1 + 2 / m)))
        std_dev_out = sensitivity * math.sqrt(t_squared) / epsilon

        # Generate random unit direction and magnitude
        h = np.random.randn(m)
        h = h / np.linalg.norm(h)
        z = np.abs(np.random.normal(0, 1))
        noise = std_dev_out * z * h

        # Add product noise to model coefficients
        f_priv = f_non_priv + noise
        classifier_copy = copy.deepcopy(classifier)
        classifier_copy.coef_ = np.array([f_priv])

        # Predict and evaluate
        y_pred_priv = classifier_copy.predict(testing_features)
        accuracy_priv = accuracy_score(testing_labels, y_pred_priv)
        acc_list.append(accuracy_priv)

    # Record average and std deviation
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
save_location = '../op_results/'
output_csv = save_location + f"op_lr_mnist_results_{lambda_param}_{timestamp}.csv"
df.to_csv(output_csv)
print(f"\nResults saved to {output_csv}")
