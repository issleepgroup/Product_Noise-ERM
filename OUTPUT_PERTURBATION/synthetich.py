# Import required libraries
import os
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (for server environments)
from sklearn.linear_model import LogisticRegression
import math
import copy
import pandas as pd
from sklearn.metrics import accuracy_score
from datetime import datetime

# ------------------------
# Load sparse input data for 'synthetich' dataset
# ------------------------
dataset_location = '../datasets/data'
data2shape = {'synthetich': (72309, 20958)}  # Sparse matrix shape

# Load compressed sparse row (CSR) format components
data = np.load(os.path.join(dataset_location, 'synthetich_processed_x.npy'))
indices = np.load(os.path.join(dataset_location, 'synthetich_processed_indices.npy'))
indptr = np.load(os.path.join(dataset_location, 'synthetich_processed_indptr.npy'))

# Construct sparse feature matrix
features = csr_matrix((data, indices, indptr), shape=data2shape['synthetich'])

# Load binary labels and convert to float
labels = np.load(os.path.join(dataset_location, 'synthetich_processed_y.npy'))
labels = labels.astype(float)

# ------------------------
# Split data into training and testing sets (80% / 20%)
# ------------------------
training_size = int(features.shape[0] * 0.8)
training_labels = labels[:training_size]
testing_labels = labels[training_size:]
training_features = features[:training_size]
testing_features = features[training_size:]

# ------------------------
# Train baseline (non-private) logistic regression model
# ------------------------
print("Training scikit-learn classifier on un-normalized data")
classifier = LogisticRegression()
classifier.fit(training_features, training_labels)
y_pred = classifier.predict(testing_features)
accuracy = accuracy_score(testing_labels, y_pred)
print("Accuracy before output perturbation: {:.8f}".format(accuracy))

theta = classifier.coef_.squeeze()  # Store coefficients

# ------------------------
# Set differential privacy hyperparameters
# ------------------------
lambda_param = 0.0001  # Regularization parameter
number_of_samples = training_features.shape[0]
sensitivity = 2 / (number_of_samples * lambda_param)  # Sensitivity of ERM solution
delta = 1e-5  # δ parameter for (ε, δ)-DP

# Define ε values for experimentation
all_eps_list = [0.0001, 0.000316227766016838, 0.001, 0.00316227766016838, 0.01, 0.0316227766016838, 0.1]

# ------------------------
# Initialize containers for accuracy results
# ------------------------
epsilons = []
gauss_acc_list = []
gauss_std_list = []
alt_acc_list = []
alt_std_list = []

# ------------------------
# Experiment 1: Gaussian noise perturbation to model coefficients
# ------------------------
print("\nExperiment 1: Gaussian noise perturbation")
for epsilon in all_eps_list:
    acc_list = []
    for i in range(10):  # Repeat for statistical robustness
        C = 1 / lambda_param
        logreg = LogisticRegression(penalty='l2', C=C, fit_intercept=False)
        logreg.fit(training_features, training_labels)
        f_non_priv = logreg.coef_[0]

        # Compute Gaussian noise standard deviation
        sigma = (np.sqrt(2 * np.log(1.25 / delta)) / epsilon) * sensitivity
        f_priv = f_non_priv + np.random.normal(0, sigma, len(f_non_priv))

        # Copy classifier and assign noisy coefficients
        classifier_copy = copy.deepcopy(classifier)
        classifier_copy.coef_ = np.array([f_priv])

        # Evaluate perturbed model
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
# Experiment 2: Product noise perturbation to model coefficients
# ------------------------
print("\nExperiment 2: Our alternative noise")
for idx, epsilon in enumerate(all_eps_list):
    acc_list = []
    for i in range(10):
        C = 1 / lambda_param
        logreg = LogisticRegression(penalty='l2', C=C, fit_intercept=False)
        logreg.fit(training_features, training_labels)
        f_non_priv = logreg.coef_[0]

        # Compute theoretical variance for product noise
        k = 1000  # Hyperparameter for controlling noise shape
        m = training_features.shape[1]  # Dimensionality of feature space
        t_squared = 2 * (k ** (4 / m)) * (((m / 4) + 1.5) ** (1 + 4 / m) / (math.e ** (1 + 2 / m)))
        std_dev_out = sensitivity * math.sqrt(t_squared) / epsilon

        # Generate noise vector with random direction and scaled magnitude
        h = np.random.randn(m)
        h = h / np.linalg.norm(h)  # Normalize to unit vector
        z = np.abs(np.random.normal(0, 1))  # Scalar magnitude
        noise = std_dev_out * z * h  # Final noise vector

        # Add noise to coefficients
        f_priv = f_non_priv + noise
        classifier_copy = copy.deepcopy(classifier)
        classifier_copy.coef_ = np.array([f_priv])

        # Evaluate perturbed model
        y_pred_priv = classifier_copy.predict(testing_features)
        accuracy_priv = accuracy_score(testing_labels, y_pred_priv)
        acc_list.append(accuracy_priv)

    avg_accuracy = np.mean(acc_list)
    std_accuracy = np.std(acc_list)
    print(f"Epsilon: {epsilon:.8f}, Alternative Average Accuracy: {avg_accuracy:.8f}, Std: {std_accuracy:.8f}")
    alt_acc_list.append(avg_accuracy)
    alt_std_list.append(std_accuracy)

# ------------------------
# Save results to CSV file
# ------------------------
df = pd.DataFrame({
    "Gaussian_Acc": gauss_acc_list,
    "Gaussian_Std": gauss_std_list,
    "Alt_Acc": alt_acc_list,
    "Alt_Std": alt_std_list,
    "Baseline": [accuracy] * len(epsilons)
}, index=epsilons)
df.index.name = "epsilon"

# Timestamped output filename
timestamp = datetime.now().strftime("%H-%M-%S")
output_csv = f"synthetich_{timestamp}_{all_eps_list[0]}.csv"
df.to_csv(output_csv)
print(f"\nResults saved to {output_csv}")
