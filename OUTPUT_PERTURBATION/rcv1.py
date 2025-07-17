import os
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
from sklearn.linear_model import LogisticRegression
import math
import copy
import pandas as pd
from sklearn.metrics import accuracy_score
from datetime import datetime

data2shape = {'rcv1': (50000, 47236)}

# load data
dataset_location = '../datasets/data'

data = np.load(os.path.join(dataset_location, 'rcv1_processed_d.npy'))
indices = np.load(os.path.join(dataset_location, 'rcv1_processed_indices.npy'))
indptr = np.load(os.path.join(dataset_location, 'rcv1_processed_indptr.npy'))
features = csr_matrix((data, indices, indptr), shape=data2shape['rcv1'])
labels = np.load(os.path.join(dataset_location, 'rcv1_processed_y.npy'))
labels = labels.astype(float)

training_size = int(features.shape[0] * 0.8)
training_labels = labels[:training_size]
testing_labels = labels[training_size:]
training_features = features[:training_size]
testing_features = features[training_size:]

print("Training scikit-learn classifier on un-normalized data")
classifier = LogisticRegression()
classifier.fit(training_features, training_labels)
y_pred = classifier.predict(testing_features)
accuracy = accuracy_score(testing_labels, y_pred)
print("Accuracy before output perturbation: {:.8f}".format(accuracy))

theta = classifier.coef_.squeeze()

# set hyperparameters
lambda_param = 0.0001
number_of_samples = training_features.shape[0]
sensitivity = 2 / (number_of_samples * lambda_param)
delta = 1e-5

# ε values to test
all_eps_list = [0.0001, 0.000316227766016838, 0.001, 0.00316227766016838, 0.01, 0.0316227766016838, 0.1]

# save results
epsilons = []
gauss_acc_list = []
gauss_std_list = []
alt_acc_list = []
alt_std_list = []

# 实验 1：Gaussian 噪声扰动
print("\nExperiment 1: Gaussian noise perturbation")
for epsilon in all_eps_list:
    acc_list = []
    for i in range(10):
        C = 1 / lambda_param
        logreg = LogisticRegression(penalty='l2', C=C, fit_intercept=False)
        logreg.fit(training_features, training_labels)
        f_non_priv = logreg.coef_[0]

        sigma = (np.sqrt(2 * np.log(1.25 / delta)) / epsilon) * sensitivity
        f_priv = f_non_priv + np.random.normal(0, sigma, len(f_non_priv))

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

# 实验 2：Our alternative noise
print("\nExperiment 2: Our alternative noise")
for idx, epsilon in enumerate(all_eps_list):
    acc_list = []
    for i in range(10):
        C = 1 / lambda_param
        logreg = LogisticRegression(penalty='l2', C=C, fit_intercept=False)
        logreg.fit(training_features, training_labels)
        f_non_priv = logreg.coef_[0]

        k = 1000
        m = training_features.shape[1]
        t_squared = 2 * (k ** (4 / m)) * (((m / 4) + (3 / 2)) ** (1 + 4 / m) / (math.e ** (1 + 2 / m)))
        std_dev_out = sensitivity * math.sqrt(t_squared) / epsilon
        h = np.random.randn(m)
        h = h / np.linalg.norm(h)
        z = np.abs(np.random.normal(0, 1))
        noise = std_dev_out * z * h

        f_priv = f_non_priv + noise
        classifier_copy = copy.deepcopy(classifier)
        classifier_copy.coef_ = np.array([f_priv])

        y_pred_priv = classifier_copy.predict(testing_features)
        accuracy_priv = accuracy_score(testing_labels, y_pred_priv)
        acc_list.append(accuracy_priv)

    avg_accuracy = np.mean(acc_list)
    std_accuracy = np.std(acc_list)
    print(f"Epsilon: {epsilon:.8f}, Alternative Average Accuracy: {avg_accuracy:.8f}, Std: {std_accuracy:.8f}")
    alt_acc_list.append(avg_accuracy)
    alt_std_list.append(std_accuracy)

# save as csv
df = pd.DataFrame({
    "Gaussian_Acc": gauss_acc_list,
    "Gaussian_Std": gauss_std_list,
    "Alt_Acc": alt_acc_list,
    "Alt_Std": alt_std_list,
    "Baseline": [accuracy] * len(epsilons)
}, index=epsilons)
df.index.name = "epsilon"

timestamp = datetime.now().strftime("%H-%M-%S")
output_csv = f"../op_results/rcv1_{timestamp}_{lambda_param}.csv"
df.to_csv(output_csv)
print(f"\nResults saved to {output_csv}")