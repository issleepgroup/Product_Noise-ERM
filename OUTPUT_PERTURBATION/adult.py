import os
import numpy as np
import copy
import math
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime


# load data
FILENAME_X = 'adult_processed_x.npy'
FILENAME_Y = 'adult_processed_y.npy'
data_location = '../datasets/data'


X = np.load(os.path.join(data_location, FILENAME_X))
y = np.load(os.path.join(data_location, FILENAME_Y))


# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# train scikit-learn classifier on un-normalized data, get non-private accuracy
print("Training scikit-learn classifier on un-normalized data")
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred), 8)
print("Accuracy before output perturbation: {:.8f}".format(accuracy))


# Set differential privacy hyperparameters
L = 1  # Lipschitz constant
delta = 1e-5
lambda_param = 0.01  # Regularization parameter
n_samples = X_train.shape[0]
sensitivity = (2 * L) / (n_samples * lambda_param)


# Epsilon values to test
all_eps_list = [0.0001, 0.000316227766016838, 0.001, 0.00316227766016838, 0.01, 0.0316227766016838, 0.1]


# save results
epsilons = []
gauss_acc_list = []
gauss_std_list = []
alt_acc_list = []
alt_std_list = []


# Experiment 1: Gaussian noise perturbation
print("\nExperiment 1: Gaussian noise perturbation")
for epsilon in all_eps_list:
    acc_list = []

    # generate noise
    # repeat experiment 10 times to get average accuracy and std
    # predict on test data with perturbed model
    for i in range(10):
        C = 1 / lambda_param
        logreg = LogisticRegression(penalty='l2', solver='saga', max_iter=500)
        logreg.fit(X_train, y_train)
        f_non_priv = logreg.coef_[0]

        # generate noise and add to model
        sigma = (np.sqrt(2 * np.log(1.25 / delta)) / epsilon) * sensitivity
        f_priv = f_non_priv + np.random.normal(0, sigma, len(f_non_priv))

        classifier_copy = copy.deepcopy(classifier)
        classifier_copy.coef_ = np.array([f_priv])

        y_pred_priv = classifier_copy.predict(X_test)
        accuracy_priv = accuracy_score(y_test, y_pred_priv)
        acc_list.append(accuracy_priv)

    avg_accuracy = np.mean(acc_list)
    std_accuracy = np.std(acc_list)

    print(f"Epsilon: {epsilon:.8f}, Gaussian Average Accuracy: {avg_accuracy:.8f}, Std: {std_accuracy:.8f}")
    epsilons.append(round(epsilon, 8))
    gauss_acc_list.append(round(avg_accuracy, 8))
    gauss_std_list.append(round(std_accuracy, 8))


# Experiment 2: Our alternative noise proposed product noise
print("\nExperiment 2: Our alternative noise")
for idx, epsilon in enumerate(all_eps_list):
    acc_list = []
    for i in range(10):
        C = 1 / lambda_param
        logreg = LogisticRegression(penalty='l2', solver='saga', max_iter=500)
        logreg.fit(X_train, y_train)
        f_non_priv = logreg.coef_[0]

        # generate noise and add to model
        k = 1000
        m = X_train.shape[1]
        t_squared = 2 * (k ** (4 / m)) * (((m / 4) + (3 / 2)) ** (1 + 4 / m) / (math.e ** (1 + 2 / m)))
        std_dev_out = sensitivity * math.sqrt(t_squared) / epsilon
        h = np.random.randn(m)
        h = h / np.linalg.norm(h)
        z = np.abs(np.random.normal(0, 1))
        noise = std_dev_out * z * h

        f_priv = f_non_priv + noise
        classifier_copy = copy.deepcopy(classifier)
        classifier_copy.coef_ = np.array([f_priv])

        y_pred_priv = classifier_copy.predict(X_test)
        accuracy_priv = accuracy_score(y_test, y_pred_priv)
        acc_list.append(accuracy_priv)

    avg_accuracy = np.mean(acc_list)
    std_accuracy = np.std(acc_list)
    print(f"Epsilon: {epsilon:.8f}, Alternative Average Accuracy: {avg_accuracy:.8f}, Std: {std_accuracy:.8f}")
    alt_acc_list.append(round(avg_accuracy, 8))
    alt_std_list.append(round(std_accuracy, 8))


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
save_location = '../op_results/'
output_csv = save_location + f"op_lr_adult_results_{lambda_param}_{timestamp}.csv"
df.to_csv(output_csv)
print(f"\nResults saved to {output_csv}")