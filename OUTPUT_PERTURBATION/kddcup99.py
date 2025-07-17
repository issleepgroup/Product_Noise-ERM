import os
import numpy as np
import copy
import math
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# load data for kddcup99
FILENAME_X = 'kddcup99_processed_x.npy'
FILENAME_Y = 'kddcup99_processed_y.npy'
data_folder = '../datasets/data'

X = np.load(os.path.join(data_folder, FILENAME_X), allow_pickle=True)
X = X.astype(float)
y = np.load(os.path.join(data_folder, FILENAME_Y), allow_pickle=True)
y = y.astype(float)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 训练原始的 scikit-learn 分类器（未经过扰动）
print("Training scikit-learn classifier on un-normalized data")
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy before output perturbation: {:.8f}".format(accuracy))

theta = classifier.coef_.squeeze()

# 设置超参数
lambda_param = 0.01
number_of_samples = X_train.shape[0]
L = 1
sensitivity = (2 * L )/ (number_of_samples * lambda_param)
delta = 1e-5

# ε 值测试集（这里使用 all_eps_list2）

all_eps_list = [0.0001, 0.000316227766016838, 0.001, 0.00316227766016838, 0.01, 0.0316227766016838, 0.1]



# 用于保存结果
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
        logreg = LogisticRegression(penalty='l2', solver='lbfgs', C=1, fit_intercept=False, max_iter=1000)
        logreg.fit(X_train, y_train)
        f_non_priv = logreg.coef_[0]

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
    epsilons.append(epsilon)
    gauss_acc_list.append(avg_accuracy)
    gauss_std_list.append(std_accuracy)

# 实验 2：Our alternative noise
print("\nExperiment 2: Our alternative noise")
for idx, epsilon in enumerate(all_eps_list):
    acc_list = []
    for i in range(10):
        C = 1 / lambda_param
        logreg = LogisticRegression(penalty='l2', solver='lbfgs', C=1, fit_intercept=False, max_iter=1000)
        logreg.fit(X_train, y_train)
        f_non_priv = logreg.coef_[0]

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
    alt_acc_list.append(avg_accuracy)
    alt_std_list.append(std_accuracy)

# 保存结果到 CSV 文件，文件名中加入当前时间（时-分-秒）和 ε 列表的首个值作为标识
df = pd.DataFrame({
    "Gaussian_Acc": gauss_acc_list,
    "Gaussian_Std": gauss_std_list,
    "Alt_Acc": alt_acc_list,
    "Alt_Std": alt_std_list,
    "Baseline": [accuracy] * len(epsilons)

}, index=epsilons)
df.index.name = "epsilon"

timestamp = datetime.now().strftime("%H-%M-%S")
output_csv = f"../op_results/kddcup99_{timestamp}_{lambda_param}.csv"
df.to_csv(output_csv)
print(f"\nResults saved to {output_csv}")
