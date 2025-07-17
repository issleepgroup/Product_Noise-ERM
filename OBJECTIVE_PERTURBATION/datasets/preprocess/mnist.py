import os
import numpy as np
import torch
from torchvision import datasets, transforms
from utils.utils_preprocessing import convert_to_binary, normalize_rows, format_output

FILENAME_X = 'mnist_processed_x.npy'
FILENAME_Y = 'mnist_processed_y.npy'


def preprocess(cache_location, output_location):

    np.random.seed(10000019)


    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train = datasets.MNIST(root=os.path.join(cache_location, "MNIST_data"),
                                 train=True, download=True, transform=transform)

    mnist_test = datasets.MNIST(root=os.path.join(cache_location, "MNIST_data"),
                                train=False, download=True, transform=transform)

    train_features = np.array([np.array(img[0]).flatten() for img in mnist_train])
    train_labels = np.array([img[1] for img in mnist_train])

    test_features = np.array([np.array(img[0]).flatten() for img in mnist_test])
    test_labels = np.array([img[1] for img in mnist_test])

    train_labels = np.eye(10)[train_labels]
    test_labels = np.eye(10)[test_labels]

    features_set = np.vstack((train_features, test_features))
    labels_set = np.vstack((train_labels, test_labels))

    label_width = labels_set.shape[1]

    combined_data = np.column_stack([features_set, labels_set])

    np.random.shuffle(combined_data)

    np.save(os.path.join(output_location, FILENAME_X), combined_data[:, :-label_width])  # 只存图像数据
    np.save(os.path.join(output_location, FILENAME_Y), combined_data[:, -label_width:])  # 只存标签

