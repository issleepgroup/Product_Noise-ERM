import os
import numpy as np
import torch
from torchvision import datasets, transforms
from utils.utils_preprocessing import convert_to_binary, normalize_rows, format_output

FILENAME_X = 'mnist_processed_x.npy'
FILENAME_Y = 'mnist_processed_y.npy'

def preprocess(cache_location, output_location):
    np.random.seed(10000019)

    # Define transformation to convert images to tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load training and test datasets
    mnist_train = datasets.MNIST(root=os.path.join(cache_location, "MNIST_data"),
                                 train=True, download=True, transform=transform)

    mnist_test = datasets.MNIST(root=os.path.join(cache_location, "MNIST_data"),
                                train=False, download=True, transform=transform)

    # Flatten the image tensors and extract labels
    train_features = np.array([np.array(img[0]).flatten() for img in mnist_train])
    train_labels = np.array([img[1] for img in mnist_train])

    test_features = np.array([np.array(img[0]).flatten() for img in mnist_test])
    test_labels = np.array([img[1] for img in mnist_test])

    # One-hot encode the labels
    train_labels = np.eye(10)[train_labels]
    test_labels = np.eye(10)[test_labels]

    # Combine training and testing data
    features_set = np.vstack((train_features, test_features))
    labels_set = np.vstack((train_labels, test_labels))

    label_width = labels_set.shape[1]

    # Combine features and labels into one array
    combined_data = np.column_stack([features_set, labels_set])

    # Shuffle the combined dataset
    np.random.shuffle(combined_data)

    # Save processed features and labels separately
    np.save(os.path.join(output_location, FILENAME_X), combined_data[:, :-label_width])  # Save only image data
    np.save(os.path.join(output_location, FILENAME_Y), combined_data[:, -label_width:])  # Save only label data
