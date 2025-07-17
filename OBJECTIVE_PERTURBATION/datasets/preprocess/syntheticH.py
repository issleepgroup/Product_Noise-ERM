import os
import numpy as np
from sklearn.datasets import make_classification
from tqdm import tqdm

FILENAME_X = 'syntheticH_processed_x.npy'
FILENAME_Y = 'syntheticH_processed_y.npy'


def preprocess(output_location):
    print("Generating dataset...")

    num_samples = 70000
    batch_size = 10000
    X_list = []
    y_list = []

    for _ in tqdm(range(num_samples // batch_size), desc="Generating Data", unit="batch"):
        X_batch, y_batch = make_classification(n_samples=batch_size, n_features=20000, n_informative=5000,
                                               n_redundant=5000, n_classes=2, random_state=None)
        X_list.append(X_batch)
        y_list.append(y_batch)

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    print("Saving dataset...")
    np.save(os.path.join(output_location, FILENAME_X), X)
    np.save(os.path.join(output_location, FILENAME_Y), y)

    print(f"Dataset saved at {output_location} as {FILENAME_X} and {FILENAME_Y}")