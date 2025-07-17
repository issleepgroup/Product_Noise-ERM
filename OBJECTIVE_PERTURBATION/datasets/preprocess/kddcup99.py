import os
import numpy as np
import sklearn.datasets
from sklearn import preprocessing
from utils.utils_preprocessing import convert_to_binary, normalize_rows, format_output

FILENAME_X = 'kddcup99_processed_x.npy'
FILENAME_Y = 'kddcup99_processed_y.npy'

def preprocess(cache_location, output_location):
    np.random.seed(10000019)
    os.environ['SCIKIT_LEARN_DATA'] = cache_location

    # Load 10% subset of KDDCup99 dataset
    subset = sklearn.datasets.fetch_kddcup99(percent10=True)

    # Randomly sample 70,000 examples
    indices = np.random.randint(subset['data'].shape[0], size=70000)
    subset['data'] = subset['data'][indices, :]
    subset['target'] = subset['target'][indices]

    symbolic_cols = []
    continuous_cols = []
    le = preprocessing.LabelEncoder()

    for i in range(subset['data'].shape[1]):
        col = subset['data'][:, i]

        if isinstance(col[0], bytes):
            # Process categorical feature by encoding to numeric values
            numeric = le.fit_transform(col).astype(np.float32)
            symbolic_cols.append(numeric)
        else:
            # Process continuous feature
            continuous_cols.append(col.astype(np.float32))

    # One-hot encode the categorical features
    symbolic_cols = convert_to_binary(symbolic_cols)

    # Concatenate categorical and continuous features
    combined_data = np.column_stack(symbolic_cols + continuous_cols).astype(np.float32)

    # Format and encode the labels
    labels = format_output(subset['target']).astype(np.float32)

    # Combine features and labels, then shuffle the full dataset
    all_data = np.column_stack([combined_data, labels])
    np.random.shuffle(all_data)

    print(all_data[:, -1])
    print(len(all_data))
    print(len(all_data[0]))

    # Save processed feature and label arrays
    np.save(os.path.join(output_location, FILENAME_X), all_data[:, :-1].astype(np.float32))
    np.save(os.path.join(output_location, FILENAME_Y), all_data[:, -1].astype(np.float32))
