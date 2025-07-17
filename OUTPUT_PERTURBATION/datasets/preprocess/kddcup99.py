# import os
# import numpy as np
# import sklearn.datasets
# from sklearn import preprocessing
# from utils.utils_preprocessing import convert_to_binary, normalize_rows, format_output
#
#
#
# FILENAME_X = 'kddcup99_processed_x.npy'
# FILENAME_Y = 'kddcup99_processed_y.npy'
#
#
# """
# Preprocess the kddcup99 dataset
#
# This process is discussed in section 7.1 of CMS11:
# \"For the KDDCup99 data set, the instances were preprocessed by converting
# each categorial attribute to a binary vector. Each column was normalized
# to ensure that the maximum value is 1, and finally, each row was normalized,
# to ensure that the norm of any example is at most 1. After preprocessing,
# each example was represented by a 119-dimensional vector, of norm at most 1.\"
#
# My code to preprocess the categorial features was somewhat based upon:
# https://biggyani.blogspot.com/2014/08/using-onehot-with-categorical.html
#
# The preprecessed data gets saved to a file hardcoded into this script.
# """
#
#
# def preprocess(cache_location, output_location):
#
#     np.random.seed(10000019)
#     os.environ['SCIKIT_LEARN_DATA'] = cache_location
#
#     subset = sklearn.datasets.fetch_kddcup99(percent10=True)
#
#     # Randomly select 70,000 elements
#     # Based on https://stackoverflow.com/a/14262743/859277
#
#     indices = np.random.randint(subset['data'].shape[0], size=70000)
#
#     subset['data'] = subset['data'][indices, :]
#     subset['target'] = subset['target'][indices]
#
#     symbolic_cols = []
#     continuous_cols = []
#     le = preprocessing.LabelEncoder()
#
#     for i in range(subset['data'].shape[1]):
#         col = subset['data'][:, i]
#
#         if type(col[0]) == bytes:
#             numeric = le.fit_transform(col)
#             symbolic_cols.append(numeric)
#         else:
#             continuous_cols.append(col)
#
#     symbolic_cols = convert_to_binary(symbolic_cols)
#
#     combined_data = np.column_stack(symbolic_cols + continuous_cols)
#     final_data = combined_data
#
#     all_data = np.column_stack([final_data, format_output(subset['target'])])
#     np.random.shuffle(all_data)
#
#
#     print(all_data[:, -1])
#     print(len(all_data))
#     print(len(all_data[0]))
#     np.save(os.path.join(output_location, FILENAME_X), all_data[:, :-1])
#     np.save(os.path.join(output_location, FILENAME_Y), all_data[:, -1])
#
#


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

    subset = sklearn.datasets.fetch_kddcup99(percent10=True)

    # 随机选择 70,000 个样本
    indices = np.random.randint(subset['data'].shape[0], size=70000)
    subset['data'] = subset['data'][indices, :]
    subset['target'] = subset['target'][indices]

    symbolic_cols = []
    continuous_cols = []
    le = preprocessing.LabelEncoder()

    for i in range(subset['data'].shape[1]):
        col = subset['data'][:, i]

        if isinstance(col[0], bytes):  # 处理类别型数据
            numeric = le.fit_transform(col).astype(np.float32)  # 转换为数值型
            symbolic_cols.append(numeric)
        else:  # 处理连续型数据
            continuous_cols.append(col.astype(np.float32))  # 转换为 float32

    symbolic_cols = convert_to_binary(symbolic_cols)  # One-hot 编码

    # 合并数值型和类别型特征
    combined_data = np.column_stack(symbolic_cols + continuous_cols).astype(np.float32)

    # 处理标签数据
    labels = format_output(subset['target']).astype(np.float32)

    # 组合所有数据
    all_data = np.column_stack([combined_data, labels])
    np.random.shuffle(all_data)

    print(all_data[:, -1])
    print(len(all_data))
    print(len(all_data[0]))

    # **转换数据格式后再保存**
    np.save(os.path.join(output_location, FILENAME_X), all_data[:, :-1].astype(np.float32))
    np.save(os.path.join(output_location, FILENAME_Y), all_data[:, -1].astype(np.float32))

