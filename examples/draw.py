import pandas as pd
import matplotlib.pyplot as plt

# Copy the 'Epsilon' column from file1 to file2 (first 100 rows only)
def copy_epsilon_column(file1, file2):
    df1 = pd.read_csv(file1).head(100)
    df2 = pd.read_csv(file2).head(100)
    df2['Epsilon'] = df1['Epsilon'].values[:len(df2)]
    df2.to_csv(file2, index=False)

# Plot training/test accuracy and epsilon values for the Adult dataset
def plot_comparisons_adult(file1, file2, file3, file4,file5):
    df1 = pd.read_csv(file1).head(30)
    df2 = pd.read_csv(file2).head(30)
    df3 = pd.read_csv(file3).head(30)
    df4 = pd.read_csv(file4).head(30)
    df5 = pd.read_csv(file5).head(30)

    # Training Accuracy Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df1['Epoch'], df1['Training Accuracy'] * 100, label='proposed product noise', color='red', linewidth=2)
    plt.plot(df2['Epoch'], df2['Training Accuracy'] * 100, label='traditional Gaussian noise', color='blue', linestyle='dashed', linewidth=2)
    plt.plot(df3['Epoch'], df3['Training Accuracy'] * 100, label='non-private baseline', color='black', linestyle='dashdot', linewidth=2)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Training Accuracy (%)', fontsize=20)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig("./adult/adult_train_accuracy.png", bbox_inches='tight')
    plt.show()

    # Test Accuracy Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df1['Epoch'], df1['Test Accuracy'] * 100, label='proposed product noise', color='red', linewidth=2)
    plt.plot(df2['Epoch'], df2['Test Accuracy'] * 100, label='traditional Gaussian noise', color='blue', linestyle='dashed', linewidth=2)
    plt.plot(df3['Epoch'], df3['Test Accuracy'] * 100, label='non-private baseline', color='black', linestyle='dashdot', linewidth=2)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Test Accuracy (%)', fontsize=20)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig("./adult/adult_test_accuracy.png", bbox_inches='tight')
    plt.show()

    # Epsilon (Privacy Budget) Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df1['Epoch'], df1['Epsilon'], label='product noise via distribution-independent composition', color='red', linewidth=2)
    plt.plot(df2['Epoch'], df2['Epsilon'], label='traditional Gaussian noise via MA', color='blue', linestyle='dashed', linewidth=2)
    plt.plot(df4['Epoch'], df4['Epsilon'], label='traditional Gaussian noise via CLT', color='black', linestyle='dashdot', linewidth=2)
    plt.plot(df5['Epoch'], df5['Epsilon'], label='traditional Gaussian noise via PRV', color='green', linestyle=(0, (3, 1, 1, 1)), linewidth=2)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Epsilon', fontsize=20)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig("./adult/adult_epsilon.png", bbox_inches='tight')
    plt.show()

# Plot for the IMDB dataset
def plot_comparisons_imdb(file1, file2, file3, file4, file5):
    df1 = pd.read_csv(file1).head(60)
    df2 = pd.read_csv(file2).head(60)
    df3 = pd.read_csv(file3).head(60)
    df4 = pd.read_csv(file4).head(60)
    df5 = pd.read_csv(file5).head(60)

    # Training Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(df1['Epoch'], df1['Train Accuracy'] * 100, label='proposed product noise', color='red', linewidth=2)
    plt.plot(df2['Epoch'], df2['Train Accuracy'] * 100, label='traditional Gaussian noise', color='blue', linestyle='dashed', linewidth=2)
    plt.plot(df3['Epoch'], df3['Train Accuracy'] * 100, label='non-private baseline', color='black', linestyle='dashdot', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig("./imdb/imdb_train_accuracy.png", bbox_inches='tight')
    plt.show()

    # Test Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(df1['Epoch'], df1['Test Accuracy'] * 100, label='proposed product noise', color='red', linewidth=2)
    plt.plot(df2['Epoch'], df2['Test Accuracy'] * 100, label='traditional Gaussian noise', color='blue', linestyle='dashed', linewidth=2)
    plt.plot(df3['Epoch'], df3['Test Accuracy'] * 100, label='non-private baseline', color='black', linestyle='dashdot', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig("./imdb/imdb_test_accuracy.png", bbox_inches='tight')
    plt.show()

    # Epsilon Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df1['Epoch'], df1['Epsilon'], label='product noise via distribution-independent composition', color='red', linewidth=2)
    plt.plot(df2['Epoch'], df2['Epsilon'], label='traditional Gaussian noise via MA', color='blue', linestyle='dashed', linewidth=2)
    plt.plot(df4['Epoch'], df4['Epsilon'], label='traditional Gaussian noise via CLT', color='black', linestyle='dashdot', linewidth=2)
    plt.plot(df5['Epoch'], df5['Epsilon'], label='traditional Gaussian noise via PRV', color='green', linestyle=(0, (3, 1, 1, 1)), linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.grid(True)
    plt.savefig("./imdb/imdb_epsilon.png", bbox_inches='tight')
    plt.show()

# Plot for the MNIST dataset
def plot_comparisons_mnist(file1, file2, file3, file4, file5):
    df1 = pd.read_csv(file1).head(50)
    df2 = pd.read_csv(file2).head(50)
    df3 = pd.read_csv(file3).head(50)
    df4 = pd.read_csv(file4).head(50)
    df5 = pd.read_csv(file5).head(50)

    # Ensure necessary columns are present
    required_columns = ['Epoch', 'Train Accuracy', 'Test Accuracy', 'Epsilon']
    for col in required_columns:
        if col not in df1.columns or col not in df2.columns:
            raise ValueError("Column missing in dataset")

    # Train Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(df1['Epoch'], df1['Train Accuracy'] * 100, label='proposed product noise', color='red', linewidth=2)
    plt.plot(df2['Epoch'], df2['Train Accuracy'] * 100, label='traditional Gaussian noise', color='blue', linestyle='dashed', linewidth=2)
    plt.plot(df3['Epoch'], df3['Train Accuracy'] * 100, label='non-private baseline', color='black', linestyle='dashdot', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig("./mnist/mnist_train_accuracy.png", bbox_inches='tight')
    plt.show()

    # Test Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(df1['Epoch'], df1['Test Accuracy'] * 100, label='proposed product noise', color='red', linewidth=2)
    plt.plot(df2['Epoch'], df2['Test Accuracy'] * 100, label='traditional Gaussian noise', color='blue', linestyle='dashed', linewidth=2)
    plt.plot(df3['Epoch'], df3['Test Accuracy'] * 100, label='non-private baseline', color='black', linestyle='dashdot', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy (%)')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    plt.legend()
    plt.grid(True)
    plt.savefig("./mnist/mnist_test_accuracy.png", bbox_inches='tight')
    plt.show()

    # Epsilon Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df1['Epoch'], df1['Epsilon'], label='product noise via distribution-independent composition', color='red', linewidth=2)
    plt.plot(df2['Epoch'], df2['Epsilon'], label='traditional Gaussian noise via MA', color='blue', linestyle='dashed', linewidth=2)
    plt.plot(df4['Epoch'], df4['Epsilon'], label='traditional Gaussian noise via CLT', color='black', linestyle='dashdot', linewidth=2)
    plt.plot(df5['Epoch'], df5['Epsilon'], label='traditional Gaussian noise via PRV', color='green', linestyle=(0, (3, 1, 1, 1)), linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.grid(True)
    plt.savefig("mnist/mnist_epsilon.png", bbox_inches='tight')
    plt.show()

# Plot for MovieLens (uses RMSE instead of accuracy)
def plot_comparisons_movielens(file1, file2, file3, file4, file5):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df4 = pd.read_csv(file4)
    df5 = pd.read_csv(file5)

    # Training RMSE
    plt.figure(figsize=(8, 6))
    plt.plot(df1['Epoch'], df1['RMSE'], label='proposed product noise', color='red', linewidth=2)
    plt.plot(df2['Epoch'], df2['RMSE'], label='traditional Gaussian noise', color='blue', linestyle='dashed', linewidth=2)
    plt.plot(df3['Epoch'], df3['RMSE'], label='non-private baseline', color='black', linestyle='dashdot', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Training RMSE')
    plt.legend()
    plt.grid(True)
    plt.savefig("./movielens/movielens_RMSE.png", bbox_inches='tight')
    plt.show()

    # Test RMSE
    plt.figure(figsize=(8, 6))
    plt.plot(df1['Epoch'], df1['Test RMSE'], label='proposed product noise', color='red', linewidth=2)
    plt.plot(df2['Epoch'], df2['Test RMSE'], label='traditional Gaussian noise', color='blue', linestyle='dashed', linewidth=2)
    plt.plot(df3['Epoch'], df3['Test RMSE'], label='non-private baseline', color='black', linestyle='dashdot', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Test RMSE')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    plt.legend()
    plt.grid(True)
    plt.savefig("./movielens/movielens_test_RMSE.png", bbox_inches='tight')
    plt.show()

    # Epsilon Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df1['Epoch'], df1['Epsilon'], label='product noise via distribution-independent composition', color='red', linewidth=2)
    plt.plot(df2['Epoch'], df2['Epsilon'], label='traditional Gaussian noise via MA', color='blue', linestyle='dashed', linewidth=2)
    plt.plot(df4['Epoch'], df4['Epsilon'], label='traditional Gaussian noise via CLT', color='black', linestyle='dashdot', linewidth=2)
    plt.plot(df5['Epoch'], df5['Epsilon'], label='traditional Gaussian noise via PRV', color='green', linestyle=(0, (3, 1, 1, 1)), linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.grid(True)
    plt.savefig("./movielens/movielens_epsilon.png", bbox_inches='tight')
    plt.show()

# Plot for CIFAR-10 dataset
def plot_comparisons_cifar10(file1, file2, file3, file4, file5):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df4 = pd.read_csv(file4)
    df5 = pd.read_csv(file5)

    required_columns = ['Epoch', 'Train Accuracy', 'Test Accuracy', 'Epsilon']
    for col in required_columns:
        if col not in df1.columns or col not in df2.columns:
            raise ValueError("Column missing in dataset")

    # Train Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(df2['Epoch'], df2['Train Accuracy'] * 100, label='proposed product noise', color='red', linewidth=2)
    plt.plot(df1['Epoch'], df1['Train Accuracy'] * 100, label='traditional Gaussian noise', color='blue', linestyle='dashed', linewidth=2)
    plt.plot(df3['Epoch'], df3['Train Accuracy'] * 100, label='non-private baseline', color='black', linestyle='dashdot', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig("./cifar10/cifar10_train_accuracy.png", bbox_inches='tight')
    plt.show()

    # Test Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(df2['Epoch'], df2['Test Accuracy'] * 100, label='proposed product noise', color='red', linewidth=2)
    plt.plot(df1['Epoch'], df1['Test Accuracy'] * 100, label='traditional Gaussian noise', color='blue', linestyle='dashed', linewidth=2)
    plt.plot(df3['Epoch'], df3['Test Accuracy'] * 100, label='non-private baseline', color='black', linestyle='dashdot', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig("./cifar10/cifar10_test_accuracy.png", bbox_inches='tight')
    plt.show()

    # Epsilon Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df2['Epoch'], df2['Epsilon'], label='product noise via distribution-independent composition', color='red', linewidth=2)
    plt.plot(df1['Epoch'], df1['Epsilon'], label='traditional Gaussian noise via MA', color='blue', linestyle='dashed', linewidth=2)
    plt.plot(df4['Epoch'], df4['Epsilon'], label='traditional Gaussian noise via CLT', color='black', linestyle='dashdot', linewidth=2)
    plt.plot(df5['Epoch'], df5['Epsilon'], label='traditional Gaussian noise via PRV', color='green', linestyle=(0, (3, 1, 1, 1)), linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.grid(True)
    plt.savefig("./cifar10/cifar10_epsilon.png", bbox_inches='tight')
    plt.show()

# Entry point for running the script
if __name__ == '__main__':
    plot_comparisons_cifar10(
                            './cifar10/cifar10_their.csv',
                            './cifar10/cifar10_us.csv',
                            './cifar10/cifar10_non-private.csv',
                            './cifar10/cifar10_gdp_epsilon.csv',
                            './cifar10/cifar10_prv_epsilon.csv',

    )

    plot_comparisons_adult("./adult/adult_us.csv",
                           "./adult/adult_their.csv",
                           "./adult/adult_baseline.csv",
                           "./adult/gdp_epsilon.csv",
                           "./adult/prv_epsilon.csv")

    plot_comparisons_imdb('./imdb/imdb_us.csv',
                          './imdb/imdb_their.csv',
                          './imdb/IMDB_0.02_0.56_1.0_512_60_baseline.csv',
                          './imdb/IMDB_gdp_60_epsilon.csv',
                          './imdb/IMDB_prv_60_epsilon.csv')

    plot_comparisons_mnist('mnist/mnist_our.csv',
                           'mnist/mnist_their.csv',
                           'mnist/mnist_base.csv',
                           'mnist/mnist_gdp_Epsilon.csv',
                           'mnist/mnist_prv_Epsilon.csv'

                           )
    #
    plot_comparisons_movielens('./movielens/MoveLens_our.csv',
                               './movielens/movielens_their.csv',
                               './movielens/movielens_baseline_0.01.csv',
                               './movielens/movielens_gdp_epsilon.csv',
                               './movielens/movielens_prv_epsilon.csv'
                               )