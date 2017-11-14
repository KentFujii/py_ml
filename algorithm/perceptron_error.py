import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import configparser
from perceptron import Perceptron


if __name__ == '__main__':
    inifile = configparser.SafeConfigParser()
    inifile.read('./config.ini')
    df = pd.read_csv(
        inifile.get('perceptron', 'iris_dataset_url'), header=None)
    print(df.tail())
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(
        range(1, len(ppn.errors_) + 1),
        ppn.errors_,
        marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassification')
    plt.show()
