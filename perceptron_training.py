import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from perceptron import Perceptron
# NOTE: LEFT OFF WITH A MARKER GETTING AN INVALID VALUE ERROR
def plot_decision_regions(X, y, classifier, resolution=0.02): 
    # setup marker generator and color map 
    markers = ('s', 'x', 'o', '^', 'v')
    # markers = enumerate(markers)
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # colors = np.array(colors)
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface 
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx1, z, alpha=0.3 ,cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples 
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y = X[y == cl, 1], 
                    alpha = 0.8, 
                    c = colors[idx], 
                    marker = markers[idx], 
                    label = cl, 
                    edgecolor = 'black')

def perceptron_trainer():
    # dataFilePath = "/datasets/iris.data"
    # df = pd.read_csv(dataFilePath, header=None)
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # select setosa and versicolor 
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    # extract sepal length and petal length 
    x = df.iloc[0:100, [0, 2]].values

    # plot data 
    # plt.scatter(x[:50, 0], x[:50, 1], colors='red', marker='o', label='setosa')
    # plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='versicolor')
    # plt.xlabel('sepal length [cm]')
    # plt.ylabel(' petal length [cm]')
    # plt.legend(loc='upper left')
    # plt.show()

    # the above plot should pull up a chart that looks like a linear classifier would suffice to separate the data 
    # Now lets train the perceptron algorithm on the iris data subset 

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(x, y)
    # plt.plot(range(1, len(ppn.errors_) + 1), 
    #         ppn.errors_, marker='o')
    # plt.xlabel('Epochs')
    # plt.ylabel("Number of updates")
    # plt.show()

    plot_decision_regions(x, y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel(' petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

def main():
    perceptron_trainer()

if __name__ == '__main__': 
    main()