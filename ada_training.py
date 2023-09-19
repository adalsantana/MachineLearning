import numpy as np 
from adaline import AdalineGD
from adaline import AdalineSGD
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
import pandas as pd

def standardize(x): 
        X_std = np.copy(x)
        X_std[:, 0] = (x[:, 0] - x[:,0].mean()) / x[:,0].std()
        X_std[:, 1] = (x[:, 1] - x[:,1].mean()) / x[:,1].std()



def plot_decision_regions(x, y, classifier, resolution=0.02): 
    # setup marker generator and color map 
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface 
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1 
    xx1, xx2 = np.meshgrid(np.arrange(x1_min, x1_max, resolution), 
                           np.arrange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx1, z, alpha=0.3 ,cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples 
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X=x[y == cl, 0], 
                    Y = x[y == cl, 1], 
                    alpha = 0.8, 
                    c = colors[idx], 
                    markers = markers[idx], 
                    label = cl, 
                    edgecolor = 'black')


dataFilePath = "/datasets/iris.data"
df = pd.read_csv(dataFilePath, header=None)

# select setosa and versicolor 
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length 
x = df.iloc[0:100, [0, 2]].values


############################################################
### plotting the epochs for two different learning rates ###
############################################################
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(10,4))

ada1 = AdalineGD(n_iter=10, eta=0.01).fit(x, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), 
           np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter=10, eta=0.001).fit(x, y)

ax[1].plot(range(1, len(ada2.cost_) + 1), 
           np.log10(ada1.cost_), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - learning rate 0.001')
plt.show()

"""
    The plots above should show that the resulting cost functions encountered two different types of problems. 
    The first chart shows what could happen if you choose a learning rate that is too large. Instead of minimizing the cost function, 
    the error becomes larger in every epoch because we overshoot the global minimum. 

    On the other hand, eta = 0.0001 is so small that the algorithm would require a large number of epochs to converge to the global cost minimum. 
"""
############################################################
###                     end                              ###
############################################################


# ----------------------------------------------------------------------------------- # 
# -------------------------- NEXT EXERCISE ------------------------------------------ #
# ----------------------------------------------------------------------------------- #


############################################################
### standardizing data and retraining adaline model      ###
############################################################
X_std = standardize(x)
ada3 = AdalineGD(n_iter=15, eta = 0.01)
ada3.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada3)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [cm]')
plt.ylabel(' petal length [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


plt.plot(range(1, len(ada3.cost_) + 1), ada3.cost_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel("Sum-squared-error")
plt.show()

"""
    The above plots should demonstrate convergence after being trained on the standardized features. Note that SSE remains zero 
    even after all the samples were correctly classified. 
"""
############################################################
###                     end                              ###
############################################################


# ----------------------------------------------------------------------------------- # 
# -------------------------- NEXT EXERCISE ------------------------------------------ #
# ----------------------------------------------------------------------------------- #


############################################################
###        Stochastic Gradient Descent Model             ###
############################################################
ada = AdalineSGD(n_iter = 15, eta = 0.01, random_state=1)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title("Adaline - Stochastic Gradient Descent")
plt.xlabel('sepal length [cm]')
plt.ylabel(' petal length [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()
