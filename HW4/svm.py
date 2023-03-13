import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def plot_results(models, titles, X, y, plot_sv=False):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, len(titles))  # 1, len(list(models)))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    if len(titles) == 1:
        sub = [sub]
    else:
        sub = sub.flatten()
    for clf, title, ax in zip(models, titles, sub):
        # print(title)
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        if plot_sv:
            sv = clf.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], c='k', s=60)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()

def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

C_hard = 1000000.0  # SVM regularization parameter
C = 10
n = 100



# Data is labeled by a circle

radius = np.hstack([np.random.random(n), np.random.random(n) + 1.5])
angles = 2 * math.pi * np.random.random(2 * n)
X1 = (radius * np.cos(angles)).reshape((2 * n, 1))
X2 = (radius * np.sin(angles)).reshape((2 * n, 1))

X = np.concatenate([X1,X2],axis=1)
y = np.concatenate([np.ones((n,1)), -np.ones((n,1))], axis=0).reshape([-1])

linear_kernel = svm.SVC(kernel="linear", C=10)
homogeneous_2 = svm.SVC(kernel='poly', gamma='auto', degree=2, C=10, coef0=0)
homogeneous_3 = svm.SVC(kernel='poly', gamma='auto', degree=3, C=10, coef0=0)

models = []
models.append(linear_kernel.fit(X, y))
models.append(homogeneous_2.fit(X,y))
models.append(homogeneous_3.fit(X,y))
models = np.array(models)

plot_results(models=models, titles=['linear kernel', 'homogeneous_2', 'homogeneous_3'], X=X, y=y)

non_homogeneous_2 = svm.SVC(kernel='poly', gamma='auto', degree=2, C=10, coef0=1)
non_homogeneous_3 = svm.SVC(kernel='poly', gamma='auto', degree=3, C=10, coef0=1)

models = []
models.append(linear_kernel.fit(X, y))
models.append(non_homogeneous_2.fit(X,y))
models.append(non_homogeneous_3.fit(X,y))
models = np.array(models)

plot_results(models=models, titles=['linear kernel', 'non homogeneous_2', 'non homogeneous_3'], X=X, y=y)

p = [0.1, 0.9]
for i in range(len(y)):
    if y[i] == 1:
        y[i] = np.random.choice(a=[-1,1], p=p)

for i in [10, 20, 50, 100]:
    non_homogeneous_2 = svm.SVC(kernel='poly', gamma='auto', degree=2, C=10, coef0=1)
    RBF = svm.SVC(kernel='rbf', gamma=i)
    models = []
    models.append(non_homogeneous_2.fit(X, y))
    models.append(RBF.fit(X,y))
    models = np.array(models)

    plot_results(models=models, titles=['non homogeneous_2', 'RBF'], X=X, y=y)
