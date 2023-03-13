#################################
# Your name: Roei Ben Zion
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import math
import matplotlib.pyplot as plt
import scipy.special
"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    n = len(labels) 
    wt = np.zeros(shape=(len(data[0])))
    eta_t = eta_0
    for t in range(1,T):
        eta_t = eta_0/t
        ind = np.random.randint(low=1, high=n)
        xi = np.array(data[ind])
        yi = (labels[ind])
        temp = yi*np.dot(xi, wt)
        if(np.isnan(temp)):
            print(t)
            break
        if temp < 1:
            a = (1-eta_t)*wt
            b = (eta_t*C*yi)*xi
            wt = np.add(a,b)
        else:
            wt = (1-eta_t)*wt
    return wt

def SGD_log(data, labels, eta_0, T, plot_norm):
    """
    Implements SGD for log loss.
    """
    if plot_norm == True:
        g = [0]*(T-1)  
        i = 0
    n = len(labels) 
    wt = np.zeros(shape=(len(data[0])))
    for t in range(1,T):
        ind = np.random.randint(low=1, high=n)
        xi = np.array(data[ind])
        yi = (labels[ind])
        gradient = calculate_log_gradient(x_ind=xi, y_ind=yi, wt=wt)
        wt = np.add(wt, -(eta_0)*gradient)
        if plot_norm == True:
            g[i] = np.linalg.norm(wt)
            i+=1
    if plot_norm:
        plt.xlabel("iteration")
        plt.ylabel("norm")
        plt.xlim(0, T-1)
        plt.plot(np.arange(0,T-1), g)
        plt.show()
    return wt

#################################
def calculate_log_gradient(x_ind:list, y_ind:int, wt:list)->list:
    return (-y_ind*x_ind)*scipy.special.expit(-y_ind*np.dot(x_ind,wt))
def accuracy(classifier: list, data: list, labels:list)->float:
    n = len(data)
    error_count = 0
    for i in range(n):
        example = data[i]
        temp = labels[i]*np.dot(classifier,example)
        if temp < 0:
            error_count+=1
    error_count /= n
    return 1-error_count

def find_best_eta(train_data:list, train_labels:list, validation_data:list, validation_labels:list, hinge:bool)->float:
    eta_0 = float("inf")
    score = float("-inf")
    g = [0]*9
    i = 0
    etas = [pow(10,i) for i in range(-5, 4)]
    for eta in etas:
        ave = 0.0
        for k in range(10):
            if hinge:
                classifier = SGD_hinge(data = train_data, labels=train_labels,C=1, eta_0= eta, T=1000)
            else:
                classifier = SGD_log(data = train_data, labels=train_labels, eta_0= eta, T=1000, plot_norm=False)
            ave += accuracy(classifier=classifier,data=validation_data,labels=validation_labels)
        ave /=10
        g[i] = ave
        if(g[i] > score):
            eta_0 = eta
            score = g[i]
        i = i+1
    plt.xlabel("10 to the power")
    plt.ylabel("accuracy")
    plt.plot(np.arange(-5,4), g)
    plt.show()
    return eta_0

def find_best_C(eta_0:float,train_data:list, train_labels:list, validation_data:list, validation_labels:list)->float:
    C_opt = float("-inf")
    score = float("-inf")
    g = [0]*9
    i = 0
    constants = [pow(10, i) for i in range(-5, 4)]
    for constant in constants:
        ave = 0.0
        for k in range(10):
            classifier = SGD_hinge(data = train_data, labels=train_labels,C=constant, eta_0= eta_0, T=1000)
            ave += accuracy(classifier=classifier,data=validation_data,labels=validation_labels)
        ave /=10
        g[i] = ave
        if(g[i] > score):
            C_opt = constant
            score = g[i]
        i = i+1
    plt.xlabel("10 to the power")
    plt.ylabel("accuracy")
    plt.plot(np.arange(-5,4), g)
    plt.show()
    return C_opt

def train_classifier(eta_0:float, C:float, train_data:list, train_labels:list, hinge:bool)->list:
    if hinge:
        classifier = SGD_hinge(data = train_data, labels=train_labels,C=C, eta_0= eta_0, T=20000)
    else:
        classifier = SGD_log(data = train_data, labels=train_labels, eta_0= eta_0, T=20000, plot_norm=False)
    return classifier
def check_accuracy(classifier:list, test_data:list, test_labels:list)-> float:
    return accuracy(classifier=classifier, data=test_data, labels=test_labels)


train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()

#Hinge
eta_0 = find_best_eta(train_data=train_data, train_labels=train_labels, validation_data=validation_data, validation_labels=validation_labels, hinge=True)
C = find_best_C(eta_0=eta_0, train_data=train_data, train_labels=train_labels, validation_data=validation_data, validation_labels=validation_labels)

classifier = train_classifier(eta_0=eta_0, C=C, train_data=train_data, train_labels=train_labels, hinge=True)
classifier = np.array(classifier)
plt.imshow(classifier.reshape((28, 28)), interpolation="nearest")
plt.show()
classifier = train_classifier(eta_0=eta_0, C=C, train_data=train_data, train_labels=train_labels, hinge=True)
print("Hinge SGD accuracy - ", check_accuracy(classifier=classifier, test_data=test_data, test_labels=test_labels))

#Log
eta_0 = find_best_eta(train_data=train_data, train_labels=train_labels, validation_data=validation_data, validation_labels=validation_labels, hinge=False)
classifier = train_classifier(eta_0=eta_0, C=0, train_data=train_data, train_labels=train_labels, hinge=False)
plt.imshow(classifier.reshape((28, 28)), interpolation="nearest")
plt.show()
print("Log SGD accuracy - ",check_accuracy(classifier=classifier, test_data=test_data, test_labels=test_labels))
SGD_log(data=train_data, labels=train_labels, eta_0=eta_0, T=20000, plot_norm=True)
