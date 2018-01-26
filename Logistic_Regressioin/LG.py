import numpy as np


def loadData():
    """
    test data
    """
    pass


def sigmoid(z):
    """
    output the sigmoid function
    """

    s = 1/(1+np.exp(-z))
    return s


def gradDescent(inMatrix, classLabel):
    """
    Gradient Descent Algorithm to optimize weights
    """

    inMat = np.mat(inMatrix)
    y = np.mat(classLabel)
    m, n = inMat.shape
    weights = np.ones((n, 1))
    alpha = 0.01
    cycles = 500

    for i in range(cycles):
        a = sigmoid(np.dot(inMat, weights))
        err = y - a
        # update weights
        # Need to be proofed this formular is correct.
        # L(y,a) = -[y*ln(a) + (1-y)*ln(1-a)]
        # L(y,a) derivative to weights is x.T*(y-a)
        weights = weights - alpha*(inMat.T.dot(err))

    return weights
