import numpy as np
import operator


def CreateDataSet():
    """
    return inputs data and the correspond labeled data
    """

    X = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    Y = ['A', 'A', 'B', 'B']
    return X, Y


def kNN(x_test, X, Y, dist, k=3):
    """
    Keywork arguments:
    x_test-- the input to be predic
    X     -- sample data
    Y     -- labeled data
    dis   -- distance func
    k     -- algorithm hyperparameter
    """

    distance = dist(x_test, X)
    sortedIndex = distance.argsort()
    count = {}
    for i in range(k):
        label = Y[sortedIndex[i]]
        count[label] = count.get(label, 0) + 1

    sortedCount = sorted(count.items(), key=operator.itemgetter(1), reverse=True)

    return sortedCount[0][0]


def dist(v1, v2):
    """
    calculate the distance from v1 to v2
    """

    return np.sqrt(np.sum((v1-v2)**2, axis=1))


def normalize(X):
    """
    normalize the sample data using feature scaling
    """

    minCol = np.min(X, axis=0)
    maxCol = np.max(X, axis=0)

    return (X - minCol)/(maxCol - minCol)
