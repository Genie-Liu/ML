import numpy as np
import random


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append(lineArr[:-1])
            labelMat.append(lineArr[-1])

    return dataMat, labelMat


def selectJrand(i, m):
    """return the j between (0, m) but not i"""
    j = i
    while j == i:
        j = int(random.uniform(0, m))

    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L

    return aj


def smoSimple(dataMat, classLables, C, toler, maxIter):
    """simple smo algorithm"""

    dataMatrix = np.mat(dataMat)
    labelMat = np.mat(classLables).transpose()
    b = 0
    m, n = dataMatrix.shape
    alphas = np.mat(np.zeros((m, n)))
    iter = 0
    while(iter < maxIter):
        alphaPairChanged = 0
        for i in range(m):
            fXi = 