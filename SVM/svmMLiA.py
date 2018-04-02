import numpy as np
import random


def loadDataSet(file_name):
    dataMat = []
    labelMat = []
    with open(file_name) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append(lineArr[:-1])
            labelMat.append(lineArr[-1])

    return dataMat, labelMat

    # X_train = []
    # y_train = []
    # with open(file_name, "r") as f:
    #     for line in f:
    #         data = line.split(',')
    #         x = data[:-1]
    #         # x.append(1)
    #         y = data[-1]
    #         # 针对鸢尾花进行分类
    #         if y.strip() == "Iris-setosa":
    #             y = [1]
    #         else:
    #             y = [-1]
    #         X_train.append(x)
    #         y_train.append(y)

    # return X_train, y_train


def selectJrand(i, m):
    """
    返回介于0到m，且不为i的整数值
    """
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


def smoSimple(dataMat, classLables, C, toler, maxIter=500):
    """
    简单的SMO算法实现，不考虑高效的选择alpha
    """

    # 进行数据预处理
    dataMatrix = np.mat(dataMat, dtype='float64')
    labelMat = np.mat(classLables, dtype='float64').T
    # labelMat = np.mat(classLables, dtype='float64')

    # 初始化参数
    b = 0
    m, n = dataMatrix.shape
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while(iter < maxIter):
        alphaPairChanged = 0
        for i in range(m):
            # 根据现有的alpha对第i个例子进行预测
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            # 模型与实际的误差
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or \
                ((labelMat[i]*Ei > toler) and alphas[i] > 0):
                # 违反KKT条件
                # 选择下一个要优化的alpha
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMat).T * \
                    (dataMatrix*dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 根据i,j对应的输出是否异号来决定边界
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                # 二次求导结果eta
                eta = 2.0*dataMatrix[i, :]*dataMatrix[j, :].T - \
                    dataMatrix[i, :]*dataMatrix[i, :].T - \
                    dataMatrix[j, :]*dataMatrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                # 更新第二个alpha
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                # 更新第一个alpha
                alphas[i] += labelMat[j]*labelMat[i] * \
                    (alphaJold - alphas[j])

                # 计算对应的b1和b2
                b1 = b - Ei - labelMat[i]*(alphas[i] - alphaIold) * \
                    dataMatrix[i, :]*dataMatrix[i, :].T - \
                    labelMat[j]*(alphas[j] - alphaJold) * \
                    dataMatrix[i, :]*dataMatrix[j, :].T

                b2 = b - Ej - labelMat[i]*(alphas[i] - alphaIold) * \
                    dataMatrix[i, :]*dataMatrix[j, :].T - \
                    labelMat[j]*(alphas[j] - alphaJold) * \
                    dataMatrix[j, :]*dataMatrix[j, :].T
                # 根据alpha是否在界内来决定b值
                if 0 < alphas[i] and C > alphas[i]:
                    b = b1
                elif 0 < alphas[j] and C > alphas[j]:
                    b = b2
                else:
                    b = (b1+b2)/2.0
                alphaPairChanged += 1
                print("iter: %d i:%d, pairs changed %d" % \
                    (iter, i, alphaPairChanged))
        if alphaPairChanged == 0:
            iter += 1
        else:
            iter = 0
    return b, alphas







