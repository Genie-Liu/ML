import numpy as np

# 载入数据
def loadData():
    """
    test data
    """
    pass

# 定义sigmoid 函数
def sigmoid(z):
    """
    output the sigmoid function
    """

    s = 1/(1+np.exp(-z))
    return s

# 梯度下降算法
def gradDescent(inMatrix, classLabel):
    """
    Gradient Descent Algorithm to optimize weights
    """

    inMat = np.mat(inMatrix)
    y = np.mat(classLabel)
    # m个例子，n个特征
    m, n = inMat.shape
    # 初始化权重weight， 学习率alpha, 循环次数cycles
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
        # 链式求导得到inMat.T.dot(err)为Loss函数对weights的求导结果
        weights = weights - alpha*(inMat.T.dot(err))

    return weights

def predict(inputs, weights):
    output = np.dot(inputs, weights)


