import numpy as np

# 载入数据, 默认为鸢尾花的数据"iris.data"
def loadData(file_name="iris.data"):
    """
    test data
    """
    X_train = []
    y_train = []
    with open(file_name, "r") as f:
        for line in f:
            data = line.split(',')
            x = data[:-1]
            x.append(1)
            y = data[-1]
            # 针对鸢尾花进行分类
            if y.strip() == "Iris-setosa":
                y = [1]
            else:
                y = [0]
            X_train.append(x)
            y_train.append(y)

    return X_train, y_train

# 定义sigmoid 函数
def sigmoid(z):
    """
    output the sigmoid function
    """
    s = 1/(1+np.exp(-z))
    return s

# 梯度下降算法
def gradDescentOptimize(inMatrix, classLabel):
    """
    Gradient Descent Algorithm to optimize weights
    """
    # 对输入的list类型进行numpy数据类型处理
    inMat = np.mat(inMatrix, dtype='float64')
    y = np.mat(classLabel, dtype='float64')
    # m个例子，n个特征
    m, n = inMat.shape
    # 初始化权重weight， 学习率alpha, 循环次数cycles
    weights = np.ones((n, 1), dtype='float64')
    alpha = 0.01
    cycles = 100

    for i in range(cycles):
        a = sigmoid(np.dot(inMat, weights))
        err = y - a
        # 链式求导得到inMat.T.dot(err)为Loss函数对weights的求导结果
        # 关于链式求导的过程可参考https://blog.csdn.net/Mophistoliu/article/details/79689840
        weights = weights + alpha*(inMat.T.dot(err))

    return weights

def predict(inputs, weights):
    # 针对输入及训练出来的权重参数进行预测
    inputs = np.mat(inputs, dtype='float64')
    output = sigmoid(np.dot(inputs, weights))
    return output

def test(X_test, y_test, weights):
    # 对测试数据使用训练好的权重进行预测，返回预测正确率
    X_test = np.mat(X_test, dtype='float64')
    y_test = np.mat(y_test, dtype='float64')
    predict = sigmoid(np.dot(X_test, weights))
    predict = (predict > 0.5) == (y_test == 1)
    accurate = np.average(predict)
    return accurate


