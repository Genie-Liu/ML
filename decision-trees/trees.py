# -*- coding: utf-8 -*-
import numpy as np
import operator


def shannonEntropy(dataSet):
    """
    calculate Shannon Entropy
    dataSet -- input X and label Y
    """

    Y = [v[-1] for v in dataSet]
    total = len(Y)
    shannonEnt = 0.0
    labelCount = {}

    for label in Y:
        labelCount[label] = labelCount.get(label, 0) + 1

    for key in labelCount.keys():
        prob = labelCount[key]/total
        shannonEnt -= np.log2(prob)*prob

    return shannonEnt


def splitData(dataSet, axis, value):
    retMatrix = []
    for v in dataSet:
        if v[axis] == value:
            splitVec = v[:axis]
            splitVec.extend(v[axis+1:])
            retMatrix.append(splitVec)
    return retMatrix


def chooseBestFeatToSplit(dataSet):
    """
    choose the best feature to be splited
    """

    numFeat = len(dataSet[0])-1
    entropyGain = 0.0
    bestFeat = -1
    baseEntropy = shannonEntropy(dataSet)

    for i in range(numFeat):
        featValues = [v[i] for v in dataSet]
        featValueSet = set(featValues)
        newEntropy = 0.0
        for value in featValueSet:
            splitedMatrix = splitData(dataSet, i, value)
            prob = len(splitedMatrix)/len(dataSet)
            newEntropy += prob*shannonEntropy(splitedMatrix)
        splitEntGain = baseEntropy - newEntropy
        if(splitEntGain > entropyGain):
            entropyGain = splitEntGain
            bestFeat = i

    return bestFeat


def majorCount(classList):
    """
    find out the most class label
    """
    classCount = {}
    for cls in classList:
        classCount[cls] = classCount.get(cls, 0) + 1
    classCount = sorted(classCount.items(), key=operator.itemgetter(1),
                        reverse=True)

    return classCount[0][0]


def createTree(dataSet, labels):
    """
    create the decision tree
    """
    featLabel = labels.copy()
    classList = [observation[-1] for observation in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorCount(classList)
    bestFeat = chooseBestFeatToSplit(dataSet)
    bestFeatLabel = featLabel[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(featLabel[bestFeat])
    featValue = [value[bestFeat] for value in dataSet]
    uniFeatValue = set(featValue)
    for value in uniFeatValue:
        subLabel = featLabel[:]
        myTree[bestFeatLabel][value] = createTree(splitData(dataSet, bestFeat, value), subLabel)

    return myTree


def CreateDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    label = ['no surfacing', 'flipper']
    return dataSet, label


def classify(myTree, featLabels, testVec):
    """
    classify the testVec with provided decision tree
    """
    initLabel = list(myTree.keys())[0]
    initDict = myTree[initLabel]
    pos = featLabels.index(initLabel)
    for key in initDict.keys():
        if testVec[pos] == key:
            if type(initDict[key]).__name__ == 'dict':
                classLabel = classify(initDict[key], featLabels, testVec)
            else:
                classLabel = initDict[key]
    return classLabel


def storeTree(myTree, filename):
    """
    store the computered decision tree into a file
    """
    import pickle
    with open(filename, 'wb') as fw:
        pickle.dump(myTree, fw)


def grapTree(filename):
    """
    retrieve the decision tree from file
    """
    import pickle
    with open(filename, 'rb') as fr:
        dic = pickle.load(fr)

    return dic
