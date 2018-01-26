import numpy as np


def loadDataSet():
    """
    load the test dataSet
    """

    posts = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
             ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
             ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
             ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
             ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
             ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classVec = [0, 1, 0, 1, 0, 1]
    return posts, classVec


def createVocabList(dataSet):
    """
    create a vocabulary list from input data
    """
    vocabSet = set([])
    for doc in dataSet:
        vocabSet = vocabSet | set(map(str.lower, doc))
    return list(vocabSet)


def createVocabVec(vocabList, inputData):
    """
    transform inputData to a N dimention vector
    with vocabList(N is the dimention of vocabList)
    """

    wordVect = [0] * len(vocabList)

    for w in inputData:
        w = str.lower(w)
        if w in vocabList:
            wordVect[vocabList.index(w)] = 1
        else:
            pass
    return wordVect


def trainNB(wordVectMatrix, classLabel):
    """
    train a binary classification
    byes model with inputdata

    p0Prob-- the probability correspond to each word in vocabList to be label0
    p1Prob-- the probability correspond to each word in vocabList to be label1
    pSam  -- the probability of being a spam
    """

    nDocs = len(wordVectMatrix)
    nWords = len(wordVectMatrix[0])
    pSpam = np.mean(classLabel)
    # p0Count = np.zeros(nWords)
    # p1Count = np.zeros(nWords)
    # p0All = 0.0
    # p1All = 0.0
    # avoid multiplying zero
    p0Count = np.ones(nWords)
    p1Count = np.ones(nWords)
    p0All = 2.0
    p1All = 2.0

    for i in range(nDocs):
        if classLabel[i] == 1:
            p1Count += wordVectMatrix[i]
            p1All += np.sum(wordVectMatrix[i])
        else:
            p0Count += wordVectMatrix[i]
            p0All += np.sum(wordVectMatrix[i])
    # p0Prob = p0Count/p0All
    # p1Prob = p1Count/p1All
    # avoid the probability too small
    p0Prob = np.log(p0Count/p0All)
    p1Prob = np.log(p1Count/p1All)

    return pSpam, p0Prob, p1Prob


def classifyNB(wordVect, p0Prob, p1Prob, pSpam):
    """
    classify the wordvect
    """

    p1 = np.sum(wordVect * p1Prob) - np.log(pSpam)
    p0 = np.sum(wordVect * p0Prob) - np.log(1-pSpam)

    if p1 > p0:
        return 1
    else:
        return 0


def testNB():
    posts, labels = loadDataSet()
    myVocabList = createVocabList(posts)
    trainMatrix = []
    for doc in posts:
        trainMatrix.append(createVocabVec(myVocabList, doc))
    pSpam, p0Prob, p1Prob = trainNB(trainMatrix, labels)
    testDoc = ['love', 'my', 'dalmation']
    thisDoc = np.array(createVocabVec(myVocabList, testDoc))
    print('This doc is classified as: %d' \
        % classifyNB(thisDoc, p0Prob, p1Prob, pSpam))
    testDoc = ['stupid', 'garbage']
    thisDoc = np.array(createVocabVec(myVocabList, testDoc))
    print('This doc is classified as: %d' \
        % classifyNB(thisDoc, p0Prob, p1Prob, pSpam))
