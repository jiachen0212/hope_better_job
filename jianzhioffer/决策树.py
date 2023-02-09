##### python 实现决策树
#coding:utf-8

from math import log

class DecisonTree:
    trainData = []
    trainLabel = []
    featureValus = {} #每个特征所有可能的取值

    def __init__(self, trainData, trainLabel, threshold):
        self.loadData(trainData, trainLabel)
        self.threshold = threshold
        self.tree = self.createTree(range(0,len(trainLabel)), range(0,len(trainData[0])))


    #加载数据
    def loadData(self, trainData, trainLabel):
        if len(trainData) != len(trainLabel):
            raise ValueError('input error')
        self.trainData = trainData
        self.trainLabel = trainLabel

        #计算 featureValus
        for data in trainData:
            for index, value in enumerate(data):
                if not index in self.featureValus.keys():
                    self.featureValus[index] = [value]
                if not value in self.featureValus[index]:
                    self.featureValus[index].append(value)

    #计算信息熵
    def caculateEntropy(self, dataset):
        labelCount = self.labelCount(dataset)
        size = len(dataset)
        result = 0
        for i in labelCount.values():
            pi = i / float(size)
            result -= pi * (log(pi) /log(2))
        return result

    #计算信息增益
    def caculateGain(self, dataset, feature):
        values = self.featureValus[feature] #特征feature 所有可能的取值
        result = 0
        for v in values:
            subDataset = self.splitDataset(dataset=dataset, feature=feature, value=v)
            result += len(subDataset) / float(len(dataset)) * self.caculateEntropy(subDataset)
        return self.caculateEntropy(dataset=dataset) - result

    #计算数据集中，每个标签出现的次数
    def labelCount(self, dataset):
        labelCount = {}
        for i in dataset:
            if trainLabel[i] in labelCount.keys():
                labelCount[trainLabel[i]] += 1
            else:
                labelCount[trainLabel[i]] = 1

        return labelCount

    '''
    dataset:数据集
    features:特征集
    '''
    def createTree(self, dataset, features):

        labelCount = self.labelCount(dataset)
        #如果特征集为空，则该树为单节点树
        #计算数据集中出现次数最多的标签
        if not features:
            return max(list(labelCount.items()),key = lambda x:x[1])[0]

        #如果数据集中，只包同一种标签，则该树为单节点树
        if len(labelCount) == 1:
            return labelCount.keys()[0]

        #计算特征集中每个特征的信息增益
        l = map(lambda x : [x, self.caculateGain(dataset=dataset, feature=x)], features)

        #选取信息增益最大的特征
        feature, gain = max(l, key = lambda x: x[1])

        #如果最大信息增益小于阈值，则该树为单节点树
        #
        if self.threshold > gain:
            return max(list(labelCount.items()),key = lambda x:x[1])[0]

        tree = {}
        #选取特征子集
        subFeatures = filter(lambda x : x != feature, features)
        tree['feature'] = feature
        #构建子树
        for value in self.featureValus[feature]:
            subDataset = self.splitDataset(dataset=dataset, feature=feature, value=value)

            #保证子数据集非空
            if not subDataset:
                continue
            tree[value] = self.createTree(dataset=subDataset, features=subFeatures)
        return tree

    def splitDataset(self, dataset, feature, value):
        reslut = []
        for index in dataset:
            if self.trainData[index][feature] == value:
                reslut.append(index)
        return reslut

    def classify(self, data):
        def f(tree, data):
            if type(tree) != dict:
                return tree
            else:
                return f(tree[data[tree['feature']]], data)
        return f(self.tree, data)

