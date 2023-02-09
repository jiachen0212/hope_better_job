# coding=utf-8
# https://www.jianshu.com/p/9bf3017e2487
import numpy as np
import random
import xlrd

def getData(dataset):
    m, n = np.shape(dataset)
    traindata = np.ones((m, n))
    traindata[:, :-1] = dataset[:, :-1]    # so the last col of traindata is ones. x0=1, for the bias
    trainlabel = dataset[:, -1]
    return traindata, trainlabel


def batchgradientdescent(x, y, theta, alpha, m, maxiterations):
    # x is data, y is label, theta is weights, alpha is lr, m is samples num
    xTrains = x.transpose()
    for i in range(maxiterations):
        hypothesis = np.dot(x, theta)          # predict
        loss = hypothesis - y
        gradient = np.dot(xTrains, loss) / m   # the diff
        theta = theta - alpha * gradient       # update the weight value
    return theta

def predict(x, theta):
    m, n = np.shape(x)
    xTest = np.ones((m, n + 1))
    xTest[:, :-1] = x
    yP = np.dot(xTest, theta)   # y = theta*x
    return yP



dataset = np.array([[ 1.1,  1.5,  2.5],
 [ 1.3,  1.9,  3.2],
 [ 1.5,  2.3,  3.9],
 [ 1.7,  2.7,  4.6],
 [ 1.9,  3.1,  4.3],
 [ 2.1,  3.5,  6. ],
 [ 2.3,  3.9,  6.7],
 [ 2.5,  4.3,  7.4],
 [ 2.7,  4.7,  8.1],
 [ 2.9,  5.1,  8.8]])


traindata, trainlabel = getData(dataset)
m, n = np.shape(traindata)
theta = np.ones(n)  # initizial weight
alpha = 0.1         # lr
maxiterations = 5000
theta = batchgradientdescent(traindata, trainlabel, theta, alpha, m, maxiterations)
print theta
x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
print predict(x, theta)
