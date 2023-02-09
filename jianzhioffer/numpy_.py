# coding=utf-8
# numpy

import numpy as np

arr = np.eye(5)
arr1 = np.identity(5)   # 两个都是生成对角线为1的矩阵 其他元素为0
print arr
print '----'
print arr1


# import random
arr2 = np.random.rand(3,4)
print '----'
print arr2
print '----'
# arr2[:, 2] = 0   # 某一列快速置0
arr2[2, :] = 0   # 某一行快速置0
print arr2