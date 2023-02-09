# coding=utf-8
# leetcode ac
# 螺旋生成矩阵
# 输入n 则螺旋式的生成nxn的一个矩阵
import numpy as np


# 类似Z型打印/对角打印矩阵的思路
# 需要找规律去实现
# 分成向右边 向下 向左 向上 四条边去打印
class Solution(object):
    def generateMatrix(self, n):
        res = [[0]*n for i in range(n)]
        count = 0
        num = 1
        for i in range(n-1, 0, -2):
            for j in range(i):
                res[count][j+count] = num
                num += 1
            for j in range(i):
                res[count+j][n-1-count] = num
                num += 1
            for k in range(i+count, count, -1):
                res[n-1-count][k] = num
                num += 1
            for k in range(i+count, count, -1):
                res[k][count] = num
                num += 1
            count += 1
        if n % 2 == 1:
            res[n/2][n/2] = n**2
        return res

s = Solution()
res = s.generateMatrix(3)
ans = np.array(res)
print(ans)
