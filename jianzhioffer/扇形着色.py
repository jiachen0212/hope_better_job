# coding=utf-8
# 阿里算法题
import numpy as np

def circle_divide(n, m):
    # n: 扇形个数
    # m: 颜色种类
    if n == 1:
        return m
    elif n == 2:
        return m*(m - 1)
    return m * np.power(m - 1, n - 1) - circle_divide(n - 1, m)


res = circle_divide(4, 3)
print(res)