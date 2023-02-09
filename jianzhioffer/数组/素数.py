# coding=utf-8
# 1000以内的最大素数/质数

'''
从上到下，从999开始往下搜，搜到就停，每次-2。
bool判断，从2到根号x开始，全求模，非0跳出False。
'''
import numpy as np
import time


# 判断n是否素数的help函数
def help(n):
    if n % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(n)), 2):  # 不能被2整除,那接着看3～√n 且每次+2
        if n % i == 0:
            return False
    return True

def main(num):
    for i in range(num-1, 1, -1):
        if help(i):
            return i

# time_start=time.time()
print main(18)
# time_end=time.time()
# print('time cost',time_end-time_start,'s')