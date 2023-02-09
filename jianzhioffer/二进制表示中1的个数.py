# coding=utf-8
# 使用位运算,得到输入整数的二进制表达的1的个数 如:9 1001 2个1
# 方法:把原数n与n-1做位的与运算,可消除n的最右边那个1. 迭代进行使它全部的1变成0,即完成n中1的数量统计


def fun(n):
    count = 0
    if n < 0:    # 考虑负数的情况
        n = n & 0xffffffff
    while n:
        n = n&(n-1)  # n和n-1与运算，消除n中最后边的1
        count += 1
    return count