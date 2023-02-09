#coding=utf-8
# 实现两个数相加,但不用+-x/等四则运算符实现.. 那么就只能用位运算了...
# 首先考虑不进位的话: 11--0 10--1 01--1 00--0 所以可以抽象为是异或运算
# 然后是考虑进位: 11--1 10--0 01--0 00--0 所以可以抽象为与运算 并且得到的结果要向左移动一位..
# ok,上代码:

def add_without_jiajianchengchu(a, b):
    while b != 0:   # 有进位，就是当前的ssum和进位的结果再继续加法
        ssum = a ^ b  # 异或
        jinwei = (a & b) << 1  # 做与运算并左移一位

        # 好了现在要把ssum和jinwei的结果相加了.但是因为没有+可以用,所以这的加也只能按照前两步一样的方法来:即进行异或和进位计算,直到没有进位为止...
        a = ssum
        b = jinwei  # 所以是进行了个循环其实... 这一点很巧妙..
    return a   # b=0就是没有进位,所以和直接等于ssum

# res = add_without_jiajianchengchu(5, 3)
res = add_without_jiajianchengchu(9, 3)
print res


#### 牛客ac版
# 讨论区看到的  对python的位运算添加一些越界检查
# -*- coding:utf-8 -*-
class Solution:
    def Add(self, num1, num2):
        while num2 != 0:
            ssum = (num1 ^ num2 )& 0xFFFFFFFF
            jw = ((num1 & num2) << 1) & 0xFFFFFFFF
            num1 = ssum
            num2 = jw
        return num1 if num1 <= 0x7FFFFFFF else ~(num1 ^ 0xFFFFFFFF)