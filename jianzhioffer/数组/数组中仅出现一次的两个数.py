#coding=utf-8
import sys
# 找到数组中仅出现一次的两个数(其他的数都成对出现)
# 方法: 如果一个list中仅有一个数只出现一次,其他均是成对出现.则把这个list做组内异或就可得到这个数.
# 所以采用的方法是: 把含两个仅出现一次的数组分成两个子组,使得每个子组中仅含一个只出现一次的数.
# 怎么分组? 就是先对大list做异或,可得到两个不同的,仅出现一次的,数的,异或结果(其他成对出现的异或都变成0了,所以结果仅体现为这两个不同数的异或结果)
# 因为是异或,所以二进制的表达中,位上的元素不相等则得到结果1. 所以分组准则就是:
# 我们求出以上异或结果的,最右边的那个1出现的位置,记为index.然后对大list中的每个值进行判断,
# 如果index位上是1,则分在list1中,否则分在list2中..
# 这样仅出现一次的两个数必定是被分到不同的两个组的了..
# 然后就可以依次对这两个组进行组内异或,得到这两个值..



# 牛客 ac 版
# -*- coding:utf-8 -*-
import sys
# 使用异或技巧
class Solution:
    # 组内异或
    def yihuo(self, s):
        res = 0
        for i in range(len(s)):
            res ^= s[i]  # ^ 是异或运算符
        return res
    # 寻找二进制表示中最右1的index
    def findFirstBit1(self, a):
        bitind = 0
        while a & 1 == 0 and bitind < 8 * sys.getsizeof(int):
            bitind += 1
            a = a >> 1
        return bitind
    # 判断index位置上是否是1
    def Isbit1(self, a, ind):
        a = a >> ind
        return a & 1
    # main 函数
    def FindNumsAppearOnce(self, array):
        if array == None:
            return
        xor = self.yihuo(array)   # 整个大数组异或先
        bit1ind = self.findFirstBit1(xor)   # 找到最右的那个1的index
        l1 = []
        l2 = []
        for i in range(len(array)):
            # 下面就进行分组了
            if self.Isbit1(array[i], bit1ind):  # 如果指定的bit1ind位是1,则存在l1中
                l1.append(array[i])
            else:
                l2.append(array[i])
        n1 = self.yihuo(l1)
        n2 = self.yihuo(l2)
        return (n1, n2)


