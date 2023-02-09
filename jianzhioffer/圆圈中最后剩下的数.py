#coding=utf-8
import numpy as np
# 环形的n个值(0~n1),一直减去第m个数,求最后剩下的数

def last_reaining_number(n, m):
    s = [x for x in range(n)]
    p = m - 1
    while len(s) != 1:
        while p > len(s) - 1:   # 超过了尾数的index
        # 这个条件要放在最前面,为了防止p一上来就设置的大于len(s)-1, 如last_reaining_number(5, 8)
            p = p - (len(s) - 1) - 1  # -1 减1是因为index从0计数..
        popnum = s.pop(p)   # popnum是被剔除的那个数
        p += (m - 1)
    return s

res = last_reaining_number(5, 3)
# res = last_reaining_number(5, 8)
print res



#  牛客 ac 版  操作直译
# -*- coding:utf-8 -*-
class Solution:
    def LastRemaining_Solution(self, n, m):
        if n < 1 or m < 1:
            return -1
        s = [x for x in range(n)]
        p = m - 1
        while len(s) != 1:
            while p > len(s) - 1:
                p = p - (len(s) - 1) - 1
            s.pop(p)
            p += (m - 1)   # p 指向被剔除的下一位，直观的操作操作就是数值+(m-1)
        return s[0]


####### 另一ac版本  剑指上总结的公式程序
# 时间复杂度 O(n)  空间复杂度O(1)
# -*- coding:utf-8 -*-
class Solution:
    def LastRemaining_Solution(self, n, m):
        if n < 1 or m < 1:
            return -1
        s = 0
        for i in range(2, n + 1):
            s = (s + m) % i
        return s