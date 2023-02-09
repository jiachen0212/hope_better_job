# coding=utf-8
##  左旋转字符串


def LeftRotateString(s, n):
    s1 = s[:n]
    s2 = s[n:]
    l1 = list(s1)
    l2 = list(s2)
    l1.reverse()   # reverse() 函数无return 直接原地改list
    l2.reverse()
    res = l1 + l2
    res.reverse()
    return ''.join(res)

s = 'abcdefg'
res = LeftRotateString(s, 2)
print(res)



######### 牛客 ac 版
# -*- coding:utf-8 -*-
class Solution:
    def LeftRotateString(self, s, n):
        if len(s) == 0:
            return ""
        if len(s) < n:
            l = list(s)
            l.revevse()
            return ''.join(l)
        s1 = s[:n]
        s2 = s[n:]
        l1 = list(s1)
        l2 = list(s2)
        l1.reverse()  # reverse() 函数无return 直接原地改list
        l2.reverse()
        l = l1 + l2
        l.reverse()
        return ''.join(l)



####### 变态牛客 ac 版
# -*- coding:utf-8 -*-
class Solution:
    def LeftRotateString(self, s, n):
        return s[n:]+s[:n]   # 不服不行!!!
