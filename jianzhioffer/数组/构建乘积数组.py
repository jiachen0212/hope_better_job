# -*- coding:utf-8 -*-
# 牛客 ac 版本
class Solution:
    def multiply(self, A):
        if A == None:
            return
        l = len(A)
        if l == 1:
            return [0]
        if l > 1:
            p1 = p2 = 1
            ans = [1 for i in range(l)]
            for i in range(l):
                ans[i] *= p1
                p1 *= A[i]
                ans[-1-i] *= p2
                p2 *= A[-1-i]
            return ans

