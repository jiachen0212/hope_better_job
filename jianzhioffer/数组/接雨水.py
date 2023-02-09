# coding=utf-8
# 接雨水   diff
'''
你是怎么想到从左边遍历加从右边遍历等于总面积加原来图像加雨水数的...
'''
class Solution(object):
    def trap(self, height):
        ans = 0
        h1 = 0
        h2 = 0
        for i in range(len(height)):
            h1 = max(h1,height[i])
            h2 = max(h2,height[-i-1])
            ans = ans + h1 + h2 -height[i]
        return  ans - len(height)*h1
