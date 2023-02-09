# coding=utf-8


class Solution(object):
    def maxArea(self, height):
        i,j = 0, len(height)-1
        res = 0
        while i<j:
            res = max(res, min(height[i], height[j]) * (j-i))
            if height[i] < height[j]:  # 左边更低，那右移左边界看看能不能出现height[i+1]>height[i]呀～
                i += 1
            else:
                j -= 1
        return res


