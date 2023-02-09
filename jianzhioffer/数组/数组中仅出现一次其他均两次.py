# coding=utf-8
# 数组中其他数都出现两次，仅一个出现一次，求出这个数
# 方法：做位运算异或


class Solution(object):
    def singleNonDuplicate(self, nums):
        res = nums[0]
        for i in range(1,len(nums)):
            res ^= nums[i]
        return res