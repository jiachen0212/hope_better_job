# coding=utf-8
# 求众数
'''
给定一个大小为 n 的数组，找出其中所有出现超过 ⌊ n/3 ⌋ 次的元素。
说明: 要求算法的时间复杂度为 O(n)，空间复杂度为 O(1)。

示例 1:
输入: [3,2,3]
输出: [3]

示例 2:
输入: [1,1,1,3,3,2,2,2]
输出: [1,2]

首先可以明确的一点是，这样的元素可能有0个、1个、或者2个，再没有别的情况了.

'''

class Solution(object):
    def majorityElement(self, nums):
        res = []
        if not nums or len(nums) < 1:
            return res
        llen = len(nums)
        maj1, maj2 = nums[0], nums[0]
        con1, con2 = 0, 0
        for num in nums:
            if num == maj1:
                con1 += 1
            elif num == maj2:
                con2 += 1
            elif con1 == 0:
                maj1 = num
                con1 = 1
            elif con2 == 0:
                con2 = 1
                maj2 = num
            else:
                con1 -= 1
                con2 -= 1
        # 以上代码是找出数组中出现次数最多的两个数  确定maj1和maj2
        # 这个和之前京东问我的，找出数组中出现一半以上数的思想是一样的
        con1, con2 = 0, 0
        for num in nums:
            if num == maj1:
                con1 += 1
            elif num == maj2:
                con2 += 1
        if con1 > llen / 3:
            res.append(maj1)
        if con2 > llen / 3 and maj2 != maj1:
            res.append(maj2)
        return res

s = Solution()
res = s.majorityElement([4,2,1,1])
print res
