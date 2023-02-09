# coding=utf-8
# leetcode 189
'''
输入: [1,2,3,4,5,6,7] 和 k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右旋转 1 步: [7,1,2,3,4,5,6]
向右旋转 2 步: [6,7,1,2,3,4,5]
向右旋转 3 步: [5,6,7,1,2,3,4]

'''
class Solution(object):
    def rotate(self, nums, k):
        nums[:] = nums[-k%len(nums): ] + nums[: -k%len(nums)]
        return nums