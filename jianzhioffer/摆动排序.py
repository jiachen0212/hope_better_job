# coding=utf-8
# 摆动排序
# nums[0] < nums[1] > nums[2] < nums[3]...
# https://leetcode-cn.com/problems/wiggle-sort-ii/
# 先排序，再穿插

class Solution(object):
     def wiggleSort(self, nums):
        nums.sort()
        half = len(nums[::2])
        nums[::2], nums[1::2] = nums[:half][::-1], nums[half:][::-1]
        return nums

s = Solution()
ans = s.wiggleSort([1,5,1,1,6,4])
print(ans)