# coding=utf-8
# 二分查找第一次出现和最后出现的位置 然后相减就是次数

class Solution(object):
    def searchRange(self, nums, target):
        if not nums:
            return [-1,-1]
        ll = len(nums)

        def left_index(nums, target):
            l, r = 0, ll-1
            while l<= r:
                mid = (l+r)/2
                if target <= nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            return l

        def right_index(nums, target):
            l,r = 0, ll-1
            while l<=r:
                mid = (l+r)/2
                if target >= nums[mid]:
                    l = mid+1
                else:
                    r = mid-1
            return r

        le, ri = left_index(nums, target), right_index(nums,target)
        return [le, ri] if (le <= ri) else [-1, -1]