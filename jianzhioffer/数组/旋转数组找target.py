# coding=utf-8
# leetcode 33

class Solution(object):
    def search(self, nums, target):
        if not nums:
            return -1
        ll = len(nums)
        l,r = 0, ll-1
        while l<=r:
            while l<r and nums[l]==nums[l+1]: # 因为数组可以有重复的数
                l += 1
            while l<r and nums[r]==nums[r-1]:
                r -= 1
            mid = (l+r)/2

            if nums[mid] == target:
                return mid  # return mid就是返回target的下标

            # mid 和 l  比较  得到两边哪一边有序
            if nums[mid]>=nums[l]:  # 左有序
                # target在有序的段内
                if target < nums[mid] and target >= nums[l]:
                    r = mid - 1
                else: # 不在有序的段内，也就是在右边咯，那把l加大
                    l = mid + 1
            else:  # 右有序
                # 在有序的段内
                if target > nums[mid] and target <= nums[r]:  # target在有序的这段
                    l = mid + 1
                else:   # 不在，那就是在左段咯，把r减少
                    r = mid - 1
        return -1


s = Solution()
print(s.search([4,5,6,7,0,1,2], 3))
