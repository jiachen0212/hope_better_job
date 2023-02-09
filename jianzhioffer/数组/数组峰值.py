# coding=utf-8
# 求数组的峰值  这个值大于其前面的所有值 小于其后面的所有值
# 很直接的二分法，但是注意 l r 的变化条件
# 时复 logn
nums = [1,3,5,7,16,9,8,10]

def fengIndex(nums):
    l,r = 0, len(nums)-1
    while l<=r:
        mid = (l+r)/2
        if nums[mid]<nums[mid-1]:  # mid选大了，数组已经在递减了
            r = mid
        elif nums[mid]<nums[mid+1]: # mid选小了，数组还在递增
            l = mid
        else:
            return mid
print fengIndex(nums), '+++'



# 进阶
# 峰值 就是该值大于左右相邻的元素即可
class Solution(object):
    def findPeakElement(self, nums):
        l = len(nums)
        l,r = 0,l-1
        while l<r:
            mid = (l+r)/2
            if nums[mid]>nums[mid+1]:
                r = mid
            else:  # nums[mid]<=nums[mid+1]
                l = mid+1
        return l


s = Solution()
res = s.findPeakElement([1,2,1,3,5,6,4])
print res
