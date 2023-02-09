# coding=utf-8


# 注意mid的更新代码
class Solution(object):
    def findMin(self, nums):
        l,r = 0, len(nums)-1
        while l<r:
            mid = (l+r)/2
            if nums[mid] < nums[r]:
                r = mid
            else:  # nums[mid]>=nums[r]证明最小值在后面那段
                l = mid+1 # 所以要把l变大
        return nums[l]

a = [3,4,5,1,2]
# a = [1, 0, 1, 1, 1]
# a = [1, 2, 4, 5, 9]
s = Solution()
ans = s.findMin(a)
print(ans)