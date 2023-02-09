# coding=utf-8
# 012乱序数组 要求O(n)时复  O(1)空复排序

'''
三指针快排：用三个指针，第一个指针指向最左边的非0，第二个指针用来遍历数组，
第三个指针指向最右边的非2元素。遍历到0，和第一个指针互换元素，遍历到2，和第三个指针互换元素，
遍历到1，就不换，接着往后走。
'''
nums = [2,2,0,0,0,0,0,0,2,2,2,2,2,2,1,1,1,1,0,0,1,1,0,2]
# leetcode ac
class Solution(object):
    def sortColors(self, nums):
        i, j, k = 0, len(nums) - 1, 0
        while k < len(nums):
            if nums[k] == 0 and k > i:
                nums[k], nums[i] = nums[i], nums[k]
                i += 1
            elif nums[k] == 2 and k < j:
                nums[k], nums[j] = nums[j], nums[k]
                j -= 1
            else:
                k += 1
s = Solution()
s.sortColors(nums)
print nums

