# coding=utf-8
'''
输入：[2,1,4,7,3,2,5]
输出：5
解释：最长的 “山脉” 是 [1,4,7,3,2]，长度为 5
'''

# O(n)遍历一次的方法
nums = [2,1,4,7,3,2,5]

class Solution(object):
    def longestMountain(self, nums):
        if not nums or len(nums)<3:
            return 0
        l = len(nums)
        res = 0
        count = 0
        decrease = False
        for i in range(l-1):
            if count != 0 and nums[i+1]<nums[i]:
                decrease = True
                count += 1   # 递减阶段
            if not decrease: # 递增阶段
                count = count + 1 if nums[i+1]>nums[i] else 0
            # elif 代表decrease是True 处理递减阶段
            elif nums[i] <= nums[i+1]:  # 好呀递减阶段要结束了
                decrease = False
                res = count+1 if count+1>res else res
                count = 0 if nums[i+1]==nums[i] else 1  # 更新count
            if decrease and nums[i]>nums[i+1]:  # count递减阶段
                res = count+1 if count+1>res else res
        return res



s = Solution()
res = s.longestMountain(nums)
print res






