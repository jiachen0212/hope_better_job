# coding=utf-8
# 所有递增子序列

'''
输入: [4, 6, 7, 7]
输出: [[4, 6], [4, 7], [4, 6, 7], [4, 6, 7, 7], [6, 7], [6, 7, 7], [7,7], [4,7,7]]
'''


# 回溯法
class Solution:
    def findSubsequences(self, nums):
        self.res = set()  # set中装一个个递增子序列
        n = len(nums)
        def helper(ind, tmp):   # tmp是可能的一个递增子序列
            if len(tmp) > 1 and tmp not in self.res:
                self.res.add(tmp)
            for i in range(ind+1, n):
                if nums[i] >= tmp[-1]:
                    helper(i, tmp + (nums[i],))

        for i in range(n):
            helper(i, (nums[i],))
        return [list(i) for i in self.res]
