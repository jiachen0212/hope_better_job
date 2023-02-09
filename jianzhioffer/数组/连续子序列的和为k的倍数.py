# coding=utf-8
'''
输入: [23,2,6,4,7], k = 6
输出: True
解释: [23,2,6,4,7]是大小为 5 的子数组，并且和为 42
'''

# 这种问题要善用dict
'''
计算当前和对k的mod  出现mod值相等，那么这两个索引之间的和肯定是k的倍数
判断这两个index是否>=2即可

'''

# 时间复杂度O(n)
class Solution(object):
    def checkSubarraySum(self, nums, k):
        if not nums or len(nums) < 2:
            return False
        l = len(nums)
        for i in range(l-1):
            if nums[i] == nums[i+1] == 0:
                return True   # 0是任何k的0倍
        if k == 0:   # 这个写在nums连续0的判断之后 是可行的
            return False
        k = abs(k)
        mods ={}
        mods[0] = -1
        sums = 0
        for i in range(l):
            sums += nums[i]
            mod = sums%k
            if mod in mods:
                if i - mods[mod] > 1:  # 是否长度>=2
                    return True
            else:
                mods[mod] = i
        return False

s = Solution()
res = s.checkSubarraySum([1,2,3], 6)
print res