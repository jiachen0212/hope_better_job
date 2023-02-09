# coding=utf-8
# 给一个数组，求一个index，使得分成的两部分方差的和最小
# 从左到右算一遍方差，从右到左再算一遍，然后再遍历一遍数组，得到方差和最小的位置
'''
时间复杂度：O(n)，空间复杂度 O(n)

方差概念：平方的均值减去均值的平方，即 D(X) = E(x^2) - [E(X)]^2

'''


class Solution:
    """
    @param nums: 数组
    @return: 最小方差划分的数组索引和最小方差
    """
    def minVariancePartition(self, nums):
        left = self.subVariance(nums[:])
        right = self.subVariance(nums[::-1])[::-1]
        minVariance, index = float("inf"), 0
        for i in range(1, len(right)):
            if left[i-1] + right[i] < minVariance:
                minVariance = left[i-1] + right[i]
                index = i - 1  # 更新划分的索引
        return index, minVariance

    def subVariance(self, nums):
        subVar = []
        subSum = subSquare = 0
        for i in range(len(nums)):
            subSum += nums[i]
            subSquare += nums[i] * nums[i]
            subVar.append(subSquare/(i+1) - (subSum/(i+1))**2) # 子数组方差
        return subVar

a = [3,5,11,2]
print(Solution().minVariancePartition(a)) # (1,1.25)

