# coding=utf-8
# dp问题

class Solution:
    def dp(self, i, preSum, a, maxSum):
        if preSum <= 0 or i == 0:
            cursum = a  # 之前的和<0的话 摒弃
        if i > 0 and preSum > 0:
            cursum = preSum + a
        maxSum = cursum if cursum > maxSum else maxSum
        return cursum, maxSum

    def maxSubArray(self, array):
        if not array:
            return
        preSum, maxSum = array[0], array[0]
        for i in range(len(array)):
            preSum, maxSum = self.dp(i, preSum, array[i], maxSum)
        return maxSum




