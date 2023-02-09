# coding=utf-8
# 寻找数组的最短连续子数组的长度，使得子数组的和大于等于t
# https://github.com/Shellbye/Shellbye.github.io/issues/41


class Solution(object):
    def shortestSubarray(self, A, K):
        minLin = len(A) + 1
        presum = [0]*minLin
        for i in range(minLin-1):
            presum[i+1] = presum[i] + A[i]

        queue = []  # 存放连续子序列的index
        for i in range(len(A)+1):   # i肯定是比当前的queue中的所有index都大的
            while queue and presum[i] <= presum[queue[-1]]:
                queue.pop()  # 前面出现负,把前面的都依次pop掉
            while queue and presum[i] - presum[queue[0]] >= K:
                res = i - queue[0]
                minLin = res if res < minLin else minLin
                queue.pop(0)  # 把更早的一些可以删除，使得子序列最短
            queue.append(i)
        return minLin if minLin < len(A)+1 else -1

s = Solution()
res = s.shortestSubarray([2,-1,2], 3)
print res