# coding=utf-8
# 最长斐波拉契长度  非连续
# 是个递增序列
# dp问题
# a[i][j]用于记录A[?]+A[i]是否能得到A[j]，如果从来都不行，然后第一次可以，就为3；
# 否则则为a[i][j]+1
'''
输入: [1,2,3,4,5,6,7,8]
输出: 5
解释:
最长的斐波那契式子序列为：[1,2,3,5,8]
'''

class Solution:
    def lenLongestFibSubseq(self, A):   # A是递增序列
        n, res = len(A), 0
        a = [[0] * n for i in range(n)]
        for i, v in enumerate(A):
            lo, hi = 0, i - 1
            while lo < hi:
                if A[lo] + A[hi] < v:
                    lo += 1
                elif A[lo] + A[hi] > v:
                    hi -= 1
                else:  # A[lo] + A[hi] == v
                    if a[lo][hi]:
                        a[hi][i] = a[lo][hi] + 1  # v可以添加在a[lo][hi]构成的序列后
                    else:
                        a[hi][i] = 3 # lo+hi=i
                    res = max(a[hi][i], res)
                    lo += 1
                    hi -= 1
        return res