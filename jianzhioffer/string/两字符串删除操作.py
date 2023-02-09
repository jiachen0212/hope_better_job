# coding=utf-8

# 两list最长公共子序列改版题   非连续子序列

# dp 做
'''
输入: "sea", "eat"
输出: 2
解释: 第一步将"sea"变为"ea"，第二步将"eat"变为"ea"
'''

class Solution(object):
    def minDistance(self, word1, word2):
        l1 = len(word1)
        l2 = len(word2)
        dp = [(l2+1)*[0] for i in range(l1+1)]

        for i in range(1, l1+1):
            for j in range(1, l2+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return l1 - dp[-1][-1] + l2 - dp[-1][-1]