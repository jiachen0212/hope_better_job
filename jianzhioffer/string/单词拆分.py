# coding=utf-8
# 单词拆分  dp问题
class Solution(object):
    def wordBreak(self, s, word_dicts):
        l = len(s)
        dp = [False]*(l+1)  # dp[i]表示前i位可以被分割处dicts中的单词
        dp[0] = True
        for i in range(1, l+1):
            for j in range(i):
                if dp[j] and s[j: i] in word_dicts:
                    dp[i] = True
                    break
        return dp[-1]

s = Solution()
res = s.wordBreak("leetcode", ["leet", "code"])
print(res)
