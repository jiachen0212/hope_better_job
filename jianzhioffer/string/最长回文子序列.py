# coding=utf-8
# 最长回文子序列

'''
神奇在于：bbbab的回文长度是4！！！
'''

# 还是动态规划做
'''
dp[i][j]: i到j回文串长度
dp[i][i] = 1
当s[i] = s[j]  则dp[i][j] = dp[i+1][j-1]+2   这里没有规定j-i<2
所以就是允许bbbab这种情况的...
当s[i] != s[j]
dp[i][j]=max(dp[i+1][j-1], dp[i][j-1])

 由于dp[i][j]需要dp[i+1][j]所以需要逆序枚举s的长度，
 而又因为j是递增的，所以在求解dp[i][j]时,dp[i][j-1]肯定已经求解过了

'''
class Solution(object):
    def longestPalindromeSubseq(self, s):
        if s == '':
            return 0
        n = len(s)
        dp = [[0]*n for i in range(n)]
        for i in range(n):
            dp[i][i] = 1
        for i in range(n-1, -1, -1):
            for j in range(i+1, n):   # i在前，j在后
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1])
        return dp[0][-1]


s = Solution()
res = s.longestPalindromeSubseq('bbbab')
print(res)
