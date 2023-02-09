# coding=utf-8

'''
输入: s1 = "sea", s2 = "eat"
输出: 231
解释: 在 "sea" 中删除 "s" 并将 "s" 的值(115)加入总和。
在 "eat" 中删除 "t" 并将 116 加入总和。
结束时，两个字符串相等，115 + 116 = 231 就是符合条件的最小和。

'''


class Solution(object):
    def minimumDeleteSum(self, s1, s2):
            m,n=len(s1),len(s2)
            dp=[[float('inf') for _ in range(n+1)] for i in range(m+1)]
            dp[0][0]=0
            for i in range(m):
                dp[i+1][0]=dp[i][0]+ord(s1[i])
            for j in range(n):
                dp[0][j+1]=dp[0][j]+ord(s2[j])

            for i in range(1,m+1):
                for j in range(1,n+1):
                    if s1[i-1]==s2[j-1]:
                        dp[i][j]=dp[i-1][j-1]
                    else:
                        dp[i][j]=min(dp[i-1][j]+ord(s1[i-1]),dp[i][j-1]+ord(s2[j-1]))
            return dp[-1][-1]