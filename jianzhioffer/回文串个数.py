# coding=utf-8
# 回文串个数
# 动态规划
'''
动态规划，dp[i][j]的含义是s[i..j]是否回文
dp[i][i] = true，dp[i][i+1] =s[i] == s[i+1]
dp[i][i] = true，if dp[i+1][j-1]
dp[i][j]= false if s[i]!=s[j]

example:
'aaa':  回文个数=6   a a a aa aa aa aaa
'''

# j 一直往 i 的后方扫

class Solution(object):
    def countSubstrings(self, s):
        ll = len(s)
        count = 0
        dp = [[0]*ll for i in range(ll)]
        for i in range(ll-1, -1, -1):   # 这里注意i的取值是逆向的
        # 因为后面要用dp[i+1][j-1]
            for j in range(i, ll):
                if s[i] == s[j] and (j-i <= 2 or dp[i+1][j-1]):
                    dp[i][j] = 1
                    count += 1
        return count


s = Solution()
res = s.countSubstrings('abcb')
print(res)