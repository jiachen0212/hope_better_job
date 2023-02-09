# coding=utf-8
# 字符串中的最长回文串

# 动态规划做
# dp[i][j]: 子串i~j是否回文 1代表是 0代表不是
class Solution:
    def longestPalindrome(self,s):
        if s == '':
            return ''
        n = len(s)
        dp = [[0]*n for i in range(n)]
        maxlen = 1
        res = s[0]
        for i in range(n):
            for j in range(i, -1, -1):  # j在前，i在后
                if s[i] == s[j] and (i-j<2 or dp[i-1][j+1]):
                    dp[i][j] = 1
                if dp[i][j] and i-j+1 > maxlen:
                    maxlen = i-j+1
                    res=s[j:i+1]
        return res, maxlen


s = Solution()
sub_str, maxlen = s.longestPalindrome('cbba')
print(sub_str, maxlen)







