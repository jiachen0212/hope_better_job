# coding=utf-8
# 两数组，最长重复子串
# dp 做
'''
3 2 1 4 7
1 0 0 1 0 0
2 0 1 0 0 0
3 1 0 0 0 0
2 0 2 0 0 0
1 0 0 3 0 0    对角add
'''

class Solution(object):
    def findLength(self, A, B):
        la = len(A)
        lb = len(B)
        dp = [[0]*(lb+1) for i in range(la+1)]
        for i in range(1, la+1):
            for j in range(1, lb+1):
                if A[i-1] == B[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
        return max(max(hang) for hang in dp)



s = Solution()
ans = s.findLength([1,2,3,2,1],[3,2,1,4,7])
print(ans)