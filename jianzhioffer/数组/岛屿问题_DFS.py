# coding=utf-8
# 岛屿问题
# 算是DFS的应用吧～

class Solution(object):

    def DFS(self, M, i, j, m, n):
        if (i<0 or j<0 or i>=m or j>=n):
            return
        if M[i][j] == '0':  # 遍历过了，标记为'0'
            return
        M[i][j] = '0'
        self.DFS(M, i-1, j, m, n)
        self.DFS(M, i, j-1, m, n)
        self.DFS(M, i+1, j, m, n)
        self.DFS(M, i, j+1, m, n)

    def maxAreaOfIsland(self, M):
        res = 0
        if not M or not M[0]:
            return res
        m, n = len(M), len(M[0])
        for i in range(m):
            for j in range(n):
                if M[i][j] == '0':  # 已经是遍历过的了
                    continue
                else:
                    res += 1
                    self.DFS(M, i, j, m, n)
        return res

M = [[1,1,1,1,0],
[1,1,0,1,0],
[1,1,0,0,0],
[0,0,0,0,0]]
s = Solution()
res = s.maxAreaOfIsland(M)
print(res)