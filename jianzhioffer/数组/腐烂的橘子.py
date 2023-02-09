# coding=utf-8

# BFS
# 注意time的叠加手法！！！

class Solution:
    def orangesRotting(self, grid):
        m,n = len(grid),len(grid[0])
        time = 0
        ners = [[-1,0],[0,-1],[0,1],[1,0]]
        stack = []
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    stack.append((i,j,0))
        while stack:
            i,j,time = stack.pop(0)
            for ner in ners:
                x,y = i+ner[0],j+ner[1]
                if 0 <=x<m and 0<=y<n and grid[x][y]==1:
                    grid[x][y] = 2
                    stack.append((x,y,time+1))
        for g in grid:
            if 1 in g:
                return -1
        return time

s = Solution()
M = [[2,1,1], [1,1,0], [0,1,1]]
print s.orangesRotting(M)

