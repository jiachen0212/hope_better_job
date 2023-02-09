# coding=utf-8
# 貌似么有ac

# 最短，那就想到bfs
class Solution(object):
    def shortestPathBinaryMatrix(self, grid):
        m = len(grid)
        n = len(grid[0])
        ners = [[1,1],[0,1],[1,0]]
        visited = [[0]*n for i in range(m)]
        if grid[0][0] == 1 or grid[-1][-1] == 1:
            return -1
        res = {}
        queue = []
        queue.append([0,0])
        res[(0,0)] = 0
        visited[0][0]=1

        while queue:
            cur = queue.pop()
            if cur == [m-1,n-1]:
                return res[(cur[0], cur[1])]
            for ner in ners:
                x = cur[0]+ner[0]
                y = cur[1]+ner[1]
                if x>=0 and y>=0 and x<m and y<n and not grid[x][y] and not visited[x][y]:
                    queue.append([x,y])
                    res[(x,y)] = res[(cur[0], cur[1])]+1
                    visited[x][y] = 1
        return -1


M = [[0,1,0,0,0,0],[0,1,1,1,1,1],[0,0,0,0,1,1],[0,1,0,0,0,1],[1,0,0,1,0,1],[0,0,1,0,1,0]]
s = Solution()
print s.shortestPathBinaryMatrix(M)