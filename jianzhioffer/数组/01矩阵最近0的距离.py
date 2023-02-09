# coding=utf-8
# 01 矩阵 每个位置到最近0的距离
# 网易考察的算法题
# 典型BFS

class Solution(object):
    def updateMatrix(self, matrix):
        m = len(matrix)
        n = len(matrix[0])
        ners = [[-1,0],[1,0],[0,1],[0,-1]]
        queue = []
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    queue.append([i,j])
                else:
                    matrix[i][j] = m+n  # 如果矩阵值是1，给成最大值target一下
        while queue:
            cur = queue.pop(0)
            for ner in ners:
                x,y=cur[0]+ner[0],cur[1]+ner[1]
                # 因为xy 和 cur 是相邻的，所以value的理论距离是1
                # 那 matrix[x][y] > matrix[cur[0]][cur[1]]+1 则说明xy位置上值为1  所以xy位置上距离值在cur上+1
                if x>=0 and y>=0 and x<m and y<n and matrix[x][y] > matrix[cur[0]][cur[1]]+1:
                    matrix[x][y] = matrix[cur[0]][cur[1]] + 1
                    queue.append([x,y])
        return matrix


M = [[0,0,0],[0,1,0],[1,1,1]]
s = Solution()
res = s.updateMatrix(M)
print res

