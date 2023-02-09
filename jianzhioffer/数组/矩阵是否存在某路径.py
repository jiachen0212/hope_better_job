#coding=utf-8
# 矩阵中的路径
############# 牛客 ac

class Solution:
    def hasPath(self, matrix, row, col, path): # path为字符串路径
        self.col, self.row = col, row
        matrix = [list(matrix[col * i:col * i + col]) for i in range(row)]  # 矩阵转list
        for i in range(row):
            for j in range(col):
                if matrix[i][j] == path[0]:  # 首先第一个路径char得在矩阵中吧..
                    self.b = False  # 表示该位置已占用
                    self.search(matrix, path[1:], [(i, j)], i, j) # i,j表示当前找到的位置
                    if self.b:
                        return True
        return False

    def search(self, matrix, path, visited, i, j):
        if path == "":
            self.b = True
            return
        # path[0],始终代表当前需要找的路径的第一个值
        if j != 0 and (i, j - 1) not in visited and matrix[i][j - 1] == path[0]: # 左
            self.search(matrix, path[1:], visited + [(i, j - 1)], i, j - 1)
        if i != 0 and (i - 1, j) not in visited and matrix[i - 1][j] == path[0]: # 上
            self.search(matrix, path[1:], visited + [(i - 1, j)], i - 1, j)
        if j != self.col - 1 and (i, j + 1) not in visited and matrix[i][j + 1] == path[0]: # 右
            self.search(matrix, path[1:], visited + [(i, j + 1)], i, j + 1)
        if i != self.row - 1 and (i + 1, j) not in visited and matrix[i + 1][j] == path[0]: # 下
            self.search(matrix, path[1:], visited + [(i + 1, j)], i + 1, j)

# test
s = Solution()
matrix = 'abtgcfysjdeh'
ss = s.hasPath(matrix, 3, 4, "acfd")
print ss